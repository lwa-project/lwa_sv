
from AdpCommon  import *
from AdpConfig  import *
from AdpLogging import *

import MCS2
from DeviceMonitor import ROACH2Device
from ConsumerThread import ConsumerThread
from SequenceDict import SequenceDict
from ThreadPool import ThreadPool
from ThreadPool import ObjectPool
#from Cache      import threadsafe_lru_cache as lru_cache
from Cache      import lru_cache_method
from AdpRoach   import AdpRoach
from iptools    import *

from Queue import Queue
import numpy as np
import time
import math
from collections import defaultdict
import logging
import struct
import subprocess
import datetime
import zmq
# Note: paramiko must be pip installed (it's also included with fabric)
import paramiko # For ssh'ing into roaches to call reboot

__version__    = "0.2"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

# Global shared resources
#_g_thread_pool = ThreadPool()
_g_zmqctx      = zmq.Context()

def wait_until_utc_sec(utcstr):
	cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)
	while cur_time != utcstr:
		time.sleep(0.01)
		cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)

class SlotCommandProcessor(object):
	def __init__(self, cmd_code, cmd_parser, exec_delay=2):
		self.cmd_sequence = defaultdict(list)
		self.exec_delay   = exec_delay
		self.cmd_code     = cmd_code
		self.cmd_parser   = cmd_parser
	def process_command(self, msg):
		assert( msg.cmd == self.cmd_code )
		exec_slot = msg.slot + self.exec_delay
		self.cmd_sequence[exec_slot].append(self.cmd_parser(msg))
		return 0
	def execute_commands(self, slot):
		try:
			cmds = self.cmd_sequence.pop(slot)
		except KeyError:
			return
		return self.execute(cmds)

class TbnCommand(object):
	def __init__(self, msg):
		self.freq, self.filt, self.gain \
		    = struct.unpack('>fhh', msg.data)
		# TODO: Check allowed range of freq
		assert( 1 <= self.filt <= 11 )
		assert( 0 <= self.gain <= 30 )
class Tbn(SlotCommandProcessor):
	def __init__(self, config, log, roaches):
		SlotCommandProcessor.__init__(self, 'TBN', TbnCommand)
		self.config  = config
		self.log     = log
		self.roaches = roaches
		self.cur_freq = self.cur_filt = self.cur_gain = 0
	#def startup(self):
	#	self.config['tbn']['capture_bandwidth']
	def start(self, freq=50e6, filt=1, gain=1, record=True):
		self.log.info("Starting TBN: freq=%f,filt=%i,gain=%i"
		              % (freq,filt,gain))
		# TODO: How/where to apply bandwidth filter?
		#       How/where to apply gain bitshift?
		#       Need to send command to pipelines to begin new
		#         observation with this info.
		#         Only do this when record==True
		bw = self.config['tbn']['capture_bandwidth']
		# TODO: Check whether pausing the data flow is necessary
		#self.roaches.disable_tbn()
		#time.sleep(1)
		self.roaches.tune_tbn(freq, bw)
		rets = self.roaches.enable_tbn_data()
		self.cur_freq = freq
		self.cur_filt = filt
		self.cur_gain = gain
		return rets
	def execute(self, cmds):
		for cmd in cmds:
			self.start(cmd.freq, cmd.filt, cmd.gain)
	def stop(self):
		self.log.info("Stopping TBN")
		self.roaches.disable_tbn_data()
		self.log.info("TBN stopped")
		return 0

class DrxCommand(object):
	def __init__(self, msg):
		self.tuning, self.freq, self.filt, self.gain \
		    = struct.unpack('>BfBh', msg.data)
		#assert( 1 <= self.tuning <= 2 ) # TODO: Check upper limit
		assert( 1 <= self.tuning <= 1 ) # TODO: Check upper limit
		# TODO: Check allowed range of freq
		assert( 0 <= self.filt   <= 8 )
		assert( 0 <= self.gain   <= 15 )
class Drx(SlotCommandProcessor):
	def __init__(self, config, log, roaches):
		SlotCommandProcessor.__init__(self, 'DRX', DrxCommand)
		self.config  = config
		self.log     = log
		self.roaches = roaches
		self.ntuning = 1
		self.cur_freq = [0]*self.ntuning
		self.cur_filt = [0]*self.ntuning
		self.cur_gain = [0]*self.ntuning
	def execute(self, cmds):
		for cmd in cmds:
			tuning = cmd.tuning-1 # Convert from 1-based to 0-based
			cfreq  = cmd.freq
			# TODO: How to deal with multiple smaller tunings?
			#       How/where to apply bandwidth filter?
			#       How/where to apply gain bitshift?
			#       Need to send command to pipelines to begin new
			#         observation with this info.
			bw = self.config['drx']['capture_bandwidth']
			# TODO: Check whether pausing the data flow is necessary
			#self.roaches.disable_drx()
			#time.sleep(1)
			self.roaches.tune_drx(cfreq, bw)
			self.roaches.enable_drx_data()
			self.cur_freq[tuning] = cmd.freq
			self.cur_filt[tuning] = cmd.filt
			self.cur_gain[tuning] = cmd.gain
	def stop(self):
		self.roaches.disable_drx_data()
		return 0
"""
class DrxCommand(object):
	def __init__(self, msg):
		#self.beam = int(ord(data[0])) # Note: IGNORED PARAMETER
		self.tuning  = int(ord(data[1]))
		self.freq    = struct.unpack('>f', (data[2:6]))[0]
		self.filt    = int(ord(data[6]))
		self.gain    = struct.unpack('>h', (data[7:9]))[0]
		self.subslot = int(ord(data[9]))
class Drx(object):
	def __init__(self):
		self.exec_delay = 2
		self.cmd_stream = defaultdict(lambda : defaultdict(list))
	def process_command(self, msg):
		assert( msg.cmd == 'DRX' )
		exec_slot = msg.slot + self.exec_delay
		self.cmd_stream[exec_slot][subslot].append(DRXCommand(msg))
	def execute_commands(self, slot):
		subslots = self.cmd_stream[slot]
		for subslot in sorted(subslots.keys()):
			cmds = subslots[subslot]
			# TODO: Compute updated channel mask
			#       Send channel mask to roaches
		del self.cmd_stream[slot]
"""

"""
class FstCommand(object):
	def __init__(self, msg):
		self.index = int(struct.unpack('>h', (msg.data[0:2]))[0])
		self.coefs = np.ndarray((16,32), dtype='>h', buffer=msg.data[2:])
class Fst(object):
	def __init__(self, config, log,
	             nupdate_save=5):
		self.config = config
		self.log    = log
		hosts = config['server_hosts']
		ports = config['fst']['control_ports']
		self.addrs = ['tcp://%s:%i'%(hosts[i/2],ports[i%2]) \
		              for i in xrange(len(hosts)*len(ports))]
		self.socks = ObjectPool([_g_zmqctx.socket(zmq.REQ) \
		                         for _ in self.addrs])
		for sock,addr in zip(self.socks,self.addrs):
			try: sock.connect(addr)
			except zmq.error.ZMQError:
				self.log.error("Invalid or non-existent address: %s" %
				                   addr)
				# TODO: How to bail out?
		self.exec_delay = 2
		self.cmd_sequence = defaultdict(list)
		self.fir_coefs = SequenceDict(lambda : np.ones((NSTAND,NPOL,
		                                                FIR_NFINE,FIR_NCOEF),
		                                               dtype=np.int16),
		                              maxlen=nupdate_save)
		self.fir_coefs[0][...] = self._load_default_fir_coefs()
		#self.future_pool = FuturePool(len(self.socks))
	def _load_default_fir_coefs(self):
		nfine = self.fir_coefs[-1].shape[-2]
		ncoef = self.fir_coefs[-1].shape[-1]
		fir_coefs = np.fromfile(self.config['fst']['default_coeffs'],
		                        dtype='>h').reshape(nfine,ncoef)
		return fir_coefs[None,None,:,:]
	def process_command(self, msg):
		assert( msg.cmd == 'FST' )
		exec_slot = msg.slot + self.exec_delay
		self.cmd_sequence[exec_slot].append(FstCommand(msg))
	def execute_commands(self, slot):
		try:
			cmds = self.cmd_sequence.pop(slot)
		except KeyError:
			return
		# Start with current coefs
		self.fir_coefs[slot][...] = self.fir_coefs.at(slot-1)
		# Merge updates into the set of coefficients
		for cmd in cmds:
			if cmd.index == -1:
				self.fir_coefs[slot][...] = self._load_default_fir_coefs()
			elif cmd.index == 0:
				# Apply same coefs to all inputs
				self.fir_coefs[slot][...] = cmd.coefs[None,None,:,:]
			else:
				stand = (cmd.index-1) / 2
				pol   = (cmd.index-1) % 2
				self.fir_coefs[slot][stand,pol] = cmd.coefs
		self._send_update(slot)
	def get_fir_coefs(self, slot):
		# Access history of updates
		return self.fir_coefs.at(slot)
	def _send_update(self, slot):
		weights = get_freq_domain_filter(self.fir_coefs[slot])
		# weights: [stand,pol,chan] complex64
		weights = weights.transpose(2,0,1)
		# weights: [chan,stand,pol] complex64
		weights /= weights.max() # Normalise to max DC gain of 1.0
		# Send update to pipelines
		# Note: We send all weights to all servers and let them extract
		#         the channels they need, rather than trying to keep
		#         track of which servers have which channels from here.
		# TODO: If msg traffic ever becomes a problem, could probably
		#         use fp16 instead of fp32 for these.
		#hdr  = struct.pack('@iihc', slot, NCHAN, NSTAND, NPOL)
		hdr = json.dumps({'slot':   slot,
		                  'nchan':  NCHAN,
		                  'nstand': NSTAND,
		                  'npol':   NPOL})
		data = weights.astype(np.complex64).tobytes()
		msg  = hdr+data

		self.socks.send_multipart([hdr, data])
		replies = self.socks.recv_json()
		#def send_msg(sock):
		#	sock.send_multipart([hdr, data])
		#	# TODO: Add receive timeout
		#	return sock.recv_json()
		#for sock in self.socks:
		#	self.future_pool.add_task(send_msg, sock)
		#replies = self.future_pool.wait()

		for reply,addr in zip(replies,self.addrs):
			if reply['status'] < 0:
				self.log.error("Gain update failed "
				               "for address %s: (%i) %s" %
				               addr, reply['status'], reply['info'])
"""
# Special response packing functions
def pack_reply_CMD_STAT(slot, cmds):
	ncmd_max = 606
	cmds = cmds[:ncmd_max]
	fmt = '>LH%dL%dB' % (len(cmds), len(cmds))
	responseParts = [slot, len(cmds)]
	responseParts.extend( [cmd[1] for cmd in cmds] )
	responseParts.extend( [cmd[2] for cmd in cmds] )
	return struct.pack(fmt, *responseParts)
def truncate_message(s, n):
		return s if len(s) <= n else s[:n-3] + '...'
def pretty_print_bytes(bytestring):
	return ' '.join(['%02x' % ord(i) for i in bytestring])

class AdpServerMonitorClient(object):
	def __init__(self, config, log, host, timeout=0.1):
		self.config = config
		self.log  = log
		self.host = host
		self.host_ipmi = self.host + "-ipmi"
		self.port = config['mcs']['server']['local_port']
		self.sock = _g_zmqctx.socket(zmq.REQ)
		addr = 'tcp://%s:%i' % (self.host,self.port)
		try: self.sock.connect(addr)
		except zmq.error.ZMQError:
			self.log.error("Invalid or non-existent address: %s" % addr)
		self.sock.SNDTIMEO = int(timeout*1000)
		self.sock.RCVTIMEO = int(timeout*1000)
	def read_sensors(self):
		ret = self._ipmi_command('sdr')
		sensors = {}
		for line in ret.split('\n'):
			if '|' not in line:
				continue
			cols = [col.strip() for col in line.split('|')]
			key = cols[0]
			val = cols[1].split()[0]
			sensors[key] = val
		return sensors
	@lru_cache_method(maxsize=4)
	def get_temperatures(self, slot):
		try:
			sensors = self.read_sensors()
			return {key: float(sensors[key])
			        for key in self.config['server']['temperatures']
			        if  key in sensors}
		except:
			return {'error': float('nan')}
	@lru_cache_method(maxsize=4)
	def get_status(self, slot):
		return self._request('STAT')
	@lru_cache_method(maxsize=4)
	def get_info(self, slot):
		return self._request('INFO')
	@lru_cache_method(maxsize=4)
	def get_software(self, slot):
		return self._request('SOFTWARE')
	def _request(self, query):
		try:
			self.sock.send(query)
			response = self.sock.recv_json()
		except zmq.error.Again:
			raise RuntimeError("Server '%s' did not respond" % self.host)
		# TODO: Choose a better form of status codes
		if response['status'] == -404:
			raise KeyError
		elif response['status'] < 0:
			raise RuntimeError(response['info'])
		else:
			return response['data']
	def get_power_state(self):
		"""Returns 'on' or 'off'"""
		return self._ipmi_command("power status").split()[-1]
	def do_power(self, op='status'):
		return self._ipmi_command("power "+op)
	def _ipmi_command(self, cmd):
		username = self.config['ipmi']['username']
		password = self.config['ipmi']['password']
		#try:
		return subprocess.check_output(['ipmitool', '-H', self.host_ipmi,
		                                '-U', username, '-P', password] +
		                                cmd.split())
		#except CalledProcessError as e:
		#	raise RuntimeError(str(e))

# TODO: Rename this (and possibly refactor)
class Roach2MonitorClient(object):
	def __init__(self, config, log, num):
		# Note: num is 1-based index of the roach
		self.config = config
		self.log    = log
		self.roach  = AdpRoach(num, config['roach']['port'])
		self.host   = self.roach.hostname
		self.device = ROACH2Device(self.host)
		self.num = num
		self.GBE_DRX = 0
		self.GBE_TBN = 1
	def reboot(self):
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect(self.host, username='root',
		            password=self.config['roach']['password'])
		ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('reboot')
		#ssh_stdout.read()
		## Note: This requires ssh authorized_keys to have been set up
		#try:
		#	subprocess.check_output(['ssh', 'root@'+self.host,
		#	                         'shutdown -r now'])
		#except subprocess.CalledProcessError:
		#	raise RuntimeError("Roach reboot command failed")
	def get_samples(self, slot, stand, pol, nsamps=None):
		return self.get_samples_all(nsamps)[stand,pol]
	@lru_cache_method(maxsize=4)
	def get_samples_all(self, slot, nsamps=None):
		"""Returns an NDArray of shape (stand,pol,sample)"""
		return self.device.samples_all(nsamps).transpose([1,2,0])
	@lru_cache_method(maxsize=4)
	def get_temperatures(self, slot):
		try:
			return self.device.temperatures()
		except:
			return {'error': float('nan')}
	def program(self):
		# Program with ADP firmware
		# Note: ADCs must be re-calibrated after doing this
		boffile      = self.config['roach']['firmware']
		max_attempts = self.config['roach']['max_program_attempts']
		adc_gain     = self.config['roach']['adc_gain']
		adc_gain_bits   = ( adc_gain       | (adc_gain <<  4) |
		                   (adc_gain << 8) | (adc_gain << 12) )
		adc_gain_reg    = 0x2a
		adc_registers = {adc_gain_reg: adc_gain_bits}
		self.roach.program(boffile, adc_registers, max_attempts)
	#def calibrate(self):
		# TODO: Implement this
	def configure_dual_mode(self):
		try:
			self.roach.stop_processing()
			# DRX on gbe0, TBN on gbe1
			drx_dst_hosts = self.config['host']['servers-data']
			tbn_dst_hosts = [self.config['host']['servers-tbn'][self.num-1]]
			src_ip_base   = self.config['roach']['data_ip_base']
			src_port_base = self.config['roach']['data_port_base']
			dst_ports     = self.config['server']['data_ports']
			drx_dst_ips   = [host2ip(host) for host in drx_dst_hosts]
			tbn_dst_ips   = [host2ip(host) for host in tbn_dst_hosts]
			macs = load_ethers()
			drx_dst_macs  = [macs[ip] for ip in drx_dst_ips]
			tbn_dst_macs  = [macs[ip] for ip in tbn_dst_ips]
			drx_arp_table = gen_arp_table(drx_dst_ips, drx_dst_macs)
			tbn_arp_table = gen_arp_table(tbn_dst_ips, tbn_dst_macs)
			drx_dst_ports = [dst_ports[0]] * len(drx_dst_ips)
			tbn_dst_ports = [dst_ports[1]] * len(tbn_dst_ips)
			
			ret0 = self.roach.configure_10gbe(self.GBE_DRX, drx_dst_ips, drx_dst_ports, drx_arp_table, src_ip_base, src_port_base)
			ret1 = self.roach.configure_10gbe(self.GBE_TBN, tbn_dst_ips, tbn_dst_ports, tbn_arp_table, src_ip_base, src_port_base)
			if not ret0 or not ret1:
				raise RuntimeError("Configuring Roach 10GbE ports failed")
		except:
			self.log.exception("Configuring roach failed")
			raise
	"""
	def configure_mode(self, mode='DRX'):
		# Configure 10GbE ports
		if mode == 'DRX':
			dst_hosts = self.config['host']['servers-data']
		elif mode == 'TBN':
			dst_hosts = self.config['host']['servers-tbn']
		else:
			raise KeyError("Invalid roach mode")
		src_ip_base   = self.config['roach']['data_ip_base']
		src_port_base = self.config['roach']['data_port_base']
		dst_ports = self.config['server']['data_ports']
		dst_ips   = [host2ip(host) for host in dst_hosts]
		macs = load_ethers()
		dst_macs  = [macs[ip] for ip in dst_ips]
		arp_table = gen_arp_table(dst_ips, dst_macs)
		dst_ports0 = [dst_ports[0]] * len(dst_ips)
		dst_ports1 = [dst_ports[1]] * len(dst_ips)
		ret0 = self.roach.configure_10gbe(0, dst_ips, dst_ports0, arp_table, src_ip_base, src_port_base)
		ret1 = self.roach.configure_10gbe(1, dst_ips, dst_ports1, arp_table, src_ip_base, src_port_base)
		if not ret0 or not ret1:
			raise RuntimeError("Configuring Roach 10GbE port(s) failed")
	def tune(self, gbe, cfreq, bw):
		bw = round(bw, 3) # Round to mHz to avoid precision errors
		#nsubband = 1 # HACK TESTING
		nsubband = len(self.config['host']['servers-data'])
		subband_nchan = int(math.ceil(bw / CHAN_BW / nsubband))
		chan0         = int(round( cfreq / CHAN_BW)) - subband_nchan//2
		#if self.num == 1:
		#	print "****", bw, CHAN_BW, math.ceil(bw / CHAN_BW / nsubband)
		#	print "chan_bw %.16f" % CHAN_BW
		#	print "error:", CHAN_BW-25000
		#	print "nchan, chan0", subband_nchan, chan0
		self.roach.configure_fengine(gbe, nsubband, subband_nchan, chan0)
		return subband_nchan, chan0
	"""
	def tune_drx(self, cfreq, bw):
		bw = round(bw, 3) # Round to mHz to avoid precision errors
		nsubband      = len(self.config['host']['servers-data'])
		subband_nchan = int(math.ceil(bw / CHAN_BW / nsubband))
		chan0         =  int(round(cfreq / CHAN_BW)) - subband_nchan//2
		self.roach.configure_fengine(self.GBE_DRX, nsubband, subband_nchan, chan0)
		return subband_nchan, chan0
	def tune_tbn(self, cfreq, bw):
		bw = round(bw, 3) # Round to mHz to avoid precision errors
		nsubband = 1
		subband_nchan = int(math.ceil(bw / CHAN_BW / nsubband))
		chan0         =  int(round(cfreq / CHAN_BW)) - subband_nchan//2
		self.roach.configure_fengine(self.GBE_TBN, nsubband, subband_nchan, chan0)
		return subband_nchan, chan0
	def start_processing(self):
		self.roach.start_processing()
	def stop_processing(self):
		self.roach.stop_processing()
	def processing_started(self):
		return roach.processing_started()
	def enable_drx_data(self):
		self.roach.enable_data(self.GBE_DRX)
	def enable_tbn_data(self):
		self.roach.enable_data(self.GBE_TBN)
	def disable_drx_data(self):
		self.roach.disable_data(self.GBE_DRX)
	def disable_tbn_data(self):
		self.roach.disable_data(self.GBE_TBN)
	def drx_data_enabled(self):
		return roach.data_enabled(self.GBE_DRX)
	def tbn_data_enabled(self):
		return roach.data_enabled(self.GBE_TBN)
	"""
	def start_data(self, mode='DRX'):
		if mode == 'DRX':
			gbe0, gbe1 = True, False
		elif mode == 'TBN':
			gbe0, gbe1 = True, False
		else:
			raise KeyError("Invalid roach mode")
		self.roach.start_data(gbe0, gbe1)
	def pause_data(self):
		self.roach.pause_data()
	def stop_data(self):
		self.roach.stop_data()
	def data_enabled(self, gbe):
		return self.roach.data_enabled(gbe)
	"""
	# TODO: Configure channel selection (based on FST)
	# TODO: start/stop data flow (remember to call roach.reset() before start)

def exception_in(vals, error_type=Exception):
	return any([isinstance(val, error_type) for val in vals])

class MsgProcessor(ConsumerThread):
	def __init__(self, config, log,
	             max_process_time=1.0, ncmd_save=4, dry_run=False):
		ConsumerThread.__init__(self)
		self.config           = config
		self.log              = log
		self.shutdown_timeout = 3.
		self.dry_run          = dry_run
		self.msg_queue        = Queue()
		max_concurrent_msgs = int(MAX_MSGS_PER_SEC*max_process_time)
		self.thread_pool = ThreadPool(max_concurrent_msgs)
		self.name = "Adp.MsgProcessor"
		self.utc_start     = None
		self.utc_start_str = "NULL"

		mcs_local_host  = self.config['mcs']['headnode']['local_host']
		mcs_local_port  = self.config['mcs']['headnode']['local_port']
		mcs_remote_host = self.config['mcs']['headnode']['remote_host']
		mcs_remote_port = self.config['mcs']['headnode']['remote_port']
		"""
		self.msg_receiver = MCS2.MsgReceiver((mcs_local_host, mcs_local_port),
		                                     subsystem=SUBSYSTEM)
		self.msg_sender   = MCS2.MsgSender((mcs_remote_host, mcs_remote_port),
		                                   subsystem=SUBSYSTEM)
		"""
		# Maps slot->[(cmd,ref,exit_code), ...]
		self.cmd_status = SequenceDict(list, maxlen=ncmd_save)
		#self.zmqctx = zmq.Context()

		self.servers = ObjectPool([AdpServerMonitorClient(config, log, host)
		                           for host in self.config['host']['servers']])
		#self.roaches = ObjectPool([Roach2MonitorClient(config, log, host)
		#                           for host in self.config['host']['roaches']])
		#nroach = len(self.config['host']['roaches'])
		nroach = NBOARD
		self.roaches = ObjectPool([Roach2MonitorClient(config, log, num+1)
		                           for num in xrange(nroach)])

		#*self.fst = Fst(config, log)
		#self.bam =
		self.drx = Drx(config, log, self.roaches)
		self.tbn = Tbn(config, log, self.roaches)

		self.serial_number = '1'
		self.version = str(__version__)
		self.state = {}
		self.state['status']  = 'BOOTING'
		self.state['info']    = 'Need to INI ADP'
		self.state['lastlog'] = ('Welcome to ADP S/N %s, version %s' %
		                         (self.serial_number, self.version))
		self.state['activeProcess'] = []

	def uptime(self):
		# Returns no. secs since data processing began (during INI)
		if self.utc_start is None:
			return 0
		secs = (datetime.datetime.utcnow() - self.utc_start).total_seconds()
		return secs
	def raise_error_state(self, cmd, state):
		# TODO: Need new codes? Need to document them?
		state_map = {'BOARD_SHUTDOWN_FAILED':      (0x08,'Board-level shutdown failed'),
		             'BOARD_PROGRAMMING_FAILED':   (0x04,'Board programming failed'),
		             'BOARD_CONFIGURATION_FAILED': (0x05,'Board configuration failed'),
		             'SERVER_STARTUP_FAILED':      (0x09,'Server startup failed'),
		             'SERVER_SHUTDOWN_FAILED':     (0x0A,'Server shutdown failed')}
		code, msg = state_map[state]
		self.state['lastlog'] = '%s: Finished with error' % cmd
		self.state['status']  = 'ERROR'
		self.state['info']    = 'SUMMARY! 0x%02X! %s' % (code, msg)
		self.state['activeProcess'].pop()
		return code

	def check_success(self, func, description, names):
		self.state['info'] = description
		rets = func()
		oks = [True for _ in rets]
		for i, (name, ret) in enumerate(zip(names, rets)):
			if isinstance(ret, Exception):
				oks[i] = False
				self.log.error("%s: %s" % (name, str(ret)))
		all_ok = all(oks)
		if not all_ok:
			symbols = ''.join(['.' if ok else 'x' for ok in oks])
			self.log.error("%s failed: %s" % (description, symbols))
			self.state['info'] = description + " failed"
		else:
			self.state['info'] = description + " succeeded"
		return all_ok
	def ini(self, arg=None):
		start_time = time.time()
		# Note: Return value from this function is not used
		self.state['activeProcess'].append('INI')
		self.state['status'] = 'BOOTING'
		self.state['info']   = 'Running INI sequence'

		if not self.check_success(lambda: self.servers.do_power('on'),
		                          'Powering on servers',
		                          self.servers.host):
			return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')
		startup_timeout = self.config['server']['startup_timeout']
		try:
			self._wait_until_servers_power('on', startup_timeout)
		except RuntimeError:
			return self.raise_error_state('INI', 'SERVER_STARTUP_FAILED')

		# Note: This is for debugging, not in spec
		if 'NOREPROGRAM' not in arg:
			if not self.check_success(lambda: self.roaches.program(),
			                          'Programming FPGAs',
			                          self.roaches.host):
				return self.raise_error_state('INI', 'BOARD_PROGRAMMING_FAILED')
		#if not self.check_success(lambda: self.roaches.configure_mode('DRX'),
		if not self.check_success(lambda: self.roaches.configure_dual_mode(),
		                          'Configuring FPGAs',
		                          self.roaches.host):
			return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')

		# TODO: Need to make sure server pipelines etc. start up and respond here

		start_delay = 5.
		utc_now   = datetime.datetime.utcnow()
		utc_start = utc_now + datetime.timedelta(0, start_delay)
		utc_init  = utc_start - datetime.timedelta(0, 1) # 1 sec before
		utc_start_str = utc_start.strftime(DATE_FORMAT)
		utc_init_str  = utc_start.strftime(DATE_FORMAT)
		self.utc_start     = utc_start
		self.utc_start_str = utc_start_str
		self.state['lastlog'] = "Starting processing at UTC "+utc_start_str

		# TODO: Tell server pipelines the value of utc_start_str and have them
		#         await imminent data.

		# Wait until we're in the middle of the init sec
		wait_until_utc_sec(utc_init_str)
		time.sleep(0.5)
		self.state['lastlog'] = "Starting processing now"
		if not self.check_success(lambda: self.roaches.start_processing(),
		                          'Starting FPGA processing',
		                          self.roaches.host):
			return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')
		if not self.check_success(lambda: self.tbn.start(record=False),
		                          'Initializing TBN',
		                          self.roaches.host):
			return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')
		time.sleep(0.1)
		if not all(self.roaches.processing_started()):
			return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')
		if not all(self.roaches.tbn_data_enabled()):
			return self.raise_error_state('INI', 'BOARD_CONFIGURATION_FAILED')

		self.state['lastlog'] = 'INI finished in %.3f s' % (time.time() - start_time)
		self.state['status']  = 'NORMAL'
		self.state['activeProcess'].pop()
		return 0
		
	def sht(self, arg):
		# TODO: Consider allowing specification of 'only servers' or 'only boards'
		start_time = time.time()
		self.state['activeProcess'].append('SHT')
		self.state['status'] = 'SHUTDWN'
		# TODO: Use self.check_success here like in ini()
		self.state['info']   = 'System is shutting down'
		if 'SCRAM' in arg:
			if 'RESTART' in arg:
				if exception_in(self.servers.do_power('reset')):
					return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
				if exception_in(self.roaches.reboot()):
					return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
			else:
				if exception_in(self.servers.do_power('off')):
					return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
				if exception_in(self.roaches.reboot()):
					return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
			self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
			self.state['status']  = 'SHUTDWN'
			self.state['info']    = 'System has been shut down'
			self.state['activeProcess'].pop()
		else:
			if 'RESTART' in arg:
				def soft_reboot():
					if exception_in(self.servers.do_power('soft')):
						return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
					if exception_in(self.roaches.reboot()):
						return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
					try:
						self._wait_until_servers_power('off')
					except RuntimeError:
						return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
					if exception_in(self.servers.do_power('on')):
						return self.raise_error_state('SHT', 'SERVER_STARTUP_FAILED')
					try:
						self._wait_until_servers_power('on')
					except RuntimeError:
						return self.raise_error_state('SHT', 'SERVER_STARTUP_FAILED')
					self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
					self.state['status']  = 'SHUTDWN'
					self.state['info']    = 'System has been shut down'
					self.state['activeProcess'].pop()
				self.thread_pool.add_task(soft_reboot)
			else:
				def soft_power_off():
					if exception_in(self.servers.do_power('soft')):
						return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
					if exception_in(self.roaches.reboot()):
						return self.raise_error_state('SHT', 'BOARD_SHUTDOWN_FAILED')
					try:
						self._wait_until_servers_power('off')
					except RuntimeError:
						return self.raise_error_state('SHT', 'SERVER_SHUTDOWN_FAILED')
					self.state['lastlog'] = 'SHT finished in %.3f s' % (time.time() - start_time)
					self.state['status']  = 'SHUTDWN'
					self.state['info']    = 'System has been shut down'
					self.state['activeProcess'].pop()
				self.thread_pool.add_task(soft_power_off)
		return 0

	def _wait_until_servers_power(self, target_state, max_wait=30):
		# TODO: Need to check for ping (or even ssh connectivity) instead of 'power is on'?
		time.sleep(6)
		wait_time = 6
		while not all( (state == target_state
		                for state in self.servers.get_power_state()) ):
			time.sleep(2)
			wait_time += 2
			if wait_time >= max_wait:
				raise RuntimeError("Timed out waiting for server(s) to turn "+target_state)

	def process(self, msg):
		if msg.cmd == 'PNG':
			self.log.info('Received PNG: '+str(msg))
			if not self.dry_run:
				self.process_msg(msg, lambda msg: True, '')
		elif msg.cmd == 'RPT':
			self.log.info('Received RPT request: '+str(msg))
			if not self.dry_run:
				# Note: RPT messages are processed asynchronously
				#         to avoid stalls.
				# TODO: Check that this doesn't cause any problems
				#         due to race conditions.
				self.thread_pool.add_task(self.process_msg,
				                          msg, self.process_report)
		else:
			self.log.info('Received command: '+str(msg))
			if not self.dry_run:
				self.process_msg(msg, self.process_command)

		next_slot = MCS2.get_current_slot() + 1
		# TODO: Could defer replies until here for better error handling
		for cmd_processor in [self.drx, self.tbn]:#, self.fst, self.bam]
			self.thread_pool.add_task(cmd_processor.execute_commands,
			                          next_slot)
		#self.drx.execute_commands(next_slot)
		#*self.fst.execute_commands(next_slot)
		#self.bam.execute_commands(next_slot)
	def shutdown(self):
		# Propagate shutdown to downstream consumers
		self.msg_queue.put(ConsumerThread.STOP)
		if not self.thread_pool.wait(self.shutdown_timeout):
			self.log.warning("Active tasks still exist and will be killed")
		print self.name, "shutdown"
	def process_msg(self, msg, process_func):
		accept, reply_data = process_func(msg)
		status = self.state['status']
		reply_msg = msg.create_reply(accept, status, reply_data)
		self.msg_queue.put(reply_msg)
	def process_report(self, msg):
		key, args = MCS2.mib_parse_label(msg.data)
		try: value = self._get_report_result(key, args, msg.slot)
		except KeyError:
			self.log.warning('Unknown MIB entry: %s' % msg.data)
			return False, 'Unknown MIB entry: %s' % msg.data
		except ValueError as e:
			self.log.warning(e)
			return False, str(e)
		#except (ValueError,RuntimeError) as e:
		except Exception as e:
			self.log.error('%s: %s'%(type(e), str(e)))
			return False, '%s: %s'%(type(e), str(e))
		reply_data = self._pack_report_result(key, value)
		log_data   = self._format_report_result(key, value)
		self.log.debug('%s = %s' % (msg.data, log_data))
		return True, reply_data
	def _get_next_fir_index(self):
		idx = self.fir_idx
		self.fir_idx += 1
		self.fir_idx %= NINPUT
		return idx
	def _get_report_result(self, key, args, slot):
		reduce_ops = {'MAX':      np.max,
		              'MIN':      np.min,
		              'AVG':      np.mean,
		              'RMS':      lambda x: np.sqrt(np.mean(x**2)),
		              'SAT':      lambda x: np.sum(np.abs(x)>=ADC_MAXVAL),
		              'DCOFFSET': np.mean,
		              'PEAK':     np.max}
		if key == 'SUMMARY':         return self.state['status']
		if key == 'INFO':            return self.state['info']
		if key == 'LASTLOG':         return self.state['lastlog']
		if key == 'SUBSYSTEM':       return SUBSYSTEM
		if key == 'SERIALNO':        return self.serial_number
		if key == 'VERSION':         return self.version
		# TODO: TBF_STATUS
		#       TBF_TUNING_MASK
		if key == 'NUM_STANDS':      return NSTAND
		if key == 'NUM_SERVERS':     return NSERVER
		if key == 'NUM_BOARDS':      return NBOARD
		if key == 'NUM_TBN_BITS':    return TBN_BITS
		if key == 'TBN_CONFIG_FREQ': return self.tbn.cur_freq
		if key == 'TBN_CONFIG_FILTER': return self.tbn.cur_filt
		if key == 'TBN_CONFIG_GAIN': return self.tbn.cur_gain
		# TODO: NUM_BEAMS
		if key == 'BEAM_FIR_COEFFS': return FIR_NCOEF
		# TODO: T_NOM
		if key == 'NUM_DRX_TUNINGS': return self.drx.ntuning
		if args[0] == 'DRX' and args[1] == 'CONFIG':
			tuning = args[2]-1
			if args[3] == 'FREQ':
				return self.drx.cur_freq[tuning]
			if args[3] == 'FILTER':
				return self.drx.cur_filt[tuning]
			if args[3] == 'GAIN':
				return self.drx.cur_gain[tuning]
		if key == 'NUM_FREQ_CHANS':  return NCHAN
		if key == 'FIR_CHAN_INDEX':  return self._get_next_fir_index()
		if key == 'FIR':
			return self.fst.get_fir_coefs(slot)[input2standpol(self.fir_idx)]
		if key == 'CLK_VAL':         return MCS2.slot2mpm(slot-1)
		if key == 'UTC_START':       return self.utc_start_str # Not in spec
		if key == 'UPTIME':          return self.uptime() # Not in spec
		if key == 'STAT_SAMP_SIZE':  return STAT_SAMP_SIZE
		if args[0] == 'ANT':
			inp = args[1]-1
			if not (0 <= inp < NINPUT):
				raise ValueError("Unknown input number %i"%(inp+1))
			board,stand,pol = input2boardstandpol(inp)
			samples = self.roaches[board].get_samples(slot, stand, pol,
			                                          STAT_SAMP_SIZE)
			# Convert from int8 --> float32 before reducing
			samples = samples.astype(np.float32)
			op = args[2]
			return reduce_ops[op](samples)
		# TODO: BEAM_*
		#  BEAM%i_DELAY
		#  BEAM%i_GAIN
		#  BEAM%i_TUNING # Note: (ADP only)
		if args[0] == 'BOARD':
			board = args[1]-1
			if not (0 <= board < NBOARD):
				raise ValueError("Unknown board number %i"%(board+1))
			if args[2] == 'STAT': return None # TODO
			if args[2] == 'INFO': return None # TODO
			if args[2] == 'TEMP':
				temps = self.roaches[board].get_temperatures(slot).values()
				op = args[3]
				return reduce_ops[op](temps)
			if args[2] == 'FIRMWARE': return self.config['roach']['firmware']
			if args[2] == 'HOSTNAME': return self.roaches[board].host
			raise KeyError
		if args[0] == 'SERVER':
			svr = args[1]-1
			if not (0 <= svr < NSERVER):
				raise ValueError("Unknown server number %i"%(svr+1))
			if args[2] == 'HOSTNAME': return self.servers[svr].host
			# TODO: This request() should raise exceptions on failure
			# TODO: Change to .status(), .info()?
			if args[2] == 'STAT': return self.servers[svr].get_status()
			if args[2] == 'INFO': return self.servers[svr].get_info()
			if args[2] == 'TEMP':
				temps = self.servers[svr].get_temperatures(slot).values()
				op = args[3]
				return reduce_ops[op](temps)
			raise KeyError
		if args[0] == 'GLOBAL':
			if args[1] == 'TEMP':
				temps = []
				# Note: Actually just flattening lists, not summing
				temps += sum(self.roaches.get_temperatures(slot).values(), [])
				temps += sum(self.servers.get_temperatures(slot).values(), [])
				# Remove error values before reducing
				temps = [val for val in temps if not math.isnan(val)]
				if len(temps) == 0: # If all values were nan (exceptional!)
					temps = [float('nan')]
				op = args[2]
				return reduce_ops[op](temps)
			raise KeyError
		if key == 'CMD_STAT': return (slot,self.cmd_status[slot-1])
		raise KeyError
	def _pack_report_result(self, key, value):
		return {
			'SUMMARY':          lambda x: x[:7],
			'INFO':             lambda x: truncate_message(x, 256),
			'LASTLOG':          lambda x: truncate_message(x, 256),
			'SUBSYSTEM':        lambda x: x[:3],
			'SERIALNO':         lambda x: x[:5],
			'VERSION':          lambda x: truncate_message(x, 256),
			#'TBF_STATUS':
			#'TBF_TUNING_MASK':
			'NUM_TBN_BITS':     lambda x: struct.pack('>B', x),
			'NUM_DRX_TUNINGS':  lambda x: struct.pack('>B', x),
			'NUM_FREQ_CHANS':   lambda x: struct.pack('>H', x),
			#'NUM_BEAMS':
			'NUM_STANDS':       lambda x: struct.pack('>H', x),
			'NUM_BOARDS':       lambda x: struct.pack('>B', x),
			'NUM_SERVERS':      lambda x: struct.pack('>B', x),
			'BEAM_FIR_COEFFS':  lambda x: struct.pack('>B', x),
			#'T_NOMn:
			'FIR_CHAN_INDEX':   lambda x: struct.pack('>H', x),
			'FIR':              lambda x: x.astype('>h').tobytes(),
			'CLK_VAL':          lambda x: struct.pack('>I', x),
			'UTC_START':        lambda x: truncate_message(x, 256), # Not in spec
			'UPTIME':           lambda x: struct.pack('>I', x),     # Not in spec
			'STAT_SAMPLE_SIZE': lambda x: struct.pack('>I', x),
			'ANT_RMS':          lambda x: struct.pack('>f', x),
			'ANT_SAT':          lambda x: struct.pack('>i', x),
			'ANT_DCOFFSET':     lambda x: struct.pack('>f', x),
			'ANT_PEAK':         lambda x: struct.pack('>i', x),
			# TODO: Implement these BEAM requests
			#         Are these actually in the spec?
			#'BEAM_RMS':         lambda x: struct.pack('>f', x),
			#'BEAM_SAT':         lambda x: struct.pack('>i', x),
			#'BEAM_DCOFFSET':    lambda x: struct.pack('>f', x),
			#'BEAM_PEAK':        lambda x: struct.pack('>i', x),
			# TODO: In the spec this is >I ?
			'BOARD_STAT':       lambda x: struct.pack('>L', x),
			'BOARD_TEMP_MAX':   lambda x: struct.pack('>f', x),
			'BOARD_TEMP_MIN':   lambda x: struct.pack('>f', x),
			'BOARD_TEMP_AVG':   lambda x: struct.pack('>f', x),
			'BOARD_FIRMWARE':   lambda x: truncate_message(x, 256),
			'BOARD_HOSTNAME':   lambda x: truncate_message(x, 256),
			# TODO: SERVER_STAT
			'SERVER_TEMP_MAX':  lambda x: struct.pack('>f', x),
			'SERVER_TEMP_MIN':  lambda x: struct.pack('>f', x),
			'SERVER_TEMP_AVG':  lambda x: struct.pack('>f', x),
			'SERVER_SOFTWARE':  lambda x: truncate_message(x, 256),
			'SERVER_HOSTNAME':  lambda x: truncate_message(x, 256),
			'GLOBAL_TEMP_MAX':  lambda x: struct.pack('>f', x),
			'GLOBAL_TEMP_MIN':  lambda x: struct.pack('>f', x),
			'GLOBAL_TEMP_AVG':  lambda x: struct.pack('>f', x),
			'CMD_STAT':         lambda x: pack_reply_CMD_STAT(*x),
			'TBN_CONFIG_FREQ':  lambda x: struct.pack('>f', x),
			'TBN_CONFIG_FILTER':lambda x: struct.pack('>H', x),
			'TBN_CONFIG_GAIN':  lambda x: struct.pack('>H', x),
			'DRX_CONFIG_FREQ':  lambda x: struct.pack('>f', x),
			'DRX_CONFIG_FILTER':lambda x: struct.pack('>H', x),
			'DRX_CONFIG_GAIN':  lambda x: struct.pack('>H', x)
		}[key](value)
	def _format_report_result(self, key, value):
		format_function = defaultdict(lambda : str)
		format_function.update({
			'FIR':      pretty_print_bytes,
			'CMD_STAT': lambda x: '%i commands in previous slot' % len(x)
		})
		return format_function[key](value)
	
	def currently_processing(self, *cmds):
		return any([cmd in self.state['activeProcess'] for cmd in cmds])
	def process_command(self, msg):
		exec_delay = 2
		exec_slot  = msg.slot + exec_delay
		accept = True
		reply_data = ""
		exit_status = 0
		if msg.cmd == 'INI':
			if self.currently_processing('INI', 'SHT'):
				# TODO: This stuff could be tidied up a bit
				self.state['lastlog'] = ('INI: %s - %s is active and blocking'%
				                         ('Blocking operation in progress',
				                          self.state['activeProcess']))
				exit_status = 0x0C
			else:
				self.thread_pool.add_task(self.ini, msg.data)
		elif msg.cmd == 'SHT':
			if self.currently_processing('INI', 'SHT'):
				self.state['lastlog'] = ('SHT: %s - %s is active and blocking'%
				                         ('Blocking operation in progress',
				                          self.state['activeProcess']))
				exit_status = 0x0C
			else:
				self.thread_pool.add_task(self.sht, msg.data)
		elif msg.cmd == 'STP':
			mode = msg.data # TBN/TBF/BEAMn/COR
			if mode == 'DRX':
				# TODO: This is not actually part of the spec (useful for debugging?)
				exit_status = self.drx.stop()
			elif mode == 'TBN':
				exit_status = self.tbn.stop()
			elif mode == 'TBF':
				self.state['lastlog'] = "UNIMPLEMENTED STP request"
				exit_status = -1 # TODO: Implement this
			elif mode.startswith('BEAM'):
				self.state['lastlog'] = "UNIMPLEMENTED STP request"
				exit_status = -1 # TODO: Implement this
			elif mode == 'COR':
				self.state['lastlog'] = "UNIMPLEMENTED STP request"
				exit_status = -1 # TODO: Implement this
			else:
				self.state['lastlog'] = "Invalid STP request"
				exit_status = -1
		elif msg.cmd == 'DRX':
			exit_status = self.drx.process_command(msg)
		elif msg.cmd == 'FST':
			exit_status = self.fst.process_command(msg)
		elif msg.cmd == 'BAM':
			exit_status = self.bam.process_command(msg)
		elif msg.cmd == 'TBN':
			exit_status = self.tbn.process_command(msg)
		else:
			exit_status = 0
			accept = False
			reply_data = 'Unknown command: %s' % msg.cmd
		if exit_status != 0:
			accept = False
			reply_data = "0x%02X! %s" % (exit_status, self.state['lastlog'])
		self.cmd_status[msg.slot].append( (msg.cmd, msg.ref, exit_status) )
		return accept, reply_data
