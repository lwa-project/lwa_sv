#!/usr/bin/env python

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from bifrost import Ring, Address, UDPSocket, CHIPSCapture, bfAffinitySetCore

import signal
import logging
import time
import os
import argparse
from threading import Event

import numpy as np
import threading
import json
import socket
import struct
import time
import datetime
from ctypes import create_string_buffer

#from numpy.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft

__version__    = "0.1"
__date__       = '$LastChangedDate: 2015-07-23 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

class CaptureOp(object):
	def __init__(self, log, *args, **kwargs):
		self.shutdown_event = threading.Event()
		self.log    = log
		self.args   = args
		self.kwargs = kwargs
		self.utc_start = self.kwargs['utc_start']
	def shutdown(self):
		self.shutdown_event.set()
	def header_callback(self, seq0, chan0, nchan, nroach, char_ptr_ptr):
		timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
		time_tag0  = timestamp0 * int(FS)
		time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
		hdr = {
			'time_tag': time_tag,
			'chan0':    chan0,
			'nchan':    nchan,
			'nroach':   nroach, # TODO: Have these passed to the callback too
			'nbit':     8       # TODO: Probably shouldn't be hard coded
		}
		hdr_str = json.dumps(hdr)
		# TODO: Can't pad with NULL because returned as C-string
		#hdr_str = json.dumps(hdr).ljust(4096, '\0')
		#hdr_str = json.dumps(hdr).ljust(4096, ' ')
		self.header_buf = create_string_buffer(hdr_str)
		char_ptr_ptr[0] = self.header_buf
		return time_tag
	def main(self):
		self.capture = CHIPSCapture(*self.args,
		                            header_callback=self.header_callback,
		                            **self.kwargs)
		
		## HACK TESTING
		#sync_pipelines = MCS.Synchronizer('TBN2')
		#print "Sleeping for 5 secs"
		#time.sleep(5)
		#sync_pipelines(0)
		
		while not self.shutdown_event.is_set():
			self.capture.recv()
		self.capture.end()

def quantize_complex8b(a):
	# Assumes data are scaled to [-1:+1]
	assert(a.dtype == np.complex64)
	orig_shape = a.shape
	iq = a.view(dtype=np.float32).reshape(orig_shape+(2,))
	iq = np.clip(iq, -1, +1)
	iq *= 127
	return np.round(iq).astype(np.int8)

class TEngineOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, oring, nchan_max=256, nroach_max=16, ntime_gulp=2500,
	             nchan_out=4, gain=2, core=-1):
		self.log   = log
		self.iring = iring
		self.oring = oring
		self.nchan_out = nchan_out # 4 => 100 MHz
		self.gain      = gain
		ninput_max = nroach_max*32#*2
		#frame_size_max = nchan_max*ninput_max
		self.ntime_gulp = ntime_gulp
		self.core = core
		self.iring.resize(self.ntime_gulp*nchan_max*ninput_max*2)#frame_size_max)
		self.oring.resize(self.ntime_gulp*self.nchan_out*ninput_max*2)
	def main(self):
		bfAffinitySetCore(self.core)
		gain      = self.gain
		nchan_out = self.nchan_out
		with self.oring.begin_writing() as oring:
			for isequence in self.iring.read():#guarantee=False):
				ihdr = json.loads(isequence.header.tostring())
				#print 'TEngineOp', ihdr
				self.log.info("TEngine: Start of new sequence: %s", str(ihdr))
				nchan  = ihdr['nchan']
				nroach = ihdr['nroach']
				nstand = nroach*16
				npol   = 2
				igulp_size = self.ntime_gulp*nchan*nstand*npol*2
				ishape = (self.ntime_gulp,nchan,nstand,npol,2)
				oshape = (self.ntime_gulp*nchan_out,nstand,npol,2)
				ogulp_size = self.ntime_gulp*nchan_out*nstand*npol*2
				ohdr = {}
				ohdr['time_tag'] = ihdr['time_tag']
				ohdr['cfreq']    = (ihdr['chan0'] + 0.5*(ihdr['nchan']-1))*CHAN_BW
				ohdr['bw']       = nchan_out*CHAN_BW
				ohdr['gain']     = gain
				ohdr['nstand']   = nstand
				ohdr['npol']     = npol
				ohdr['complex']  = True
				ohdr['nbit']     = 8
				ohdr_str = json.dumps(ohdr)
				with oring.begin_sequence(time_tag=isequence.time_tag,
				                          header=ohdr_str) as osequence:
					for ispan in isequence.read(igulp_size):
						#print "TENGINE processing"
						if ispan.size < igulp_size:
							continue # Ignore final gulp
						with osequence.reserve(ogulp_size) as ospan:
							data = ispan.data_view(np.int8).reshape(ishape)
							data  = data >> 4 # Correct for 8->16bit padding
							data  = data.astype(np.float32)
							data  = data[...,0] + 1j*data[...,1]
							tdata = data[:,nchan/2-nchan_out/2:nchan/2+nchan_out/2]
							tdata = ifft(tdata, axis=1).astype(np.complex64)
							tdata *= 1./(2**gain * np.sqrt(nchan_out))
							tdata = tdata.reshape((-1,nstand,npol))
							mean, stddev = tdata.mean(), tdata.std()
							#print "Mean, stddev: %f+%fj\t%f" % (np.round(mean.real, 3), np.round(mean.imag, 3), stddev)
							
							if stddev > 0.5:
								self.log.warning("May need to increase gain to avoid excess clipping")
								self.log.warning("Mean, stddev: %f+%fj\t%f" % (np.round(mean.real, 3), np.round(mean.imag, 3), stddev))
							elif stddev < 0.15:
								self.log.warning("May need to decrease gain to avoid excess quantization noise")
								self.log.warning("Mean, stddev: %f+%fj\t%f" % (np.round(mean.real, 3), np.round(mean.imag, 3), stddev))
							
							tdata = quantize_complex8b(tdata) # Note: dtype is now real
							odata = ospan.data_view(np.int8).reshape((1,)+oshape)
							odata[...] = tdata

def gen_tbn_header(stand, pol, cfreq, gain, time_tag, time_tag0, bw=100e3):
	nframe_per_sample = int(FS) // int(bw)
	nframe_per_packet = nframe_per_sample * TBN_NSAMPLE_PER_PKT
	sync_word    = 0xDEC0DE5C
	idval        = 0x0
	frame_num_wrap = 10 * int(bw) # 10 secs = 4e6, fits within a uint24
	frame_num    = ((time_tag - time_tag0) // nframe_per_packet) % frame_num_wrap + 1 # Packet sequence no.
	id_frame_num = idval << 24 | frame_num
	assert( 0 <= cfreq < FS )
	tuning_word  = int(round(cfreq / FS * 2**32))
	tbn_id       = (pol + NPOL*stand) + 1
	gain         = gain
	#if stand == 0 and pol == 0:
	#	print cfreq, bw, gain, time_tag, time_tag0
	#	print nframe_per_sample, nframe_per_packet
	return struct.pack('>IIIhhq',
	                   sync_word,
	                   id_frame_num,
	                   tuning_word,
	                   tbn_id,
	                   gain,
	                   time_tag)

class PacketizeOp(object):
	# Note: Input data are: [time,stand,pol,iq]
	def __init__(self, log, iring, addr, port, nroach, roach0, npkt_gulp=128,
	             core=-1):
		self.log   = log
		self.iring = iring
		self.sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.connect((addr,port))
		self.npkt_gulp = npkt_gulp
		self.nroach = nroach
		self.roach0 = roach0
		self.sync_pipelines = MCS.Synchronizer('TBN')
		self.core = core
	def main(self):
		bfAffinitySetCore(self.core)
		stand0 = self.roach0 * 16 # TODO: Get this less hackily
		ntime_pkt     = TBN_NSAMPLE_PER_PKT
		ntime_gulp    = self.npkt_gulp * ntime_pkt
		ninput_max    = self.nroach*32
		gulp_size_max = ntime_gulp * ninput_max * 2
		self.iring.resize(gulp_size_max)
		#ofile = open('/data1/test_tbn.pkt', 'wb') # HACK TESTING
		for isequence in self.iring.read():
			hdr = json.loads(isequence.header.tostring())
			#print 'PacketizeOp', hdr
			cfreq  = hdr['cfreq']
			bw     = hdr['bw']
			gain   = hdr['gain']
			nstand = hdr['nstand']
			#stand0 = hdr['stand0']
			npol   = hdr['npol']
			time_tag0 = isequence.time_tag
			time_tag  = time_tag0
			gulp_size = ntime_gulp*nstand*npol*2
			for ispan in isequence.read(gulp_size):
				if ispan.size < gulp_size:
					continue # Ignore final gulp
				#print 'Packetizing', stand0, nstand
				shape = (-1,nstand,npol,2)
				data = ispan.data_view(np.int8).reshape(shape)
				#self.sync_pipelines(time_tag)
				for t in xrange(0, ntime_gulp, ntime_pkt):
					self.sync_pipelines(time_tag)
					for stand in xrange(nstand):
						for pol in xrange(npol):
							pktdata = data[t:t+ntime_pkt,stand,pol,:]
							#pktdata = pktdata[...,::-1] # WAR: Swap I/Q
							#assert( len(pktdata) == ntime_pkt )
							time_tag_cur = time_tag + int(round(float(t)/bw*FS))
							hdr = gen_tbn_header(stand0+stand, pol, cfreq, gain,
							                     time_tag_cur, time_tag0, bw)
							pkt = hdr + pktdata.tostring()
							self.sock.send(pkt)
							#ofile.write(pkt) # HACK TESTING

							## HACK TESTING double packets fake
							#hdr = gen_tbn_header(48+stand0+stand, pol, cfreq, gain,
							#                     time_tag_cur, time_tag0, bw)
							#pkt = hdr + pktdata.tostring()
							#self.sock.send(pkt)

				time_tag += int(round(float(ntime_gulp)/bw*FS))
		#ofile.close()

def get_utc_start():
	got_utc_start = False
	while not got_utc_start:
		try:
			with MCS.Communicator() as adp_control:
				utc_start = adp_control.report('UTC_START')
				# Check for valid timestamp
				utc_start_dt = datetime.datetime.strptime(utc_start, DATE_FORMAT)
			got_utc_start = True
		except Exception as ex:
			print ex
			time.sleep(1)
	#print "UTC_START:", utc_start
	#return utc_start
	return utc_start_dt

def get_numeric_suffix(s):
	i = 0
	while True:
		if len(s[i:]) == 0:
			raise ValueError("No numeric suffix in string '%s'" % s)
		try: return int(s[i:])
		except ValueError: i += 1

def partition_balanced(nitem, npart, part_idx):
	rem = nitem % npart
	part_nitem  = nitem / npart + (part_idx < rem)
	part_offset = (part_idx*part_nitem if part_idx < rem else
	               rem*(part_nitem+1) + (part_idx-rem)*part_nitem)
	return part_nitem, part_offset

def partition_packed(nitem, npart, part_idx):
	part_nitem  = (nitem-1) / npart + 1
	part_offset = part_idx * part_nitem
	part_nitem  = min(part_nitem, nitem-part_offset)
	return part_nitem, part_offset

def main(argv):
	parser = argparse.ArgumentParser(description='LWA-SV ADP control service')
	parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
	parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
	parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
	parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
	parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
	args = parser.parse_args()
	
	config = Adp.parse_config_file(args.configfile)
	
	log = logging.getLogger(__name__)
	logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
	                              datefmt='%Y-%m-%d %H:%M:%S')
	logFormat.converter = time.gmtime
	if args.logfile is None:
		logHandler = logging.StreamHandler(sys.stdout)
	else:
		logHandler = Adp.AdpFileHandler(config, args.logfile)
	logHandler.setFormatter(logFormat)
	log.addHandler(logHandler)
	verbosity = args.verbose - args.quiet
	if   verbosity >  0: log.setLevel(logging.DEBUG)
	elif verbosity == 0: log.setLevel(logging.INFO)
	elif verbosity <  0: log.setLevel(logging.WARNING)
	
	short_date = ' '.join(__date__.split()[1:4])
	log.info("Starting %s with PID %i", argv[0], os.getpid())
	log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
	log.info("Version:      %s", __version__)
	log.info("Last changed: %s", short_date)
	log.info("Current MJD:  %f", Adp.MCS2.slot2mjd())
	log.info("Current MPM:  %i", Adp.MCS2.slot2mpm())
	log.info("Config file:  %s", args.configfile)
	log.info("Log file:     %s", args.logfile)
	log.info("Dry run:      %r", args.dryrun)
	
	shutdown_event = threading.Event()
	def handle_signal_terminate(signum, frame):
		SIGNAL_NAMES = dict((k, v) for v, k in \
		                    reversed(sorted(signal.__dict__.items()))
		                    if v.startswith('SIG') and \
		                    not v.startswith('SIG_'))
		log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
		ops[0].shutdown()
		shutdown_event.set()
	for sig in [signal.SIGHUP,
	            signal.SIGINT,
	            signal.SIGQUIT,
	            signal.SIGTERM,
	            signal.SIGTSTP]:
		signal.signal(sig, handle_signal_terminate)
	
	log.info("Waiting to get UTC_START")
	utc_start_dt = get_utc_start()
	log.info("UTC_START:    %s", utc_start_dt.strftime(DATE_FORMAT))
	
	hostname = socket.gethostname()
	server_idx = get_numeric_suffix(hostname) - 1
	log.info("Hostname:     %s", hostname)
	log.info("Server index: %i", server_idx)
	
	pipeline_idx = config['tbn']['pipeline_idx']
	recorder_idx = config['tbn']['recorder_idx']
	iaddr  = config['server']['data_ifaces'][pipeline_idx]
	iport  = config['server']['data_ports' ][pipeline_idx]
	oaddr  = config['host']['recorders'][recorder_idx]
	oport  = config['recorder']['port']
	nroach_tot = len(config['host']['roaches'])
	nserver    = len(config['host']['servers'])
	#nserver = 4 # HACK TESTING
	#nroach, roach0 = partition_balanced(nroach_tot, nserver, server_idx)
	#nroach, roach0 = partition_packed(nroach_tot, nserver, server_idx)
	tbn_servers = config['host']['servers-tbn']
	server_data_host = config['host']['servers-data'][server_idx]
	nroach = len([srv for srv in tbn_servers if srv == server_data_host])
	roach0 = [i for (i,srv) in enumerate(tbn_servers) if srv == server_data_host][0]
	core0 = config['tbn']['first_cpu_core']
	
	## HACK TESTING
	#nroach = 4
	
	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("Dst address:  %s:%i", oaddr, oport)
	log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
	
	# Note: Capture uses Bifrost address+socket objects, while output uses
	#         plain Python address+socket objects.
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	
	capture_ring = Ring()
	tengine_ring = Ring()
	
	ops = []
	ops.append(CaptureOp(log, isock, capture_ring, core=core0+0,
	                     nroach=nroach, roach0=roach0, utc_start=utc_start_dt,
	                     #buffer_ntime=2500, slot_ntime=25000))
	                     buffer_ntime=25000, slot_ntime=25000))
	# **TODO: Setting nroach_max=16 here (which just causes a larger ring.resize()),
	#           breaks the capture code! It seems that regular gaps appear in
	#           the pkt seq numbers, which screws the sequence tracking and 
	#           causes endless 'early' packets. No idea what's going on!
	ops.append(TEngineOp(log, capture_ring, tengine_ring, nroach_max=nroach,
	                     core=core0+1))
	ops.append(PacketizeOp(log, tengine_ring,
	                       nroach=nroach, roach0=roach0,
	                       addr=oaddr, port=oport,
	                       npkt_gulp=10, # LASI OK
	                       #npkt_gulp=128, # LASI OK
	                       core=core0+2))
	                       #npkt_gulp=1280)) # LASI drops packets?
	                       #npkt_gulp=12800)) # Breaks LASI
	
	threads = [threading.Thread(target=op.main) for op in ops]
	
	log.info("Launching %i thread(s)", len(threads))
	for thread in threads:
		thread.daemon = True
		thread.start()
	log.info("Waiting for threads to finish")
	while not shutdown_event.is_set():
		signal.pause()
	for thread in threads:
		thread.join()
	log.info("All done")
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
