#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.udp_capture import UDPCapture
from bifrost.udp_transmit import UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.fft import Fft
from bifrost.quantize import quantize as Quantize
from bifrost.libbifrost import bf

#import numpy as np
import signal
import logging
import time
import os
import argparse
import ctypes
import threading
import json
import socket
import struct
#import time
import datetime
import pickle

#from numpy.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft

import pyfftw

FILTER2BW = {1:   250000, 
		   2:   500000, 
		   3:  1000000, 
		   4:  2000000, 
		   5:  4900000, 
		   6:  9800000, 
		   7: 19600000}
FILTER2CHAN = {1:   250000/25000, 
			2:   500000/25000, 
			3:  1000000/25000, 
			4:  2000000/25000, 
			5:  4900000/25000, 
			6:  9800000/25000, 
			7: 19600000/25000}

__version__    = "0.1"
__date__       = '$LastChangedDate: 2016-08-09 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"
#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
	def __init__(self, log, *args, **kwargs):
		self.log    = log
		self.args   = args
		self.kwargs = kwargs
		self.utc_start = self.kwargs['utc_start']
		del self.kwargs['utc_start']
		self.shutdown_event = threading.Event()
		## HACK TESTING
		#self.seq_callback = None
	def shutdown(self):
		self.shutdown_event.set()
	def seq_callback(self, seq0, chan0, nchan, nsrc,
	                 time_tag_ptr, hdr_ptr, hdr_size_ptr):
		timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
		time_tag0  = timestamp0 * int(FS)
		time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
		print "++++++++++++++++ seq0     =", seq0
		print "                 time_tag =", time_tag
		time_tag_ptr[0] = time_tag
		hdr = {
			'time_tag': time_tag,
			'chan0':    chan0,
			'nsrc':     nsrc,
			'nchan':    nchan*nsrc, 
			'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
			'bw':       nchan*nsrc*CHAN_BW,
			'nstand':   1,
			#'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
			'npol':     2,
			'complex':  True,
			'nbit':     64
		}
		print "******** CFREQ:", hdr['cfreq']
		hdr_str = json.dumps(hdr)
		# TODO: Can't pad with NULL because returned as C-string
		#hdr_str = json.dumps(hdr).ljust(4096, '\0')
		#hdr_str = json.dumps(hdr).ljust(4096, ' ')
		self.header_buf = ctypes.create_string_buffer(hdr_str)
		hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
		hdr_size_ptr[0] = len(hdr_str)
		return 0
	def main(self):
		seq_callback = bf.BFudpcapture_sequence_callback(self.seq_callback)
		with UDPCapture(*self.args,
		                sequence_callback=seq_callback,
		                **self.kwargs) as capture:
			while not self.shutdown_event.is_set():
				status = capture.recv()
				#print status
		del capture

class ReorderChannelsOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500,# ntime_buf=None,
	             guarantee=True, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		#if ntime_buf is None:
		#	ntime_buf = self.ntime_gulp*3
		#self.ntime_buf = ntime_buf
		self.guarantee = guarantee
		self.core = core
		
	@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())
				
				self.log.info("Reorder: Start of new sequence: %s", str(ihdr))
				
				nsrc   = ihdr['nsrc']
				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				
				igulp_size = self.ntime_gulp*nchan*nstand*npol*16				# complex128
				ishape = (self.ntime_gulp,nchan/nsrc,nsrc,npol)
				ogulp_size = self.ntime_gulp*nchan*nstand*npol*8				# complex64
				oshape = (self.ntime_gulp,nchan,1,npol)
				self.iring.resize(igulp_size)
				self.oring.resize(ogulp_size)#, obuf_size)
				ohdr = ihdr.copy()
				ohdr_str = json.dumps(ohdr)
				with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
					for ispan in iseq.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
							
						with oseq.reserve(ogulp_size) as ospan:
							idata = ispan.data_view(np.complex128).reshape(ishape)
							odata = ospan.data_view(np.complex64).reshape(oshape)
							rdata = idata.astype(np.complex64)	
							odata[...] = np.swapaxes(rdata, 1, 2).reshape((self.ntime_gulp,nchan,1,npol))

def quantize_complex4b(a):
	# Assumes data are scaled to [-1:+1]
	assert(a.dtype == np.complex64)
	orig_shape = a.shape
	iq = a.view(dtype=np.float32).reshape(orig_shape+(2,))
	iq = np.clip(iq, -1, +1)
	iq *= 7
	iq = np.round(iq).astype(np.int8)
	iq *= 16
	return ((iq[...,0])&0xF0) | ((iq[...,1]>>4)&0xF)
	
class TEngineOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500,# ntime_buf=None,
	             guarantee=True, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		#if ntime_buf is None:
		#	ntime_buf = self.ntime_gulp*3
		#self.ntime_buf = ntime_buf
		self.guarantee = guarantee
		self.core = core
		
		self.configMessage = ISC.DRXConfigurationClient(addr=('adp',5832))
		self.gain = 7
		self.filt = 6
		self.nchan_out = FILTER2CHAN[self.filt]
		self.phaseRot = 1
		
	@ISC.logException
	def updateConfig(self, config, hdr):
		if config:
			self.log.info("TEngine: New configuration received: %s", str(config))
			tuning, freq, filt, gain = config
			self.rFreq = freq
			self.filt = filt
			self.nchan_out = FILTER2CHAN[filt]
			self.gain = gain
			
			fDiff = freq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
			self.log.info("TEngine: Tuning offset is %.3f kHz to be corrected with phase rotation", fDiff)
			
			self.phaseRot = np.exp(-2j*np.pi*fDiff/(self.nchan_out*CHAN_BW)*np.arange(self.ntime_gulp*self.nchan_out))
			self.phaseRot.shape += (1,1)
			
			return True
		else:
			return False
			
	@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		
		#try:
		#	fh = open('/home/adp/wisdom.pkl', 'r')
		#	pyfftw.import_wisdom(pickle.load(fh))
		#	fh.close()
		#except :
		#	pass
		
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())
				
				self.updateConfig( self.configMessage(), ihdr )
				
				self.log.info("TEngine: Start of new sequence: %s", str(ihdr))
				
				nsrc   = ihdr['nsrc']
				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				
				igulp_size = self.ntime_gulp*nchan*nstand*npol*8				# complex64
				ishape = (self.ntime_gulp,nchan,1,npol)
				ogulp_size = self.ntime_gulp*self.nchan_out*nstand*npol*1		# 4+4 complex
				oshape = (self.ntime_gulp*self.nchan_out,1,npol)
				self.iring.resize(igulp_size)
				self.oring.resize(ogulp_size)#, obuf_size)
				ohdr = {}
				ohdr['time_tag'] = ihdr['time_tag']
				try:
					ohdr['cfreq']    = self.rFreq
				except AttributeError:
					ohdr['cfreq']    = (ihdr['chan0'] + 0.5*(ihdr['nchan']-1))*CHAN_BW - CHAN_BW / 2 + CHAN_BW
				ohdr['bw']       = self.nchan_out*CHAN_BW
				ohdr['gain']     = self.gain
				ohdr['filter']   = self.filt
				ohdr['nstand']   = nstand
				ohdr['npol']     = npol
				ohdr['complex']  = True
				ohdr['nbit']     = 4
				ohdr_str = json.dumps(ohdr)
				
				#pb = pyfftw.empty_aligned((self.ntime_gulp,self.nchan_out,1,npol), dtype=np.complex64)
				#pf = pyfftw.FFTW(pb, pb, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',), axes=(1,))
				
				with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
					for ispan in iseq.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
							
						self.updateConfig( self.configMessage(), ihdr )
						
						with oseq.reserve(ogulp_size) as ospan:
							## Setup and load
							idata = ispan.data_view(np.complex64).reshape(ishape)
							odata = ospan.data_view(np.int8).reshape((1,)+oshape)
							
							## Prune and shift the data ahead of the IFFT
							tdata = idata[:,nchan/2-self.nchan_out/2:nchan/2+self.nchan_out/2,:,:]
							tdata = np.fft.fftshift(tdata, axes=1)
							
							## IFFT
							#pb[...] = tdata.astype(np.complex64)
							#tdata = pf()
							tdata = BFArray(tdata, space='system')
							gdata = tdata.copy(space='cuda')
							try:
								bfft.execute(gdata, gdata, inverse=True)
							except NameError:
								bfft = Fft()
								bfft.init(gdata, gdata, axes=1)
								bfft.execute(gdata, gdata, inverse=True)
							tdata = gdata.copy(space='system')
							
							## Phase rotation
							tdata = tdata.reshape((-1,nstand,npol))
							tdata *= self.phaseRot
							
							## Scaling
							tdata *= 8./(2**self.gain * np.sqrt(self.nchan_out))
							
							## Quantization
							try:
								Quantize(tdata, qdata)
							except NameError:
								qdata = BFArray(shape=tdata.shape, native=False, dtype='ci4')
								Quantize(tdata, qdata)
								
							## Save
							odata[...] = qdata.view(np.int8).reshape((1,)+oshape)
							
							#tdata = quantize_complex4b(tdata) # Note: dtype is now real
							#odata = ospan.data_view(np.int8).reshape((1,)+oshape)
							#odata[...] = tdata
							
				# Clean-up
				try:
					del bfft
					del qdata
				except NameError:
					pass
					
		#fh = open('/home/adp/wisdom.pkl', 'w')
		#w = pyfftw.export_wisdom()
		#pickle.dump(w, fh)
		#fh.close()

def gen_drx_header(beam, pol, cfreq, filter, time_tag):
	bw = FILTER2BW[filter]
	decim = int(FS) / bw
	sync_word    = 0xDEC0DE5C
	idval        = ((pol&0x1)<<7) | ((1&0x7)<<3) | (beam&0x7)
	frame_num    = 0
	id_frame_num = idval << 24 | frame_num
	assert( 0 <= cfreq < FS )
	tuning_word  = int(round(cfreq / FS * 2**32))
	#if stand == 0 and pol == 0:
	#	print cfreq, bw, gain, time_tag, time_tag0
	#	print nframe_per_sample, nframe_per_packet
	return struct.pack('>IIIhhqII',
	                   sync_word,
	                   id_frame_num,
	                   0, 
	                   decim, 
	                   0, 
	                   time_tag, 
	                   tuning_word, 
	                   0)

class PacketizeOp(object):
	# Note: Input data are: [time,beam,pol,iq]
	def __init__(self, log, iring, osock, nbeam, beam0, npkt_gulp=128, core=-1):
		self.log   = log
		self.iring = iring
		self.sock  = osock
		self.nbeam = nbeam
		self.beam0 = beam0
		self.npkt_gulp = npkt_gulp
		self.core = core
		
	@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		
		ntime_pkt     = DRX_NSAMPLE_PER_PKT
		ntime_gulp    = self.npkt_gulp * ntime_pkt
		ninput_max    = self.nbeam * 2
		gulp_size_max = ntime_gulp * ninput_max * 2
		self.iring.resize(gulp_size_max)
		
		local = True
		if local:
			ofile = open('/data1/test_drx.pkt', 'wb') # HACK TESTING
			
		for isequence in self.iring.read():
			ihdr = json.loads(isequence.header.tostring())
			
			self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
			
			#print 'PacketizeOp', ihdr
			cfreq  = ihdr['cfreq']
			bw     = ihdr['bw']
			gain   = ihdr['gain']
			filt   = ihdr['filter']
			nbeam  = ihdr['nstand']
			npol   = ihdr['npol']
			time_tag0 = isequence.time_tag
			time_tag  = time_tag0
			gulp_size = ntime_gulp*nbeam*npol
			
			with UDPTransmit(sock=self.sock, core=self.core) as udt:
				for ispan in isequence.read(gulp_size):
					if ispan.size < gulp_size:
						continue # Ignore final gulp
						
					shape = (-1,nbeam,npol)
					data = ispan.data_view(np.int8).reshape(shape)
					
					pkts = []
					for t in xrange(0, ntime_gulp, ntime_pkt):
						for beam in xrange(nbeam):
							for pol in xrange(npol):
								pktdata = data[t:t+ntime_pkt,beam,pol]
								time_tag_cur = time_tag + int(round(float(t)/bw*FS))
								hdr = gen_drx_header(beam+self.beam0, pol, cfreq, filt, 
											time_tag_cur)
								try:
									pkt = hdr + pktdata.tostring()
									pkts.append( pkt )
									if local:
										ofile.write(pkt) # HACK TESTING
								except Exception as e:
									print str(e)
									
					if not local:
						try:
							udt.sendmany(pkts)
						except Exception as e:
							print str(e)
							pass
					time_tag += int(round(float(ntime_gulp)/bw*FS))
					
			del udt
			
		if local:
			ofile.close() # HACK TESTING

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

def main(argv):
	parser = argparse.ArgumentParser(description='LWA-SV ADP T-Engine Service')
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
	try:
		server_idx = get_numeric_suffix(hostname) - 1
	except ValueError:
		server_idx = 0 # HACK to allow testing on head node "adp"
	log.info("Hostname:     %s", hostname)
	log.info("Server index: %i", server_idx)
	
	pipeline_idx = config['drx']['pipeline_idx']
	recorder_idx = config['drx']['recorder_idx']
	iaddr  = 'p5p1' #config['server']['data_ifaces'][pipeline_idx]
	iport  = 4019 #config['server']['data_ports' ][pipeline_idx]
	oaddr  = config['host']['recorders'][recorder_idx]
	oport  = config['recorder']['port']
	nserver = len(config['host']['servers'])
	server0 = 0
	core0 = config['drx']['first_cpu_core']
	
	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("Dst address:  %s:%i", oaddr, oport)
	log.info("Servers:      %i-%i", server0+1, server0+nserver)
	
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	isock.timeout = 0.5
	
	capture_ring = Ring()
	swap_ring = Ring()
	tengine_ring = Ring()
	
	oaddr = Address(oaddr, oport)
	osock = UDPSocket()
	osock.connect(oaddr)
	
	ops = []
	core = core0
	ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
	                     nsrc=nserver, src0=server0, max_payload_size=9000,
	                     buffer_ntime=2500, slot_ntime=25000, core=core,
	                     utc_start=utc_start_dt))
	core += 1
	ops.append(ReorderChannelsOp(log, capture_ring, swap_ring, 
	                             ntime_gulp=2500, 
	                             core=core))
	core += 1
	ops.append(TEngineOp(log, swap_ring, tengine_ring,
	                     ntime_gulp=2500, 
	                     core=core))
	core += 1
	ops.append(PacketizeOp(log, tengine_ring,
	                       osock=osock, 
	                       nbeam=1, beam0=1, 
	                       npkt_gulp=10, core=core))
	core += 1
	
	threads = [threading.Thread(target=op.main) for op in ops]
	
	log.info("Launching %i thread(s)", len(threads))
	for thread in threads:
		#thread.daemon = True
		thread.start()
	while not shutdown_event.is_set():
		signal.pause()
	log.info("Shutdown, waiting for threads to join")
	for thread in threads:
		thread.join()
	log.info("All done")
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
