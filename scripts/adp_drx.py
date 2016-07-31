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

__version__    = "0.1"
__date__       = '$LastChangedDate: 2015-07-23 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

def get_time_tag(dt=datetime.datetime.utcnow(), seq_offset=0):
	timestamp = int((dt - ADP_EPOCH).total_seconds())
	time_tag  = timestamp*int(FS) + seq_offset*(int(FS)//int(CHAN_BW))
	return time_tag
def seq_to_time_tag(seq):
	return seq*(int(FS)//int(CHAN_BW))

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
			'nroach':   nroach,
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
		self.capture = CHIPSCapture(*self.args, header_callback=self.header_callback,
		                            **self.kwargs)
		while not self.shutdown_event.is_set():
			self.capture.recv()
		self.capture.end()

class TBFOp(object):
	def __init__(self, log, osock, iring,# oring,
	             core=-1):
		self.log = log
		self.sock = osock
		self.iring = iring
		#self.oring = oring
		self.core  = core
		self.ntime_gulp = 2500
		self.ntime_buf  = int(round(4.0*25000))
	def main(self):
		bfAffinitySetCore(self.core)
		nchan_max = 72#144
		ninput    = 512
		ntime_pkt = 1
		self.iring.resize(self.ntime_gulp*nchan_max*ninput*2,
		                  self.ntime_buf *nchan_max*ninput*2)
		# **TODO: Why does time_tag come out < isequence.time_tag even when
		#           the sleep is 8 seconds?!
		time.sleep(16) # HACK TESTING
		#self.oring.resize(self.ntime_gulp*nchan_max*ninput)
		# TODO: Add remote triggering here
		print "TBF DUMPING"
		# HACK TESTING
		time_tag = get_time_tag(datetime.datetime.utcnow())
		#with self.oring.begin_writing() as oring:
		isequence = self.iring.open_sequence_at(time_tag, guarantee=True)
		time_tag0 = isequence.time_tag
		print "Opened sequence, time_tag0:", time_tag0
		ihdr = json.loads(isequence.header.tostring())
		print "Header:", ihdr
		nchan  = ihdr['nchan']
		chan0  = ihdr['chan0']
		ishape = (-1,nchan,ninput,2)
		frame_nbyte = nchan*ninput*2
		igulp_size = self.ntime_gulp*nchan*ninput*2
		# HACK TESTING
		f = open('/data1/test.tbf', 'wb')
		for ispan in isequence.read(igulp_size):
			seq_offset = ispan.offset // frame_nbyte
			data = ispan.data.reshape(ishape)
			for t in xrange(0, self.ntime_gulp, ntime_pkt):
				time_tag = time_tag0 + seq_to_time_tag(seq_offset + t)
				for c in xrange(0, nchan, TBF_NCHAN_PER_PKT):
					pktdata = data[t:t+ntime_pkt,c:c+TBF_NCHAN_PER_PKT]
					hdr = gen_tbf_header(chan0, time_tag, time_tag0)
					pkt = hdr + pktdata.tostring()
					#*self.sock.send(pkt)
					# HACK TESTING
					f.write(pkt)
		f.close()
		print "TBF DUMP COMPLETE"

def gen_tbf_header(chan0, time_tag, time_tag0):
	sync_word    = 0xDEC0DE5C
	idval        = 0x01
	#frame_num    = (time_tag % int(FS)) // NFRAME_PER_SPECTRUM # Spectrum no.
	frame_num_wrap = 10*60 * int(CHAN_BW) # 10 mins = 15e6, just fits within a uint24
	frame_num    = ((time_tag - time_tag0) // NFRAME_PER_SPECTRUM) % frame_num_wrap + 1 # Spectrum no.
	id_frame_num = idval << 24 | frame_num
	secs_count   = time_tag // int(FS) - M5C_OFFSET
	freq_chan    = chan0
	unassigned   = 0
	return struct.pack('>IIIhhq',
	                   sync_word,
	                   id_frame_num,
	                   secs_count,
	                   freq_chan,
	                   unassigned,
	                   time_tag)

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
	
	pipeline_idx = config['drx']['pipeline_idx']
	recorder_idx = config['drx']['recorder_idx']
	iaddr  = config['server']['data_ifaces'][pipeline_idx]
	iport  = config['server']['data_ports' ][pipeline_idx]
	oaddr  = config['host']['recorders'][recorder_idx]
	oport  = config['recorder']['port']
	nroach_tot = len(config['host']['roaches'])
	nserver    = len(config['host']['servers'])
	nroach, roach0 = nroach_tot, 0
	core0 = config['drx']['first_cpu_core']
	
	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("Dst address:  %s:%i", oaddr, oport)
	log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
	
	# Note: Capture uses Bifrost address+socket objects, while output uses
	#         plain Python address+socket objects.
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	
	capture_ring = Ring()
	#tengine_ring = Ring()
	
	osock = None # TODO
	
	ops = []
	ops.append(CaptureOp(log, isock, capture_ring,
	                     core=core0+0, buffer_ntime=2500,
	                     nroach=nroach, utc_start=utc_start_dt))
	ops.append(TBFOp(log=log, osock=osock, iring=capture_ring,
	                 core=core0+1))
	
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

