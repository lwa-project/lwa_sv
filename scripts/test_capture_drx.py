#!/usr/bin/env python

from adp.ThreadPool import ObjectPool
from adp import Adp
from adp.Adp import Roach2MonitorClient
#from iptools import *

import sys
import os
#import subprocess
#import signal
import time
import datetime
import logging

DATE_FORMAT = "%Y_%m_%dT%H_%M_%S"

def wait_until_utc_sec(utcstr):
	cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)
	while cur_time != utcstr:
		time.sleep(0.01)
		cur_time = datetime.datetime.utcnow().strftime(DATE_FORMAT)

def init(roaches):
	print "Configuring roach 10gbe ports"
	print "Results:", roaches.configure_dual_mode()
	
	start_delay = 3.
	utc_now   = datetime.datetime.utcnow()
	utc_start = utc_now + datetime.timedelta(0, start_delay)
	utc_init  = utc_start - datetime.timedelta(0, 1) # 1 sec before
	utc_start_str = utc_start.strftime(DATE_FORMAT)
	utc_init_str  = utc_start.strftime(DATE_FORMAT)
	print "Starting processing at UTC "+utc_start_str
	wait_until_utc_sec(utc_init_str)
	time.sleep(0.5)
	#roaches.disable_drx_data()
	roaches.start_processing()
	roaches.disable_drx_data()
	freq = 59.98e6
	bw   = 400e3
	roaches.tune_drx(freq, bw)

def main(argv):
	configfile = "/usr/local/share/adp/adp_config.json"
	config     = Adp.parse_config_file(configfile)
	
	log = logging.getLogger(__name__)
	logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
	                              datefmt='%Y-%m-%d %H:%M:%S')
	logFormat.converter = time.gmtime
	logHandler = logging.StreamHandler(sys.stdout)
	logHandler.setFormatter(logFormat)
	log.addHandler(logHandler)
	log.setLevel(logging.INFO)
	
	print "Connecting to roaches"
	nroach  = len(config['host']['roaches'])
	roaches = ObjectPool([Roach2MonitorClient(config, log, i+1)
	                      for i in xrange(nroach)])
	args = [arg.lower() for arg in argv[1:]]
	
	if 'ini' in args:
		init(roaches)
	elif 'start' in args:
		print "Enabling DRX data"
		print roaches.enable_drx_data()
	elif 'tune' in args:
		freq = float(args[-2])*1e6
		bw   = float(args[-1])*1e3
		print "Setting observing bandwidth to", bw
		print roaches.tune_drx(freq, bw)
	elif 'stop' in args:
		print "Disabling DRX data"
		print roaches.disable_drx_data()
	elif 'sht' in args:
		print "Stopping roach processing"
		roaches.stop_processing()
	else:
		print "No command specified (available cmds: ini|start|tune cfreq bw|stop|sht)"
	
	print "All done"
	return 0

if __name__ == "__main__":
	import sys
	sys.exit(main(sys.argv))
