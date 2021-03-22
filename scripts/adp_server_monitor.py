#!/usr/bin/env python
"""
ADP server monitor service
To be run on ADP servers

Responds to requests from the ADP main control script for:
  TEMP_MIN/MAX/AVG temperatures
  SOFTWARE version
  STAT code and INFO string
"""

from __future__ import print_function

__version__    = "0.1"
__date__       = '$LastChangedDate: 2015-07-23 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

from StoppableThread import StoppableThread
import zmq
import json
import platform
from DeviceMonitor import CPUDevice, GPUDevice, DiskDevice, GPUSystem

class ServerMonitor(object):
	def __init__(self, cpu_ids=[], gpu_ids=[], disk_ids=[]):
		self.gpu_system  = GPUSystem()
		self.cpus  = [ CPUDevice(idx)  for idx  in cpu_ids]
		self.gpus  = [ GPUDevice(idx)  for idx  in gpu_ids]
		self.disks = [DiskDevice(path) for path in disk_ids]
	def get_device_ids(self):
		cpu_ids = ['CPU%i'%cpu.id() for cpu in self.cpus]
		gpu_ids = ['GPU%i'%gpu.id() for gpu in self.gpus]
		return cpu_ids + gpu_ids
	def get_device_temps(self):
		cpu_temps = [cpu.temperature() for cpu in self.cpus]
		gpu_temps = [gpu.temperature() for gpu in self.gpus]
		return cpu_temps + gpu_temps
	def get_disk_ids(self):
		return [disk.path for disk in self.disks]
	def get_disk_usages(self):
		return [disk.usage() for disk in self.disks]
	
class AdpServerMonitor(StoppableThread):
	def __init__(self, config, log, monitor, timeout=0.1):
		StoppableThread.__init__(self)
		self.config  = config
		self.log     = log
		self.monitor = monitor
		self.zmqctx  = zmq.Context()
		self.sock    = self.zmqctx.socket(zmq.REP)
		addr = "tcp://%s:%i" % (config['mcs']['server']['local_host'],
		                        config['mcs']['server']['local_port'])
		self.log.info("Binding socket to: %s", addr)
		self.sock.bind(addr)
		self.sock.RCVTIMEO = int(timeout*1000)
	def run(self):
		self.log.info("Listening for requests")
		while not self.stop_requested():
			self.log.debug("Waiting for requests")
			try:
				req = self.sock.recv()
			except zmq.error.Again:
				pass
			except zmq.error.ZMQError: # For "Interrupted system call"
				pass
			else:
				self.log.debug("Received request: %s", req)
				try:
					reply = self._process_request(req)
				except Exception as e:
					reply = {'status': -500,
					         'info':   "Internal server error: %s"%e,
					         'data':   None}
				self.log.debug("Replying with: %s", json.dumps(reply))
				self.sock.send_json(reply)
	def _process_request(self, req):
		status = 0
		info   = 'OK'
		data   = None
		if req == 'PNG':
			pass
		elif req == 'STAT':
			# TODO: What information to encode here?
			pass
		elif req == 'INFO':
			# TODO: What information to give here?
			pass
		elif req.startswith('TEMP'):
			temps = self.monitor.get_device_temps()
			if req == 'TEMP_MAX':
				data = max(temps)
			elif req == 'TEMP_MIN':
				data = min(temps)
			elif req == 'TEMP_AVG':
				data = sum(temps)/float(len(temps))
			else:
				status = -404
				info   = "Unknown MIB entry: %s" % req
				self.log.error(info)
		elif req == 'SOFTWARE':
			# TODO: Get pipeline version etc.
			kernel_version = platform.uname()[2]
			data = ("krnl:%s,adp_srv_mon:%s,gpu_drv:%s" %
			        (kernel_version,
			         __version__,
			         self.monitor.gpu_system.driver_version()))
		else:
			status = -1
			info   = "Unknown MIB entry: %s" % req
			self.log.error(info)
		return {'status': status,
		        'info':   info,
		        'data':   data}

import AdpConfig
import MCS2
import logging
from logging.handlers import TimedRotatingFileHandler
import time
import os
import signal

def main(argv):
	import sys
	if len(sys.argv) <= 1:
		print("Usage:", sys.argv[0], "config_file")
		sys.exit(-1)
	config_filename = sys.argv[1]
	config = AdpConfig.parse_config_file(config_filename)
	
	# TODO: Try to encapsulate this in something simple in AdpLogging
	log = logging.getLogger(__name__)
	logFormat = logging.Formatter(config['log']['msg_format'],
	                              datefmt=config['log']['date_format'])
	logFormat.converter = time.gmtime
	logHandler = logging.StreamHandler(sys.stdout)
	logHandler.setFormatter(logFormat)
	log.addHandler(logHandler)
	log.setLevel(logging.INFO)
	
	short_date = ' '.join(__date__.split()[1:4])
	log.info("Starting %s with PID %i", argv[0], os.getpid())
	log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
	log.info("Version:      %s", __version__)
	log.info("Last changed: %s", short_date)
	log.info("Current MJD:  %f", MCS2.slot2mjd())
	log.info("Current MPM:  %i", MCS2.slot2mpm())
	log.info('All dates and times are in UTC unless otherwise noted')
	
	log.debug("Creating server monitor")
	monitor = ServerMonitor(config['server']['cpu_ids'],
							config['server']['gpu_ids'],
							config['server']['disk_ids'])
	
	log.debug("Creating ADP server monitor")
	adpmon = AdpServerMonitor(config, log, monitor)
	
	def handle_signal_terminate(signum, frame):
		SIGNAL_NAMES = dict((k, v) for v, k in \
		                    reversed(sorted(signal.__dict__.items()))
		                    if v.startswith('SIG') and \
		                    not v.startswith('SIG_'))
		log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
		adpmon.request_stop()
	log.debug("Setting signal handlers")
	signal.signal(signal.SIGHUP,  handle_signal_terminate)
	signal.signal(signal.SIGINT,  handle_signal_terminate)
	signal.signal(signal.SIGQUIT, handle_signal_terminate)
	signal.signal(signal.SIGTERM, handle_signal_terminate)
	signal.signal(signal.SIGTSTP, handle_signal_terminate)
	
	log.debug("Running ADP server monitor")
	adpmon.run()
	
	log.info("All done, exiting")
	sys.exit(0)
	
if __name__ == "__main__":
	import sys
	sys.exit(main(sys.argv))
