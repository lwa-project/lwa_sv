#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ADP main control service
To be run on ADP headnode

Responds to requests and commands from the telescope MCS

 ** TODO: Implement a script to run on each server that responds to requests
            for MIN/MAX/AVG temperatures, SOFTWARE version, STAT and INFO.
            Temperatures should be taken from the two GPUs in the server

  ** TODO: Enable persistence mode on all GPUs in the cluster
  ** TODO: Can get wake-on-lan working to allow remote startup of cluster?
             Use to implement INI and SHT(?)
  
  TODO: INI/SHT could use IPMI commands to control the servers:
                             power status
          INI:               power on
          SHT:               power soft
          SHT RESTART:       power soft, power status, power on
          SHT SCRAM:         power off
          SHT SCRAM RESTART: power reset
        The pipeline services on the servers would then receive
          terminate signals and automatically shut down (or be
          be killed by a hard shutdown).

  TODO: Need to support disk-space usage queries?

  TODO: Creating a custom service/daemon (e.g., "leda-lock-manager")
  Use the new 'upstart' approach
    https://help.ubuntu.com/community/UbuntuBootupHowto
	http://upstart.ubuntu.com/cookbook/
	http://manpages.ubuntu.com/manpages/natty/en/man5/init.5.html
  Create /etc/init/leda-lock-manager.conf
  sudo initctl reload-configuration
  sudo service leda-lock-manager start
Note that on stop, SIGTERM is sent (by default), and then SIGKILL after 5 secs (by default)

TODO: Timing: Receive commands in second t
              Process commands in second t+1
              Apply   commands in second t+2

Note: The only param changes required for the ROACHes are: TBN mode or DRX chan mask change

Recv thread:
  while not stop_requested():
    cmds_this_second = [[] for _ in range(self.nsubslot)]
    while time_left_this_second > 0:
      new_cmd = wait_for_new_cmd(timeout=time_left_this_second)
      cmds_this_second[new_cmd.subslot].append( new_cmd )
    process_queue.push_back(cmds_this_second)
Process thread:
  while not stop_requested():
    cmd_list = process_queue.pop_front(timeout=0)
    for cmd in cmd_list:
      process_cmd(cmd)
      *** TODO: How to manage cmds in different subslots?
                  Channel mask changes need to be sorted by subslot and then accumulated into ordered subslots
                  Basically just build up a list of commands in each subslot, then
                    merge the commands into combined changes for each subslot.
                    E.g., Multiple DRX commands --> one combined chan_mask update (per subslot)
                          Multiple BAM commands --> one combined beam weights update (per subslot)
                          Multiple FST commands --> one combined stand weights update (per slot)
    apply_changes_next_second()
    sleep_until(next_second)

"""

from __future__ import print_function
import sys
if sys.version_info < (3,):
    range = xrange
    
from adp import MCS2, Adp

import signal
import logging
import time
import os
import argparse
from threading import Event

__version__    = "0.1"
__date__       = '$LastChangedDate: 2015-07-23 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

def try_join(thread, timeout=0.):
	thread.join(timeout)
	return not thread.is_alive()
# Utility function for joining a collection of threads with a timeout
def join_all(threads, timeout):
	deadline = time.time() + timeout
	alive_threads = list(threads)
	while True:
		alive_threads = [t for t in alive_threads if not try_join(t)]
		available_time = max(deadline - time.time(), 0)
		if (len(alive_threads) == 0 or
		    available_time == 0):
			return alive_threads
		alive_threads[0].join(available_time)

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
	
	# TODO: Change these names to something more easily understood
	mcs_local_host  = config['mcs']['headnode']['local_host']
	mcs_local_port  = config['mcs']['headnode']['local_port']
	mcs_remote_host = config['mcs']['headnode']['remote_host']
	mcs_remote_port = config['mcs']['headnode']['remote_port']
	recv_addr = (mcs_local_host, mcs_local_port)
	# Note: This remote host address is no longer used; the reply address
	#         is now obtained from the source IP of each received pkt.
	#         The remote port is still required though.
	send_addr = (mcs_remote_host, mcs_remote_port)
	
	log.info("Local addr:   %s:%i", mcs_local_host,  mcs_local_port)
	log.info("Remote addr:  %s:%i", mcs_remote_host, mcs_remote_port)
	log.info('All dates and times are in UTC unless otherwise noted')
	
	try:
		services = {
			'msg_receiver':  MCS2.MsgReceiver(recv_addr, subsystem=Adp.SUBSYSTEM),
			'msg_processor':  Adp.MsgProcessor(config, log, dry_run=args.dryrun),
			'msg_sender':    MCS2.MsgSender(  send_addr, subsystem=Adp.SUBSYSTEM)
		}
	except MCS2.socket.error as e:
		if e.errno == 98:
			log.error('Socket address in use; '
			          'is adp-control already running?')
			sys.exit(-e.errno)
	except Exception as e:
		log.error('Subservice initialisation failed: %s' % str(e))
		sys.exit(-1)
	services['msg_processor'].input_queue = services['msg_receiver'].msg_queue
	services['msg_sender'   ].input_queue = services['msg_processor'].msg_queue
	
	shutdown_event = Event()
	def handle_signal_terminate(signum, frame):
		SIGNAL_NAMES = dict((k, v) for v, k in \
		                    reversed(sorted(signal.__dict__.items()))
		                    if v.startswith('SIG') and \
		                    not v.startswith('SIG_'))
		log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
		#for service in services.values():
		#	service.request_stop()
		# Note: Services each propagate the stop request downstream
		services['msg_receiver'].request_stop()
		shutdown_event.set()
	log.debug("Setting signal handlers")
	for sig in [signal.SIGHUP,
	            signal.SIGINT,
	            signal.SIGQUIT,
	            signal.SIGTERM,
	            signal.SIGTSTP]:
		signal.signal(sig, handle_signal_terminate)
	
	for name, service in services.items():
		log.debug("Starting service '%s'" % name)
		service.daemon = True
		service.start()
	
	#shutdown_event.wait()
	# WAR for Event.wait() preventing signals from being received
	while not shutdown_event.is_set():
		signal.pause()
	
	join_all(services.values(), config['shutdown_timeout'])
	for name, service in services.items():
		if service.is_alive():
			log.warning("Service %s did not shut down on time "
			            " and will be killed" % name)#service.name)
	
	log.info("All done, exiting")
	return 0


if __name__ == "__main__":
	import sys
	print("--- Start of application ---")
	ret = main(sys.argv)
	print("--- End of application ---")
	sys.exit(ret)
	
