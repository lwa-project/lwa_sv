#!/usr/bin/env python

from __future__ import print_function

from AdpLogging import AdpFileLogger
import AdpConfig
import subprocess
import re
from ThreadPool import ThreadPool

def read_hosts():
    """ Parse contents of system hosts file into list of (ip, hostname) pairs """
    with open('/etc/hosts', 'r') as hostfile:
        hostlines = hostfile.readlines()
    hosts = []
    for hostline in hostlines:
        hostline = re.sub('\t', ' ', hostline).strip('\n').strip()
        if hostline == '':
            continue
        if not ':' in hostline: # Skip IPv6 entries
            host_cols = hostline.split()
            if not host_cols[0].startswith('#'):
                #hosts.append((host_cols[0], host_cols[1]))
                hostname = host_cols[1]
                hosts.append(hostname)
    return hosts

def ping(ip):
    ret = subprocess.call(['ping', '-c', '1', ip],
            shell=True,
            stdout=open('/dev/null', 'w'),
            stderr=subprocess.STDOUT)
    if ret == 0: # Host responded
        return True
    else: # Host didn't respond
	    return False

class AdpDownHostsLogger(AdpFileLogger):
	def __init__(self, config, filename):
		fileheader = ['#'+'DOWN_HOSTS']
		AdpFileLogger.__init__(self, config, filename, fileheader)
		self.thread_pool = ThreadPool()
	def update(self):
		hosts = read_hosts()
		for host in hosts:
			self.thread_pool.add_task(ping, host)
		responses = self.thread_pool.wait()
		down_hosts = []
		for host, response in zip(hosts, responses):
			if not response:
				down_hosts.append(host)
		logstr = ','.join(down_hosts)
		self.log(logstr)

if __name__ == "__main__":
	import sys
	if len(sys.argv) <= 1:
		print("Usage:", sys.argv[0], "config_file")
		sys.exit(-1)
	config_filename = sys.argv[1]
	config = AdpConfig.parse_config_file(config_filename)
	filename = config['log']['files']['down_hosts']
	logger = AdpDownHostsLogger(config, filename)
	logger.update()
	sys.exit(0)
