#!/usr/bin/env python

from __future__ import print_function

from AdpLogging import AdpFileLogger
import AdpConfig
from DeviceMonitor import DiskDevice
import math

def round_up(x, n=0):
	scale = 10.**n
	return math.ceil(x * scale) / scale

class AdpDiskUsageLogger(AdpFileLogger):
	def __init__(self, config, filename):
		paths = config['server']['disk_ids']
		fileheader = ['#'+'\t'.join(paths)]
		super(AdpDiskUsageLogger, self).__init__(config, filename, fileheader)
		self.disks = [DiskDevice(path) for path in paths]
	def update(self):
		usages   = [disk.usage() for disk in self.disks]
		percents = [round_up(100.*usage[1]/usage[0], 1) for usage in usages]
		logstr   = '\t'.join(['%.1f%%'%percent for percent in percents])
		self.log(logstr)

if __name__ == "__main__":
	import sys
	if len(sys.argv) <= 1:
		print("Usage:", sys.argv[0], "config_file")
		sys.exit(-1)
	config_filename = sys.argv[1]
	config = AdpConfig.parse_config_file(config_filename)
	
	filename = config['log']['files']['disk_usage']
	logger = AdpDiskUsageLogger(config, filename)
	logger.update()
	sys.exit(0)
