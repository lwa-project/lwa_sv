#!/usr/bin/env python

from __future__ import print_function

from AdpLogging import AdpFileLogger
import AdpConfig
from DeviceMonitor import ROACH2Device

class AdpRoachTempLogger(AdpFileLogger):
	def __init__(self, config, filename, host):
		self.temp_ids = config['roach']['temperatures']
		fileheader = ['#'+'\t'.join(self.temp_ids)]
		AdpFileLogger.__init__(self, config, filename, fileheader)
		self.roach = ROACH2Device(host)
	def update(self):
		"""
		try:
			temps = self.roach.temperatures()
		except:
			temps = {name: float('nan')
			         for name in self.temp_ids}
		"""
		temps = self.roach.temperatures()
		logstr = '\t'.join(['%.1f'%temps[name]
		                    for name in self.temp_ids])
		self.log(logstr)

if __name__ == "__main__":
	import sys
	if len(sys.argv) <= 2:
		print("Usage:", sys.argv[0], "config_file roach_host")
		sys.exit(-1)
	config_filename = sys.argv[1]
	roach_host      = sys.argv[2]
	config = AdpConfig.parse_config_file(config_filename)
	filename = config['log']['files']['roach_temps']
	logger = AdpRoachTempLogger(config, filename, roach_host)
	logger.update()
	sys.exit(0)
