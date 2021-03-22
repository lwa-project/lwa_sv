#!/usr/bin/env python

from __future__ import print_function

from AdpLogging import AdpFileLogger
import AdpConfig
from DeviceMonitor import CPUDevice, GPUDevice

class AdpServerTempLogger(AdpFileLogger):
	def __init__(self, config, filename):
		cpu_ids = config['server']['cpu_ids']
		gpu_ids = config['server']['gpu_ids']
		cpu_id_strs = ['CPU%i'%i for i in cpu_ids]
		gpu_id_strs = ['GPU%i'%i for i in gpu_ids]
		fileheader = ['#'+'\t'.join(cpu_id_strs+gpu_id_strs)]
		AdpFileLogger.__init__(self, config, filename, fileheader)
		self.cpus = [CPUDevice(i) for i in cpu_ids]
		self.gpus = [GPUDevice(i) for i in gpu_ids]
	def update(self):
		try:
			cpu_temps = [cpu.temperature() for cpu in self.cpus]
		except:
			cpu_temps = [float('nan') for cpu in self.cpus]
		try:
			gpu_temps = [gpu.temperature() for gpu in self.gpus]
		except:
			gpu_temps = [float('nan') for gpu in self.gpus]
		temps = cpu_temps + gpu_temps
		logstr   = '\t'.join(['%.1f'%temp for temp in temps])
		self.log(logstr)

if __name__ == "__main__":
	import sys
	if len(sys.argv) <= 1:
		print("Usage:", sys.argv[0], "config_file")
		sys.exit(-1)
	config_filename = sys.argv[1]
	config = AdpConfig.parse_config_file(config_filename)
	filename = config['log']['files']['server_temps']
	logger = AdpServerTempLogger(config, filename)
	logger.update()
	sys.exit(0)
