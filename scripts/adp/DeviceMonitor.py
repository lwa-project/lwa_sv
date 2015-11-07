
"""
Disk, CPU, GPU and ROACH2 device monitoring

TODO: `impmitool sensor list` shows there are also thermal-region and PSU temps available
"""

import os
# TODO: Replace this with katcp (commands remain the same)
#from telnetlib import Telnet # For reading ROACH2 sensors
import corr
import time
import subprocess            # For calling adc16_dump_chans
import numpy as np
# Available at https://pypi.python.org/pypi/nvidia-ml-py/
try:
	from pynvml import *
	nvmlInit()
	import atexit
	atexit.register(nvmlShutdown)
except ImportError:
	pass

class DiskDevice(object):
	def __init__(self, path):
		self.path = path
	def id(self):
		return self.path
	def usage(self):
		statvfs = os.statvfs(self.path)
		total_bytes = statvfs.f_frsize * statvfs.f_blocks
		free_bytes  = statvfs.f_frsize * statvfs.f_bfree
		avail_bytes = statvfs.f_frsize * statvfs.f_bavail
		# Note: This matches what df does
		used_bytes = total_bytes - free_bytes
		total_avail_bytes = used_bytes + avail_bytes
		return (total_avail_bytes, used_bytes)

class CPUDevice(object):
	def __init__(self, socket=0):
		self.socket = socket
	def id(self):
		return self.socket
	def temperature(self):
		filename = "/sys/class/thermal/thermal_zone%i/temp" % self.socket
		try:
			with open(filename, 'r') as f:
				contents = f.read()
			return float(contents) / 1000.
		except IOError:
			return float('nan')

class GPUSystem(object):
	def __init__(self):
		#nvmlInit()
		pass
	def __del__(self):
		#nvmlShutdown()
		pass
	def driver_version(self):
		return nvmlSystemGetDriverVersion()
	def device_count(self):
		return nvmlDeviceGetCount()
	#def device(self, idx):
	#	return NVMLDevice(idx)
	def devices(self):
		return [GPUDevice(i) for i in xrange(self.device_count())]

class GPUDevice(object):
	def __init__(self, idx):
		self.idx = idx
		if isinstance(idx, int):
			self.handle = nvmlDeviceGetHandleByIndex(idx)
		else:
			self.handle = nvmlDeviceGetHandleByPciBusId(idx)
	def id(self):
		return self.idx
	def name(self):
		return nvmlDeviceGetName(self.handle)
	def memory_info(self):
		"""Returned object provides .total, .free and .used in bytes"""
		return nvmlDeviceGetMemoryInfo(self.handle)
	def temperature(self):
		"""Returns degrees Celcius"""
		return nvmlDeviceGetTemperature(self.handle,
		                                NVML_TEMPERATURE_GPU)
	def temperature_threshold(self):
		"""Returns degrees Celcius"""
		return nvmlDeviceGetTemperatureThreshold(self.handle,
		         NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
	def fan_speed(self):
		"""Returns percentage"""
		return nvmlDeviceGetFanSpeed(self.handle)

from collections import defaultdict

class ROACH2Device(object):
	def __init__(self, host, port=7147):
		self.host = host
		self.port = port
		self.fpga = corr.katcp_wrapper.FpgaClient(host, port)
		time.sleep(0.1)
	def samples(self, stand, pol, nsamps=None):
		if nsamps > 1024:
			raise ValueError("Requested nsamps exceeds limit of 1024")
		data = self._read_samples(nsamps)
		return data[:,stand,pol]
	def samples_all(self, nsamps=None):
		if nsamps is None:
			nsamps = 1024
		if nsamps > 1024:
			raise ValueError("Requested nsamps exceeds limit of 1024")
		return self._read_samples(nsamps)
	def _read_samples(self, nsamps=1024):
		"""Returns an array of shape (nsamps, nstand, npol) dtype=np.int8"""
		nstand = 16 # Not changeable
		npol   = 2  # Not changeable
		cmd    = "adc16_dump_chans.rb"
		out = subprocess.check_output([cmd, "-l", str(nsamps), self.host])#,
		                              #shell=True)
		#cmd = ' '.join([cmd, "-l", str(nsamps), self.host])
		#print cmd
		#out = subprocess.check_output(cmd)
		data = np.fromstring(out, sep=' ', dtype=np.int8)
		data_shape = (nsamps, nstand, npol)
		try:
			data = data.reshape(data_shape)
		except ValueError:
			return np.zeros(data_shape, dtype=np.int8)
		return data
	def temperatures(self):
		sensors = self._read_sensors()
		return {name: sensors['raw.temp.'+name]['value']/1000.
		        for name in ['ambient', 'ppc', 'fpga', 'inlet', 'outlet']}
	def fan_speeds(self):
		sensors = self._read_sensors()
		return {name: sensors['raw.fan.'+name]['value']/1.
		        for name in ['chs0', 'chs1', 'chs2', 'fpga']}
	def voltages(self):
		sensors = self._read_sensors()
		return {name: sensors['raw.voltage.'+name]['value']/1000.
		        for name in ['1v', '1v5', '1v8', '2v5', '3v3',
		                     '5v', '12v', '3v3aux', '5vaux']}
	def currents(self):
		sensors = self._read_sensors()
		return {name: sensors['raw.current.'+name]['value']/1000.
		        for name in ['1v', '1v5', '1v8', '2v5', '3v3',
		                     '5v', '12v']}
	def _read_raw(self, command, *args):
		"""Connects via Telnet and reads the raw text output of a command"""
		tel = Telnet(self.host, self.port)
		command_line = command
		argstr = ' '.join(args)
		if len(args):
			command_line += ' '+argstr
			tel.write('?%s\n' % command_line)
			response = tel.read_until('!%s' % command)
			status = tel.read_until('\n')[:-1].split()[0]
			if status != 'ok':
				raise KeyError(argstr+': '+status)
			return response
	def _read_sensors(self):
		"""Returns a dictionary of all current sensor values
		     along with descriptions, units, types and statuses."""
		results = {}
		"""
		try:
			response = self._read_raw('sensor-list')
		except:
			# HACK to avoid having to catch exceptions all the time
			return defaultdict(lambda : {'value': float('nan')})
		for line in response.split('\n'):
		"""
		status, response = self.fpga._request('sensor-list', 1)
		response = [str(x) for x in response]
		for line in response:
			if line.startswith('#sensor-list'):
				vals = line.split()
				name = vals[1]
				results[name] = {}
				results[name]['desc']  = vals[2].replace(r'\_',' ')
				results[name]['units'] = vals[3]
				results[name]['type']  = vals[4]
		"""
		try:
			response = self._read_raw('sensor-value')
		except:
			# HACK see above
			return defaultdict(lambda : {'value': float('nan')})
		for line in response.split('\n'):
		"""
		status, response = self.fpga._request('sensor-value', 1)
		response = [str(x) for x in response]
		for line in response:
			if line.startswith('#sensor-value'):
				vals = line.split()
				name   = vals[-3]
				results[name]['status'] = vals[-2]
				if results[name]['type'] == 'integer':
					results[name]['value']  = int(vals[-1])
				else:
					results[name]['value']  = vals[-1]
		return results
