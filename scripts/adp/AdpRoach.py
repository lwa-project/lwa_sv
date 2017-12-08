# -*- coding: utf-8 -*-

#try:
import corr
#except ImportError:
#	print "ERROR: Module 'corr' not installed; roaches cannot be controlled"
import subprocess
import time
import numpy as np
import struct
import logging

from iptools import *


# TODO: Consider refactoring into generic ADC16Roach base class and AdpRoach specialisation
class AdpRoach(object):
	def __init__(self, number, port=7147):
		# Note: number is the 1-based index of the roach
		self.num  = number
		self.port = port
		self.connect()
		
		# State variable
		self._fpgaState = {}
		
	@property
	def hostname(self):
		# TODO: Should really get this from an argument to __init__
		#return "roach%i" % self.num
		return "rofl%i" % self.num
		
	def connect(self):
		self.fpga = corr.katcp_wrapper.FpgaClient(self.hostname, self.port, timeout=1.0)
		time.sleep(0.1)
		
	def program(self, boffile, nsubband0, subband_nchan0, nsubband1, subband_nchan1, nsubband2, subband_nchan2, adc_registers={}, max_attempts=5, bypass_pfb=False):
		# Validate
		assert( 0 <= subband_nchan0 and subband_nchan0 < 256 )
		assert( 0 <= subband_nchan1 and subband_nchan1 < 256 )
		assert( 0 <= subband_nchan2 and subband_nchan2 < 256 )
		
		if len(adc_registers) > 0:
			regstring = ','.join(["0x%x=0x%x" % (key,val)
			                      for key,val in adc_registers.items()])
		else:
			regstring = ""
		cmd_line = ["adc16_init.rb", "-r", regstring, self.hostname, boffile]
		ok = False
		attempt = 0
		while not ok:
			if attempt == max_attempts:
				raise RuntimeError("SERDES calibration failed after %i attempt(s)" % max_attempts)
			attempt += 1
			try:
				out = subprocess.check_output(cmd_line)
			except subprocess.CalledProcessError as e:
				raise RuntimeError("Firmware programming on %s failed: %s" % (self.hostname(), str(e)))
			time.sleep(1.0) # Note: Problems arose when this was set to only 0.1s
			try:
				ok, _ = self.check_serdes()
			except ValueError:
				ok = True # Fine if non-ADC16 firmware
			print out
			
		###
		### NOTE:  These next several parameters are only active after adc_rst
		### is called, i.e., after start_processing()
		###
		
		# Reset the FPGA state
		self._fpgaState = {}
		
		# Zero out the ADC delays (or 512 them out, as it were)
		for i in xrange(32):
			self.fpga.write_int('adc_delay%i' % i, 512)
			#self._fpgaState['adc_delay%i' % i] = 512
			
		# Configure the F-engine with basic parameters
		self.fpga.write_int('pkt_roach_id', self.num)
		self.fpga.write_int('pkt_n_roach', 16)
		self.fpga.write_int('fft_use_fengine', 1)		#### Used for test firmware ###
		
		# Configure the PFB
		if bypass_pfb:
			self.disable_pfb()
		else:
			self.enable_pfb()
			
		# Update the FFT shift
		self.fpga.write_int('fft_f1_fft_shift', 65535)
		self.fpga.write_int('fft_f2_fft_shift', 65535)
		self.fpga.write_int('fft_f3_fft_shift', 65535)
		self.fpga.write_int('fft_f4_fft_shift', 65535)
		
		# Configure the allowed bandwidth...
		self.fpga.write_int('pkt_gbe0_n_chan_per_sub', subband_nchan0)
		self.fpga.write_int('pkt_gbe0_n_subband', nsubband0)
		self.fpga.write_int('pkt_gbe1_n_chan_per_sub', subband_nchan1)
		self.fpga.write_int('pkt_gbe1_n_subband', nsubband1)
		self.fpga.write_int('pkt_gbe2_n_chan_per_sub', subband_nchan2)
		self.fpga.write_int('pkt_gbe2_n_subband', nsubband2)
		
		# ... and save these to the internal state so that we can use them later
		self._fpgaState['pkt_gbe0_n_chan_per_sub'] = subband_nchan0
		self._fpgaState['pkt_gbe0_n_subband'] = nsubband0
		self._fpgaState['pkt_gbe1_n_chan_per_sub'] = subband_nchan1
		self._fpgaState['pkt_gbe1_n_subband'] = nsubband1
		self._fpgaState['pkt_gbe2_n_chan_per_sub'] = subband_nchan2
		self._fpgaState['pkt_gbe2_n_subband'] = nsubband2
		
		return out
		
	def unprogram(self):
		try:
			# Note: This throws an exception but still works to unload firmware
			self.fpga.progdev("")
		except RuntimeError:
			pass
			
		# Reset the FPGA state
		self._fpgaState = {}
		
	def check_serdes(self):
		cmd_line =["adc16_status.rb", "-c", self.hostname]
		sp = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
		out, err = sp.communicate()
		if 'device not found' in out:
			raise ValueError("Not an ADC16 firmware")
		if err is not None or 'deskew' not in out:
			print out
			raise RuntimeError("Firmware status request failed: "+str(err))
		ok = not ('X' in out or 'BAD' in out)
		return ok, out
		
	def configure_10gbe(self, gbe_idx, dst_ips, dst_ports, arp_table, src_ip_base="192.168.40.50", src_port_base=4000):
		if isinstance(dst_ports, int):
			dst_ports = [dst_ports] * len(dst_ips)
		mac_base    = mac2int("02:02:00:00:00:00")
		src_ip_base = ip2int(src_ip_base)
		src_port    = src_port_base + gbe_idx
		src_ip      = src_ip_base + (self.num-1)*3 + gbe_idx
		src_mac     = mac_base + src_ip
		self.fpga.config_10gbe_core('pkt_gbe%i' % gbe_idx, src_mac, src_ip, src_port, arp_table)
		ip_addr_bram_vals = np.zeros(1024, 'L')
		ip_addr_bram_vals[:len(dst_ips)] = [ip2int(ip) for ip in dst_ips]
		ip_addr_bram_packed = struct.pack('>1024L', *ip_addr_bram_vals)
		self.fpga.write('pkt_gbe%i_ip_addr_bram' % gbe_idx, ip_addr_bram_packed)
		ip_port_bram_vals = np.zeros(1024, 'L')
		ip_port_bram_vals[:len(dst_ports)] = [int(port) for port in dst_ports]
		ip_port_bram_packed = struct.pack('>1024L', *ip_port_bram_vals)
		self.fpga.write('pkt_gbe%i_ip_port_bram' % gbe_idx, ip_port_bram_packed)
		return self.check_link(gbe_idx)
		
	def reset(self, syncFunction=None):
		if syncFunction is not None:
			syncFunction()
			
		self.fpga.write_int('adc_rst', 0)
		self.fpga.write_int('adc_rst', 1)
		self.fpga.write_int('adc_rst', 0)
		
	def check_link(self, gbe_idx):
		gbe_link = self.fpga.read_int('pkt_gbe%i_linkup' % gbe_idx)
		return bool(gbe_link)
		
	def configure_adc_delay(self, adc_idx, delay_clocks):
		# Note: adc_idx is 0-based and a value of "32" sets all signal paths on the roach
		
		# Validate the inputs
		assert( 0 <= adc_idx <= 32 )
		assert( 0 <= delay_clocks < 1024 )
		
		# Go
		if adc_idx == 32:
			adc_idx = range(32)
		else:
			adc_idx = [adc_idx,]
			
		updated = False
		for adc in adc_idx:
			register = 'adc_delay%i' % adc
			try:
				currDelay = self._fpgaState[register]
			except KeyError:
				currDelay = -1
			if currDelay != delay_clocks:
				self.fpga.write_int(register, delay_clocks)
				self._fpgaState[register] = delay_clocks
				updated |= True
				
		return updated
		
	def read_adc_delay(self, adc_idx):
		# Note: adc_idx is 0-based and a value of "32" sets all signal paths on the roach
		
		# Validate the inputs
		assert( 0 <= adc_idx <= 32 )
		
		# Go
		if adc_idx == 32:
			adc_idx = range(32)
		else:
			adc_idx = [adc_idx,]
			
		delays = []
		for adc in adc_idx:
			register = 'adc_delay%i' % adc
			delays.append( self.fpga.read_int(register) )
		if len(adc_idx) == 1:
			delays = delays[0]
			
		return delays
		
	def configure_fengine(self, gbe_idx, start_chan, scale_factor=1.948, shift_factor=27):
		# Note: gbe_idx is the 0-based index of the gigabit ethernet core
		
		# Validate the inputs
		assert( 0 <= gbe_idx and gbe_idx < 3 )
		assert( 0 <= start_chan and start_chan < 4096 )
		
		# Compute the stop channel and updated packetizer registries as needed
		stop_chan = start_chan + self._fpgaState['pkt_gbe%i_n_chan_per_sub' % gbe_idx] * \
							self._fpgaState['pkt_gbe%i_n_subband' % gbe_idx]
		updated = False
		for baseReg,value in zip(('start_chan', 'stop_chan'), (start_chan, stop_chan)):
			register = 'pkt_gbe%i_%s' % (gbe_idx, baseReg)
			
			## Do no cache the start and stop channels
			#try:
			#	currValue = self._fpgaState[register]
			#except KeyError:
			currValue = -1
			
			if value != currValue:
				self.fpga.write_int(register, value)
				self._fpgaState[register] = value
				updated |= True
				
		# Update the FFT scale as needed
		try:
			currScale = self._fpgaState['scale_factor']
			currShift = self._fpgaState['shift_factor']
		except KeyError:
			currScale, currShift = -1, -1
		if currScale != scale_factor or currShift != shift_factor:
			scaledata = np.ones(4096, 'l') * ((1<<shift_factor) - 1) * scale_factor
			cstr = struct.pack('>4096l', *scaledata)
			self.fpga.write('fft_f1_cg_bpass_bram', cstr)
			self.fpga.write('fft_f2_cg_bpass_bram', cstr)
			self.fpga.write('fft_f3_cg_bpass_bram', cstr)
			self.fpga.write('fft_f4_cg_bpass_bram', cstr)
			self._fpgaState['scale_factor'] = scale_factor
			self._fpgaState['shift_factor'] = shift_factor
			updated |= True
			
		return updated
		
	def _read_pkt_tx_enable(self):
		bitset = self.fpga.read_int('pkt_tx_enable')
		gbe_bitset  = bitset & 0b111
		return gbe_bitset
		
	def _write_pkt_tx_enable(self, gbe_bitset):
		bitset = gbe_bitset & 0b111
		try:
			txReady = self._fpgaState['tx_ready']
		except KeyError:
			txReady = False
		if not txReady:
			self.fpga.write_int('pkt_tx_rst', 0b000)
			self.fpga.write_int('pkt_tx_rst', 0b111)
			self.fpga.write_int('pkt_tx_rst', 0b000)
			
			self._fpgaState['tx_ready'] = True
			
		self.fpga.write_int('pkt_tx_enable', bitset)
		
	# Note: Starting processing without starting data flow
	#         means that we can't get the first 1s of data, because
	#         the data enable takes another 1s to take effect.
	def start_processing(self, syncFunction=None):
		self.stop_processing()
		
		self.reset(syncFunction=syncFunction)
		
		# Ready the packetizer
		gbe_bitset  = 0b111
		self._write_pkt_tx_enable(gbe_bitset)
		
	def enable_data(self, gbe):
		gbe_bitset = self._read_pkt_tx_enable()
		gbe_bitset |= (1<<gbe)
		self._write_pkt_tx_enable(gbe_bitset)
		
	def disable_data(self, gbe):
		gbe_bitset = self._read_pkt_tx_enable()
		gbe_bitset &= ~(1<<gbe)
		self._write_pkt_tx_enable(gbe_bitset)
		
	def stop_processing(self):
		gbe_bitset  = 0b000
		self._write_pkt_tx_enable(gbe_bitset)
		
	def processing_started(self):
		ret = (self.fpga.read_int('pkt_tx_enable') == 0b111)
		return ret
		
	def data_enabled(self, gbe):
		gbe_bitset = self._read_pkt_tx_enable()
		return bool(gbe_bitset & (1<<gbe))
		
	def check_overflow(self):
		gbe0_oflow = self.fpga.read_int('pkt_gbe0_oflow_cnt')
		gbe1_oflow = self.fpga.read_int('pkt_gbe1_oflow_cnt')
		gbe2_oflow = self.fpga.read_int('pkt_gbe2_oflow_cnt')
		return (gbe0_oflow, gbe1_oflow, gbe2_oflow)
		
	def disable_pfb(self):
		"""
		Turn off the PFB frontend.  Returns True.
		"""
		
		self.fpga.write_int('fft_f1_bypass', 1)
		self.fpga.write_int('fft_f2_bypass', 1)
		self.fpga.write_int('fft_f3_bypass', 1)
		self.fpga.write_int('fft_f4_bypass', 1)
		
		return True
		
	def enable_pfb(self):
		"""
		Turn on the PFB frontend.  Returns True.
		"""
		
		self.fpga.write_int('fft_f1_bypass', 0)
		self.fpga.write_int('fft_f2_bypass', 0)
		self.fpga.write_int('fft_f3_bypass', 0)
		self.fpga.write_int('fft_f4_bypass', 0)
		
		return True
		
	def wait_for_pps(self):
		"""
		Function to wait until the PPS hits the roach board.  It blocks until
		the PPS increments 'adc_sync_count' and then returns True.
		"""
		
		p0 = self.fpga.read_int('adc_sync_count')
		p1 = self.fpga.read_int('adc_sync_count')
		
		while p1 <= p0:
			time.sleep(1e-6)
			p1 = self.fpga.read_int('adc_sync_count')
			
		return True
