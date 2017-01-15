# -*- coding: utf-8 -*-

#try:
import corr
#except ImportError:
#	print "ERROR: Module 'corr' not installed; roaches cannot be controlled"
import subprocess
import time
import numpy as np
import struct

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
		self.fpga = corr.katcp_wrapper.FpgaClient(self.hostname, self.port)
		time.sleep(0.1)
		
	def program(self, boffile, adc_registers={}, max_attempts=5):
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
			
		# Configure the F-engine with basic parameters
		self.fpga.write_int('pkt_roach_id', self.num)
		self.fpga.write_int('pkt_n_roach', 16)
		self.fpga.write_int('fft_use_fengine', 1)		#### Used for test firmware ###
		self._update_pkt_registers()
		
		return out
		
	def unprogram(self):
		try:
			# Note: This throws an exception but still works to unload firmware
			self.fpga.progdev("")
		except RuntimeError:
			pass
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
		src_ip      = src_ip_base + (self.num-1)*2 + gbe_idx
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
		
	def reset(self):
		self.fpga.write_int('adc_rst', 0)
		self.fpga.write_int('adc_rst', 1)
		self.fpga.write_int('adc_rst', 0)
		
	def check_link(self, gbe_idx):
		gbe_link = self.fpga.read_int('pkt_gbe%i_linkup' % gbe_idx)
		return bool(gbe_link)
		
	def configure_fengine(self, gbe_idx, nsubband, subband_nchan, start_chan, scale_factor=1.948, shift_factor=27, bypass_pfb=False):
		# Note: gbe_idx is the 0-based index of the gigabit ethernet core
		
		# Validate the inputs
		assert( 0 <= gbe_idx and gbe_idx < 3 )
		assert( 0 <= start_chan and start_chan < 4096 )
		assert( 0 <= subband_nchan and subband_nchan < 256 )
		
		# Compute the stop channel and updated packetizer registries as needed
		stop_chan = start_chan + nsubband*subband_nchan
		updated = False
		for baseReg,value in zip(('n_chan_per_sub', 'n_subband', 'start_chan', 'stop_chan'), 
							(subband_nchan, nsubband, start_chan, stop_chan)):
			register = 'pkt_gbe%i_%s' % (gbe_idx, baseReg)
			
			try:
				currValue = self._fpgaState[register]
			except KeyError:
				currValue = -1
				
			if value != currValue:
				self.fpga.write_int(register, value)
				self._fpgaState[register] = value
				updated = True
				
		# Update the PFB setting as needed
		if bypass_pfb:
			updated |= self.disable_pfb()
		else:
			updated |= self.enable_pfb()
			
		# Update the FFT shift as needed
		try:
			currShift = self._fpgaState['fft_shift']
		except KeyError:
			currShift = -1
		if currShift != 65535:
			self.fpga.write_int('fft_f1_fft_shift', 65535)
			self.fpga.write_int('fft_f2_fft_shift', 65535)
			self.fpga.write_int('fft_f3_fft_shift', 65535)
			self.fpga.write_int('fft_f4_fft_shift', 65535)
			self._fpgaState['fft_shift'] = 65535
			updated = True
			
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
			updated = True
			
		# Flush any changes to the system
		if updated:
			self._update_pkt_registers()
			
	def _update_pkt_registers(self):
		self.fpga.write_int('pkt_update_reg', 0)
		self.fpga.write_int('pkt_update_reg', 1)
		self.fpga.write_int('pkt_update_reg', 0)
		
	def _read_pkt_tx_enable(self):
		bitset = self.fpga.read_int('pkt_tx_enable')
		#fifo_bitset =  bitset & 0b0011
		#gbe_bitset  = (bitset & 0b1100) >> 2
		fifo_bitset = 0 # TODO: Remove this
		gbe_bitset  = bitset & 0b11
		return gbe_bitset, fifo_bitset
	def _write_pkt_tx_enable(self, gbe_bitset, fifo_bitset):
		#bitset = ((gbe_bitset & 0b11) << 2) | (fifo_bitset & 0b11)
		bitset = gbe_bitset & 0b11
		try:
			fifoEnabled = self._fpgaState['fifo_enabled']
		except KeyError:
			fifoEnabled = False
		if not fifoEnabled:
			self.fpga.write_int('pkt_fifo_enable', 0b11)
			self.fpga.write_int('pkt_tx_rst', 0b00)
			self.fpga.write_int('pkt_tx_rst', 0b11)
			self.fpga.write_int('pkt_tx_rst', 0b00)
			
			self._fpgaState['fifo_enabled'] = True
			
		self.fpga.write_int('pkt_tx_enable', bitset)
		
	# Note: Starting processing without starting data flow
	#         means that we can't get the first 1s of data, because
	#         the data enable takes another 1s to take effect.
	def start_processing(self):
		self.stop_processing()
		self.reset_sequence_number() # TODO: Check that this is the right place to do this
		self.reset() # Must call this here to initialise things
		self._update_pkt_registers()
		
		## Note: We restore the previous gbe states
		#gbe_bitset, fifo_bitset = self._read_pkt_tx_enable()
		fifo_bitset = 0b11
		gbe_bitset  = 0b00
		#gbe_bitset  = 0b11
		self._write_pkt_tx_enable(gbe_bitset, fifo_bitset)
		
	def enable_data(self, gbe):
		gbe_bitset, fifo_bitset = self._read_pkt_tx_enable()
		#if fifo_bitset != 0b11:
		#	raise RuntimeError("Enable data requested but data not started")
		gbe_bitset |= (1<<gbe)
		self._write_pkt_tx_enable(gbe_bitset, fifo_bitset)
		
	def disable_data(self, gbe):
		gbe_bitset, fifo_bitset = self._read_pkt_tx_enable()
		#if fifo_bitset != 0b11:
		#	raise RuntimeError("Disable data requested but data not started")
		gbe_bitset &= ~(1<<gbe)
		self._write_pkt_tx_enable(gbe_bitset, fifo_bitset)
		
	def stop_processing(self):
		fifo_bitset = 0b00
		gbe_bitset  = 0b00
		self._write_pkt_tx_enable(gbe_bitset, fifo_bitset)
		
	def processing_started(self):
		#gbe_bitset, fifo_bitset = self._read_pkt_tx_enable()
		#return (fifo_bitset == 0b11)
		ret = (self.fpga.read_int('pkt_fifo_enable') == 0b11)
		return ret
		
	def data_enabled(self, gbe):
		gbe_bitset, fifo_bitset = self._read_pkt_tx_enable()
		return bool(gbe_bitset & (1<<gbe))
		
	def check_overflow(self):
		gbe0_oflow = self.fpga.read_int('pkt_gbe0_oflow_cnt')
		gbe1_oflow = self.fpga.read_int('pkt_gbe1_oflow_cnt')
		return (gbe0_oflow, gbe1_oflow)
		
	def reset_sequence_number(self):
		self.fpga.write_int('pkt_rst_seq_num', 0)
		self.fpga.write_int('pkt_rst_seq_num', 1)
		self.fpga.write_int('pkt_rst_seq_num', 0)
		
	def disable_pfb(self):
		"""
		Turn off the PFB frontend.  Returns True if the PFB is turned off, 
		False if it is already off.
		"""
		
		try:
			pfbOn = self._fpgaState['pfb']
		except KeyError:
			# Default state is on
			pfbOn = True
			
		if pfbOn:
			self.fpga.write_int('fft_f1_bypass', 1)
			self.fpga.write_int('fft_f2_bypass', 1)
			self.fpga.write_int('fft_f3_bypass', 1)
			self.fpga.write_int('fft_f4_bypass', 1)
			
			self._fpgaState['pfb'] = False
			return True
		else:
			return False
			
	def enable_pfb(self):
		"""
		Turn on the PFB frontend.  Returns True if the PFB is turned on, 
		False if it is already on.
		"""
		
		try:
			pfbOn = self._fpgaState['pfb']
		except KeyError:
			# Default state is on
			pfbOn = True
			
		if not pfbOn:
			self.fpga.write_int('fft_f1_bypass', 0)
			self.fpga.write_int('fft_f2_bypass', 0)
			self.fpga.write_int('fft_f3_bypass', 0)
			self.fpga.write_int('fft_f4_bypass', 0)
			
			self._fpgaState['pfb'] = True
			return True
		else:
			return False
