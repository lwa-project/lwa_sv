
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
	@property
	def hostname(self):
		# TODO: Should really get this from an argument to __init__
		#return "roach%i" % self.num
		return "rofl%i" % self.num
	def connect(self):
		self.fpga = corr.katcp_wrapper.FpgaClient(self.hostname, self.port)
		time.sleep(0.1)
	def program(self, boffile, adc_registers={}, max_attempts=5):
		#self.fpga.progdev(boffile)
		#time.sleep(1.0)
		if len(adc_registers) > 0:
			regstring = ','.join(["0x%x=0x%x" % (key,val)
			                      for key,val in adc_registers.items()])
		else:
			regstring = ""
		#cmd_line = ["adc16_init.rb", self.hostname, boffile, '-g', '%s' % gain]
		cmd_line = ["adc16_init.rb", "-r", regstring, self.hostname, boffile]
		ok = False
		attempt = 0
		while not ok:
			if attempt == max_attempts:
				raise RuntimeError("SERDES calibration failed "
				                   "after %i attempt(s)" % max_attempts)
			attempt += 1
			try:
				out = subprocess.check_output(cmd_line)
			except subprocess.CalledProcessError as e:
				raise RuntimeError("Firmware programming failed: "+str(e))
			time.sleep(1.0) # Note: Problems arose when this was set to only 0.1s
			try:
				ok, _ = self.check_serdes()
			except ValueError:
				ok = True # Fine if non-ADC16 firmware
		self.fpga.write_int('pkt_roach_id', self.num)
		return out
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
	def configure_10gbe(self, gbe_idx, dst_ips, dst_ports, arp_table,
	                    src_ip_base="192.168.40.50", src_port_base=4000):
		if isinstance(dst_ports, int):
			dst_ports = [dst_ports] * len(dst_ips)
		mac_base    = mac2int("02:02:00:00:00:00")
		src_ip_base = ip2int(src_ip_base)
		src_port    = src_port_base + gbe_idx
		src_ip      = src_ip_base + (self.num-1)*2 + gbe_idx
		src_mac     = mac_base + src_ip
		self.fpga.config_10gbe_core('pkt_gbe%i' % gbe_idx,
		                            src_mac, src_ip, src_port, arp_table)
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
		self.fpga.write_int('adc_rst', 0x0)
		self.fpga.write_int('adc_rst', 0x3)
		self.fpga.write_int('adc_rst', 0x0)
		time.sleep(0.1)
	def check_link(self, gbe_idx):
		gbe_link = self.fpga.read_int('pkt_gbe%i_linkup' % gbe_idx)
		return bool(gbe_link)
	def configure_fengine(self, gbe_idx, nsubband, subband_nchan, start_chan,
						  scale_factor=1.948, shift_factor=27):
		# Note: gbe_idx is the 0-based index of the gigabit ethernet core
		assert( 0 <= gbe_idx and gbe_idx < 3 )
		assert( 0 <= start_chan and start_chan < 4096 )
		self.fpga.write_int('pkt_roach_id', self.num)
		stop_chan = start_chan + nsubband*subband_nchan
		self.fpga.write_int('pkt_gbe%i_n_chan_per_sub' % gbe_idx, subband_nchan)
		self.fpga.write_int('pkt_gbe%i_n_subband'      % gbe_idx, nsubband)
		self.fpga.write_int('pkt_gbe%i_start_chan'     % gbe_idx, start_chan)
		self.fpga.write_int('pkt_gbe%i_stop_chan'      % gbe_idx, stop_chan)
		self.fpga.write_int('fft_f1_fft_shift', 65535)
		self.fpga.write_int('fft_f2_fft_shift', 65535)
		self.fpga.write_int('fft_f3_fft_shift', 65535)
		self.fpga.write_int('fft_f4_fft_shift', 65535)
		scaledata = np.ones(4096, 'l') * ((1<<shift_factor) - 1) * scale_factor
		cstr = struct.pack('>4096l', *scaledata)
		self.fpga.write('fft_f1_cg_bpass_bram', cstr)
		self.fpga.write('fft_f2_cg_bpass_bram', cstr)
		self.fpga.write('fft_f3_cg_bpass_bram', cstr)
		self.fpga.write('fft_f4_cg_bpass_bram', cstr)
	def start_data(self, gbe0=True, gbe1=True, gbe2=False):
		self.stop_data()
		gbe_bitset = (int(gbe0)<<0) | (int(gbe1)<<1) | (int(gbe2)<<2)
		self.fpga.write_int('pkt_tx_enable', gbe_bitset)
	def stop_data(self):
		self.fpga.write_int('pkt_tx_enable', 0)
		self.reset()
	def data_enabled(self, gbe):
		bitmask = 1<<gbe
		return bool(self.fpga.read_int('pkt_tx_enable') & bitmask)
	def check_overflow(self):
		gbe0_oflow = self.fpga.read_int('pkt_gbe0_oflow_cnt')
		gbe1_oflow = self.fpga.read_int('pkt_gbe1_oflow_cnt')
		return (gbe0_oflow, gbe1_oflow)
		#fifo_pc_full = self.fpga.read_int('fifo_pc_full')
		#fifo_percent = (fifo_pc_full * 1.0 / 2**12) * 100
		#return fifo_percent, (gbe0_oflow, gbe1_oflow)
