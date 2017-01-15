#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from bifrost import Ring, Address, UDPSocket, CHIPSCapture, bfAffinitySetCore

from adp import ISC

import signal
import logging
import time
import math
import os
import argparse
from threading import Event

import numpy as np
import threading
import json
import socket
import struct
import time
import datetime
from ctypes import create_string_buffer

#from np.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft

import pycuda.driver as cuda
import pycuda.gpuarray as gpa
from pycuda.compiler import SourceModule
from pycuda import tools

__version__    = "0.1"
__date__       = '$LastChangedDate: 2015-07-23 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

NFRAME_PER_SPECTRUM = int(FS) // int(CHAN_BW) # 7840
NFRAME_PER_SUBSLOT  = int(FS) // NSUBSLOT_PER_SEC

class CaptureOp(object):
	def __init__(self, log, *args, **kwargs):
		self.shutdown_event = threading.Event()
		self.log    = log
		self.args   = args
		self.kwargs = kwargs
		self.utc_start = self.kwargs['utc_start']
		with open('.log.txt', 'w') as f:
			f.write('HELLO WORLD')
	def shutdown(self):
		self.shutdown_event.set()
	def header_callback(self, seq0, chan0, nchan, nroach, char_ptr_ptr):
		timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
		time_tag0  = timestamp0 * int(FS)
		time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
		hdr = {
			'time_tag': time_tag,
			'chan0':    chan0,
			'nchan':    nchan,
			'nroach':   nroach, # TODO: Have these passed to the callback too
			'nbit':     8       # TODO: Probably shouldn't be hard coded
		}
		hdr_str = json.dumps(hdr)
		# TODO: Can't pad with NULL because returned as C-string
		#hdr_str = json.dumps(hdr).ljust(4096, '\0')
		#hdr_str = json.dumps(hdr).ljust(4096, ' ')
		self.header_buf = create_string_buffer(hdr_str)
		char_ptr_ptr[0] = self.header_buf
		return time_tag
	def main(self):
		self.capture = CHIPSCapture(*self.args,
		                            header_callback=self.header_callback,
		                            **self.kwargs)
		
		## HACK TESTING
		#sync_pipelines = MCS.Synchronizer('TBN2')
		#print "Sleeping for 5 secs"
		#time.sleep(5)
		#sync_pipelines(0)
		
		while not self.shutdown_event.is_set():
			self.capture.recv()
		self.capture.end()


class CopyOp(object):
	def __init__(self, log, iring, oring, nchan_max=256, nroach_max=16, ntime_gulp=2500, core=-1):
		self.log   = log
		self.iring = iring
		self.oring = oring
		ninput_max = nroach_max*32#*2
		self.ntime_gulp = ntime_gulp
		self.core = core
		self.iring.resize(self.ntime_gulp*nchan_max*ninput_max*2)#frame_size_max)
		self.oring.resize(self.ntime_gulp*nchan_max*ninput_max*2)#frame_size_max)
		
	def main(self):
		bfAffinitySetCore(self.core)
		
		with self.oring.begin_writing() as oring:
			for isequence in self.iring.read():#guarantee=False):
				ihdr = json.loads(isequence.header.tostring())
				
				self.log.info("Copy: Start of new sequence: %s", str(ihdr))
				nchan  = ihdr['nchan']
				nroach = ihdr['nroach']
				nstand = nroach*16
				npol   = 2
				
				# Copy to the trigger buffer
				igulp_size = self.ntime_gulp*nchan*nstand*npol*2
				ishape = (self.ntime_gulp,nchan,nstand,npol,2)
				oshape = (self.ntime_gulp,nchan,nstand,npol,2)
				ogulp_size = self.ntime_gulp*nchan*nstand*npol*2
				ohdr_str = json.dumps(ihdr)
				with oring.begin_sequence(time_tag=isequence.time_tag, header=ohdr_str) as osequence:
					for ispan in isequence.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
							
						with osequence.reserve(ogulp_size) as ospan:
							data = ispan.data_view(np.int8).reshape(ishape)
							odata = ospan.data_view(np.int8).reshape(oshape)
							odata[...] = data


def gen_tbf_header(chan0, time_tag, time_tag0):
	sync_word    = 0xDEC0DE5C
	idval        = 0x01
	#frame_num    = (time_tag % int(FS)) // NFRAME_PER_SPECTRUM # Spectrum no.
	frame_num_wrap = 10*60 * int(CHAN_BW) # 10 mins = 15e6, just fits within a uint24
	frame_num    = ((time_tag - time_tag0) // NFRAME_PER_SPECTRUM) % frame_num_wrap + 1 # Spectrum no.
	id_frame_num = idval << 24 | frame_num
	secs_count   = time_tag // int(FS) - M5C_OFFSET
	freq_chan    = chan0
	unassigned   = 0
	return struct.pack('>IIIhhq', sync_word, id_frame_num, secs_count, freq_chan, unassigned, time_tag)


class TriggerOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, addr, port, nchan_max=256, nroach_max=16, ntime_gulp=2500, core=-1):

		self.log   = log
		self.iring = iring
		ninput_max = nroach_max*32#*2
		self.ntime_gulp = ntime_gulp
		self.core = core
		#self.iring.resize(self.ntime_gulp*nchan_max*ninput_max*2)#frame_size_max)
		
		self.configMessage = ISC.TriggerClient(addr=('adp',5832))
		
	@ISC.logException
	def main(self):
		bfAffinitySetCore(self.core)
		
		triggerSequence = 0
		
		while True:
			config = self.configMessage()
			if config:
				print "Trigger: New trigger received: %s" % str(config)
			samplesToDump = config[1]
			print "HERE: Dumping", samplesToDump, "samples"
			triggerSequence += 1
			
			fh = open('/data1/tbf_trigger_%s_%i.dat' % (socket.gethostname(), triggerSequence), 'wb')
			
			done = False
			samples = 0
			for isequence in self.iring.read():#guarantee=False):
				ihdr = json.loads(isequence.header.tostring())
				time_tag0 = isequence.time_tag
				
				self.log.info("Trigger: Start of new sequence: %s", str(ihdr))
				chan0  = ihdr['chan0']
				nchan  = ihdr['nchan']
				nroach = ihdr['nroach']
				nstand = 256
				npol   = 2
				igulp_size = self.ntime_gulp*nchan*nstand*npol*2
				ishape = (self.ntime_gulp,nchan,nstand,npol,2)
				
				payload = np.zeros((12,256,2), dtype=np.uint8)
				
				for ispan in isequence.read(igulp_size):
					#print "TENGINE processing"
					if ispan.size < igulp_size:
						continue # Ignore final gulp
						
					data = ispan.data_view(np.uint8).reshape(ishape)
					data  = data >> 4 # Correct for 8->16bit padding
					
					if samples == 100:
						self.log.info("First 10 stand values are are:")
						for sbc in xrange(20):
							sbs = sbc/2
							sbp = sbc%2
							self.log.info("  %i @ %i %i -> %s %s", sbc+1, sbs+1, sbp+1, bin(data[0,0,sbs,sbp,0]), bin(data[0,0,sbs,sbp,1]))
							
					for t in xrange(self.ntime_gulp):
						time_tag = time_tag0 + t*NFRAME_PER_SPECTRUM
						samples += 1
						
						for c in xrange(0, (nchan/12)*12, 12):
							cSt = c
							cSp = cSt + 12
							
							header = gen_tbf_header(cSt + chan0, time_tag, time_tag0)
							payload[:,:,:]  = ((data[t,cSt:cSp,:,:,0] & 0xF) << 4)
							payload[:,:,:] |=  (data[t,cSt:cSp,:,:,1] & 0xF)
							packet = header + payload.tostring()
							fh.write(packet)
							
						if samples >= samplesToDump:
							done = True
							break
							
					if done:
						break
						
				if done:
					break
					
			fh.close()


class gpuBeam(object):
	"""
	GPU beamformer class that takes in data, frequencies, gains, and delays 
	to form a beam.
	"""
	
	def __init__(self, nTime, nChan, nStand, log=None):
		# Basic data size parameters needed for just-in-time compilation
		self.nTime = nTime
		self.nChan = nChan
		self.nStand = nStand
		self.log = log
		
		#  PyCUDA setup and device initialization
		cuda.init()
		self.ctx = tools.make_default_context()
		self.dev = self.ctx.get_device()
		self.tpb =  self.dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
		if self.log is not None:
			self.log.info("Initializing GPU beamformer for %i times, %i channels, and %i stands", self.nTime, self.nChan, self.nStand)
			self.log.info("  Device: %s", self.dev.name())
			self.log.info("    Compute capability: %s", self.dev.compute_capability())
			self.log.info("    Max Threads per block: %i", self.dev.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK))
			
		# Beamformer block and grid sizes
		self.bSize = (self.tpb/32, 
				    32,  
				    1)
		self.gSize = (self.nTime/self.bSize[0] + self.nTime%self.bSize[0], 
				    self.nChan/self.bSize[1] + self.nChan%self.bSize[1], 
				    1)
					
		# Defaul phase rotation  and gain arrays
		self.phaseRot = gpa.to_gpu(np.zeros((self.nChan,self.nStand,2), dtype=np.complex64))
		self.gains = gpa.to_gpu(np.zeros((self.nStand, 2, 2), dtype=np.uint16))
		
		# Beam output setup
		self.beam = gpa.to_gpu(np.zeros((self.nTime, self.nChan, 2), dtype=np.complex64))
		
		# Build the CUDA code for this problem
		self._compile()
		
	def setChannels(self, chans):
		"""
		Given a NumPy 1-D array of channel numbers as numpy.int16 values, 
		update the internal frequency channel cache.
		"""
		
		# Validate and send
		assert(chans.size == self.nChan)
		assert(chans.dtype == np.int16)
		self.chans = gpa.to_gpu(chans)
		
	def setDelays(self, delays, chans=None):
		"""
		Given a NumPy 1-D array of DP packed delays as numpy.uint16 values, 
		update the internal phase rotation cache.  Optionally, the frequency
		channel information may also be updated at this point via the 'chans'
		keyword.
		"""
		
		assert(delays.size == self.nStand*2)
		assert(delays.dtype == np.uint16)
		if chans is not None:
			self.setChannels(chans)
			
		# Validate
		self.delays = gpa.to_gpu(delays)
		# Setup the problem sizes
		bSize = (self.tpb/self.nStand, self.nStand, 1)
		gSize = (self.nChan/bSize[0]+self.nChan%bSize[0], 1, 1)
		# Compute the phase rotations
		self._computePhaseRotation(self.chans, self.delays, self.phaseRot, block=bSize, grid=gSize)
		return True
		
	def setPhaseRotation(self, phaseRot):
		"""
		Given a NumPy 3-D array of phase rotators as numpy.complex64 values, 
		update the internal phase rotation cache.
		
		..note::
		  This overrides any channel/delay information provided by
		  setChannels() and setDelays().
		"""
		
		# Validate and send
		assert(phaseRot.shape[0] == self.nChan)
		assert(phaseRot.shape[1] == self.nStand)
		assert(phaseRot.shape[2] == 2)
		assert(phaseRot.dtype == np.complex64)
		self.phaseRot = gpa.to_gpu(phaseRot)
		return True
		
	def setGains(self, gains):
		"""
		Given a NumPy 3-D array of DP packed gains as numpy.uint16 values, 
		update the internal gain cache.
		"""
		
		# Validate and send
		assert(gains.shape[0] == self.nStand)
		assert(gains.shape[1] == 2)
		assert(gains.shape[2] == 2)
		assert(gains.dtype == np.uint16)
		self.gains = gpa.to_gpu(gains)
		return True
		
	def _compile(self):
		"""
		Build the CUDA code necessary to beamform data of this size.
		"""
		
		kernels = SourceModule("""
#include <stdio.h>
#include <pycuda-complex.hpp>

#define PI (float) 3.1415926535898

// Conversion macros
/* Channel number-ish to frequency in Hz #### FIX THIS #### */
#define CHAN_TO_DOUBLE(X) (((double) X + 3) * 25e3)
/* Packed DP delay to delay in s */
#define PACKEDDELAY_TO_DOUBLE(X)  (((double) (((X>>4)&0xFFF) + ((double) (X&0xF))/16.0)) / 196e6)
/* Packed DP gain to float gain */
#define PACKEDGAIN_TO_FLOAT(X)  (((float) X) / ((float) 32767.0))
/* Packed Bifrost data to float */
#define BIFROST_TO_FLOAT(X)  ((float) (X>>4))


__global__ void computePhaseRotation(short int *chan, short int *packedDelays, pycuda::complex<float> *phaseRot) {
	int fid = threadIdx.x + blockDim.x * blockIdx.x;
	int sid = threadIdx.y + blockDim.y * blockIdx.y;
	
	short int temp;
	double delay;
	pycuda::complex <float> CI(0.,1.);
	
	if( fid < %(nChan)i && sid < %(nStand)i ) {
		temp = packedDelays[2*sid + 0];
		delay = PACKEDDELAY_TO_DOUBLE(temp);
		phaseRot[fid*%(nStand)i*2 + sid*2 + 0] = exp(-2*PI*CI * (float) (CHAN_TO_DOUBLE(chan[fid])*delay));
		
		temp = packedDelays[2*sid + 1];
		delay = PACKEDDELAY_TO_DOUBLE(temp);
		phaseRot[fid*%(nStand)i*2 + sid*2 + 1] = exp(-2*PI*CI * (float) (CHAN_TO_DOUBLE(chan[fid])*delay));
	}
}


//__global__ void computeGains(short int *gains) {
//	int sid = threadIdx.x + blockDim.x * blockIdx.x;
//	
//	float temp;
//	if( sid < %(nStand)i ) {
//		temp = (float) gains[sid*2*2 + 0*2 + 0] / (float) 32767.0;
//		something texture related
//		temp = (float) gains[sid*2*2 + 0*2 + 1] / (float) 32767.0;
//		something texture related
//		temp = (float) gains[sid*2*2 + 1*2 + 0] / (float) 32767.0;
//		something texture related
//		temp = (float) gains[sid*2*2 + 1*2 + 1] / (float) 32767.0;
//		something texture related
//	}
//}


__global__ void beamformer(signed char *data, pycuda::complex<float> *phaseRot, short int *gains,  pycuda::complex<float> *beam) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int fid = threadIdx.y + blockDim.y * blockIdx.y;
	
	int i;
	pycuda::complex <float> CI(0.,1.);
	pycuda::complex <float> tempP(0.,0.);
	pycuda::complex <float> beamX(0.,0.);
	pycuda::complex <float> beamY(0.,0.);
	
	if( tid < %(nTime)i && fid < %(nChan)i) {
		for(i=0; i<%(nStand)i; i++) {
			tempP =     BIFROST_TO_FLOAT(data[tid*%(nChan)i*%(nStand)i*2*2 + fid*%(nStand)i*2*2 + i*2*2 + 0*2 + 0]);
			tempP += CI*BIFROST_TO_FLOAT(data[tid*%(nChan)i*%(nStand)i*2*2 + fid*%(nStand)i*2*2 + i*2*2 + 0*2 + 1]);
			tempP *= phaseRot[ fid*%(nStand)i*2 + i*2 + 0];
			beamX += tempP * (float) 1.0; //PACKEDGAIN_TO_FLOAT(gains[i*%(nStand)i*2 + 0*2 + 0]);
			beamY += tempP * (float) 0.0; //PACKEDGAIN_TO_FLOAT(gains[i*%(nStand)i*2 + 0*2 + 1]);
			
			tempP =     BIFROST_TO_FLOAT(data[tid*%(nChan)i*%(nStand)i*2*2 + fid*%(nStand)i*2*2 + i*2*2 + 1*2 + 0]);
			tempP += CI*BIFROST_TO_FLOAT(data[tid*%(nChan)i*%(nStand)i*2*2 + fid*%(nStand)i*2*2 + i*2*2 + 1*2 + 1]);
			tempP *= phaseRot[ fid*%(nStand)i*2 + i*2 + 1];
			beamX += tempP * (float) 0.0; //PACKEDGAIN_TO_FLOAT(gains[i*%(nStand)i*2 + 1*2 + 0]);
			beamY += tempP * (float) 1.0; //PACKEDGAIN_TO_FLOAT(gains[i*%(nStand)i*2 + 1*2 + 1]);
		}
		
		beam[tid*%(nChan)i*2 + fid*2 + 0] = beamX;
		beam[tid*%(nChan)i*2 + fid*2 + 1] = beamY;
	}
}
""" % {'nTime':self.nTime, 'nChan':self.nChan, 'nStand':self.nStand}, nvcc="/usr/local/cuda/bin/nvcc")
		self._computePhaseRotation = kernels.get_function('computePhaseRotation')
		self._beamformer = kernels.get_function('beamformer')
		
		if self.log is not None:
			self.log.info("Beamformer CUDA code compiled and ready")
			
		return True
		
	def __call__(self, data):
		"""
		Apply the necessary beamforming to the provided data and return the 
		result as a NumPy 3-D numpy.complex64 array.  The axes are time, 
		frequency, and polarization.
		"""
		
		# Validate and send
		assert(data.shape[0] == self.nTime)
		assert(data.shape[1] == self.nChan)
		assert(data.shape[2] == self.nStand)
		assert(data.shape[3] == 2)
		assert(data.dtype == np.int8)
		data = gpa.to_gpu(data)
		# Compute
		self._beamformer(data, self.phaseRot, self.gains, self.beam, block=self.bSize, grid=self.gSize)
		# Done
		return self.beam.get()


class BeamformerOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, nchan_max=256, nroach_max=16, ntime_gulp=2500, core=-1):
		self.log   = log
		self.iring = iring
		ninput_max = nroach_max*32#*2
		self.ntime_gulp = ntime_gulp
		self.core = core
		self.nchan_max = nchan_max
		#self.iring.resize(self.ntime_gulp*nchan_max*ninput_max*2)#frame_size_max)
		
		self.configMessage = ISC.BAMConfigurationClient(addr=('adp',5832))
		
	@ISC.logException
	def updateConfig(self, config, hdr):
		if config:
			self.log.info("Beamformer: New configuration received for beam %i", config[0])
			beam, delays, gains, tuning = config
			
			# Get the frequency channels and update the beamformer
			chans = hdr['chan0'] + np.arange(hdr['nchan'])
			self.gbf.setChannels(chans.astype(np.int16))
			self.log.info('  Channels set to %i through %i (%i channels)', chans[0], chans[-1], chans.size)
			
			# Update the delays in the beamformer
			self.gbf.setDelays(delays.astype(np.uint16))
			self.log.info('  Delays set')
			
			# Upate the gains in the beamformer
			self.gbf.setGains(gains.astype(np.uint16))
			self.log.info('  Gains set')
			
			return True
		else:
			return False
		
	@ISC.logException
	def main(self):
		bfAffinitySetCore(self.core)
		
		self.gbf = gpuBeam(self.ntime_gulp, self.nchan_max, 256, log=self.log)
		
		fh = open('/data1/beamformer_%s.dat' % socket.gethostname(), 'w')
		
		for isequence in self.iring.read():#guarantee=False):
			ihdr = json.loads(isequence.header.tostring())
			
			status = self.updateConfig( self.configMessage(), ihdr )
			if status:
				fh.write('### BAM ###\n')
				
			self.log.info("Beamformer: Start of new sequence: %s", str(ihdr))
			
			nchan  = ihdr['nchan']
			nroach = ihdr['nroach']
			nstand = 256
			npol   = 2
			igulp_size = self.ntime_gulp*nchan*nstand*npol*2
			ishape = (self.ntime_gulp,nchan,nstand,npol,2)
			
			time_tag0 = isequence.time_tag
			
			t = 0
			for ispan in isequence.read(igulp_size):
				#print "TENGINE processing"
				if ispan.size < igulp_size:
					continue # Ignore final gulp
					
				status = self.updateConfig( self.configMessage(), ihdr )
				if status:
					fh.write('### BAM ###\n')
					
				data = ispan.data_view(np.int8).reshape(ishape)
				time_tag  = time_tag0 + t*NFRAME_PER_SPECTRUM*self.ntime_gulp
				t += 1
				
				try:
					beam = self.gbf(data)
					#fh.write('%.6f  %.6f  %.6f\n' % (time_tag/196e6, (np.abs(beam[:,:,0])**2).mean(), (np.abs(beam[:,:,1])**2).mean()))
					fh.write('%.6f  %s\n' % (time_tag/196e6, str(beam[0,:,0].imag)))
					if t % 10 == 0:
						fh.flush()
				except AttributeError:
					pass
		fh.close()


def get_utc_start():
	got_utc_start = False
	while not got_utc_start:
		try:
			with MCS.Communicator() as adp_control:
				utc_start = adp_control.report('UTC_START')
				# Check for valid timestamp
				utc_start_dt = datetime.datetime.strptime(utc_start, DATE_FORMAT)
			got_utc_start = True
		except Exception as ex:
			print ex
			time.sleep(1)
	#print "UTC_START:", utc_start
	#return utc_start
	return utc_start_dt

def get_numeric_suffix(s):
	i = 0
	while True:
		if len(s[i:]) == 0:
			raise ValueError("No numeric suffix in string '%s'" % s)
		try: return int(s[i:])
		except ValueError: i += 1

def partition_balanced(nitem, npart, part_idx):
	rem = nitem % npart
	part_nitem  = nitem / npart + (part_idx < rem)
	part_offset = (part_idx*part_nitem if part_idx < rem else
	               rem*(part_nitem+1) + (part_idx-rem)*part_nitem)
	return part_nitem, part_offset

def partition_packed(nitem, npart, part_idx):
	part_nitem  = (nitem-1) / npart + 1
	part_offset = part_idx * part_nitem
	part_nitem  = min(part_nitem, nitem-part_offset)
	return part_nitem, part_offset

def main(argv):
	parser = argparse.ArgumentParser(description='LWA-SV ADP DRX service')
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
	log.info("Version:  %s", __version__)
	log.info("Last changed: %s", short_date)
	log.info("Current MJD:  %f", Adp.MCS2.slot2mjd())
	log.info("Current MPM:  %i", Adp.MCS2.slot2mpm())
	log.info("Config file:  %s", args.configfile)
	log.info("Log file:     %s", args.logfile)
	log.info("Dry run:      %r", args.dryrun)
	
	shutdown_event = threading.Event()
	def handle_signal_terminate(signum, frame):
		SIGNAL_NAMES = dict((k, v) for v, k in \
		                    reversed(sorted(signal.__dict__.items()))
		                    if v.startswith('SIG') and \
		                    not v.startswith('SIG_'))
		log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
		ops[0].shutdown()
		shutdown_event.set()
	for sig in [signal.SIGHUP,
	            signal.SIGINT,
	            signal.SIGQUIT,
	            signal.SIGTERM,
	            signal.SIGTSTP]:
		signal.signal(sig, handle_signal_terminate)
	
	log.info("Waiting to get UTC_START")
	utc_start_dt = get_utc_start()
	log.info("UTC_START:    %s", utc_start_dt.strftime(DATE_FORMAT))
	
	hostname = socket.gethostname()
	server_idx = get_numeric_suffix(hostname) - 1
	log.info("Hostname:     %s", hostname)
	log.info("Server index: %i", server_idx)
	
	pipeline_idx = config['drx']['pipeline_idx']
	recorder_idx = config['drx']['recorder_idx']
	iaddr  = config['server']['data_ifaces'][pipeline_idx]
	iport  = config['server']['data_ports' ][pipeline_idx]
	oaddr  = config['host']['recorders'][recorder_idx]
	oport  = config['recorder']['port']
	nroach_tot = len(config['host']['roaches'])
	nserver    = len(config['host']['servers'])
	drx_servers = config['host']['servers-data']
	server_data_host = config['host']['servers-data'][server_idx]
	nroach = 16#len([srv for srv in drx_servers if srv == server_data_host])
	roach0 = 0#[i for (i,srv) in enumerate(drx_servers) if srv == server_data_host][0]
	core0 = config['drx']['first_cpu_core']
	nChanMax = int(math.ceil(config['drx']['capture_bandwidth'] / CHAN_BW / len(config['host']['servers-data'])))
	
	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("Dst address:  %s:%i", oaddr, oport)
	log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
	log.info("Max Channels: %i", nChanMax)
	
	# Note: Capture uses Bifrost address+socket objects, while output uses
	#         plain Python address+socket objects.
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	
	capture_ring = Ring()
	trigger_ring = Ring()
	beam_ring = Ring()
	
	ops = []
	ops.append(CaptureOp(log, isock, capture_ring, core=core0+0, #nchan_max=nChanMax*6,  
	                     nroach=nroach, roach0=roach0, utc_start=utc_start_dt,
	                     #buffer_ntime=2500, slot_ntime=25000))
	                     buffer_ntime=25000, slot_ntime=25000))
	# **TODO: Setting nroach_max=16 here (which just causes a larger ring.resize()),
	#           breaks the capture code! It seems that regular gaps appear in
	#           the pkt seq numbers, which screws the sequence tracking and 
	#           causes endless 'early' packets. No idea what's going on!
	#TODO: Turn on and change rings on TriggerOp after testing
	ops.append(CopyOp(log, capture_ring, trigger_ring, 
					nchan_max=nChanMax, nroach_max=nroach, ntime_gulp=25000, 
					core=core0+1))
	ops.append(TriggerOp(log, trigger_ring, oaddr, oport, 
						nchan_max=nChanMax, nroach_max=nroach, ntime_gulp=25000, 
						core=core0+2))

	# TODO: TURN ME ON FOR BEAMFORMER
	#ops.append(BeamformerOp(log, capture_ring, 
						#nchan_max=nChanMax, nroach_max=nroach, ntime_gulp=25000, 
						#core=core0+3))
	
	threads = [threading.Thread(target=op.main) for op in ops]
	
	log.info("Launching %i thread(s)", len(threads))
	for thread in threads:
		thread.daemon = True
		thread.start()
	log.info("Waiting for threads to finish")
	while not shutdown_event.is_set():
		signal.pause()
	for thread in threads:
		thread.join()
	log.info("All done")
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
