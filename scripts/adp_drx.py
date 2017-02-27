#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.udp_capture import UDPCapture
from bifrost.udp_transmit import UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.unpack import unpack as Unpack
from bifrost.libbifrost import bf

#import numpy as np
import signal
import logging
import time
import os
import argparse
import ctypes
import threading
import json
import socket
import struct
#import time
import datetime

import pycuda.driver as cuda
import pycuda.gpuarray as gpa
from pycuda.compiler import SourceModule
from pycuda import tools

__version__    = "0.1"
__date__       = '$LastChangedDate: 2016-08-09 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"
#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
	def __init__(self, log, *args, **kwargs):
		self.log    = log
		self.args   = args
		self.kwargs = kwargs
		self.utc_start = self.kwargs['utc_start']
		del self.kwargs['utc_start']
		self.shutdown_event = threading.Event()
		## HACK TESTING
		#self.seq_callback = None
	def shutdown(self):
		self.shutdown_event.set()
	def seq_callback(self, seq0, chan0, nchan, nsrc,
	                 time_tag_ptr, hdr_ptr, hdr_size_ptr):
		timestamp0 = int((self.utc_start - ADP_EPOCH).total_seconds())
		time_tag0  = timestamp0 * int(FS)
		time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
		print "++++++++++++++++ seq0     =", seq0
		print "                 time_tag =", time_tag
		time_tag_ptr[0] = time_tag
		hdr = {
			'time_tag': time_tag,
			'seq0':     seq0, 
			'chan0':    chan0,
			'nchan':    nchan,
			'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
			'bw':       nchan*CHAN_BW,
			'nstand':   nsrc*16,
			#'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
			'npol':     2,
			'complex':  True,
			'nbit':     4
		}
		print "******** CFREQ:", hdr['cfreq']
		hdr_str = json.dumps(hdr)
		# TODO: Can't pad with NULL because returned as C-string
		#hdr_str = json.dumps(hdr).ljust(4096, '\0')
		#hdr_str = json.dumps(hdr).ljust(4096, ' ')
		self.header_buf = ctypes.create_string_buffer(hdr_str)
		hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
		hdr_size_ptr[0] = len(hdr_str)
		return 0
	def main(self):
		seq_callback = bf.BFudpcapture_sequence_callback(self.seq_callback)
		with UDPCapture(*self.args,
		                sequence_callback=seq_callback,
		                **self.kwargs) as capture:
			while not self.shutdown_event.is_set():
				status = capture.recv()
				#print status
		del capture

class UnpackOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		self.core = core
	def main(self):
		cpu_affinity.set_core(self.core)
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read():
				#print "HEADER:", iseq.header.tostring()
				ihdr = json.loads(iseq.header.tostring())
				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				igulp_size = self.ntime_gulp*nchan*nstand*npol
				ishape = (self.ntime_gulp,nchan,nstand,npol,1)
				ogulp_size = igulp_size * 2
				oshape = (self.ntime_gulp,nchan,nstand,npol,2)
				self.iring.resize(igulp_size)
				self.oring.resize(ogulp_size)
				ohdr = ihdr.copy()
				ohdr['nbit'] = 8
				ohdr_str = json.dumps(ohdr)
				with oring.begin_sequence(time_tag=iseq.time_tag,
				                          header=ohdr_str) as oseq:
					for ispan in iseq.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
						with oseq.reserve(ogulp_size) as ospan:
							idata = ispan.data_view(np.int8).reshape(ishape)
							odata = ospan.data_view(np.int8).reshape(oshape)
							
							bfidata = BFArray(shape=idata.shape, dtype='ci4', native=False, buffer=idata.ctypes.data, space='cuda_host')
							bfodata = BFArray(shape=idata.shape, dtype='ci8', space='cuda_host')
							
							Unpack(bfidata, bfodata)
							odata[...] = bfodata.view(np.int8)

class CopyOp(object):
	def __init__(self, log, iring, oring, ntime_gulp=2500,# ntime_buf=None,
	             guarantee=True, core=-1):
		self.log = log
		self.iring = iring
		self.oring = oring
		self.ntime_gulp = ntime_gulp
		#if ntime_buf is None:
		#	ntime_buf = self.ntime_gulp*3
		#self.ntime_buf = ntime_buf
		self.guarantee = guarantee
		self.core = core
	def main(self):
		cpu_affinity.set_core(self.core)
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())
				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				igulp_size = self.ntime_gulp*nchan*nstand*npol
				ogulp_size = igulp_size
				#obuf_size  = self.ntime_buf*nchan*nstand*npol
				self.iring.resize(igulp_size)
				self.oring.resize(ogulp_size)#, obuf_size)
				ohdr = ihdr.copy()
				ohdr_str = json.dumps(ohdr)
				with oring.begin_sequence(time_tag=iseq.time_tag,
				                          header=ohdr_str) as oseq:
					for ispan in iseq.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
						with oseq.reserve(ogulp_size) as ospan:
							idata = ispan.data_view(np.uint8)
							odata = ospan.data_view(np.uint8)
							odata[...] = idata
							#print "COPY"

def get_time_tag(dt=datetime.datetime.utcnow(), seq_offset=0):
	timestamp = int((dt - ADP_EPOCH).total_seconds())
	time_tag  = timestamp*int(FS) + seq_offset*(int(FS)//int(CHAN_BW))
	return time_tag
def seq_to_time_tag(seq):
	return seq*(int(FS)//int(CHAN_BW))
def time_tag_to_seq_float(time_tag):
	return time_tag*CHAN_BW/FS

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
	return struct.pack('>IIIhhq',
	                   sync_word,
	                   id_frame_num,
	                   secs_count,
	                   freq_chan,
	                   unassigned,
	                   time_tag)

class TriggeredDumpOp(object):
	def __init__(self, log, osock, iring, ntime_gulp, ntime_buf, core=-1, max_bytes_per_sec=None):
		self.log = log
		self.sock = osock
		self.iring = iring
		self.core  = core
		self.ntime_gulp = ntime_gulp
		self.ntime_buf = ntime_buf
		self.configMessage = ISC.TriggerClient(addr=('adp',5832))
		
		if max_bytes_per_sec is None:
			max_bytes_per_sec = 100*1024**2
		self.max_bytes_per_sec = max_bytes_per_sec
		
	def main(self):
		cpu_affinity.set_core(self.core)
		
		nchan_max  = 144
		ninput_max = 512
		frame_nbyte_max = nchan_max*ninput_max
		self.iring.resize(self.ntime_gulp*frame_nbyte_max,
		                  self.ntime_buf *frame_nbyte_max)
		while not self.iring.writing_ended():
			#print "TBFOp waiting for trigger"
			config = self.configMessage(block=False)
			if not config:
				time.sleep(0.1)
				continue
			print "Trigger: New trigger received: %s" % str(config)
			try:
				self.dump(samples=config[1], local=config[3])
			except RuntimeError as e:
				print "Error on TBF dump: %s" % str(e)
		print "Writing ended, TBFOp exiting"
		
	def dump(self, samples, time_tag=None, mask=None, local=False):
		ntime_pkt = 1 # TODO: Should be TBF_NTIME_PER_PKT?
		
		# HACK TESTING
		#time.sleep(3)
		#time.sleep(1)
		dump_time_tag = time_tag
		if dump_time_tag is None:
			utc_now = datetime.datetime.utcnow()
			time_offset_secs = -3
			time_offset = datetime.timedelta(seconds=time_offset_secs)
			dump_time_tag = get_time_tag(utc_now + time_offset)
		#print "********* dump_time_tag =", dump_time_tag
		#time.sleep(3)
		#ntime_dump = 0.25*1*25000
		#ntime_dump = 0.1*1*25000
		ntime_dump = int(round(time_tag_to_seq_float(samples)))
		
		print "TBF DUMPING %f secs at time_tag = %i (%s)%s" % (samples/FS, dump_time_tag, datetime.datetime.utcfromtimestamp(dump_time_tag/FS), (' locallay' if local else ''))
		with self.iring.open_sequence_at(dump_time_tag, guarantee=True) as iseq:
			time_tag0 = iseq.time_tag
			ihdr = json.loads(iseq.header.tostring())
			nchan  = ihdr['nchan']
			chan0  = ihdr['chan0']
			nstand = ihdr['nstand']
			npol   = ihdr['npol']
			ninput = nstand*npol
			print "*******", nchan, ninput
			ishape = (-1,nchan,ninput)#,2)
			frame_nbyte = nchan*ninput#*2
			igulp_size = self.ntime_gulp*nchan*ninput#*2
			
			dump_seq_offset  = int(time_tag_to_seq_float(dump_time_tag - time_tag0))
			dump_byte_offset = dump_seq_offset * frame_nbyte
			
			# HACK TESTING write to file instead of socket
			local = True
			if local:
				filename = '/data1/test_%s_%020i.tbf' % (socket.gethostname(), dump_time_tag)#time_tag0
				ofile = open(filename, 'wb')
			ntime_dumped = 0
			nchan_rounded = nchan // TBF_NCHAN_PER_PKT * TBF_NCHAN_PER_PKT
			bytesSent, bytesStart = 0.0, time.time()
			
			print "Opening read space of %i bytes at offset = %i" % (igulp_size, dump_byte_offset)
			for ispan in iseq.read(igulp_size, begin=dump_byte_offset):
				print "**** ispan.size, offset", ispan.size, ispan.offset
				print "**** Dumping at", ntime_dumped
				if ntime_dumped >= ntime_dump:
					break
				ntime_dumped += self.ntime_gulp
				#print ispan.offset, seq_offset
				seq_offset = ispan.offset // frame_nbyte
				data = ispan.data.reshape(ishape)
				for t in xrange(0, self.ntime_gulp, ntime_pkt):
					time_tag = time_tag0 + seq_to_time_tag(seq_offset + t)
					if t == 0:
						print "**** first timestamp is", time_tag
					for c in xrange(0, nchan_rounded, TBF_NCHAN_PER_PKT):
						pktdata = data[t:t+ntime_pkt,c:c+TBF_NCHAN_PER_PKT]
						hdr = gen_tbf_header(chan0+c, time_tag, time_tag0)
						pkt = hdr + pktdata.tostring()
						if local:
							ofile.write(pkt) # HACK TESTING
							bytesSent += len(pkt)
						else:
							bytesSent += self.sock.send(pkt)
							while bytesSent/(time.time()-bytesStart) >= self.max_bytes_per_sec:
								time.sleep(0.001)
		if local:
			ofile.close()
		print "TBF DUMP COMPLETE - average rate was %.3f MB/s" % (bytesSent/(time.time()-bytesStart)/1024**2,)

		
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
		self.phaseRot = gpa.to_gpu(np.zeros((self.nChan,self.nStand,2), dtype=np.complex128))
		self.gains = gpa.to_gpu(np.zeros((self.nStand, 2, 2), dtype=np.uint16))
		
		# Beam output setup
		self.beam = gpa.to_gpu(np.zeros((self.nTime, self.nChan, 2), dtype=np.complex64))
		
		# Setup the streams so we can try to be a little asynchronous
		self._meta_stream = cuda.Stream()
		self._data_stream = cuda.Stream()
		
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
		self.chans = gpa.to_gpu_async(chans, stream=self._meta_stream)
		
		# Are there delays so that we can update the phase rotators as well?
		try:
			self.setDelays(self.delays)
		except AttributeError:
			pass
		return True
		
	def getChannels(self):
		self._meta_stream.synchronize()
		return self.chans.get()
		
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
		self.delays = gpa.to_gpu_async(delays, stream=self._meta_stream)
		# Setup the problem sizes
		bSize = (self.tpb/self.nStand, self.nStand, 1)
		gSize = (self.nChan/bSize[0]+self.nChan%bSize[0], 1, 1)
		# Compute the phase rotations
		self._computePhaseRotation(self.chans, self.delays, self.phaseRot, block=bSize, grid=gSize, stream=self._meta_stream)
		return True
		
	def getDelays(self):
		self._meta_stream.synchronize()
		return self.delays.get()
		
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
		assert(phaseRot.dtype == np.complex128)
		self.phaseRot = gpa.to_gpu_async(phaseRot, stream=self._meta_stream)
		return True
		
	def getPhaseRotation(self):
		self._meta_stream.synchronize()
		return self.phaseRot.get()
		
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
		self.gains = gpa.to_gpu_async(gains, stream=self._meta_stream)
		return True
		
	def getGains(self):
		self._meta_stream.synchronize()
		return self.gains.get()
		
	def _compile(self):
		"""
		Build the CUDA code necessary to beamform data of this size.
		"""
		
		kernels = SourceModule("""
#include <stdio.h>
#include <pycuda-complex.hpp>

#define PI (double) 3.1415926535898

// Conversion macros
/* Channel number-ish to frequency in Hz #### FIX THIS #### */
#define CHAN_TO_DOUBLE(X) (((double) X) * 25e3)
/* Packed DP delay to delay in s */
#define PACKEDDELAY_TO_DOUBLE(X)  (((double) (((X>>4)&0xFFF)) + ((double) (X&0xF))/16.0) / 196e6)
/* Packed DP gain to float gain */
#define PACKEDGAIN_TO_FLOAT(X)  (((float) X) / ((float) 32767.0))
/* Packed DP gain to double gain */
#define PACKEDGAIN_TO_DOUBLE(X)  (((double) X) / ((double) 32767.0))
/* Packed Bifrost data to float */
#define BIFROST_TO_FLOAT(X)  ((float) X)
/* Packed Bifrost 4+4 data to complex double */
#define BIFROST_TO_CDOUBLE(X)  ((double) ((X>>4)&0xF-2*((X>>4)&8)) + CI*((double) ((X&0xF)-2*(X&8))))


__global__ void computePhaseRotation(short int *chan, short int *packedDelays, pycuda::complex<double> *phaseRot) {
	int fid = threadIdx.x + blockDim.x * blockIdx.x;
	int sid = threadIdx.y + blockDim.y * blockIdx.y;
	
	int p;
	double delay;
	pycuda::complex <double> CI(0.,1.);
	
	if( fid < %(nChan)i && sid < %(nStand)i ) {
		for(p=0; p<2; p++) {
			delay = PACKEDDELAY_TO_DOUBLE(packedDelays[2*sid + p]);
			phaseRot[fid*%(nStand)i*2 + sid*2 + p] = exp(-2*PI*CI * (CHAN_TO_DOUBLE(chan[fid])*delay));
		}
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


__global__ void beamformer(signed char *data, pycuda::complex<double> *phaseRot, short int *gains,  pycuda::complex<float> *beam) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int fid = threadIdx.y + blockDim.y * blockIdx.y;
	
	int s, p;
	pycuda::complex <double> CI(0.,1.);
	pycuda::complex <double> tempP(0.,0.);
	pycuda::complex <double> beamX(0.,0.);
	pycuda::complex <double> beamY(0.,0.);
	
	if( tid < %(nTime)i && fid < %(nChan)i) {
		beamX = 0.0 + CI*0.0;
		beamY = 0.0 + CI*0.0;
		
		for(s=0; s<%(nStand)i; s++) {
			for(p=0; p<2; p++) { 
				tempP =  BIFROST_TO_CDOUBLE(data[tid*%(nChan)i*%(nStand)i*2 + fid*%(nStand)i*2 + s*2 + p]);
				tempP *= phaseRot[ fid*%(nStand)i*2 + s*2 + p];
				beamX += tempP * PACKEDGAIN_TO_DOUBLE(gains[s*2*2 + p*2 + 0]);
				beamY += tempP * PACKEDGAIN_TO_DOUBLE(gains[s*2*2 + p*2 + 1]);
			}
		}
		
		beam[tid*%(nChan)i*2 + fid*2 + 0] = (pycuda::complex <float>) beamX;
		beam[tid*%(nChan)i*2 + fid*2 + 1] = (pycuda::complex <float>) beamY;
	}
}
""" % {'nTime':self.nTime, 'nChan':self.nChan, 'nStand':self.nStand}, nvcc="/usr/local/cuda/bin/nvcc")
		self._computePhaseRotation = kernels.get_function('computePhaseRotation')
		self._beamformer = kernels.get_function('beamformer')
		
		if self.log is not None:
			self.log.info("Beamformer compiled and ready")
			
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
		
	def stop(self):
		self.ctx.pop()

class BeamformerOp(object):
	# Note: Input data are: [time,chan,ant,pol,cpx,8bit]
	def __init__(self, log, iring, oring, nchan_max=256, nroach_max=16, ntime_gulp=2500, guarantee=True, core=-1):
		self.log   = log
		self.iring = iring
		self.oring = oring
		ninput_max = nroach_max*32#*2
		self.ntime_gulp = ntime_gulp
		self.guarantee = guarantee
		self.core = core
		
		self.nchan_max = nchan_max
		self.configMessage = ISC.BAMConfigurationClient(addr=('adp',5832))
		
	@ISC.logException
	def updateConfig(self, config, hdr, forceUpdate=False):
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
			
			## Validate
			#phaseRot = self.gbf.getPhaseRotation()
			#print phaseRot.shape, (36, 512)
			#dly = delays[2*231:2*232]
			#dlyU = (1.0*((dly>>4)&0xFFF) + 1.0*(dly&0xF)/16.0) / 196e6
			#print dly, ((dly>>4)&0xFFF), (dly&0xF), dlyU, phaseRot[7,231,:], np.exp(-2j*np.pi*(chans[7]+2)*25e3*dlyU)
			
			return True
			
		elif forceUpdate:
			self.log.info("Beamformer: New sequence configuration received")
			
			# Get the frequency channels and update the beamformer
			chans = hdr['chan0'] + np.arange(hdr['nchan'])
			self.gbf.setChannels(chans.astype(np.int16))
			self.log.info('  Channels set to %i through %i (%i channels)', chans[0], chans[-1], chans.size)
			
			return True
			
		else:
			return False
		
	@ISC.logException
	def main(self):
		cpu_affinity.set_core(self.core)
		
		with self.oring.begin_writing() as oring:
			for iseq in self.iring.read(guarantee=self.guarantee):
				ihdr = json.loads(iseq.header.tostring())
				try:
					relaunchGBF = True
					if ihdr['nchan'] != self.gbf.nChan or ihdr['nstand'] != self.gbf.nStand:
						relaunchGBF = True
						self.gbf.close()
					else:
						relaunchGBF = False
				except AttributeError:
					pass
				if relaunchGBF:
					self.gbf = gpuBeam(self.ntime_gulp, ihdr['nchan'], ihdr['nstand'], log=self.log)
					
				status = self.updateConfig( self.configMessage(), ihdr, forceUpdate=True )
				
				self.log.info("Beamformer: Start of new sequence: %s", str(ihdr))
				
				nchan  = ihdr['nchan']
				nstand = ihdr['nstand']
				npol   = ihdr['npol']
				igulp_size = self.ntime_gulp*nchan*nstand*npol
				ogulp_size = self.ntime_gulp*nchan*1*npol*16	# complex128
				ishape = (self.ntime_gulp,nchan,nstand,npol)
				oshape = (self.ntime_gulp,nchan,npol)
				
				ohdr = ihdr.copy()
				ohdr['nstand'] = 1
				ohdr['nbit'] = 64
				ohdr['complex'] = True
				ohdr_str = json.dumps(ohdr)
				
				self.oring.resize(ogulp_size)
				
				with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
					for ispan in iseq.read(igulp_size):
						if ispan.size < igulp_size:
							continue # Ignore final gulp
							
						status = self.updateConfig( self.configMessage(), ihdr, forceUpdate=False )
						
						with oseq.reserve(ogulp_size) as ospan:
							idata = ispan.data_view(np.int8).reshape(ishape)
							odata = ospan.data_view(np.complex128).reshape(oshape)
							try:
								bdata = self.gbf(idata).astype(np.complex128)
								odata[...] = bdata
							except AttributeError:
								pass
		try:
			self.gbf.stop()
		except AttributeError:
			pass

def gen_chips_header(server, nchan, chan0, seq, nservers=6):
	return struct.pack('>BBBBBBHQ', 
				    server, 
				    0, 
				    nchan,
				    1, 
				    0,
				    nservers,
				    chan0-nchan*(server-1), 
				    seq)
	
class RetransmitOp(object):
	def __init__(self, log, osock, iring, nchan_max=256, ntime_gulp=2500, guarantee=True, core=-1):
		self.log   = log
		self.sock = osock
		self.iring = iring
		self.ntime_gulp = ntime_gulp
		self.guarantee = guarantee
		self.core = core
		
		self.server = int(socket.gethostname().replace('adp', '0'), 10)
		self.nchan_max = nchan_max
		
	def main(self):
		cpu_affinity.set_core(self.core)
		
		for iseq in self.iring.read():
			ihdr = json.loads(iseq.header.tostring())
			
			self.log.info("Retransmit: Start of new sequence: %s", str(ihdr))
			
			chan0  = ihdr['chan0']
			nchan  = ihdr['nchan']
			nstand = ihdr['nstand']
			npol   = ihdr['npol']
			igulp_size = self.ntime_gulp*nchan*nstand*npol*16		# complex128
			igulp_shape = (self.ntime_gulp,nchan,npol)
			
			seq0 = ihdr['seq0']
			seq = seq0
			
			with UDPTransmit(sock=self.sock, core=self.core) as udt:
				for ispan in iseq.read(igulp_size):
					if ispan.size < igulp_size:
						continue # Ignore final gulp
						
					idata = ispan.data_view(np.complex128).reshape(igulp_shape)
					pkts = []
					for t in xrange(0, self.ntime_gulp):
						pktdata = idata[t,:,:]
						seq_curr = seq + t
						hdr = gen_chips_header(self.server, nchan, chan0, seq_curr)
						pkt = hdr + pktdata.tostring()
						pkts.append( pkt )
						
					try:
						udt.sendmany(pkts)
					except Exception as e:
						pass
					seq += self.ntime_gulp
					
			del udt


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

def main(argv):
	parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
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
	try:
		server_idx = get_numeric_suffix(hostname) - 1
	except ValueError:
		server_idx = 0 # HACK to allow testing on head node "adp"
	log.info("Hostname:     %s", hostname)
	log.info("Server index: %i", server_idx)
	
	pipeline_idx = config['drx']['pipeline_idx']
	recorder_idx = config['drx']['recorder_idx']
	iaddr  = config['server']['data_ifaces'][pipeline_idx]
	iport  = config['server']['data_ports' ][pipeline_idx]
	oaddr  = config['host']['recorders'][recorder_idx]
	oport  = config['recorder']['port']
	obw    = config['recorder']['max_bytes_per_sec']
	taddr  = 'adp-data'
	tport  = 4019
	nroach_tot = len(config['host']['roaches'])
	nserver    = len(config['host']['servers'])
	nroach, roach0 = nroach_tot, 0
	core0 = config['drx']['first_cpu_core']
	
	log.info("Src address:  %s:%i", iaddr, iport)
	log.info("Dst address:  %s:%i", oaddr, oport)
	log.info("Ten address:  %s:%i", taddr, tport)
	log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
	
	# Note: Capture uses Bifrost address+socket objects, while output uses
	#         plain Python address+socket objects.
	iaddr = Address(iaddr, iport)
	isock = UDPSocket()
	isock.bind(iaddr)
	isock.timeout = 0.5
	
	capture_ring = Ring()
	unpack_ring  = Ring()
	tbf_ring     = Ring()
	tengine_ring = Ring()
	
	# TODO: Put this into config file instead
	tbf_buffer_secs = 5
	
	oaddr = Address(taddr, tport)
	osock = UDPSocket()
	osock.connect(oaddr)
	
	ops = []
	core = core0
	ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
	                     nsrc=nroach, src0=roach0, max_payload_size=9000,
	                     buffer_ntime=2500, slot_ntime=25000, core=core,
	                     utc_start=utc_start_dt))
	core += 1
	#ops.append(UnpackOp(log, capture_ring, unpack_ring, core=core))
	#core += 1
	ops.append(CopyOp(log, capture_ring, tbf_ring,
	                  ntime_gulp=2500, #ntime_buf=25000*tbf_buffer_secs,
	                  guarantee=False, core=core))
	core += 1
	ops.append(TriggeredDumpOp(log=log, osock=osock, iring=tbf_ring, 
	                           ntime_gulp=2500, ntime_buf=25000*tbf_buffer_secs,
	                           core=core, max_bytes_per_sec=obw/6))
	core += 1
	ops.append(BeamformerOp(log=log, iring=capture_ring, oring=tengine_ring, 
	                        ntime_gulp=2500,# ntime_buf=25000*tbf_buffer_secs,
	                        core=core))
	core += 1
	ops.append(RetransmitOp(log=log, osock=osock, iring=tengine_ring, 
	                        ntime_gulp=125,# ntime_buf=25000*tbf_buffer_secs,
	                        core=core))
	core += 1
	
	threads = [threading.Thread(target=op.main) for op in ops]
	
	log.info("Launching %i thread(s)", len(threads))
	for thread in threads:
		#thread.daemon = True
		thread.start()
	while not shutdown_event.is_set():
		signal.pause()
	log.info("Shutdown, waiting for threads to join")
	for thread in threads:
		thread.join()
	log.info("All done")
	return 0


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
