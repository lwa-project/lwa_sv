#!/usr/bin/env python
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.packet_writer import HeaderInfo, UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.fft import Fft
from bifrost.fir import Fir
from bifrost.quantize import quantize as Quantize
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

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
import pickle
from collections import deque

#from numpy.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft

ACTIVE_DRX_CONFIG = threading.Event()

FILTER2BW = {1:   250000, 
             2:   500000, 
             3:  1000000, 
             4:  2000000, 
             5:  4900000, 
             6:  9800000, 
             7: 19600000}
FILTER2CHAN = {1:   250000/25000, 
               2:   500000/25000, 
               3:  1000000/25000, 
               4:  2000000/25000, 
               5:  4900000/25000, 
               6:  9800000/25000, 
               7: 19600000/25000}

__version__    = "0.1"
__date__       = '$LastChangedDate: 2016-08-09 15:44:00 -0600 (Fri, 25 Jul 2014) $'
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

UTC_START = 0

def chips_callback(seq0, chan0, nchan, nsrc, time_tag_ptr, hdr_ptr, hdr_size_ptr):
    global UTC_START
    
    timestamp0 = int((UTC_START - ADP_EPOCH).total_seconds())
    time_tag0  = timestamp0 * int(FS)
    time_tag   = time_tag0 + seq0*(int(FS)//int(CHAN_BW))
    print "++++++++++++++++ seq0     =", seq0
    print "                 time_tag =", time_tag
    time_tag_ptr[0] = time_tag
    hdr = {'time_tag': time_tag,
           'chan0':    chan0,
           'nsrc':     nsrc,
           'nchan':    nchan*nsrc,                              # Each server since part of the bandwidth
           'cfreq':    (chan0 + 0.5*(nchan*nsrc-1))*CHAN_BW,    # Each server since part of the bandwidth
           'bw':       nchan*nsrc*CHAN_BW,                      # Each server since part of the bandwidth
           'nstand':   2,#self.nbeam_max,
           'npol':     2,
           'complex':  True,
           'nbit':     32}#64 if self.nbeam_max == 1 else 32}
    print "******** CFREQ:", hdr['cfreq']
    hdr_str = json.dumps(hdr)
    # TODO: Can't pad with NULL because returned as C-string
    #hdr_str = json.dumps(hdr).ljust(4096, '\0')
    #hdr_str = json.dumps(hdr).ljust(4096, ' ')
    header_buf = ctypes.create_string_buffer(hdr_str)
    hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
    hdr_size_ptr[0] = len(hdr_str)
    return 0

#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
    def __init__(self, log, *args, **kwargs):
        global UTC_START
        
        self.log    = log
        self.args   = args
        self.kwargs = kwargs
        self.nbeam_max = self.kwargs['nbeam_max']
        del self.kwargs['nbeam_max']
        UTC_START = self.utc_start = self.kwargs['utc_start']
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
        hdr = {'time_tag': time_tag,
               'chan0':    chan0,
               'nsrc':     nsrc,
               'nchan':    nchan*nsrc,                              # Each server since part of the bandwidth
               'cfreq':    (chan0 + 0.5*(nchan*nsrc-1))*CHAN_BW,    # Each server since part of the bandwidth
               'bw':       nchan*nsrc*CHAN_BW,                      # Each server since part of the bandwidth
               'nstand':   self.nbeam_max,
               'npol':     2,
               'complex':  True,
               'nbit':     64 if self.nbeam_max == 1 else 32}
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
        seq_callback = PacketCaptureCallback()
        #seq_callback.set_chips(lambda args: self.seq_callback(*args))
        seq_callback.set_chips(chips_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                #print status
        del capture

class ReorderChannelsOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500,# ntime_buf=None,
                 guarantee=True, core=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        #if ntime_buf is None:
        #    ntime_buf = self.ntime_gulp*3
        #self.ntime_buf = ntime_buf
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
    #@ISC.logException
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("Reorder: Start of new sequence: %s", str(ihdr))
                
                nsrc   = ihdr['nsrc']
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                
                if ihdr['nbit'] == 64:
                    igulp_size = self.ntime_gulp*nchan*nstand*npol*16           # complex128
                else:
                    igulp_size = self.ntime_gulp*nchan*nstand*npol*8            # complex64
                ishape = (self.ntime_gulp,nchan/nsrc,nsrc,nstand*npol)
                ogulp_size = self.ntime_gulp*nchan*nstand*npol*8                # complex64
                oshape = (self.ntime_gulp,nchan,nstand,npol)
                self.iring.resize(igulp_size, igulp_size*10)
                self.oring.resize(ogulp_size)#, obuf_size)
                
                ohdr = ihdr.copy()
                ohdr['nbit'] = 32
                ohdr['complex'] = True
                ohdr_str = json.dumps(ohdr)
                
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            if ihdr['nbit'] == 64:
                                idata = ispan.data_view(np.complex128).reshape(ishape)
                            else:
                                idata = ispan.data_view(np.complex64).reshape(ishape)
                            odata = ospan.data_view(np.complex64).reshape(oshape)
                            
                            rdata = idata.astype(np.complex64)
                            
                            odata[...] = np.swapaxes(rdata, 1, 2).reshape((self.ntime_gulp,nchan,nstand,npol))
                            
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})

class TEngineOp(object):
    def __init__(self, log, iring, oring, tuning=0, ntime_gulp=2500, nchan_max=864, nbeam=1, # ntime_buf=None,
                 guarantee=True, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.tuning = tuning
        self.ntime_gulp = ntime_gulp
        self.nchan_max = nchan_max
        #if ntime_buf is None:
        #    ntime_buf = self.ntime_gulp*3
        #self.ntime_buf = ntime_buf
        self.guarantee = guarantee
        self.core = core
        self.gpu  = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self.configMessage = ISC.DRXConfigurationClient(addr=('adp',5832))
        self._pending = deque()
        self.gain = 7
        self.filt = filter(lambda x: FILTER2CHAN[x]<=self.nchan_max, FILTER2CHAN)[-1]
        self.nchan_out = FILTER2CHAN[self.filt]
        
        coeffs = np.array([ 0.0111580, -0.0074330,  0.0085684, -0.0085984,  0.0070656, -0.0035905, 
                           -0.0020837,  0.0099858, -0.0199800,  0.0316360, -0.0443470,  0.0573270, 
                           -0.0696630,  0.0804420, -0.0888320,  0.0941650,  0.9040000,  0.0941650, 
                           -0.0888320,  0.0804420, -0.0696630,  0.0573270, -0.0443470,  0.0316360, 
                           -0.0199800,  0.0099858, -0.0020837, -0.0035905,  0.0070656, -0.0085984,  
                            0.0085684, -0.0074330,  0.0111580], dtype=np.float64)
        
        # Setup the T-engine
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Metadata
        nstand, npol = nbeam, 2
        ## Coefficients
        coeffs.shape += (1,)
        coeffs = np.repeat(coeffs, nstand*npol, axis=1)
        coeffs.shape = (coeffs.shape[0],nstand,npol)
        self.coeffs = BFArray(coeffs, space='cuda')
        ## Phase rotator state
        phaseState = np.array((0,), dtype=np.float64)
        self.phaseState = BFArray(phaseState, space='cuda')
        sampleCount = np.array((0,), dtype=np.int64)
        self.sampleCount = BFArray(sampleCount, space='cuda')
        
    #@ISC.logException
    def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
        global ACTIVE_DRX_CONFIG
        
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Pull out the tuning (something unique to DRX/BAM/COR)
            tuning = config[0]
            if tuning != self.tuning:
                return False
                
            ## Set the configuration time - DRX commands are for the first slot in the next second
            slot = 0 / 100.0
            config_time = int(time.time()) + 1 + slot
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                self._pending.append( (config_time, config) )
                config = None
                
                ### Is there something pending?
                try:
                    stored_time, stored_config = self._pending[0]
                    if pipeline_time >= stored_time:
                        config_time, config = self._pending.popleft()
                except IndexError:
                    pass
            else:
                ### Nope, this is something we can use now
                pass
                
        else:
            ## Is there something pending?
            try:
                stored_time, stored_config = self._pending[0]
                if pipeline_time >= stored_time:
                    config_time, config = self._pending.popleft()
            except IndexError:
                #print "No pending configuation at %.1f" % pipeline_time
                pass
                
        if config:
            self.log.info("TEngine: New configuration received for tuning %i (delta = %.1f subslots)", config[0], (pipeline_time-config_time)*100.0)
            tuning, freq, filt, gain = config
            if tuning != self.tuning:
                self.log.info("TEngine: Not for this tuning, skipping")
                return False
                
            self.rFreq = freq
            self.filt = filt
            self.nchan_out = FILTER2CHAN[filt]
            self.gain = gain
            
            fDiff = freq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu != -1:
                BFSetGPU(self.gpu)
                
            phaseState = fDiff/(self.nchan_out*CHAN_BW)
            phaseRot = np.exp(-2j*np.pi*phaseState*np.arange(self.ntime_gulp*self.nchan_out, dtype=np.float64))
            phaseRot = phaseRot.astype(np.complex64)
            copy_array(self.phaseState, np.array([phaseState,], dtype=np.float64))
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            
            ACTIVE_DRX_CONFIG.set()
            
            return True
            
        elif forceUpdate:
            self.log.info("TEngine: New sequence configuration received")
            
            try:
                fDiff = self.rFreq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
            except AttributeError:
                self.rFreq = (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW + CHAN_BW / 2
                fDiff = 0.0
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu != -1:
                BFSetGPU(self.gpu)
                
            phaseState = fDiff/(self.nchan_out*CHAN_BW)
            phaseRot = np.exp(-2j*np.pi*phaseState*np.arange(self.ntime_gulp*self.nchan_out, dtype=np.float64))
            phaseRot = phaseRot.astype(np.complex64)
            copy_array(self.phaseState, np.array([phaseState,], dtype=np.float64))
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            
            return False
            
        else:
            return False
            
    #@ISC.logException
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
                             
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )
                
                nsrc   = ihdr['nsrc']
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                
                igulp_size = self.ntime_gulp*nchan*nstand*npol*8                # complex64
                ishape = (self.ntime_gulp,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*self.nchan_out*nstand*npol*1       # 4+4 complex
                oshape = (self.ntime_gulp*self.nchan_out,nstand,npol)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)#, obuf_size)
                
                ticksPerTime = int(FS) / int(CHAN_BW)
                base_time_tag = iseq.time_tag
                sample_count = 0
                copy_array(self.sampleCount, np.array([sample_count,], dtype=np.int64))
                
                ohdr = {}
                ohdr['nstand']   = nstand
                ohdr['npol']     = npol
                ohdr['complex']  = True
                ohdr['nbit']     = 4
                ohdr['fir_size'] = self.coeffs.shape[0]
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    ohdr['time_tag'] = base_time_tag
                    ohdr['cfreq']    = self.rFreq
                    ohdr['bw']       = self.nchan_out*CHAN_BW
                    ohdr['gain']     = self.gain
                    ohdr['filter']   = self.filt
                    ohdr_str = json.dumps(ohdr)
                    
                    # Adjust the gain to make this ~compatible with LWA1
                    act_gain = self.gain + 2
                    
                    with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                ## Setup and load
                                idata = ispan.data_view(np.complex64).reshape(ishape)
                                odata = ospan.data_view(np.int8).reshape((1,)+oshape)
                                
                                ## Prune and shift the data ahead of the IFFT
                                if idata.shape[1] != self.nchan_out:
                                    try:
                                        pdata[...] = idata[:,nchan/2-self.nchan_out/2:nchan/2+self.nchan_out/2]
                                    except NameError:
                                        pshape = (self.ntime_gulp,self.nchan_out,nstand,npol)
                                        pdata = BFArray(shape=pshape, dtype=np.complex64, space='system')
                                        pdata[...] = idata[:,nchan/2-self.nchan_out/2:nchan/2+self.nchan_out/2]
                                else:
                                    pdata = idata
                                    
                                ### From here until going to the output ring we are on the GPU
                                tdata = pdata.copy(space='cuda')
                                
                                ## IFFT
                                try:
                                    gdata = gdata.reshape(*t.shape)
                                    bfft.execute(tdata, gdata, inverse=True)
                                except NameError:
                                    gdata = BFArray(shape=tdata.shape, dtype=np.complex64, space='cuda')
                                    
                                    bfft = Fft()
                                    bfft.init(tdata, gdata, axes=1, apply_fftshift=True)
                                    bfft.execute(tdata, gdata, inverse=True)
                                    
                                ## Phase rotation
                                gdata = gdata.reshape((-1,nstand*npol))
                                BFMap("a(i,j) *= exp(Complex<float>(0.0, -2*BF_PI_F*fmod(g(0)*s(0), 1.0)))*b(i)", 
                                      {'a':gdata, 'b':self.phaseRot, 'g':self.phaseState, 's':self.sampleCount}, 
                                      axis_names=('i','j'), 
                                      shape=gdata.shape, 
                                      extra_code="#define BF_PI_F 3.141592654f")
                                gdata = gdata.reshape((-1,nstand,npol))
                                
                                ## FIR filter
                                try:
                                    bfir.execute(gdata, fdata)
                                except NameError:
                                    fdata = BFArray(shape=gdata.shape, dtype=gdata.dtype, space='cuda')
                                    
                                    bfir = Fir()
                                    bfir.init(self.coeffs, 1)
                                    bfir.execute(gdata, fdata)
                                    
                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=8./(2**act_gain * np.sqrt(self.nchan_out)))
                                except NameError:
                                    qdata = BFArray(shape=gdata.shape, native=False, dtype='ci4', space='cuda')
                                    Quantize(fdata, qdata, scale=8./(2**act_gain * np.sqrt(self.nchan_out)))
                                    
                                ## Save
                                tdata = qdata.copy('system')
                                odata[...] = tdata.view(np.int8).reshape((1,)+oshape)
                                
                            ## Update the base time tag
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            ## Update the sample counter
                            sample_count += oshape[0]
                            copy_array(self.sampleCount, np.array([sample_count,], dtype=np.int64))
                            
                            ## Check for an update to the configuration
                            if self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False ):
                                reset_sequence = True
                                sample_count = 0
                                copy_array(self.sampleCount, np.array([sample_count,], dtype=np.int64))
                                
                                ### New output size/shape
                                ngulp_size = self.ntime_gulp*self.nchan_out*nstand*npol*1               # 4+4 complex
                                nshape = (self.ntime_gulp*self.nchan_out,nstand,npol)
                                if ngulp_size != ogulp_size:
                                    ogulp_size = ngulp_size
                                    oshape = nshape
                                    
                                    self.oring.resize(ogulp_size)
                                    
                                ### Clean-up
                                try:
                                    del pdata
                                    del gdata
                                    del fdata
                                    del qdata
                                except NameError:
                                    pass
                                    
                                break
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                                                 
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        ## Clean-up
                        try:
                            del pdata
                            del gdata
                            del fdata
                            del qdata
                        except NameError:
                            pass
                            
                        break

def gen_drx_header(beam, tune, pol, cfreq, filter, time_tag):
    bw = FILTER2BW[filter]
    decim = int(FS) / bw
    sync_word    = 0xDEC0DE5C
    idval        = ((pol&0x1)<<7) | ((tune&0x7)<<3) | (beam&0x7)
    frame_num    = 0
    id_frame_num = idval << 24 | frame_num
    assert( 0 <= cfreq < FS )
    tuning_word  = int(round(cfreq / FS * 2**32))    # This should already be on the DP frequency grid from the control software so we need to use round()
    #if stand == 0 and pol == 0:
    #    print cfreq, bw, gain, time_tag, time_tag0
    #    print nframe_per_sample, nframe_per_packet
    return struct.pack('>IIIhhqII',
                       sync_word,
                       id_frame_num,
                       0, 
                       decim, 
                       0, 
                       time_tag, 
                       tuning_word, 
                       0)

class PacketizeOp(object):
    # Note: Input data are: [time,beam,pol,iq]
    def __init__(self, log, iring, osock, nbeam_max=1, beam0=1, tuning=0, npkt_gulp=128, core=-1):
        self.log   = log
        self.iring = iring
        self.sock  = osock
        self.nbeam_max = nbeam_max
        self.beam0 = beam0
        self.tuning = tuning
        self.npkt_gulp = npkt_gulp
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
        self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))
        
        self.sync_drx_pipelines = MCS.Synchronizer('DRX')
        
    #@ISC.logException
    def main(self):
        global ACTIVE_DRX_CONFIG
        
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        ntime_pkt     = DRX_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nbeam_max * 2
        gulp_size_max = ntime_gulp * ninput_max * 2
        self.iring.resize(gulp_size_max)
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        for isequence in self.iring.read():
            ihdr = json.loads(isequence.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
            
            #print 'PacketizeOp', ihdr
            cfreq  = ihdr['cfreq']
            bw     = ihdr['bw']
            gain   = ihdr['gain']
            filt   = ihdr['filter']
            nbeam  = ihdr['nstand']
            npol   = ihdr['npol']
            fdly   = (ihdr['fir_size'] - 1) / 2.0
            time_tag0 = isequence.time_tag
            time_tag  = time_tag0
            gulp_size = ntime_gulp*nbeam*npol
            
            # Figure out where we need to be in the buffer to be at a frame boundary
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % int(ntime_pkt)
            if soffset != 0:
                soffset = ntime_pkt - soffset
            boffset = soffset*nbeam*npol
            print '!!', toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset
            
            time_tag += soffset*ticksPerSample                  # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
            
            prev_time = time.time()
            with UDPTransmit(sock=self.sock, core=self.core) as udt:
                for ispan in isequence.read(gulp_size, begin=boffset):
                    if ispan.size < gulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    shape = (-1,nbeam,npol)
                    data = ispan.data_view(np.int8).reshape(shape)
                    
                    for t in xrange(0, ntime_gulp, ntime_pkt):
                        time_tag_cur = time_tag + int(t)*ticksPerSample
                        try:
                            self.sync_drx_pipelines(time_tag_cur)
                        except ValueError:
                            continue
                        except (socket.timeout, socket.error):
                            pass
                            
                        pkts = []
                        for beam in xrange(nbeam):
                            for pol in xrange(npol):
                                pktdata = data[t:t+ntime_pkt,beam,pol]
                                hdr = gen_drx_header(beam+self.beam0, self.tuning+1, pol, cfreq, filt, 
                                            time_tag_cur)
                                try:
                                    pkt = hdr + pktdata.tostring()
                                    pkts.append( pkt )
                                except Exception as e:
                                    print type(self).__name__, 'Packing Error', str(e)
                                    
                        try:
                            if ACTIVE_DRX_CONFIG.is_set():
                                if not self.tbfLock.is_set():
                                    udt.sendmany(pkts)
                        except Exception as e:
                            print type(self).__name__, 'Sending Error', str(e)
                            
                    time_tag += int(ntime_gulp)*ticksPerSample
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': -1, 
                                              'process_time': process_time,})
                    
            del udt

class SinglePacketizeOp(object):
    # Note: Input data are: [time,beam,pol,iq]
    def __init__(self, log, iring, osock, nbeam_max=1, beam0=1, beam=1, tuning=0, npkt_gulp=128, core=-1):
        self.log   = log
        self.iring = iring
        self.sock  = osock
        self.nbeam_max = nbeam_max
        self.beam0 = beam0
        self.beam = beam
        self.tuning = tuning
        self.npkt_gulp = npkt_gulp
        self.core = core
        
        assert(self.beam > 0)
        assert(self.beam <= self.nbeam_max)
        assert(self.npkt_gulp % 2 == 0)
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
        self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))
        
        self.sync_drx_pipelines = MCS.Synchronizer('DRX%i' % self.beam)
        
    #@ISC.logException
    def main(self):
        global ACTIVE_DRX_CONFIG
        
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        ntime_pkt     = DRX_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nbeam_max * 2
        gulp_size_max = ntime_gulp * ninput_max * 2
        self.iring.resize(gulp_size_max)
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        for isequence in self.iring.read():
            ihdr = json.loads(isequence.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
            
            #print 'PacketizeOp', ihdr
            cfreq  = ihdr['cfreq']
            bw     = ihdr['bw']
            gain   = ihdr['gain']
            filt   = ihdr['filter']
            nbeam  = ihdr['nstand']
            npol   = ihdr['npol']
            fdly   = (ihdr['fir_size'] - 1) / 2.0
            time_tag0 = isequence.time_tag
            time_tag  = time_tag0
            gulp_size = ntime_gulp*nbeam*npol
            
            # Figure out where we need to be in the buffer to be at a frame boundary
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % int(ntime_pkt)
            if soffset != 0:
                soffset = ntime_pkt - soffset
            boffset = soffset*nbeam*npol
            print '!!', '@', self.beam0, toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset
            
            time_tag += soffset*ticksPerSample                  # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
            
            prev_time = time.time()
            with UDPTransmit(sock=self.sock, core=self.core) as udt:
                pkts = []
                for ispan in isequence.read(gulp_size, begin=boffset):
                    if ispan.size < gulp_size:
                        continue # Ignore final gulp
                    curr_time = time.time()
                    acquire_time = curr_time - prev_time
                    prev_time = curr_time
                    
                    shape = (-1,nbeam,npol)
                    data = ispan.data_view(np.int8).reshape(shape)
                    
                    for t in xrange(0, ntime_gulp, ntime_pkt):
                        time_tag_cur = time_tag + int(t)*ticksPerSample
                        #try:
                        #    self.sync_drx_pipelines(time_tag_cur)
                        #except ValueError:
                        #    continue
                        #except (socket.timeout, socket.error):
                        #    pass
                            
                        #pkts = []
                        for pol in xrange(npol):
                            pktdata = data[t:t+ntime_pkt,self.beam-1,pol]
                            hdr = gen_drx_header(self.beam-1+self.beam0, self.tuning+1, pol, cfreq, filt, 
                                                 time_tag_cur)
                            try:
                                pkt = hdr + pktdata.tostring()
                                pkts.append( pkt )
                            except Exception as e:
                                print type(self).__name__, 'Packing Error', str(e)
                                
                        if len(pkts) >= 4:
                            try:
                                self.sync_drx_pipelines(time_tag_cur)
                            except ValueError:
                                continue
                            except (socket.timeout, socket.error):
                                pass
                                
                            try:
                                if ACTIVE_DRX_CONFIG.is_set():
                                    if not self.tbfLock.is_set():
                                        udt.sendmany(pkts)
                            except Exception as e:
                                print type(self).__name__, 'Sending Error', str(e)
                                
                            pkts = []
                            
                    time_tag += int(ntime_gulp)*ticksPerSample
                    
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': acquire_time, 
                                              'reserve_time': -1, 
                                              'process_time': process_time,})
                    
            del udt

def izip(*iterables):
    while True:
        yield [it.next() for it in iterables]
        
class DualPacketizeOp(object):
    # Note: Input data are: [time,beam,pol,iq]
    def __init__(self, log, iring, osocks, nbeam_max=1, beam0=1, tuning=0, npkt_gulp=128, core=-1):
        self.log   = log
        self.iring = iring
        self.socks  = osocks
        self.nbeam_max = nbeam_max
        self.beam0 = beam0
        self.tuning = tuning
        self.npkt_gulp = npkt_gulp
        self.core = core
        
        assert(len(self.socks) == 2)
        assert(self.nbeam_max  == 2)
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
        self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))
        
        self.sync_drx_pipelines = MCS.Synchronizer('DRX')
        
    #@ISC.logException
    def main(self):
        global ACTIVE_DRX_CONFIG
        
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        ntime_pkt     = DRX_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nbeam_max * 2
        gulp_size_max = ntime_gulp * ninput_max * 2
        self.iring.resize(gulp_size_max)
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        for isequence in self.iring.read():
            ihdr = json.loads(isequence.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
            
            #print 'PacketizeOp', ihdr
            cfreq  = ihdr['cfreq']
            bw     = ihdr['bw']
            gain   = ihdr['gain']
            filt   = ihdr['filter']
            nbeam  = ihdr['nstand']
            npol   = ihdr['npol']
            fdly   = (ihdr['fir_size'] - 1) / 2.0
            time_tag0 = isequence.time_tag
            time_tag  = time_tag0
            gulp_size = ntime_gulp*nbeam*npol
            
            # Figure out where we need to be in the buffer to be at a frame boundary
            NPACKET_SET = 8
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % (NPACKET_SET*int(ntime_pkt))
            if soffset != 0:
                soffset = NPACKET_SET*ntime_pkt - soffset
            boffset = soffset*nbeam*npol
            print '!!', '@', self.beam0, toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset
            
            time_tag += soffset*ticksPerSample                  # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
            
            prev_time = time.time()
            udt0 = UDPTransmit('drx', sock=self.socks[0], core=self.core)
            udt1 = UDPTransmit('drx', sock=self.socks[1], core=self.core)
            desc = HeaderInfo()
            desc.set_decimation(int(FS)//int(bw))
            desc.set_tuning(int(round(cfreq / FS * 2**32)))
            desc_src = (((self.tuning+1)&0x7)<<3)
            pkts0, pkts1 = [], []
            for ispan in isequence.read(gulp_size, begin=boffset):
                if ispan.size < gulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                shape = (-1,nbeam,npol)
                data = ispan.data_view('ci4').reshape(shape)
                
                #for t in xrange(0, ntime_gulp, ntime_pkt):
                #    time_tag_cur = time_tag + int(t)*ticksPerSample
                #    #try:
                #    #    self.sync_drx_pipelines(time_tag_cur)
                #    #except ValueError:
                #    #    pass
                #    #except (socket.timeout, socket.error):
                #    #    pass
                #        
                #    #pkts0, pkts1 = [], []
                #    for pol in xrange(npol):
                #        ## First beam
                #        pktdata0 = data[t:t+ntime_pkt,0,pol]
                #        hdr0 = gen_drx_header(0+self.beam0, self.tuning+1, pol, cfreq, filt, 
                #                              time_tag_cur)
                #        try:
                #            pkt0 = hdr0 + pktdata0.tostring()
                #            pkts0.append( pkt0 )
                #        except Exception as e:
                #            print type(self).__name__, 'Packing Error - beam 0', str(e)
                #            
                #        ## Second beam
                #        pktdata1 = data[t:t+ntime_pkt,1,pol]
                #        hdr1 = gen_drx_header(1+self.beam0, self.tuning+1, pol, cfreq, filt, 
                #                              time_tag_cur)
                #        try:
                #            pkt1 = hdr1 + pktdata1.tostring()
                #            pkts1.append( pkt1 )
                #        except Exception as e:
                #            print type(self).__name__, 'Packing Error - beam 1', str(e)
                #            
                #    # Changed on 2019/7/2 from 4 to 4 to fix synchronization at 19.8 MHz of bandwidth
                #    if len(pkts0) >= 4:
                #        #try:
                #        #    self.sync_drx_pipelines(time_tag_cur)
                #        #except ValueError:
                #        #    continue
                #        #except (socket.timeout, socket.error):
                #        #    pass
                #            
                #        try:
                #            if ACTIVE_DRX_CONFIG.is_set():
                #                if not self.tbfLock.is_set():
                #                    udt0.sendmany(pkts0)
                #                    udt1.sendmany(pkts1)
                #        except Exception as e:
                #            print type(self).__name__, 'Sending Error', str(e)
                #            
                #        pkts0, pkts1 = [], []
                
                
                data0 = data[:,0,:].reshape(-1,ntime_pkt,npol).transpose(0,2,1).copy()
                data1 = data[:,1,:].reshape(-1,ntime_pkt,npol).transpose(0,2,1).copy()
                
                for t in xrange(0, data0.shape[0], NPACKET_SET):
                    time_tag_cur = time_tag + t*ticksPerSample*ntime_pkt
                    try:
                        self.sync_drx_pipelines(time_tag_cur)
                    except ValueError:
                        print 'skip', t, data0.shape[0]
                        continue
                    except (socket.timeout, socket.error):
                        pass
                        
                    try:
                        if ACTIVE_DRX_CONFIG.is_set():
                            if not self.tbfLock.is_set():
                                udt0.send(desc, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+(0+self.beam0), 128, 
                                          data0[t:t+NPACKET_SET,:,:])
                                udt1.send(desc, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+(1+self.beam0), 128, 
                                          data1[t:t+NPACKET_SET,:,:])
                    except Exception as e:
                        print type(self).__name__, 'Sending Error', str(e)
                            
                time_tag += int(ntime_gulp)*ticksPerSample
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
            del udt0
            del udt1

def get_utc_start(shutdown_event=None):
    got_utc_start = False
    while not got_utc_start:
        if shutdown_event is not None:
            if shutdown_event.is_set():
                raise RuntimeError("Shutting down without getting the start time")
                
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
    parser = argparse.ArgumentParser(description='LWA-SV ADP T-Engine Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-t', '--tuning',     default=0, type=int,       help='DRX tuning (0 or 1)')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    args = parser.parse_args()
    tuning = args.tuning
    
    # Fork, if requested
    if args.fork:
        stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], tuning)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    config = Adp.parse_config_file(args.configfile)
    ntuning = len(config['drx'])
    drxConfig = config['drx'][tuning]
    
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
    
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
    
    log.info("Waiting to get UTC_START")
    utc_start_dt = get_utc_start(shutdown_event)
    log.info("UTC_START:    %s", utc_start_dt.strftime(DATE_FORMAT))
    
    hostname = socket.gethostname()
    try:
        server_idx = get_numeric_suffix(hostname) - 1
    except ValueError:
        server_idx = 0 # HACK to allow testing on head node "adp"
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    
    # Network - input
    tengine_idx  = drxConfig['tengine_idx']
    tngConfig    = config['tengine'][tengine_idx]
    iaddr        = config['host']['tengines'][tengine_idx]
    iport        = config['server']['data_ports' ][tngConfig['pipeline_idx']]
    # Network - output
    recorder_idx = tngConfig['recorder_idx']
    try:
        recConfig    = [config['recorder'][idx] for idx in recorder_idx]
        oaddr        = [rc['host']              for rc in recConfig]
        oport        = [rc['port']              for rc in recConfig]
        obw          = [rc['max_bytes_per_sec'] for rc in recConfig]
        split_beam = True
    except TypeError:
        recConfig    = config['recorder'][recorder_idx]
        oaddr        = recConfig['host']
        oport        = recConfig['port']
        obw          = recConfig['max_bytes_per_sec']
        split_beam = False
        
    nserver = len(config['host']['servers'])
    server0 = 0
    nbeam = drxConfig['beam_count']
    cores = tngConfig['cpus']
    gpus  = tngConfig['gpus']
    
    log.info("Src address:  %s:%i", iaddr, iport)
    try:
        for b,a,p in zip(range(len(oaddr)), oaddr, oport):
            log.info("Dst address:  %i @ %s:%i", b, a, p)
    except TypeError:
        log.info("Dst address:  %s:%i", oaddr, oport)
    log.info("Servers:      %i-%i", server0+1, server0+nserver)
    log.info("Tunings:      %i (of %i)", tuning+1, ntuning)
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    
    iaddr = Address(iaddr, iport)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5
    
    capture_ring = Ring(name="capture-%i" % tuning)
    reorder_ring = Ring(name="reorder-%i" % tuning)
    tengine_ring = Ring(name="tengine-%i" % tuning)
    
    GSIZE = 2500
    nchan_max = int(round(drxConfig['capture_bandwidth']/CHAN_BW))    # Subtly different from what is in adp_drx.py
    
    ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
                         nsrc=nserver, src0=server0, max_payload_size=9000,
                         nbeam_max=nbeam, 
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
                         utc_start=utc_start_dt))
    ops.append(ReorderChannelsOp(log, capture_ring, reorder_ring, 
                                ntime_gulp=GSIZE, 
                                core=cores.pop(0)))
    ops.append(TEngineOp(log, reorder_ring, tengine_ring,
                         tuning=tuning, ntime_gulp=GSIZE, 
                         nchan_max=nchan_max, nbeam=nbeam, 
                         core=cores.pop(0), gpu=gpus.pop(0)))
    if split_beam:
        # Switched on 2019/7/2 to the DualPacketizeOp block
        if False:
            for beam in xrange(nbeam):
                raddr = Address(oaddr[beam], oport[beam])
                rsock = UDPSocket()
                rsock.connect(raddr)
                ops.append(SinglePacketizeOp(log, tengine_ring,
                                             osock=rsock,
                                             nbeam_max=nbeam, beam0=1, beam=beam+1, tuning=tuning, 
                                             npkt_gulp=32, core=cores.pop(0)))
        else:
            rsocks = []
            for beam in xrange(nbeam):
                raddr = Address(oaddr[beam], oport[beam])
                rsocks.append(UDPSocket())
                rsocks[-1].connect(raddr)
            # Changed on 2019/7/2 from 32 to 32 for npkt_gulp
            ops.append(DualPacketizeOp(log, tengine_ring,
                                       osocks=rsocks,
                                       nbeam_max=nbeam, beam0=1, tuning=tuning, 
                                       npkt_gulp=32, core=cores.pop(0)))
    else:
        oaddr = Address(oaddr, oport)
        osock = UDPSocket()
        osock.connect(oaddr)
        
        ops.append(PacketizeOp(log, tengine_ring,
                               osock=osock, 
                               nbeam_max=nbeam, beam0=1, tuning=tuning, 
                               npkt_gulp=16, core=cores.pop(0)))
    
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
