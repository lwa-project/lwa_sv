#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPVerbsCapture as UDPCapture
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

import numpy as np
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
from scipy.signal import get_window as scipy_window, firwin as scipy_firwin

ACTIVE_DRX_CONFIG = threading.Event()

FILTER2BW = {1:   250000, 
             2:   500000, 
             3:  1000000, 
             4:  2000000, 
             5:  4900000, 
             6:  9800000, 
             7: 19600000}
FILTER2CHAN = {1:   250000//25000, 
               2:   500000//25000, 
               3:  1000000//25000, 
               4:  2000000//25000, 
               5:  4900000//25000, 
               6:  9800000//25000, 
               7: 19600000//25000}

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2016, The LWA-SV Project"
__credits__    = ["Ben Barsdell", "Daniel Price", "Jayce Dowell"]
__license__    = "Apache v2"
__maintainer__ = "Jayce Dowell"
__email__      = "jdowell at unm"
__status__     = "Development"

def pfb_window(P):
    win_coeffs = scipy_window("hamming", 4*P)
    sinc       = scipy_firwin(4*P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    win_coeffs /= win_coeffs.max()
    return win_coeffs

#{"nbit": 4, "nchan": 136, "nsrc": 16, "chan0": 1456, "time_tag": 288274740432000000}
class CaptureOp(object):
    def __init__(self, log, *args, **kwargs):
        self.log    = log
        self.args   = args
        self.kwargs = kwargs
        self.nbeam_max = self.kwargs['nbeam_max']
        del self.kwargs['nbeam_max']
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
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {'time_tag': time_tag,
               'chan0':    chan0,
               'nsrc':     nsrc,
               'nchan':    nchan,
               'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
               'bw':       nchan*CHAN_BW,
               'nstand':   nsrc,
               'npol':     2,
               'complex':  True,
               'nbit':     32}
        print("******** CFREQ:", hdr['cfreq'])
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        self.header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(self.header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_ibeam(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
        del capture

class TEngineOp(object):
    def __init__(self, log, iring, oring, tuning=0, ntime_gulp=2500, nchan_max=864, nbeam=1, # ntime_buf=None,
                 pfb_inverter=True, guarantee=True, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.tuning = tuning
        self.ntime_gulp = ntime_gulp
        self.nchan_max = nchan_max
        #if ntime_buf is None:
        #    ntime_buf = self.ntime_gulp*3
        #self.ntime_buf = ntime_buf
        self.pfb_inverter = pfb_inverter
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
        self.filt = list(filter(lambda x: FILTER2CHAN[x]<=self.nchan_max, FILTER2CHAN))[-1]
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
        ## PFB data
        self.bdata = BFArray(shape=(self.ntime_gulp,self.nchan_max,nstand,npol), dtype=np.complex64, space='cuda')
        self.gdata = BFArray(shape=(self.ntime_gulp,self.nchan_max,nstand,npol), dtype=np.complex64, space='cuda')
        self.gdata2 = BFArray(shape=(self.ntime_gulp//4,4,self.nchan_max*nstand*npol), dtype=np.complex64, space='cuda')
        ## PFB inversion matrix
        matrix = BFArray(shape=(self.ntime_gulp//4,4,self.nchan_max,nstand*npol), dtype=np.complex64)
        self.imatrix = BFArray(shape=(self.ntime_gulp//4,4,self.nchan_max,nstand*npol), dtype=np.complex64, space='cuda')
        
        pfb = pfb_window(self.nchan_max)
        pfb = pfb.reshape(4, -1)
        pfb.shape += (1,)
        pfb.shape = (1,)+pfb.shape
        matrix[:,:4,:,:] = pfb
        matrix = matrix.copy(space='cuda')
        
        pfft = Fft()
        pfft.init(matrix, self.imatrix, axes=1)
        pfft.execute(matrix, self.imatrix, inverse=False)
        
        wft = 0.3
        BFMap(f"""
              a = (a.mag2() / (a.mag2() + {wft}*{wft})) * (1+{wft}*{wft}) / a.conj();
              """,
              {'a':self.imatrix})
        
        self.imatrix = self.imatrix.reshape(-1, 4, self.nchan_max*nstand*npol)
        del matrix
        del pfft
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
                #print("No pending configuration at %.1f" % pipeline_time)
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
                self.iring.resize(igulp_size, 15*igulp_size)
                self.oring.resize(ogulp_size)#, ogulp_size)
                
                ticksPerTime = int(FS) // int(CHAN_BW)
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
                    act_gain = self.gain + 1 + self.pfb_inverter
                    
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
                                
                                ### From here until going to the output ring we are on the GPU
                                copy_array(self.bdata, idata)
                                
                                if self.pfb_inverter:
                                    ## PFB inversion
                                    ### Initial IFFT
                                    self.gdata = self.gdata.reshape(self.bdata.shape)
                                    try:
                                        
                                        bfft.execute(self.bdata, self.gdata, inverse=True)
                                    except NameError:
                                        bfft = Fft()
                                        bfft.init(self.bdata, self.gdata, axes=1, apply_fftshift=True)
                                        bfft.execute(self.bdata, self.gdata, inverse=True)
                                        
                                    ## The actual inversion
                                    self.gdata = self.gdata.reshape(self.imatrix.shape)
                                    try:
                                        pfft2.execute(self.gdata, self.gdata2, inverse=False)
                                    except NameError:
                                        pfft2 = Fft()
                                        pfft2.init(self.gdata, self.gdata2, axes=1)
                                        pfft2.execute(self.gdata, self.gdata2, inverse=False)
                                        
                                    BFMap("a *= b / (%i*2)" % nchan,
                                          {'a':self.gdata2, 'b':self.imatrix})
                                         
                                    pfft2.execute(self.gdata2, self.gdata, inverse=True)
                                    
                                    ## FFT to re-channelize
                                    self.gdata = self.gdata.reshape(-1, nchan, nstand, npol)
                                    try:
                                        ffft.execute(self.gdata, self.bdata, inverse=False)
                                    except NameError:
                                        ffft = Fft()
                                        ffft.init(self.gdata, self.bdata, axes=1, apply_fftshift=True)
                                        ffft.execute(self.gdata, self.bdata, inverse=False)
                                        
                                ## Prune and shift the data ahead of the IFFT
                                if self.bdata.shape[1] != self.nchan_out:
                                    try:
                                        pdata[...] = self.bdata[:,nchan//2-self.nchan_out//2:nchan//2+self.nchan_out//2,:,:]
                                    except NameError:
                                        pshape = (self.ntime_gulp,self.nchan_out,nstand,npol)
                                        pdata = BFArray(shape=pshape, dtype=np.complex64, space='cuda')
                                        pdata[...] = self.bdata[:,nchan//2-self.nchan_out//2:nchan//2+self.nchan_out//2,:,:]
                                else:
                                    pdata = self.bdata
                                    
                                ## IFFT
                                try:
                                    gdata3 = gdata3.reshape(*pdata.shape)
                                    bfft2.execute(pdata, gdata3, inverse=True)
                                except NameError:
                                    gdata3 = BFArray(shape=pdata.shape, dtype=np.complex64, space='cuda')
                                    
                                    bfft2 = Fft()
                                    bfft2.init(pdata, gdata3, axes=1, apply_fftshift=True)
                                    bfft2.execute(pdata, gdata3, inverse=True)
                                    
                                ## Phase rotation
                                gdata3 = gdata3.reshape((-1,nstand*npol))
                                BFMap("a(i,j) *= exp(Complex<float>(0.0, -2*BF_PI_F*fmod(g(0)*s(0), 1.0)))*b(i)", 
                                      {'a':gdata3, 'b':self.phaseRot, 'g':self.phaseState, 's':self.sampleCount}, 
                                      axis_names=('i','j'), 
                                      shape=gdata3.shape, 
                                      extra_code="#define BF_PI_F 3.141592654f")
                                gdata3 = gdata3.reshape((-1,nstand,npol))
                                
                                ## FIR filter
                                try:
                                    bfir.execute(gdata3, fdata)
                                except NameError:
                                    fdata = BFArray(shape=gdata3.shape, dtype=gdata3.dtype, space='cuda')
                                    
                                    bfir = Fir()
                                    bfir.init(self.coeffs, 1)
                                    bfir.execute(gdata3, fdata)
                                    
                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=8./(2**act_gain * np.sqrt(self.nchan_out)))
                                except NameError:
                                    qdata = BFArray(shape=gdata3.shape, native=False, dtype='ci4', space='cuda')
                                    Quantize(fdata, qdata, scale=8./(2**act_gain * np.sqrt(self.nchan_out)))
                                    
                                ## Save
                                try:
                                    copy_array(tdata, qdata)
                                except NameError:
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
                                    del gdata3
                                    del bfft2
                                    del fdata
                                    del bfir
                                    del qdata
                                    del tdata
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
                            del gdata3
                            del fdata
                            del qdata
                            del tdata
                        except NameError:
                            pass
                            
                        break

class MultiPacketizeOp(object):
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
        
        assert(len(self.socks) == self.nbeam_max)
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
        self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))
        
        self.sync_drx_pipelines = MCS.Synchronizer('DRX')
        
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
        
        udts = [UDPTransmit('drx', sock=sock, core=self.core) for sock in self.socks]
        desc = HeaderInfo()
        
        for isequence in self.iring.read():
            ihdr = json.loads(isequence.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
            
            #print('PacketizeOp', ihdr)
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
            NPACKET_SET = 16
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % (NPACKET_SET*int(ntime_pkt))
            if soffset != 0:
                soffset = NPACKET_SET*ntime_pkt - soffset
            boffset = soffset*nbeam*npol
            print('!!', '@', self.beam0, toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset)
            
            time_tag += soffset*ticksPerSample                  # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
            
            prev_time = time.time()
            desc.set_decimation(int(FS)//int(bw))
            desc.set_tuning(int(round(cfreq / FS * 2**32)))
            desc_src = (((self.tuning+1)&0x7)<<3)
            for ispan in isequence.read(gulp_size, begin=boffset):
                if ispan.size < gulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                shape = (-1,nbeam,npol)
                data = ispan.data_view('ci4').reshape(shape)
                pkts = [data[:,b,:].reshape(-1,ntime_pkt,npol).transpose(0,2,1).copy() for b in range(nbeam)]
                
                for t in range(0, pkts[0].shape[0], NPACKET_SET):
                    time_tag_cur = time_tag + t*ticksPerSample*ntime_pkt
                    try:
                        self.sync_drx_pipelines(time_tag_cur)
                    except ValueError:
                        print('speedup', t, pkts[0].shape[0])
                        pass
                    except (socket.timeout, socket.error):
                        pass
                        
                    try:
                        if ACTIVE_DRX_CONFIG.is_set():
                            if not self.tbfLock.is_set():
                                for b in range(nbeam):
                                    udts[b].send(desc, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+(b+self.beam0), 128, 
                                                 pkts[b][t:t+NPACKET_SET,:,:])
                    except Exception as e:
                        print(type(self).__name__, 'Sending Error', str(e))
                            
                time_tag += int(ntime_gulp)*ticksPerSample
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})

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
            print(ex)
            time.sleep(1)
    #print("UTC_START:", utc_start)
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
    
    log.info("Starting %s with PID %i", argv[0], os.getpid())
    log.info("Cmdline args: \"%s\"", ' '.join(argv[1:]))
    log.info("Version:      %s", __version__)
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
    pfb_inverter = True
    if 'pfb_inverter' in tngConfig:
        pfb_inverter = tngConfig['pfb_inverter']
        
    log.info("Src address:  %s:%i", iaddr, iport)
    try:
        for b,a,p in zip(range(len(oaddr)), oaddr, oport):
            bstat = ''
            if b >= nbeam:
                bstat = ' (not used)'
            log.info("Dst address:  %i @ %s:%i%s", b, a, p, bstat)
    except TypeError:
        log.info("Dst address:  %s:%i", oaddr, oport)
    log.info("Servers:      %i-%i", server0+1, server0+nserver)
    log.info("Tuning:       %i (of %i)", tuning+1, ntuning)
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    log.info("PFB inverter: %s", str(pfb_inverter))
    
    iaddr = Address(iaddr, iport)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5
    
    capture_ring = Ring(name="capture-%i" % tuning, space="cuda_host")
    tengine_ring = Ring(name="tengine-%i" % tuning, space="system")
    
    GSIZE = 2500
    nchan_max = int(round(drxConfig['capture_bandwidth']/CHAN_BW))    # Subtly different from what is in adp_drx.py
    
    ops.append(CaptureOp(log, fmt="ibeam%i" % nbeam, sock=isock, ring=capture_ring,
                         nsrc=nserver, src0=server0, max_payload_size=9000,
                         nbeam_max=nbeam, 
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
                         utc_start=utc_start_dt))
    ops.append(TEngineOp(log, capture_ring, tengine_ring,
                         tuning=tuning, ntime_gulp=GSIZE, 
                         nchan_max=nchan_max, nbeam=nbeam, 
                         pfb_inverter=pfb_inverter,
                         core=cores.pop(0), gpu=gpus.pop(0)))
    rsocks = []
    for beam in range(nbeam):
        raddr = Address(oaddr[beam], oport[beam])
        rsocks.append(UDPSocket())
        rsocks[-1].connect(raddr)
    ops.append(MultiPacketizeOp(log, tengine_ring,
                                osocks=rsocks,
                                nbeam_max=nbeam, beam0=1, tuning=tuning,
                                npkt_gulp=32, core=cores.pop(0)))
    
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
