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
from bifrost.unpack import unpack as Unpack
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
from collections import deque

#from numpy.fft import ifft
#from scipy import ifft
from scipy.fftpack import ifft

ACTIVE_TBN_CONFIG = threading.Event()

FILTER2BW = { 1:    1000, 
              2:    3125, 
              3:    6250, 
              4:   12500, 
              5:   25000, 
              6:   50000, 
              7:  100000,
              8:  200000, 
              9:  400000,
             10:  800000,
             11: 1600000}
FILTER2CHAN = { 1:    1000//25000, 
                2:    3125//25000, 
                3:    6250//25000, 
                4:   12500//25000, 
                5:   25000//25000, 
                6:   50000//25000, 
                7:  100000//25000,
                8:  200000//25000, 
                9:  400000//25000,
               10:  800000//25000,
               11: 1600000//25000}

__version__    = "0.1"
__author__     = "Ben Barsdell, Daniel Price, Jayce Dowell"
__copyright__  = "Copyright 2015, The LWA-SV Project"
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
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'nchan':    nchan,
               'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
               'bw':       nchan*CHAN_BW,
               'nsrc':     nsrc, 
               'nstand':   nsrc*16,
               #'stand0':   src0*16, # TODO: Pass src0 to the callback too(?)
               'npol':     2,
               'complex':  True,
               'nbit':     4}
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
        seq_callback.set_chips(self.seq_callback)
        with UDPCapture(*self.args,
                        sequence_callback=seq_callback,
                        **self.kwargs) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                #print(status)
        del capture

class TEngineOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_max=8, nroach=3, # ntime_buf=None,
                 guarantee=True, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
        self.nchan_max = nchan_max
        #if ntime_buf is None:
        #    ntime_buf = self.ntime_gulp*3
        #self.ntime_buf = ntime_buf
        self.guarantee = guarantee
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self.configMessage = ISC.TBNConfigurationClient(addr=('adp',5832))
        self._pending = deque()
        self.gain = 2
        self.filt = 7#list(filter(lambda x: FILTER2CHAN[x]<=self.nchan_max, FILTER2CHAN))[-1]
        self.nchan_out = FILTER2CHAN[self.filt]
        
        #coeffs = np.array([-0.0179700,  0.0144130, -0.0111240, -0.0017506, 0.0254560, 
        #                   -0.0581470,  0.0950000, -0.1290700,  0.1531900, 0.8380900, 
        #                    0.1531900, -0.1290700,  0.0950000, -0.0581470, 0.0254560,
        #                   -0.0017506, -0.0111240,  0.0144130, -0.0179700], dtype=np.float64)
        #coeffs = np.array([-0.0050417, 0.012287, -0.0221660,  0.0303950, -0.0301380,
        #                    0.0120970, 0.036904, -0.1499600,  0.6144000,  0.6144000, 
        #                   -0.1499600, 0.036904,  0.0120970, -0.0301380,  0.0303950,
        #                   -0.0221660, 0.012287, -0.0050417], dtype=np.float64)
        coeffs = np.array([-3.22350e-05, -0.00021433,  0.0017756, -0.0044913,  0.0040327, 
                            0.00735870, -0.03218100,  0.0553980, -0.0398360, -0.0748920, 
                            0.58308000,  0.58308000, -0.0748920, -0.0398360,  0.0553980,
                           -0.03218100,  0.00735870,  0.0040327, -0.0044913,  0.0017756,
                           -0.00021433, -3.2235e-05], dtype=np.float64)
        
        # Setup the T-engine
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Metadata
        nstand, npol = nroach*16, 2
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
        global ACTIVE_TBN_CONFIG
        
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Set the configuration time
            config_time = time.time()
            
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
            self.log.info("TEngine: New configuration received: %s", str(config))
            freq, filt, gain = config
            self.rFreq = freq
            self.filt = filt
            self.nchan_out = FILTER2CHAN[filt]
            self.gain = gain
            
            fDiff = self.rFreq - (hdr['chan0'] + 0.5*(hdr['nchan']-1))*CHAN_BW - CHAN_BW / 2
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu != -1:
                BFSetGPU(self.gpu)
                
            phaseState = fDiff/(self.nchan_out*CHAN_BW)
            phaseRot = np.exp(-2j*np.pi*phaseState*np.arange(self.ntime_gulp*self.nchan_out, dtype=np.float64))
            phaseRot = phaseRot.astype(np.complex64)
            copy_array(self.phaseState, np.array([phaseState,], dtype=np.float64))
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            
            ACTIVE_TBN_CONFIG.set()
            
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
                
                igulp_size = self.ntime_gulp*nchan*nstand*npol*1                # 4+4 complex
                ishape = (self.ntime_gulp,nchan,nstand,npol)
                ogulp_size = self.ntime_gulp*self.nchan_out*nstand*npol*2       # 8+8 complex
                oshape = (self.ntime_gulp*self.nchan_out,nstand,npol,2)
                self.iring.resize(igulp_size)
                self.oring.resize(ogulp_size)#, obuf_size)
                
                ticksPerTime = int(FS) // int(CHAN_BW)
                base_time_tag = iseq.time_tag
                sample_count = 0
                copy_array(self.sampleCount, np.array([sample_count,], dtype=np.int64))
                
                ohdr = {}
                ohdr['nstand']   = nstand
                ohdr['npol']     = npol
                ohdr['complex']  = True
                ohdr['nbit']     = 8
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
                    act_gain = self.gain - 18
                    
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
                                idata = ispan.data_view('ci4').reshape(ishape)
                                odata = ospan.data_view(np.int8).reshape((1,)+oshape)
                                
                                ## Prune the data
                                if idata.shape[1] != self.nchan_out:
                                    try:
                                        pdata[...] = idata[:,nchan//2-self.nchan_out//2:nchan//2+self.nchan_out//2]
                                    except NameError:
                                        pshape = (self.ntime_gulp,self.nchan_out,nstand,npol)
                                        pdata = BFArray(shape=pshape, dtype='ci4', native=False, space='system')
                                        pdata[...] = idata[:,nchan//2-self.nchan_out//2:nchan//2+self.nchan_out//2]
                                else:
                                    pdata = idata
                                    
                                ## Copy the data to the GPU - from here on out we are on the GPU
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
                                BFMap("a(i,j) *= 1./16.*exp(Complex<float>(0.0, -2*BF_PI_F*fmod(g(0)*s(0), 1.0)))*b(i)", 
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
                                    Quantize(fdata, qdata, scale=26.*128./(2**act_gain * np.sqrt(self.nchan_out)))
                                except NameError:
                                    qdata = BFArray(shape=gdata.shape, native=False, dtype='ci8', space='cuda')
                                    Quantize(fdata, qdata, scale=26.*128./(2**act_gain * np.sqrt(self.nchan_out)))
                                    
                                ## Save
                                cdata = qdata.copy(space='system')
                                odata[...] = cdata.view(np.int8).reshape((1,)+oshape)
                                
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
                                ngulp_size = self.ntime_gulp*self.nchan_out*nstand*npol*2    # 8+8 complex
                                nshape = (self.ntime_gulp*self.nchan_out,nstand,npol,2)
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
                                                 
                    # Clean-up
                    try:
                        del pdata
                        del gdata
                        del fdata
                        del qdata
                    except NameError:
                        pass
                        
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        break

class PacketizeOp(object):
    # Note: Input data are: [time,beam,pol,iq]
    def __init__(self, log, iring, osock, nroach, roach0, npkt_gulp=128, core=-1):
        self.log   = log
        self.iring = iring
        self.sock  = osock
        self.nroach = nroach
        self.roach0 = roach0
        self.npkt_gulp = npkt_gulp
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
        self.sync_tbn_pipelines = MCS.Synchronizer('TBN')
        
    def main(self):
        global ACTIVE_TBN_CONFIG
        
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        stand0 = self.roach0 * 16 # TODO: Get this less hackily
        ntime_pkt     = TBN_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nroach*32
        gulp_size_max = ntime_gulp * ninput_max * 2
        self.iring.resize(gulp_size_max)
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        udt = UDPTransmit('tbn', sock=self.sock, core=self.core)
        desc = HeaderInfo()
        
        for iseq in self.iring.read():
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
            
            #print('PacketizeOp', ihdr)
            cfreq  = ihdr['cfreq']
            bw     = ihdr['bw']
            gain   = ihdr['gain']
            nstand = ihdr['nstand']
            #stand0 = ihdr['stand0']
            npol   = ihdr['npol']
            fdly   = (ihdr['fir_size'] - 1) / 2.0
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            gulp_size = ntime_gulp*nstand*npol*2
            
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % int(ntime_pkt)
            if soffset != 0:
                soffset = ntime_pkt - soffset
            boffset = soffset*nstand*npol*2
            print('!!', toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset)
            
            time_tag += soffset*ticksPerSample              # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))     # Correct for FIR filter delay
            
            prev_time = time.time()
            desc.set_tuning(int(round(cfreq / FS * 2**32)))
            desc.set_gain(gain)
            
            free_run = False
            for ispan in iseq.read(gulp_size, begin=boffset):
                if ispan.size < gulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                shape = (-1,nstand,npol)
                data = ispan.data_view('ci8').reshape(shape)
                
                for t in range(0, ntime_gulp, ntime_pkt):
                    time_tag_cur = time_tag + int(t)*ticksPerSample
                    try:
                        self.sync_tbn_pipelines(time_tag_cur)
                    except ValueError:
                        continue
                    except (socket.timeout, socket.error):
                        pass
                        
                    sdata = data[t:t+ntime_pkt,...].transpose(1,2,0).copy()
                    sdata = sdata.reshape(-1,nstand*npol,ntime_pkt)
                    try:
                        if ACTIVE_TBN_CONFIG.is_set():
                            udt.send(desc, time_tag_cur, ticksPerSample*ntime_pkt, 2*stand0, 1, sdata)
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

def partition_balanced(nitem, npart, part_idx):
    rem = nitem % npart
    part_nitem  = nitem // npart + (part_idx < rem)
    part_offset = (part_idx*part_nitem if part_idx < rem else
                   rem*(part_nitem+1) + (part_idx-rem)*part_nitem)
    return part_nitem, part_offset

def partition_packed(nitem, npart, part_idx):
    part_nitem  = (nitem-1) // npart + 1
    part_offset = part_idx * part_nitem
    part_nitem  = min(part_nitem, nitem-part_offset)
    return part_nitem, part_offset

def main(argv):
    parser = argparse.ArgumentParser(description='LWA-SV ADP TBN Service')
    parser.add_argument('-f', '--fork',       action='store_true',       help='Fork and run in the background')
    parser.add_argument('-c', '--configfile', default='adp_config.json', help='Specify config file')
    parser.add_argument('-l', '--logfile',    default=None,              help='Specify log file')
    parser.add_argument('-d', '--dryrun',     action='store_true',       help='Test without acting')
    parser.add_argument('-v', '--verbose',    action='count', default=0, help='Increase verbosity')
    parser.add_argument('-q', '--quiet',      action='count', default=0, help='Decrease verbosity')
    args = parser.parse_args()
    
    # Fork, if requested
    if args.fork:
        stderr = '/tmp/%s.stderr' % (os.path.splitext(os.path.basename(__file__))[0],)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    config = Adp.parse_config_file(args.configfile)
    tbnConfig = config['tbn']
    
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
    server_idx = get_numeric_suffix(hostname) - 1
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    
    ## Network - input
    pipeline_idx = tbnConfig['pipeline_idx']
    if config['host']['servers-data'][server_idx].startswith('adp'):
        iaddr    = config['server']['data_ifaces'][pipeline_idx]
    else:
        iaddr    = config['host']['servers-data'][server_idx]
    iport        = config['server']['data_ports' ][pipeline_idx]
    ## Network - output
    recorder_idx = tbnConfig['recorder_idx']
    recConfig    = config['recorder'][recorder_idx]
    oaddr        = recConfig['host']
    oport        = recConfig['port']
    
    nroach_tot = len(config['host']['roaches'])
    nserver    = len(config['host']['servers'])
    tbn_servers = config['host']['servers-tbn']
    server_data_host = config['host']['servers-data'][server_idx]
    nroach = len([srv for srv in tbn_servers if srv == server_data_host])
    roach0 = [i for (i,srv) in enumerate(tbn_servers) if srv == server_data_host][0]
    cores = config['tbn']['cpus']
    gpus  = config['tbn']['gpus']
    
    log.info("Src address:  %s:%i", iaddr, iport)
    log.info("Dst address:  %s:%i", oaddr, oport)
    log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
    log.info("Cores:        %s", ' '.join([str(v) for v in cores]))
    
    iaddr = Address(iaddr, iport)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5
    
    capture_ring = Ring(name="capture")
    tengine_ring = Ring(name="tengine")
    
    oaddr = Address(oaddr, oport)
    osock = UDPSocket()
    osock.connect(oaddr)
    
    GSIZE = 2500
    ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
                         nsrc=nroach, src0=roach0, max_payload_size=9000,
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
                         utc_start=utc_start_dt))
    ops.append(TEngineOp(log, capture_ring, tengine_ring,
                         ntime_gulp=GSIZE, nroach=nroach, 
                         core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(PacketizeOp(log, tengine_ring, osock=osock, 
                           nroach=nroach, roach0=roach0,
                           npkt_gulp=7, core=cores.pop(0)))
    
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
