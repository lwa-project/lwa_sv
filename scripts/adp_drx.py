#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from adp import MCS2 as MCS
from adp import Adp
from adp.AdpCommon import *
from adp import ISC

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPVerbsCapture as UDPCapture
from bifrost.packet_writer import HeaderInfo, DiskWriter, UDPTransmit
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.unpack import unpack as Unpack
from bifrost.reduce import reduce as Reduce
from bifrost.quantize import quantize as Quantize
from bifrost.transpose import transpose as Transpose
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.linalg import LinAlg
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

from btcc import Btcc

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

ACTIVE_COR_CONFIG = threading.Event()

__version__    = "0.2"
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
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        time_tag_ptr[0] = time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'nchan':    nchan,
               'cfreq':    (chan0 + 0.5*(nchan-1))*CHAN_BW,
               'bw':       nchan*CHAN_BW,
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

class BufferCopyOp(object):
    def __init__(self, log, iring, oring, tuning=0, ntime_gulp=2500,# ntime_buf=None,
                 guarantee=True, core=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.tuning = tuning
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
        
        tid = "%s-drx-%i" % (socket.gethostname(), tuning)
        self.internal_trigger = ISC.InternalTrigger(id=tid)
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("BufferCopy: Start of new sequence: %s", str(ihdr))
                
                chan0  = ihdr['chan0']
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                igulp_size = self.ntime_gulp*nchan*nstand*npol
                ogulp_size = igulp_size
                #obuf_size  = 5*25000*nchan*nstand*npol
                ishape = (self.ntime_gulp,nchan,nstand*npol)
                self.iring.resize(igulp_size, igulp_size*10)
                #self.oring.resize(ogulp_size)#, obuf_size)
                
                ticksPerTime = int(FS / CHAN_BW)
                base_time_tag = iseq.time_tag
                
                clear_to_trigger = False
                if chan0*CHAN_BW > 60e6 and self.tuning == 1:
                    clear_to_trigger = True
                to_keep = [6,7, 224,225, 494,495]
                tchan = min([72, nchan])
                udata = BFArray(shape=(self.ntime_gulp, tchan, len(to_keep)), dtype=np.complex64)
                
                ohdr = ihdr.copy()
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    ohdr['timetag'] = base_time_tag
                    ohdr_str = json.dumps(ohdr)
                    
                    with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                # Is this really needed or is ispan.size always zero when we hit this?
                                print("too small at %i vs %i" % (ispan.size, igulp_size))
                                base_time_tag += (ispan.size//(nchan*nstand*npol))*ticksPerTime
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            try:
                                with oseq.reserve(ogulp_size, nonblocking=True) as ospan:
                                    curr_time = time.time()
                                    reserve_time = curr_time - prev_time
                                    prev_time = curr_time
                                    
                                    idata = ispan.data_view(np.uint8)
                                    odata = ospan.data_view(np.uint8)    
                                    copy_array(odata, idata)
                                    
                                    # Internal triggering code
                                    if clear_to_trigger:
                                        t0 = time.time()
                                        sdata = idata.reshape(ishape)
                                        sdata = sdata[:,:,to_keep]
                                        if tchan != nchan:
                                            sdata = sdata[:,:tchan,:]
                                        sdata = sdata.copy(space='system')
                                        sdata = BFArray(shape=sdata.shape, dtype='ci4', native=False, buffer=sdata.ctypes.data)
                                        
                                        Unpack(sdata, udata)
                                        
                                        pdata = udata.real*udata.real + udata.imag*udata.imag
                                        pdata = pdata.reshape(self.ntime_gulp,-1)
                                        pdata = pdata.mean(axis=1)
                                        
                                        s = np.argmax(pdata)
                                        m = pdata.mean()
                                        
                                        if m > 50.0:
                                            self.internal_trigger(base_time_tag+s*ticksPerTime)
                                            print(m, '@', base_time_tag+s*ticksPerTime, '>', time.time()-t0)
                                            
                            except IOError:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                reset_sequence = True
                                
                            ## Update the base time tag
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                            
                            # Reset to move on to the next input sequence?
                            if reset_sequence:
                                break
                                
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        break

class GPUCopyOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=2500, guarantee=True, core=-1, gpu=-1):
        self.log   = log
        self.iring = iring
        self.oring = oring
        self.ntime_gulp = ntime_gulp
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
                
                self.log.info("GPUCopy: Start of new sequence: %s", str(ihdr))
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                igulp_size = self.ntime_gulp*nchan*nstand*npol              # 4+4 complex
                ogulp_size = igulp_size
                
                ticksPerTime = int(FS) // int(CHAN_BW)
                base_time_tag = iseq.time_tag
                
                ohdr = ihdr.copy()
                ohdr_str = json.dumps(ohdr)
                
                self.oring.resize(ogulp_size, 10*ogulp_size)
                
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
                            
                            ## Setup and load
                            idata = ispan.data_view(np.uint8)
                            odata = ospan.data_view(np.uint8)
                            
                            ## Copy
                            copy_array(odata, idata)
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp*ticksPerTime
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})

def get_time_tag(dt=datetime.datetime.utcnow(), seq_offset=0):
    timestamp = int((dt - ADP_EPOCH).total_seconds())
    time_tag  = timestamp*int(FS) + seq_offset*(int(FS)//int(CHAN_BW))
    return time_tag
def seq_to_time_tag(seq):
    return seq*(int(FS)//int(CHAN_BW))
def time_tag_to_seq_float(time_tag):
    return time_tag*CHAN_BW/FS

class TriggeredDumpOp(object):
    def __init__(self, log, osock, iring, ntime_gulp, ntime_buf, tuning=0, nchan_max=256, core=-1, max_bytes_per_sec=None):
        self.log = log
        self.sock = osock
        self.iring = iring
        self.tuning = tuning
        self.nchan_max = nchan_max
        self.core  = core
        self.ntime_gulp = ntime_gulp
        self.ntime_buf = ntime_buf
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
        self.configMessage = ISC.TriggerClient(addr=('adp',5832))
        self.tbfLock       = ISC.PipelineEventClient(addr=('adp',5834))
        
        if max_bytes_per_sec is None:
            max_bytes_per_sec = 104857600        # default to 100 MB/s
        self.max_bytes_per_sec = max_bytes_per_sec
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        ninput_max = 512
        frame_nbyte_max = self.nchan_max*ninput_max
        #self.iring.resize(self.ntime_gulp*frame_nbyte_max,
        #                  self.ntime_buf *frame_nbyte_max)
                          
        self.udt = UDPTransmit('tbf', sock=self.sock, core=self.core)
        self.desc = HeaderInfo()
        while not self.iring.writing_ended():
            config = self.configMessage(block=False)
            if not config:
                time.sleep(0.1)
                continue
                
            print("Trigger: New trigger received: %s" % str(config))
            try:
                self.dump(time_tag=config[0], samples=config[1], mask=config[2], local=config[3])
            except Exception as e:
                print("Error on TBF dump: %s" % str(e))
                
        del self.udt
        
        print("Writing ended, TBFOp exiting")
        
    def dump(self, samples, time_tag=0, mask=None, local=False):
        if mask is None:
            mask = 0b11
        if (mask >> self.tuning) & 1 == 0:
            self.log.info('Not for us: %i -> %i @ %i', mask, (mask >> self.tuning) & 1, self.tuning)
            return False
        speed_factor = 2 // sum([mask>>i&1 for i in range(2)])        # TODO: Slightly hacky
        
        ntime_pkt = 1 # TODO: Should be TBF_NTIME_PER_PKT?
        
        # HACK TESTING
        dump_time_tag = time_tag
        if (dump_time_tag == 0 and local) or not local:
            time_offset    = -4.0
            time_offset_s  = int(time_offset)
            time_offset_us = int(round((time_offset-time_offset_s)*1e6))
            time_offset    = datetime.timedelta(seconds=time_offset_s, microseconds=time_offset_us)
            
            utc_now = datetime.datetime.utcnow()
            dump_time_tag = get_time_tag(utc_now + time_offset)
        #print("********* dump_time_tag =", dump_time_tag)
        #time.sleep(3)
        #ntime_dump = 0.25*1*25000
        #ntime_dump = 0.1*1*25000
        ntime_dump = int(round(time_tag_to_seq_float(samples)))
        
        max_bytes_per_sec = self.max_bytes_per_sec
        if local:
            if os.path.exists(TRIGGERING_ACTIVE_FILE):
                max_bytes_per_sec = 8388608 # Limit to 8 MB/s
                speed_factor = 1
            else:
                max_bytes_per_sec = 104857600 # Limit to 100 MB/s
                speed_factor = 1
                
        print("TBF DUMPING %f secs at time_tag = %i (%s)%s" % (samples/FS, dump_time_tag, datetime.datetime.utcfromtimestamp(dump_time_tag/FS), (' locally' if local else '')))
        if not local:
            self.tbfLock.set()
        with self.iring.open_sequence_at(dump_time_tag, guarantee=True) as iseq:
            time_tag0 = iseq.time_tag
            ihdr = json.loads(iseq.header.tostring())
            nchan  = ihdr['nchan']
            chan0  = ihdr['chan0']
            nstand = ihdr['nstand']
            npol   = ihdr['npol']
            ninput = nstand*npol
            print("*******", nchan, ninput)
            ishape = (-1,nchan,ninput)#,2)
            frame_nbyte = nchan*ninput#*2
            igulp_size = self.ntime_gulp*nchan*ninput#*2
            
            dump_seq_offset  = int(time_tag_to_seq_float(dump_time_tag - time_tag0))
            dump_byte_offset = dump_seq_offset * frame_nbyte
            
            # HACK TESTING write to file instead of socket
            if local:
                filename = '/data0/test_%s_%i_%020i.tbf' % (socket.gethostname(), self.tuning, dump_time_tag)#time_tag0
                ofile = open(filename, 'wb')
                ldw = DiskWriter('tbf', ofile, core=self.core)
            ntime_dumped = 0
            nchan_rounded = nchan // TBF_NCHAN_PER_PKT * TBF_NCHAN_PER_PKT
            bytesSent, bytesStart = 0.0, time.time()
            
            print("Opening read space of %i bytes at offset = %i" % (igulp_size, dump_byte_offset))
            pkts = []
            for ispan in iseq.read(igulp_size, begin=dump_byte_offset):
                print("**** ispan.size, offset", ispan.size, ispan.offset)
                print("**** Dumping at", ntime_dumped)
                if ntime_dumped >= ntime_dump:
                    break
                #print(ispan.offset, seq_offset)
                seq_offset = ispan.offset // frame_nbyte
                data = ispan.data_view('ci4').reshape(ishape)
                data = data[:,:nchan_rounded,:].copy()
                
                for t in range(0, self.ntime_gulp, ntime_pkt):
                    if ntime_dumped >= ntime_dump:
                        break
                    ntime_dumped += 1
                    
                    #pkts = []
                    time_tag = time_tag0 + seq_to_time_tag(seq_offset + t)
                    if t == 0:
                        print("**** first timestamp is", time_tag)
                        
                    sdata = data[t:t+ntime_pkt,...]
                    sdata = sdata.reshape(1,nchan//TBF_NCHAN_PER_PKT,-1)
                    if local:
                        ldw.send(self.desc,
                                 time_tag, int(FS)//int(CHAN_BW), 
                                 chan0, TBF_NCHAN_PER_PKT, sdata)
                        bytesSent += sdata.size // 6144 * 6168   # data size -> packet size
                    else:
                        try:
                            self.udt.send(self.desc,
                                          time_tag, int(FS)//int(CHAN_BW), 
                                          chan0, TBF_NCHAN_PER_PKT, sdata)
                            bytesSent += sdata.size // 6144 * 6168   # data size -> packet size
                        except Exception as e:
                            print(type(self).__name__, 'Sending Error', str(e))
                            
                    while bytesSent/(time.time()-bytesStart) >= max_bytes_per_sec*speed_factor:
                        time.sleep(0.001)
                        
            if local:
                del ldw
                ofile.close()
                
                # Try to make sure that everyone releases the ring lock at the same time
                ts = time.time()
                if ts-int(ts) < 0.75:
                    while time.time() < int(ts)+1:
                        time.sleep(0.001)
                else:
                    while time.time() < int(ts)+2:
                        time.sleep(0.001)
                        
        if not local:
            self.tbfLock.clear()
            
        print("TBF DUMP COMPLETE at %.3f - average rate was %.3f MB/s" % (time.time(), bytesSent/(time.time()-bytesStart)/1024**2))

class BeamformerOp(object):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nbeam_max=1, nroach=16, ntime_gulp=2500, guarantee=True, core=-1, gpu=-1):
        self.log   = log
        self.iring = iring
        self.oring = oring
        self.tuning = tuning
        ninput_max = nroach*16#*2
        self.ntime_gulp = ntime_gulp
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
        
        self.nchan_max = nchan_max
        self.nbeam_max = nbeam_max
        self.configMessage = ISC.BAMConfigurationClient(addr=('adp',5832))
        self._pending = deque()
        
        # Setup the beamformer
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Metadata
        nchan = self.nchan_max
        nstand, npol = nroach*16, 2
        ## Object
        self.bfbf = LinAlg()
        ## Delays and gains
        self.delays = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
        self.gains = np.zeros((self.nbeam_max*2,nstand*npol), dtype=np.float64)
        self.cgains = BFArray(shape=(self.nbeam_max*2,nchan,nstand*npol), dtype=np.complex64, space='cuda')
        ## Intermidiate arrays
        ## NOTE:  This should be OK to do since the snaps only output one bandwidth per INI
        self.bdata = BFArray(shape=(nchan,self.nbeam_max*2,self.ntime_gulp), dtype=np.complex64, space='cuda')
        self.ldata = BFArray(shape=(self.ntime_gulp,self.nbeam_max,nchan,2), dtype=self.bdata.dtype, space='cuda')
        self.bdata_reorder_shape = (nchan,self.nbeam_max,2,self.ntime_gulp)
        self.bdata_reorder_axes = (3,1,0,2)
        self.odata_shape = self.ldata.shape
        
        try:
            ##Read in the precalculated complex gains for all of the steps of the achromatic beams.
            hostname = socket.gethostname()
            cgainsFile = '/home/adp/complexGains_%s.npz' % hostname
            self.complexGains = np.load(cgainsFile)['cgains'][:,:8,:,:]
    	
            ##Figure out which indices to pull for the given tuning
            good = np.where(np.arange(self.complexGains.shape[1]) // 2 % 2 == self.tuning)[0]
            self.complexGains = self.complexGains[:,good,:,:]
        except Exception as e:
            self.log.warning("Failed to load custom beamforming coefficients: %s", str(e))
            self.complexGains = None
            
    def updatePacketizerPreferences(self, pkt_op):
        ntime = self.ntime_gulp
        nchan = self.nchan_max
        nbeam = self.nbeam_max
        npol  = 2
        
        self.pkt_gulp = pkt_op.ntime_gulp
        self.pkt_nchan = pkt_op.nchan_send
        assert(ntime % self.pkt_gulp == 0)
        assert(nchan % self.pkt_nchan == 0)
        
        self.ldata =  BFArray(shape=(ntime//self.pkt_gulp,
                                     nbeam,
                                     nchan//self.pkt_nchan,
                                     self.pkt_gulp,
                                     self.pkt_nchan,
                                     npol), dtype=self.bdata.dtype, space='cuda')
        self.bdata_reorder_shape = (nchan//self.pkt_nchan,
                                    self.pkt_nchan,
                                    nbeam,
                                    npol,
                                    ntime//self.pkt_gulp,
                                    self.pkt_gulp)
        self.bdata_reorder_axes = (4,2,0,5,1,3)
        self.odata_shape = self.ldata.shape
        
    def updateConfig(self, config, hdr, time_tag, forceUpdate=False):

        if self.gpu != -1:
            BFSetGPU(self.gpu)
            
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Pull out the tuning (something unique to DRX/BAM/COR)
            beam = config[1]
            if beam >= self.nbeam_max:
                return False
                
            ## Set the configuration time - BAM commands are for the specified subslot two seconds from when it was received
            slot = config[0] + config[4] / 100.0
            config_time = slot + 2
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                idx = bisect.bisect_right(self._pending, (config_time,))
                self._pending.insert(idx, (config_time, config))
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
            self.log.info("Beamformer: New configuration received for beam %i (delta = %.1f subslots)", config[1], (pipeline_time-config_time)*100.0)
            _, beam, delays, gains, slot = config
            
            #Search for the "code word" gain pattern which specifies an achromatic observation.
            if ( gains[0,:,:] == np.array([[8191, 16383],[32767,65535]]) ).all():
                #The pointing index is stored in the second gains entry. Pointings start at 1.
                pointing = gains[1,0,0]

                #Set the custom complex gains.
                try:
                    self.cgains[2*beam+0,:,:] = self.complexGains[pointing-1,2*beam+0,:,:]
                    self.cgains[2*beam+1,:,:] = self.complexGains[pointing-1,2*beam+1,:,:]
                    self.log.info("Beamformer: Custom complex gains set for pointing number %i of beam %i", pointing, beam)
                except (TypeError, IndexError):
                    self.cgains[2*beam+0,:,:] = np.zeros( (self.cgains.shape[1],self.cgains.shape[2]) )
                    self.cgains[2*beam+1,:,:] = np.zeros( (self.cgains.shape[1],self.cgains.shape[2]) )
                    self.log.info("Beamformer: Ran out of pointings...setting complex gains to zero.")
                    
            else:
                # Byteswap to get into little endian
                delays = delays.byteswap().newbyteorder()
                gains = gains.byteswap().newbyteorder()
            
                # Unpack and re-shape the delays (to seconds) and gains (to floating-point)
                delays = (((delays>>4)&0xFFF) + (delays&0xF)/16.0) / FS
                gains = gains/32767.0
                gains.shape = (gains.size//2, 2)
                
                # Trim down the the correct size for our number of stands
                delays = delays[:self.delays.shape[1]]
                gains = gains[:self.gains.shape[1],:]
            
                # Update the internal delay and gain cache so that we can use these later
                self.delays[2*beam+0,:] = delays
                self.delays[2*beam+1,:] = delays
                self.gains[2*beam+0,:] = gains[:,0]
                self.gains[2*beam+1,:] = gains[:,1]
            
                # Compute the complex gains needed for the beamformer
                freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
                freqs.shape = (freqs.size, 1)
                self.cgains[2*beam+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*beam+0,:]) * \
                                                self.gains[2*beam+0,:]).astype(np.complex64)
                self.cgains[2*beam+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*beam+1,:]) * \
                                                self.gains[2*beam+1,:]).astype(np.complex64)
            BFSync()
            self.log.info('  Complex gains set - beam %i' % beam)
            
            return True
            
        elif forceUpdate:
            self.log.info("Beamformer: New sequence configuration received")
            
            # Compute the complex gains needed for the beamformer
            freqs = CHAN_BW * (hdr['chan0'] + np.arange(hdr['nchan']))
            freqs.shape = (freqs.size, 1)
            for beam in range(self.nbeam_max):
                self.cgains[2*beam+0,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*beam+0,:]) \
                                                * self.gains[2*beam+0,:]).astype(np.complex64)
                self.cgains[2*beam+1,:,:] = (np.exp(-2j*np.pi*freqs*self.delays[2*beam+1,:]) \
                                                * self.gains[2*beam+1,:]).astype(np.complex64)
                BFSync()
                self.log.info('  Complex gains set - beam %i' % beam)
                
            return True
            
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
                
                status = self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )
                
                self.log.info("Beamformer: Start of new sequence: %s", str(ihdr))
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                igulp_size = self.ntime_gulp*nchan*nstand*npol              # 4+4 complex
                ogulp_size = self.ntime_gulp*nchan*self.nbeam_max*npol*8    # complex64
                ishape = (self.ntime_gulp,nchan,nstand*npol)
                oshape = self.odata_shape #(self.ntime_gulp,self.nbeam_max,nchan,2)
                
                ticksPerTime = int(FS) // int(CHAN_BW)
                base_time_tag = iseq.time_tag
                
                ohdr = ihdr.copy()
                ohdr['nstand'] = self.nbeam_max
                ohdr['nbit'] = 32
                ohdr['complex'] = True
                ohdr['pkt_gulp'] = self.pkt_gulp
                ohdr['pkt_nchan'] = self.pkt_nchan
                ohdr_str = json.dumps(ohdr)
                
                self.oring.resize(ogulp_size, 10*ogulp_size)
                
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
                            
                            ## Setup and load
                            idata = ispan.data_view('ci4').reshape(ishape)
                            odata = ospan.data_view(np.complex64).reshape(oshape)
                            
                            ## Beamform
                            self.bdata = self.bdata.reshape(nchan,self.nbeam_max*2,self.ntime_gulp)
                            self.bdata = self.bfbf.matmul(1.0, self.cgains.transpose(1,0,2), idata.transpose(1,2,0), 0.0, self.bdata)
                            
                            ## Transpose and save
                            self.bdata = self.bdata.reshape(self.bdata_reorder_shape)
                            Transpose(self.ldata, self.bdata, axes=self.bdata_reorder_axes)
                            copy_array(odata, self.ldata)
                            
                        ## Update the base time tag
                        base_time_tag += self.ntime_gulp*ticksPerTime
                        
                        ## Check for an update to the configuration
                        self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False )
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})

class CorrelatorOp(object):
    # Note: Input data are: [time,chan,ant,pol,cpx,8bit]
    def __init__(self, log, iring, oring, tuning=0, nchan_max=256, nroach=16, ntime_gulp=2500, utc_start_tt=None, guarantee=True, core=-1, gpu=-1):
        self.log   = log
        self.iring = iring
        self.oring = oring
        self.tuning = tuning
        ninput_max = nroach*16#*2
        self.ntime_gulp = ntime_gulp
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
        
        self.nchan_max = nchan_max
        def null_config():
            return None
        self.configMessage = null_config#ISC.CORConfigurationClient(addr=('adp',5832))
        self._pending = deque()
        self.navg_tt = int(round(5 * FS // (2*NCHAN*self.ntime_gulp))) * (2*NCHAN*self.ntime_gulp)
        self.navg_seq = self.navg_tt // (2*NCHAN)
        self.gain = 0
        
        # Setup the correlator
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Start time reference
        if utc_start_tt is None:
            utc_start_tt = int(round(time.time() * FS))
        self.start_time_tag = int(round(utc_start_tt // (2*NCHAN*self.ntime_gulp))) * 2*NCHAN*self.ntime_gulp
        ## Metadata
        self.decim = 4
        nchan = self.nchan_max
        ochan = nchan//self.decim
        nstand, npol = nroach*16, 2
        ## Object
        self.bfcc = Btcc()
        self.bfcc.init(8, int(np.ceil((self.ntime_gulp/16.0))*16), ochan, nstand, npol)
        ## Intermediate arrays
        ## NOTE:  This should be OK to do since the snaps only output one bandwidth per INI
        self.udata = BFArray(shape=(int(np.ceil((self.ntime_gulp/16.0))*16),ochan,nstand*npol), dtype='ci8', space='cuda')
        self.cdata = BFArray(shape=(ochan,nstand*(nstand+1)//2*npol*npol), dtype='ci32', space='cuda')
        self.tdata = BFArray(shape=(nstand*(nstand+1)//2,ochan,npol*npol), dtype='ci32', space='cuda')
        self.cdata_reorder_shape = (ochan,nstand*(nstand+1)//2,npol*npol)
        self.cdata_reorder_axes = (1,0,2)
        self.odata_shape = self.tdata.shape
        
    def updatePacketizerPreferences(self, pkt_op):
        ochan, nblpol = self.cdata.shape
        nbl, npol = nblpol//4, 4
        
        self.pkt_nchan = pkt_op.nchan_send
        assert(ochan % self.pkt_nchan == 0)
        
        self.tdata =  BFArray(shape=(ochan//self.pkt_nchan,
                                     nbl,
                                     self.pkt_nchan,
                                     npol), dtype=self.cdata.dtype, space='cuda')
        self.cdata_reorder_shape = (ochan//self.pkt_nchan,
                                    self.pkt_nchan,
                                    nbl,
                                    npol)
        self.cdata_reorder_axes = (0,2,1,3)
        self.odata_shape = self.tdata.shape
        
    def updateConfig(self, config, hdr, time_tag, forceUpdate=False):
        if self.gpu != -1:
            BFSetGPU(self.gpu)
            
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Can we act on this configuration change now?
        if config:
            ## Set the configuration time - COR commands are for the specified subslot two seconds from it was received
            slot = config[0] + config[3] / 100.0
            config_time = slow + 2
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                idx = bisect.bisect_right(self._pending, (config_time,))
                self._pending.insert(idx, (config_time, config))
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
            self.log.info("Correlator: New configuration received for tuning (delta = %.1f subslots)", (pipeline_time-config_time)*100.0)
            _, navg, gain, slot = config
            
            self.navg_tt = int(round(navg/100 * FS // (2*NCHAN*self.ntime_gulp))) * (2*NCHAN*self.ntime_gulp)
            self.navg_seq = self.navg_tt // (2*NCHAN)
            self.log.info('  Averaging time set')
            self.gain = gain
            self.log.info('  Gain set')
            
            return True
            
        elif forceUpdate:
            self.log.info("Correlator: New sequence configuration received")
            
            return True
            
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
        
        last_base_time_tag = 0
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.updateConfig( self.configMessage(), ihdr, iseq.time_tag, forceUpdate=True )
                
                self.log.info("Correlator: Start of new sequence: %s", str(ihdr))
                
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                ochan  = nchan // self.decim
                igulp_size = self.ntime_gulp*nchan*nstand*npol        # 4+4 complex
                ogulp_size = ochan*nstand*(nstand+1)//2*npol*npol*8   # 32+32 complex
                ishape = (self.ntime_gulp,nchan,nstand*npol)
                oshape = self.odata_shape
                
                # Figure out where we need to be in the buffer to be a integration boundary
                ticksPerTime = 2*NCHAN
                sOffset = ((iseq.time_tag - self.start_time_tag) // (2*NCHAN)) % self.navg_seq
                sOffset = sOffset % self.ntime_gulp
                sOffset = self.ntime_gulp - sOffset
                bOffset = sOffset * nchan*nstand*npol
                print('!! @ cor', iseq.time_tag, self.start_time_tag, '->', sOffset, '&', bOffset)
                
                base_time_tag = iseq.time_tag + sOffset*ticksPerTime
                
                ohdr = ihdr.copy()
                ohdr['nchan'] = ochan
                ohdr['nbit'] = 32
                ohdr['complex'] = True
                ohdr_str = json.dumps(ohdr)
                
                self.oring.resize(ogulp_size)
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size, begin=bOffset)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    nAccumulate = ((base_time_tag - self.start_time_tag) // (2*NCHAN)) % self.navg_seq
                    if base_time_tag == last_base_time_tag:
                        base_time_tag = base_time_tag + self.navg_tt
                        nAccumulate = -self.navg_seq
                    
                    gain_act = 1.0 / 2**self.gain / self.navg_seq
                    
                    ohdr['time_tag']  = base_time_tag
                    ohdr['start_tag'] = self.start_time_tag
                    ohdr['navg']      = self.navg_tt
                    ohdr['gain']      = self.gain
                    ohdr_str = json.dumps(ohdr)
                    
                    with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Pre-update the base time tag - we need this so that we can dump at the right times
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            ## Setup and load
                            idata = ispan.data_view('ci4').reshape(ishape)
                            
                            ## Decimate
                            BFMap(f"""
                                  // Unpack into real and imaginary, and then sum
                                  int jF;
                                  signed char sample, re, im;
                                  re = im = 0;

                                  #pragma unroll
                                  for(int l=0; l<{self.decim}; l++) {{
                                      jF = j*{self.decim} + l;
                                      sample = a(i,jF,k).real_imag;
                                      re += ((signed char)  (sample & 0xF0))       / 16;
                                      im += ((signed char) ((sample & 0x0F) << 4)) / 16;
                                  }}

                                  // Save
                                  b(i,j,k) = Complex<signed char>(re, im);
                                  """,
                                  {'a': idata, 'b': self.udata},
                                  axis_names=('i','j','k'),
                                  shape=(self.ntime_gulp,ochan,nstand*npol)
                                 )
                            
                            ## Correlate
                            corr_dump = 0
                            if nAccumulate == self.navg_seq - self.ntime_gulp:
                                corr_dump = 1
                            self.bfcc.execute(self.udata, self.cdata, corr_dump)
                            nAccumulate += self.ntime_gulp
                            
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Dump?
                            if nAccumulate == self.navg_seq:
                                ### Transpose and save
                                self.cdata = self.cdata.reshape(self.cdata_reorder_shape)
                                Transpose(self.tdata, self.cdata, axes=self.cdata_reorder_axes)
                                
                                with oseq.reserve(ogulp_size) as ospan:
                                    odata = ospan.data_view('ci32').reshape(oshape)
                                    copy_array(odata, self.tdata)
                                    print('dump @', base_time_tag)
                                    
                                nAccumulate = 0
                                self.cdata = self.cdata.reshape(ochan,nstand*(nstand+1)//2*npol*npol)
                            else:
                                BFSync()
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Check for an update to the configuration
                            if self.updateConfig( self.configMessage(), ihdr, base_time_tag, forceUpdate=False ):
                                reset_sequence = True
                                last_base_time_tag = base_time_tag
                                break
                                
                            curr_time = time.time()
                            process_time += curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                            
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        last_base_time_tag = base_time_tag
                        break

class RetransmitOp(object):
    def __init__(self, log, osocks, iring, tuning=0, ntuning=4, nchan_max=256, ntime_gulp=2500, nbeam_max=1, guarantee=True, core=-1):
        self.log   = log
        self.socks = osocks
        self.iring = iring
        self.tuning = tuning
        self.ntuning = ntuning
        self.ntime_gulp = ntime_gulp
        self.nbeam_max = nbeam_max
        self.guarantee = guarantee
        self.core = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self.server = int(socket.gethostname().replace('adp', '0'), 10)
        self.nchan_max = nchan_max
        
        self.udts = []
        self.nchan_send = min([self.nchan_max, 384])
        self.nblock_send = self.nchan_max // self.nchan_send
        for sock in self.socks:
            udt = UDPTransmit('ibeam%i_%i' % (1, self.nchan_send), sock=sock, core=self.core)
            #udt.set_rate_limit(430000)
            for i in range(self.nblock_send):
                # Recycle transmitters so that we can index easier later on
                self.udts.append(udt)
                
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        desc = []
        src_id = []
        for i in range(len(self.socks)):
            for j in range(self.nblock_send):
                desc.append(HeaderInfo())
                desc[-1].set_tuning(1+i)
                desc[-1].set_nchan(self.nchan_send)
                desc[-1].set_nsrc(self.ntuning*self.nblock_send*4)
                
                src_id.append(self.nblock_send*self.tuning*4 + j*self.server)
                self.log("tuning: %i, server %i -> src_id: %i", self.tuning, self.server, src_id[0])
                
        for iseq in self.iring.read():
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Retransmit: Start of new sequence: %s", str(ihdr))
            
            chan0   = ihdr['chan0']
            nchan   = ihdr['nchan']
            nstand  = ihdr['nstand']
            npol    = ihdr['npol']
            nstdpol = nstand * npol
            us_pkt_gulp = ihdr['pkt_gulp']
            us_pkt_nchan = ihdr['pkt_nchan']
            assert(us_pkt_gulp == self.ntime_gulp)
            assert(us_pkt_nchan == self.nchan_send)
            
            igulp_size = nstand*self.ntime_gulp*nchan*npol*8        # complex64
            igulp_shape = (nstand*self.nblock_send,self.ntime_gulp,1,self.nchan_send*npol)
            
            seq0 = ihdr['seq0']
            seq = seq0
            
            for i in range(nstand):
                for j in range(self.nblock_send):
                    desc[i*self.nblock_send + j].set_chan0(chan0 + j*self.nchan_send)
                    
            # Set the output ordering such that we send one subband to all beams
            # before moving on to the next subband.
            output_ordering = []
            for i in range(nstand):
                for j in range(self.nblock_send):
                    output_ordering.append(j*nstand + i)
                    
            # Offset the pipelines
            if self.tuning % 2 == 1:
                output_ordering = deque(output_ordering)
                output_ordering.rotate(-1)
                output_ordering = list(output_ordering)
                
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(np.complex64).reshape(igulp_shape)
                
                pkt_time_start = time.time()
                for i in output_ordering:
                    try:
                        self.udts[i].send(desc[i], seq, 1, src_id[i], 1, idata[i,...])
                    except Exception as e:
                        print(type(self).__name__, "Sending Error beam %i, block %i" % (i//self.nblock_send+1, i%self.nblock_send+1), str(e))
                        
                seq += self.ntime_gulp
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
        while len(self.udts):
            udt = self.udts.pop()
            del udt
            

class PacketizeOp(object):
    # Note: Input data are: [time,beam,pol,iq]
    def __init__(self, log, iring, osock, tuning=0, nchan_max=256, nroach=16, npkt_gulp=128, core=-1, gpu=-1, max_bytes_per_sec=None):
        self.log   = log
        self.iring = iring
        self.sock  = osock
        self.tuning = tuning
        self.npkt_gulp = npkt_gulp
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
        self.server = int(socket.gethostname().replace('adp', '0'), 10)
        self.nchan_max = nchan_max
        if max_bytes_per_sec is None:
            max_bytes_per_sec = 104857600        # default to 100 MB/s
        self.max_bytes_per_sec = max_bytes_per_sec
        
        # Setup the packetizer
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        ## Metadata
        self.nchan_send = min([self.nchan_max, 192])
        self.nblock_send = self.nchan_max // self.nchan_send
        nstand, npol = nroach*16, 2
        
        # Output packet rate
        ## nchan_send + npol^2 -> samples per packet
        samps_per_pkt = self.nchan_send * npol*npol
        ## dtype (cf32) -> bytes per sample
        bytes_per_samp = 4*2
        ## B/s -> pkts/s
        self.max_pkts_per_sec = int(self.max_bytes_per_sec / bytes_per_samp / samps_per_pkt)
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        with UDPTransmit('cor_%i' % self.nchan_send, sock=self.sock, core=self.core) as udt:
            #udt.set_rate_limit(self.max_pkts_per_sec)
            
            desc = []
            for i in range(self.nblock_send):
                desc.append(HeaderInfo())
                desc[-1].set_tuning((4 << 16) | (16 << 8) | (self.nblock_send*self.tuning*self.server + i + 1))
            
            for iseq in self.iring.read():
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("Packetizer: Start of new sequence: %s", str(ihdr))
                
                #print('PacketizeOp', ihdr)
                chan0  = ihdr['chan0']
                nchan  = ihdr['nchan']
                nstand = ihdr['nstand']
                npol   = ihdr['npol']
                navg   = ihdr['navg']
                gain   = ihdr['gain']
                time_tag0 = iseq.time_tag
                time_tag  = time_tag0
                igulp_size = nchan*nstand*(nstand+1)//2*npol*npol*8    # 32+32 complex
                ishape = (self.nblock_send,1,nstand*(nstand+1)//2,self.nchan_send*npol*npol)
                
                for i in range(self.nblock_send):
                    desc[i].set_chan0(chan0 + i*self.nchan_send)
                    desc[i].set_gain(gain)
                    desc[i].set_decimation(navg)
                    desc[i].set_nsrc(nstand*(nstand+1)//2)
                    
                ticksPerFrame = navg
                tInt = navg/FS
                
                scale_factor = navg / (2*NCHAN)
                
                reset_sequence = True
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    for ispan in iseq_spans:
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        idata = ispan.data_view(np.int32).reshape(ishape+(2,))
                        t0 = time.time()
                        try:
                            odata.real[...] = idata[...,0]
                            odata.imag[...] = idata[...,1]
                        except NameError:
                            odata = idata[...,0] + 1j*idata[...,1]
                            odata = odata.astype(np.complex64)
                            odata = odata.view(np.complex64)
                        odata /= scale_factor
                        
                        bytesSent, bytesStart = 0, time.time()
                        
                        time_tag_cur = time_tag + 0*ticksPerFrame
                        npkt = nstand*(nstand+1)//2
                        i = 0
                        while npkt > 0:
                            nsend = min([128, npkt])
                            for j in range(self.nblock_send):
                                udt.send(desc[j], time_tag_cur, ticksPerFrame, i, 1, odata[j,[0],i:i+nsend,:])
                            i += nsend
                            npkt -= nsend
                            
                        time_tag += ticksPerFrame
                        
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                    'reserve_time': -1, 
                                                    'process_time': process_time,})
                          
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        try:
                            del odata
                        except NameError:
                            pass
                            
                        break
        del udt

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
    parser = argparse.ArgumentParser(description='LWA-SV ADP DRX Service')
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
    utc_start_tt = utc_start_dt.timestamp()
    log.info("UTC_START:    %s", utc_start_dt.strftime(DATE_FORMAT))
    
    hostname = socket.gethostname()
    try:
        server_idx = get_numeric_suffix(hostname) - 1
    except ValueError:
        server_idx = 0 # HACK to allow testing on head node "adp"
    log.info("Hostname:     %s", hostname)
    log.info("Server index: %i", server_idx)
    
    ## Network - input
    pipeline_idx = drxConfig['pipeline_idx']
    if config['host']['servers-data'][server_idx].startswith('adp'):
        iaddr    = config['server']['data_ifaces'][pipeline_idx]
    else:
        iaddr    = config['host']['servers-data'][server_idx]
    iport        = config['server']['data_ports' ][pipeline_idx]
    ## Network - TBF - data recorder
    recorder_idx = drxConfig['tbf_recorder_idx']
    recConfig    = config['recorder'][recorder_idx]
    oaddr        = recConfig['host']
    oport        = recConfig['port']
    obw          = recConfig['max_bytes_per_sec']
    ## Network - COR - data recorder
    recorder_idx = drxConfig['cor_recorder_idx']
    recConfig    = config['recorder'][recorder_idx]
    vaddr        = recConfig['host']
    vport        = recConfig['port']
    vbw          = recConfig['max_bytes_per_sec']
    ## Network - T engine
    tengine_ids  = drxConfig['tengine_idx']
    taddrs, tports = [], []
    for i,tengine_idx in enumerate(tengine_ids):
        tngConfig    = config['tengine'][tengine_idx]
        taddrs.append( config['host']['tengines'][tengine_idx] )
        tports.append( config['server']['data_ports' ][tngConfig['pipeline_idx']] + i + 1)
        
    nroach_tot = len(config['host']['roaches'])
    nserver    = len(config['host']['servers'])
    nroach, roach0 = nroach_tot, 0
    nbeam = drxConfig['beam_count']
    cores = drxConfig['cpus']
    gpus  = drxConfig['gpus']
    
    log.info("Src address:  %s:%i", iaddr, iport)
    log.info("TBF address:  %s:%i", oaddr, oport)
    log.info("COR address:  %s:%i", vaddr, vport)
    log.info("Roaches:      %i-%i", roach0+1, roach0+nroach)
    log.info("Tunings:      %i (of %i)", tuning+1, ntuning)
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    
    iaddr = Address(iaddr, iport)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 0.5
    
    capture_ring = Ring(name="capture-%i" % tuning, space='cuda_host')
    tbf_ring     = Ring(name="buffer-%i" % tuning)
    gpu_ring     = Ring(name="gpu-%i" % tuning, space='cuda')
    tengine_ring = Ring(name="tengine-%i" % tuning, space='cuda_host')
    vis_ring     = Ring(name="vis-%i" % tuning, space='cuda_host')
    
    tbf_buffer_secs = int(round(config['tbf']['buffer_time_sec']))
    
    oaddr = Address(oaddr, oport)
    osock = UDPSocket()
    osock.connect(oaddr)
    
    vaddr = Address(vaddr, vport)
    vsock = UDPSocket()
    vsock.connect(vaddr)
    
    tsocks = []
    for taddr,tport in zip(taddrs, tports):
        ctaddr = Address(taddr, tport)
        tsocks.append( UDPSocket() )
        tsocks[-1].connect(ctaddr)
        
    nchan_max = int(round(drxConfig['capture_bandwidth']/CHAN_BW/nserver))
    tbf_bw_max    = obw/nserver/ntuning
    cor_bw_max    = vbw/nserver/ntuning
    
    # TODO:  Figure out what to do with this resize
    GSIZE = 500
    ogulp_size = GSIZE *nchan_max*256*2
    obuf_size  = tbf_buffer_secs*25000 *nchan_max*256*2
    tbf_ring.resize(ogulp_size, obuf_size)
    
    ops.append(CaptureOp(log, fmt="chips", sock=isock, ring=capture_ring,
                         nsrc=nroach, src0=roach0, max_payload_size=9000,
                         buffer_ntime=GSIZE, slot_ntime=25000, core=cores.pop(0),
                         utc_start=utc_start_dt))
    ops.append(BufferCopyOp(log, capture_ring, tbf_ring,
                            tuning=tuning, ntime_gulp=GSIZE, #ntime_buf=25000*tbf_buffer_secs,
                            guarantee=False, core=cores.pop(0)))
    ops.append(TriggeredDumpOp(log=log, osock=osock, iring=tbf_ring, 
                               ntime_gulp=GSIZE, ntime_buf=int(25000*tbf_buffer_secs/2500)*2500,
                               tuning=tuning, nchan_max=nchan_max, 
                               core=cores.pop(0), 
                               max_bytes_per_sec=tbf_bw_max))
    ops.append(GPUCopyOp(log, capture_ring, gpu_ring,
                         ntime_gulp=GSIZE, core=cores[0], gpu=gpus[0]))
    ops.append(BeamformerOp(log=log, iring=gpu_ring, oring=tengine_ring,
                            tuning=tuning, ntime_gulp=GSIZE,
                            nroach=16, nchan_max=nchan_max, nbeam_max=nbeam,
                            core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(RetransmitOp(log=log, osocks=tsocks, iring=tengine_ring,
                            tuning=tuning, ntuning=ntuning, nchan_max=nchan_max,
                            ntime_gulp=GSIZE, nbeam_max=nbeam,
                            core=cores.pop(0)))
    ops[-2].updatePacketizerPreferences(ops[-1])
    if tuning == 0:
        ccore = ops[2].core
        try:
            pcore = cores.pop(0)
        except IndexError:
            pcore = ccore
        ops.append(CorrelatorOp(log=log, iring=gpu_ring, oring=vis_ring, 
                                tuning=tuning, ntime_gulp=GSIZE,
                                nroach=16, nchan_max=nchan_max,
                                utc_start_tt=utc_start_tt,
                                core=ccore, gpu=tuning % 2))
        ops.append(PacketizeOp(log=log, iring=vis_ring, osock=vsock,
                               tuning=tuning, nroach=16, nchan_max=nchan_max//4,
                               npkt_gulp=1, core=pcore, gpu=tuning % 2,
                               max_bytes_per_sec=cor_bw_max))
        ops[-2].updatePacketizerPreferences(ops[-1])
        
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
