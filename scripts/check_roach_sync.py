#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
if sys.version_info < (3,):
    range = xrange
    
import os
import sys
import ephem
import numpy

from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/s').value

from lsl.reader import tbf, errors
from lsl.astro import MJD_OFFSET
from lsl.correlator._core import XEngine2


SKY_FREQ_HZ = 38.0e6

def main(args):
    filenames = args
    
    for filename in filenames:
        fh = open(filename, 'rb')
        nFrames = os.path.getsize(filename) // tbf.FRAME_SIZE
        if nFrames < 3:
            fh.close()
            continue
            
        # Read in the first frame and get the date/time of the first sample 
        # of the frame.  This is needed to get the list of stands.
        junkFrame = tbf.read_frame(fh)
        fh.seek(0)
        beginJD = junkFrame.time.mjd + MJD_OFFSET
        beginDate = junkFrame.time.datetime
        
        # Figure out how many frames there are per observation and the number of
        # channels that are in the file
        nFramesPerObs = tbf.get_frames_per_obs(fh)
        nchannels = tbf.get_channel_count(fh)
        nSamples = 7840
        
        # Figure out how many chunks we need to work with
        nChunks = nFrames // nFramesPerObs
        
        # Pre-load the channel mapper
        mapper = []
        for i in range(2*nFramesPerObs):
            cFrame = tbf.read_frame(fh)
            if cFrame.header.first_chan not in mapper:
                mapper.append( cFrame.header.first_chan )
        fh.seek(-2*nFramesPerObs*tbf.FRAME_SIZE, 1)
        mapper.sort()
        
        # Calculate the frequencies
        freq = numpy.zeros(nchannels)
        for i,c in enumerate(mapper):
            freq[i*12:i*12+12] = c + numpy.arange(12)
        freq *= 25e3
        
        # Validate and skip over files that don't contain the sky frequency
        # we are interested in looking at
        if freq.min() > SKY_FREQ_HZ or freq.max() < SKY_FREQ_HZ:
            fh.close()
            continue
            
        # File summary
        print("Filename: %s" % filename)
        print("Date of First Frame: %s" % str(beginDate))
        print("Frames per Observation: %i" % nFramesPerObs)
        print("Channel Count: %i" % nchannels)
        print("Frames: %i" % nFrames)
        print("===")
        print("Chunks: %i" % nChunks)
        print("===")
        
        data = numpy.zeros((256*2,nChunks,nchannels), dtype=numpy.complex64)
        clipping = 0
        for i in range(nChunks):
            # Inner loop that actually reads the frames into the data array
            for j in range(nFramesPerObs):
                # Read in the next frame and anticipate any problems that could occur
                try:
                    cFrame = tbf.read_frame(fh)
                except errors.EOFError:
                    break
                except errors.SyncError:
                    print("WARNING: Mark 5C sync error on frame #%i" % (int(fh.tell())//tbf.FRAME_SIZE-1))
                    continue
                if not cFrame.header.is_tbf:
                    continue
                    
                first_chan = cFrame.header.first_chan
                
                # Figure out where to map the channel sequence to
                try:
                    aStand = mapper.index(first_chan)
                except ValueError:
                    mapper.append(first_chan)
                    aStand = mapper.index(first_chan)
                
                # Actually load the data.
                if i == 0 and j == 0:
                    refCount = cFrame.header.frame_count
                count = cFrame.header.frame_count - refCount
                if count < 0:
                    continue
                subData = cFrame.payload.data
                subData.shape = (12,512)
                subData = subData.T
                
                clipping += len( numpy.where( subData**2 >= 98 )[0] )
                
                data[:,count,aStand*12:aStand*12+12] = subData
                
        fh.close()
        
        # Report on clipping
        print("Clipping: %i samples (%.1f%%)" % (clipping, 100.0*clipping/data.size))
        print("===")
        
        # Make a time domain data set out of these
        tdd = numpy.fft.ifft(data, axis=2)
        tdd = numpy.reshape(tdd, (tdd.shape[0], -1))
        
        # Correlate
        refX = numpy.fft.fft(tdd[0,:]).conj()
        refY = numpy.fft.fft(tdd[1,:]).conj()
        cc  = numpy.abs( numpy.fft.ifft( numpy.fft.fft(tdd[0::2,:], axis=1) * refX ) )**2
        cc += numpy.abs( numpy.fft.ifft( numpy.fft.fft(tdd[1::2,:], axis=1) * refY ) )**2
        cc = numpy.fft.fftshift(cc, axes=1)
        ccF = (numpy.arange(cc.shape[1]) - cc.shape[1]/2) / (nchannels*25e3) * 1e6
        
        valid = numpy.where( numpy.abs(ccF) < 150 )[0]
        ccF = ccF[valid]
        for i in range(16):
            subCC = cc[i*16:(i+1)*16,valid].sum(axis=0)
            #print(i, subCC)
            
            peak = numpy.argmax(subCC)
            print('roach%i  %i' % (i+1, int(round(ccF[peak]/40.0))*40))


if __name__ == "__main__":
    main(sys.argv[1:])
    
