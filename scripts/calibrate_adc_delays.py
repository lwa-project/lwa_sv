#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import ephem
import numpy

from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/s').value

from lsl.reader import tbf, errors
from lsl.astro import MJD_OFFSET
from lsl.correlator._core import XEngine2


TONE_FREQ_HZ = 32.768e6


ARX_PATH_IN_CM  = [17, 15, 11,    8,  6,  3,  1,  1,  3,  5,  8, 11, 15, 17, 19, 21]
ARX_PATH_OUT_CM = [44, 42, 38.5, 36, 38, 35, 32, 30, 29, 26, 24, 22, 22, 19, 17, 14]
ARX_PATH_VF     = 0.5


def main(args):
    filenames = args
    
    for filename in filenames:
        fh = open(filename, 'rb')
        nFrames = os.path.getsize(filename) / tbf.FRAME_SIZE
        if nFrames < 3:
            fh.close()
            continue
            
        # Read in the first frame and get the date/time of the first sample 
        # of the frame.  This is needed to get the list of stands.
        junkFrame = tbf.read_frame(fh)
        fh.seek(0)
        beginJD = junkFrame.time.mjd
        beginDate = junkFrame.time.datetime
        
        # Figure out how many frames there are per observation and the number of
        # channels that are in the file
        nFramesPerObs = tbf.get_frames_per_obs(fh)
        nchannels = tbf.get_channel_count(fh)
        nSamples = 7840
        
        # Figure out how many chunks we need to work with
        nChunks = nFrames / nFramesPerObs
        
        # Pre-load the channel mapper
        mapper = []
        for i in xrange(2*nFramesPerObs):
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
        
        # Validate and skip over files that don't contain the tone
        if freq.min() > TONE_FREQ_HZ or freq.max() < TONE_FREQ_HZ:
            fh.close()
            continue
            
        # File summary
        print "Filename: %s" % filename
        print "Date of First Frame: %s" % str(beginDate)
        print "Frames per Observation: %i" % nFramesPerObs
        print "Channel Count: %i" % nchannels
        print "Frames: %i" % nFrames
        print "==="
        print "Chunks: %i" % nChunks
        print "==="
        
        data = numpy.zeros((256*2,nchannels,nChunks), dtype=numpy.complex64)
        clipping = 0
        for i in xrange(nChunks):
            # Inner loop that actually reads the frames into the data array
            for j in xrange(nFramesPerObs):
                # Read in the next frame and anticipate any problems that could occur
                try:
                    cFrame = tbf.read_frame(fh)
                except errors.EOFError:
                    break
                except errors.SyncError:
                    print "WARNING: Mark 5C sync error on frame #%i" % (int(fh.tell())/tbf.FRAME_SIZE-1)
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
                subData = cFrame.payload.data
                subData.shape = (12,512)
                subData = subData.T
                
                clipping += len( numpy.where( subData**2 >= 98 )[0] )
                
                data[:,aStand*12:aStand*12+12,count] = subData
                
        fh.close()
        
        # Cross-correlate the data (although we only use a small fraction of this)
        valid = numpy.ones((data.shape[0],data.shape[2]), dtype=numpy.uint8)
        data[numpy.where(numpy.abs(data)==0)] = 1+1j
        bls = [(i,j) for i in xrange(512) for j in xrange(i,512)]
        cross = XEngine2(data, data, valid, valid)
        
        # Find the peak power, aka the tone, and report on clipping
        spec = numpy.abs(cross[cross.shape[0]/2,:])**2
        peak = numpy.where(spec == spec.max())[0][0]
        print "Peak Power: %.3f MHz (channel %i)" % (freq[peak]/1e6, peak)
        print "Clipping: %i samples (%.1f%%)" % (clipping, 100.0*clipping/data.size)
        print "==="
        
        # Solve
        ds = {}
        for i,(a0,a1) in enumerate(bls):
            ## Only work with the first 512 inputs, i.e., baselines that are 
            ## with the first input on the first roach board
            if i >= 512:
                continue
                
            ## Input -> ??? mappings
            r0, r1 = a0/32, a1/32        # input -> roach
            c0, c1 = a0%16, a1%16        # input -> ARX channel
            s0, s1 = a0/ 2, a1 /2        # input -> stand
            l0, l1 = a0/ 4, a1/ 4        # input -> ADC lane
            i0, i1 = a0% 4, a1% 4        # input -> ADC lane input
            if r0 != 0:
                continue
                
            ## Differential ARX calibration path delay correction
            cd0 = (ARX_PATH_IN_CM[c0] + ARX_PATH_OUT_CM[c0]) / 100.0 / speedOfLight / ARX_PATH_VF
            cd1 = (ARX_PATH_IN_CM[c1] + ARX_PATH_OUT_CM[c1]) / 100.0 / speedOfLight / ARX_PATH_VF
            rcd = cd0 - cd1
            
            ## Compute the delay and convert to samples @ 204.8 MHz
            a = numpy.angle(cross[i,peak])
            d = a / 2.0 / numpy.pi / TONE_FREQ_HZ + rcd
            d /= (1/204.8e6)
            
            ## Save
            try:
                ds[s1].append( d )
            except KeyError:
                ds[s1] = [d,]
                
        # Report on each measurement group
        nMeasuresPerDelay = 512 / len(ds.keys())
        j = 0
        for i in sorted(ds.keys()):
            try:
                d = numpy.array(ds[i], dtype=numpy.float32)
                print "# %s +/- %s, %s" % (d.mean(), d.std(), numpy.median(d))
                d = numpy.round(d.mean())
            except KeyError:
                d = 0.0
            for i in xrange(nMeasuresPerDelay):
                print 'roach%i input%i  %i' % (j/32+1, j%32+1, d)
                j += 1
                
        break


if __name__ == "__main__":
    main(sys.argv[1:])
    
