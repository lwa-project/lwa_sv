#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import ephem
import numpy

from lsl.reader import tbf, errors
from lsl.correlator._core import XEngine2
from lsl.astro import unix_to_utcjd, DJD_OFFSET


def main(args):
	filenames = args
	
	for filename in filenames:
		fh = open(filename, 'rb')
		nFrames = os.path.getsize(filename) / tbf.FrameSize
		
		# Read in the first frame and get the date/time of the first sample 
		# of the frame.  This is needed to get the list of stands.
		junkFrame = tbf.readFrame(fh)
		fh.seek(0)
		beginJD = unix_to_utcjd(junkFrame.getTime())
		beginDate = ephem.Date(unix_to_utcjd(junkFrame.getTime()) - DJD_OFFSET)
		
		# Figure out how many frames there are per observation and the number of
		# channels that are in the file
		nFramesPerObs = tbf.getFramesPerObs(fh)
		nChannels = tbf.getChannelCount(fh)
		nSamples = 7840
		
		# Figure out how many chunks we need to work with
		nChunks = nFrames / nFramesPerObs
		
		# Pre-load the channel mapper
		mapper = []
		for i in xrange(2*nFramesPerObs):
			cFrame = tbf.readFrame(fh)
			if cFrame.header.firstChan not in mapper:
				mapper.append( cFrame.header.firstChan )
		fh.seek(-2*nFramesPerObs*tbf.FrameSize, 1)
		mapper.sort()
		
		# Calculate the frequencies
		freq = numpy.zeros(nChannels)
		for i,c in enumerate(mapper):
			freq[i*12:i*12+12] = c + numpy.arange(12)
		freq *= 25e3
		
		# Validate and skip over files that don't contain the tone
		if freq.min() > 60e6 or freq.max() < 60e6:
			fh.close()
			continue
			
		# File summary
		print "Filename: %s" % filename
		print "Date of First Frame: %s" % str(beginDate)
		print "Frames per Observation: %i" % nFramesPerObs
		print "Channel Count: %i" % nChannels
		print "Frames: %i" % nFrames
		print "==="
		print "Chunks: %i" % nChunks
		print "==="
		
		data = numpy.zeros((256*2,nChannels,nChunks), dtype=numpy.complex64)
		for i in xrange(nChunks):
			# Inner loop that actually reads the frames into the data array
			for j in xrange(nFramesPerObs):
				# Read in the next frame and anticipate any problems that could occur
				try:
					cFrame = tbf.readFrame(fh)
				except errors.eofError:
					break
				except errors.syncError:
					print "WARNING: Mark 5C sync error on frame #%i" % (int(fh.tell())/tbf.FrameSize-1)
					continue
				if not cFrame.header.isTBF():
					continue
					
				firstChan = cFrame.header.firstChan
				
				# Figure out where to map the channel sequence to
				try:
					aStand = mapper.index(firstChan)
				except ValueError:
					mapper.append(firstChan)
					aStand = mapper.index(firstChan)
				
				# Actually load the data.
				if i == 0 and j == 0:
					refCount = cFrame.header.frameCount
				count = cFrame.header.frameCount - refCount
				subData = cFrame.data.fDomain
				subData.shape = (12,512)
				subData = subData.T
				
				data[:,aStand*12:aStand*12+12,count] = subData
		valid = numpy.ones((data.shape[0],data.shape[2]), dtype=numpy.uint8)
		data[numpy.where(numpy.abs(data)==0)] = 1+1j
		bls = [(i,j) for i in xrange(512) for j in xrange(i,512)]
		cross = XEngine2(data, data, valid, valid)
		
		spec = numpy.abs(cross[cross.shape[0]/2,:])**2
		peak = numpy.where(spec == spec.max())[0][0]
		print "Peak Power: %.3f MHz (channel %i)" % (freq[peak]/1e6, peak)
		print "==="
		
		ds = {}
		for i,(a0,a1) in enumerate(bls):
			r0, r1 = a0/32, a1/32
			i0, i1 = a0%32, a1%32
			if r0 != 0:
				continue
			if a0 == a1:
				continue
			if (a0 % 32) != (a1 % 32):
				continue
				
			#print a0,r0,i0, a1,r1,i1
			a = numpy.angle(cross[i,peak])
			d = a / 2.0 / numpy.pi / 60e6
			d /= (1/204.8e6)
			try:
				ds[r1].append( d )
			except KeyError:
				ds[r1] = [d,]
				
		for r in xrange(16):
			try:
				d = numpy.array(ds[r], dtype=numpy.float32)
				print "# %s +/- %s, %s" % (d.mean(), d.std(), numpy.median(d))
				d = numpy.round(d.mean())
			except KeyError:
				d = 0.0
			print "roach%i %i" % (r, d)
			
		fh.close()
		break


if __name__ == "__main__":
	main(sys.argv[1:])
	