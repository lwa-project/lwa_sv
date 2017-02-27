#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a collection of TBF files, process them into a NumPy file that is 
compatible with the PASI TBW processing.
"""

import os
import sys
import struct
import numpy as np
from datetime import datetime

from lsl.reader import tbf


def main(args):
	filenames = args
	
	superFreq, superData = [], []
	for filename in filenames:
		fh = open(filename, 'rb')
		nFrames = os.path.getsize(filename) / tbf.FrameSize
		
		# Read in the first frame and get the date/time of the first sample 
		# of the frame.  This is needed to get the list of stands.
		junkFrame = tbf.readFrame(fh)
		fh.seek(0)
		try:
			beginDate
		except NameError:
			beginDate = datetime.utcfromtimestamp(junkFrame.getTime())
			
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
		freq = np.zeros(nChannels)
		for i,c in enumerate(mapper):
			freq[i*12:i*12+12] = c + np.arange(12)
		freq *= 25e3
		
		# File summary
		print "Filename: %s" % filename
		print "Date of First Frame: %s" % str(beginDate)
		print "Frames per Observation: %i" % nFramesPerObs
		print "Channel Count: %i" % nChannels
		print "Frames: %i" % nFrames
		print "==="
		print "Chunks: %i" % nChunks
		
		data = np.zeros((nChannels,256,2))
		norm = np.zeros_like(data)
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
				count = cFrame.header.frameCount - 1
				#print cFrame.data.fDomain[:,7,0]
				try:
					data[aStand*12:aStand*12+12,:,:] += np.abs(cFrame.data.fDomain)**2
					norm[aStand*12:aStand*12+12,:,:] += 1
				except ValueError:
					pass
		data /= norm
		fh.close()
		
		superFreq.append( freq )
		superData.append( data )
	
	superFreq = np.array(superFreq)
	superData = np.array(superData)
	
	superFreq = superFreq.ravel()
	superData.shape = (superFreq.size, superData.shape[2], superData.shape[3])
	
	freqs = np.arange(4096) * 25e3
	_, u = np.unique(superFreq, return_index=True)
	specs = np.zeros((superData.shape[1], superData.shape[2], freqs.size), dtype=np.float32)
	for i in u:
		best = np.argmin( np.abs(freqs - superFreq[i]) )
		specs[:,:,best] = superData[i,:,:]
		
	outname = beginDate.strftime('%y%m%d_%H%M%S_tbfspecs.dat')
	fh = open(outname, 'wb')
	fh.write( struct.pack('ll', specs.shape[0], specs.shape[2]) )
	specs.tofile(fh)
	fh.close()


if __name__ == "__main__":
	main(sys.argv[1:])
	