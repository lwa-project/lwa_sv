#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------#
#   First pass at diagnosing antenna health via a TBW capture                                   #
#                                                                                               #
#   Created:  May, 2012  by Sean Cutchin                                                        #
#                                                                                               #
#   Last Modified:                                                                              #
#      October 15, 2013, Jayce Dowell                                                           #
#        * Reordered part of the loops to decrease the runtime                                  #
#                                                                                               #
#      October 4, 2012, Sean Cutchin                                                            #
#        * Changed 3dB criteria to 6 dB.                                                        #
#        * Changed format of output table.  Also added columns for the percentage of channels   #
#          flagged in both Steps 1 and 2.                                                       #
#        * Added MastDir and DataDir to make file grabbing easier                               #
#        * The filename is now given on the command line                                        #
#        * Output is a tab-delimited ASCII text file that lists:                                #
#          DP Board #, Stand #, Pol., Status                                                    #
#          Status codes = 'G' good, 'B' bad, 'S' suspect                                        #
#-----------------------------------------------------------------------------------------------#
import numpy as np
from lsl.common import stations
import math
import time
import sys
import struct

class OutputSource():

      dpboard = None	# DP Board 
      stand   = None	# Stand number
      pol     = None	# Antenna Polarization
      status  = None	# Status code [Good (G), Bad (B), Suspect (S)]
      metric1  = None    # percentage of bad channels in step 1.  How a stand is classified as bad
      metric2  = None    # percentage of bad channels in step 2.  How a stand is classified as suspect


      formatter = "   {0.dpboard:02d}      {0.stand:03d}      {0.pol:01d}          {0.status:1s}     {0.metric1:0.4f}    {0.metric2:0.4f}\n " 

      def __str__(self):
          return self.formatter.format(self)


if len(sys.argv) != 3:
    sys.stderr.write('Usage: AntennaHealthTest.py INFILE OUTFILE')
    sys.exit(2)
inFileName = sys.argv[1]
outFileName = sys.argv[2]


# Read the spectra from the TBW data file specified on the command line.
with open(inFileName, 'r') as f:
    nStands, nChans = struct.unpack('ll', f.read(16))
    spec = np.fromfile(f, count = nStands * 2 * nChans, dtype = np.float32
                       ).reshape(nStands * 2, nChans)

freq = np.arange(nChans) * 25e3
# Apply cable loss corrections
lwa1 = stations.parseSSMIF('/home/adp/tbfspecs/SSMIF_CURRENT.txt')
antennas = lwa1.getAntennas()
for i in xrange(spec.shape[0]):
    Cable_gain = antennas[i].cable.gain(freq)
    spec[i] /= Cable_gain


medSpec = np.median(spec, axis=0)          # Take the median; this is the composite spectrum

                                           # Note that both spec and medSpec are in linear units
                                           # need to convert to dB if you want to replicate spectra plots
                                           # as seen in smGUI.py

freq = freq/10**6                               # Converting frequencies to MHz
spec = 10.*np.log10(spec)                       # convert to dB
medSpec = 10.*np.log10(medSpec)                 # convert to dB

#-----------------------------------------------------------------------------------------------#
#                               Now, let's do some work.                                        #
#  Array definitions:                                                                           #
#   flag:   This has the shape of spec.  It holds either a '0' or '1' at each frequency, for    #
#           each stand.                                                                         #
#  sflag:   1d array of length=number of stands.  This also holds either '0' or '1' for each    #
#           stand.  A stand is good if sflag =0, bad if sflag=1                                 #
#  mean, std:  These are the arrays for the mean and standard deviation in each channel.        #   
#              they exist purely for testing purposes, and in the end will not be arrays.       #
#  fflag:  2d array of shape flag.  Same as flag, except for starting step 2.  In the end, this #
#          will not exist either, it is only here for testing purposes.                         # 
#-----------------------------------------------------------------------------------------------#
fbegin,fend = 1462,3552
#fbegin,fend = 1000,3500
freq = freq[fbegin:fend]
spec = spec[:,fbegin:fend]
medSpec = medSpec[fbegin:fend]

tstart=time.clock()
status = ['G'] * len(spec)

badchan1 = np.zeros(len(spec))
badchan2 = np.zeros(len(spec))
flag = np.zeros(spec.shape)
for i in xrange(len(spec)):    
    toFlag = np.where( np.abs(spec[i,:] - medSpec) > 6 )[0]             # Flag the channel for that stand if it is more than 6 dB off the median
    flag[i,toFlag] = 1

metric = flag.sum(axis=1)/len(freq)                                     # Sum up the 1's and 0's
sflag=np.zeros(len(spec))                                               # initialize the array
for i in range(len(spec)):
    badchan1[i] = metric[i]
    if metric[i]>0.3333:                                                # if the sum for a stand is greater than 0.3333 (~a third of the band)
       #print i,antennas[i].stand.id, metric[i]                         # then flag the stand as bad.  The 0.3333 can be changed.
       sflag[i] = 1
       status[i] = 'B'

mean = np.zeros(len(freq))                                             
std = np.zeros(len(freq))
flag = np.zeros(spec.shape)
toUse = np.where( sflag == 0 )[0]

for i in xrange(len(freq)):
    n, meansum, stdsum = 0, 0, 0.0
    mean[i] = (spec[toUse,i] - medSpec[i]).mean()                                    # calculate the mean and standard deviation leaving out
    stdsum  = ((spec[toUse,i] - medSpec[i] - mean[i])**2).sum(dtype=np.float64)      # the flagged stands.  
    std[i]  = math.sqrt(stdsum / float(len(toUse)-1))

for i in xrange(len(spec)):
    toFlag = np.where( np.abs((spec[i,:]-medSpec-mean)/std) > 1.75 )[0]              # if the flattened spectrum deviates by more than
    flag[i,toFlag] = 1                                                               # 1.75 sigma, flag the channel. 1.75 is tunable.

#plt.plot(freq,spec[284]-medSpec-mean)
#plt.errorbar(freq,mean,yerr=1.75*std)
#plt.savefig('flag',format='png')
del(mean,std)                                                           # We no longer need these arrays
metric = flag.sum(axis=1)/len(freq)      
for i in range(len(sflag)):
    badchan2[i] = metric[i]
    if sflag[i]==0.0:
       if metric[i] >0.3333:                                            # If the number of flagged chanels is greater than 
          sflag[i]=1                                                    # 0.3333 (~a third of the band), flag the stand as
          status[i] = 'S'                                               # suspect
       #print i, antennas[i].stand.id

#bflag = np.zeros(29)
#for i in range(len(sflag)):
#    if sflag[i]==1:
#       bflag[antennas[i].board]+=1


# This will write the output in a tab-delimited ASCII file for the operator
header = """# This is TBW Antenna Health Test version {0}
# on File {1}
#
# DPBoard Stand Polarization Status  Metric1  Metric2
#====================================================="""
#outfile = open("test_output.txt",'w')
outfile = open(outFileName,'w')
print >> outfile,header.format(3,inFileName)
for i in range(len(spec)):
    ant = OutputSource()
    ant.dpboard = antennas[i].board
    ant.stand   = antennas[i].stand.id
    ant.pol     = antennas[i].pol
    ant.status   = status[i]
    ant.metric1  = badchan1[i]
    ant.metric2  = badchan2[i]
    outfile.write(ant.formatter.format(ant)[:-1])


outfile.close()
#for i in range(len(spec)):
#    head.append((str(antennas[i].board),str(antennas[i].stand.id),str(antennas[i].pol),status[i]))


#with open(outFileName, "w") as fp:
#    fp.writelines('%s\n' % '\t'.join(items) for items in head)
#
tend=time.clock()
elapsedtime=tend-tstart
print "Elapsed time  =", elapsedtime

