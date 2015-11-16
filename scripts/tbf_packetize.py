#!/usr/bin/env python

"""
Converts raw TBF-like data dumps (as recorded during commissioning)
  into proper TBF packet format and writes them to a file.
  This is intended for use during commissioning only, as the completed
    ADP pipeline will perform the conversion itself.
"""

from adp.AdpCommon import *

import sys
import os
import datetime
import struct
import math
import numpy as np
from cStringIO import StringIO

# TODO: Move into AdpCommon?
ADP_EPOCH  = datetime.datetime(1970, 1, 1)
M5C_EPOCH  = datetime.datetime(1990, 1, 1)
#M5C_OFFSET = int((M5C_EPOCH - ADP_EPOCH).total_seconds())
M5C_OFFSET = 0 # LWA convention re-defines this to use the 1970 epoch too

NFRAME_PER_SPECTRUM = int(FS) // int(CHAN_BW) # 7840
NFRAME_PER_SUBSLOT  = int(FS) // NSUBSLOT_PER_SEC

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

def gen_cor_header(chan0, navg_subslots, gain, i, j,
                   time_tag, time_tag0):
	sync_word    = 0xDEC0DE5C
	idval        = 0x02
	frame_num_wrap = 10*60 * int(CHAN_BW) # 10 mins = 15e6, just fits within a uint24
	nframe_per_integration = NFRAME_PER_SUBSLOT * navg_subslots
	frame_num    = ((time_tag - time_tag0) // nframe_per_integration) + 1 # Integration no.
	id_frame_num = idval << 24 | (frame_num & ((1<<24)-1))
	secs_count   = time_tag // int(FS) - M5C_OFFSET
	freq_chan    = chan0
	cor_gain     = gain
	cor_navg     = navg_subslots
	stand_i      = i+1
	stand_j      = j+1
	return struct.pack('>IIIhhqihh',
	                   sync_word,
	                   id_frame_num,
	                   secs_count,
	                   freq_chan,
	                   cor_gain,
	                   time_tag,
	                   cor_navg,
	                   stand_i,
	                   stand_j)

if __name__ == "__main__":
	nchan_pkt = 12
	infilepath = sys.argv[1]
	
	inpath, infilename = os.path.split(infilepath)
	filetag            = os.path.splitext(infilename)[0]
	outfilepath        = os.path.join(inpath, filetag+".pkt")
	filetag = filetag.split('_', 1)[1] # Remove 'adp_' prefix
	utc_start_str, chan0, nchan = filetag.rsplit('_', 2)
	chan0 = int(chan0)
	nchan = int(nchan)
	print utc_start_str, chan0, nchan
	utc_start = datetime.datetime.strptime(utc_start_str, "%Y_%m_%dT%H_%M_%S")
	timestamp = int((utc_start - ADP_EPOCH).total_seconds())
	assert( datetime.datetime.utcfromtimestamp(timestamp) == utc_start )
	time_tag0 = timestamp * int(FS)
	
	frame_shape = (NBOARD,nchan,NSTAND/NBOARD,NPOL)
	#frame_shape = (nchan,NSTAND,NPOL)
	frame_size  =  nchan*NSTAND*NPOL
	ntime = os.path.getsize(infilepath) / frame_size
	print "Observation length:", ntime / CHAN_BW, "s"
	print "Writing output to", outfilepath
	
	with open( infilepath, 'r') as infile, \
	     open(outfilepath, 'w') as outfile:
		for t in xrange(ntime):
			if t % int(CHAN_BW) == 0:
				print "Processing sec", (t // int(CHAN_BW)) + 1, "/", int(math.ceil(ntime/CHAN_BW))
			time_tag = time_tag0 + t*NFRAME_PER_SPECTRUM
			data = np.fromfile(infile, count=frame_size, dtype=np.uint8)
			if data.size < frame_size:
				break
			data = data.reshape(frame_shape)
			data = data.transpose([1,0,2,3])
			data = data.reshape((nchan,NSTAND,NPOL))
			data = np.roll(data, -81, axis=0) # TODO: Why is this needed?!
			outdata = StringIO()
			for pkt in xrange(nchan//nchan_pkt):
				chan    = pkt*nchan_pkt
				payload = data[chan:chan+nchan_pkt]
				header  = gen_tbf_header(chan0+chan, time_tag, time_tag0)
				#outfile.write(header)
				#payload.tofile(outfile)
				outdata.write(header)
				outdata.write(payload.tostring())
			outfile.write(outdata.getvalue())
			outdata.close()
	print "All done"
