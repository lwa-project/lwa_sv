#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import getopt

def usage(exitCode=None):
	print """pipeline_monitor.py - Monitor the packets capture/transmit status of a 
bifrost pipeline.

Usage: pipeline_monitor.py [OPTIONS] pid

Options:
-h, --help                  Display this help information
-c, --capture               Display information on packet capture (default)
-t, --transmit              Display information on packet transmit
"""
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['mode'] = 'capture'
	config['args'] = []
	
	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hct", ["help", "capture", "transmit"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-c', '--capture'):
			config['mode'] = 'capture'
		elif opt in ('-t', '--transmit'):
			config['mode'] = 'transmit'
		else:
			assert False
			
	# Add in arguments
	config['args'] = args
	
	# Validate
	if len(config['args']) != 1:
		raise RuntimeError("Must supply a PID to monitor")
		
	# Return configuration
	return config


def readFileContents(filename):
	fh = open(filename, 'r')
	data = fh.read().strip().rstrip()
	fh.close()
	
	return data


def main(args):
	config = parseOptions(args)
	pid = int(config['args'][0], 10)
	
	if config['mode'] == 'capture':
		baseDir = '/dev/shm/bifrost/%i' % pid
	elif config['mode'] == 'transmit':
		baseDir = '/dev/shm/bifrost_transmit/%i' % pid
	else:
		raise RuntimeError("Unknown reporting mode: %s" % config['mode'])
		
	if not os.path.exists(baseDir):
		raise RuntimeError("Cannot find bifrost directory: %s" % baseDir)
	if not os.path.isdir(baseDir):
		raise RuntimeError("Bifrost directory is not a directory: %s" % baseDir)
		
	typename = os.path.join(baseDir, 'type')
	sizename = os.path.join(baseDir, 'sizes')
	channame = os.path.join(baseDir, 'chans')
	statname = os.path.join(baseDir, 'stats')
	
	reportFiles = [typename,]
	if config['mode'] == 'capture':
		reportFiles.append( sizename )
	for filename in reportFiles:
		print readFileContents(filename)
		
	try:
		while True:
			t = time.time()
			stats = readFileContents(statname)
			if config['mode'] == 'capture':
				chans = readFileContents(channame)
				
				try:
					if chans != chansOld:
						print chans
				except NameError:
					print chans
					
				chansOld = chans
				
			_, stats = stats.split('=', 1)
			stats = [int(v, 10) for v in stats.split(',')]
			
			good = stats[0]
			invalid = stats[3]
			late = stats[4]
			try:
				missing = stats[1] / float(stats[0]+stats[1])
			except ZeroDivisionError:
				missing = 1.0
			try:
				rate = (stats[0] - statsOld[0]) / (t-tOld)
			except NameError:
				rate = 0.0
			print '%i at %.3f Gb/s with %.3f%% packet loss and %i invalid, %i late' % (good, rate*8/1024.0**3, 100.0*missing, invalid, late)
			
			statsOld = stats
			tOld = t
			
			time.sleep(1.0)
			
	except IOError:
		raise RuntimeError("Could not read pipeline status, has the pipeline exited?")
		
	except KeyboardInterrupt:
		pass


if __name__ == "__main__":
	main(sys.argv[1:])
	