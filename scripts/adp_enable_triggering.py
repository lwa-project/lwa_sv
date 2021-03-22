#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import getopt

from adp.AdpCommon import TRIGGERING_ACTIVE_FILE


def usage(exitCode=None):
	print("""adp_enable_triggering.py - Toggle the internal triggering state of ADP
	
Usage: adp_enable_triggering.py [OPTIONS]

Options:
-h, --help             Display this help information
-q, --query            Display the current state but do not change it
-e, --enable           Enabled internal triggering (Default = yes)
-d, --disable          Disable internal triggering
""")
	
	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseConfig(args):
	config = {}
	# Command line flags - default values
	config['query'] = False
	config['enable'] = True
	
	# Read in and process the command line flags
	try:
		opts, arg = getopt.getopt(args, "hqed", ["help", "query", "enable", "disable"])
	except getopt.GetoptError as err:
		# Print help information and exit:
		print(str(err)) # will print something like "option -a not recognized"
		usage(exitCode=2)
		
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-q', '--query'):
			config['query'] = True
		elif opt in ('-e', '--enable'):
			config['enable'] = True
		elif opt in ('-d', '--disable'):
			config['enable'] = False
		else:
			assert False
			
	# Return the configuration
	return config


def main(args):
	# Parse the command line
	config = parseConfig(args)
	
	# Always query the state, regardless of what we are asked to do
	if os.path.exists(TRIGGERING_ACTIVE_FILE):
		active = True
	else:
		active = False
		
	
	if config['query']:
		# Already done
		pass
		
	elif config['enable']:
		if not active:
			try:
				fh = open(TRIGGERING_ACTIVE_FILE, 'w')
				fh.write("%s" % time.time())
				fh.close()
				active = True
			except IOError:
				pass
				
	else:
		if active:
			try:
				os.unlink(TRIGGERING_ACTIVE_FILE)
				active = False
			except OSError:
				pass
				
	print("ADP Triggering: %s" % ('enabled' if active else 'disabled',))


if __name__ == '__main__':
	main(sys.argv[1:])
	
