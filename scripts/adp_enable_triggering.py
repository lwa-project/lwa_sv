#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import numpy
import getopt
import struct

import corr

from adp.AdpCommon import NCHAN, TRIGGERING_ACTIVE_FILE


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
		
	else:
		# Load in the base ADP configuration
		with open('/usr/local/share/adp/adp_config.json', 'r') as fh:
			adp_conf = json.load(fh)
			roach_config = adp_config['roach']
			roach_config['host'] = adp_config['host']['roaches']
			
		# Pull out the equalizer settings and coefficients
		scale_factor = roach_config['scale_factor']
		shift_factor = roach_config['shift_factor']
		equalizer_coeffs = roach_config['equalizer_coeffs']
		try:
			equalizer_coeffs = numpy.loadtxt(equalizer_coeffs)
		except Exception as e:
			print("WARNING: Failed to load default equalizer coefficients: %s" % str(e))
			equalizer_coeffs = numpy.ones(NCHAN, 'l')
			
		# Load in the special lightning mode coefficients
		try:
			lightning_coeffs = numpy.loadtxt('/home/adp/lwa_sv/config/equalizer_lightning.txt')
		except Exception as e:
			print("WARNING: Failed to load lightning mode equalizer coefficients: %s" % str(e))
			lightning_coeffs = numpy.ones(NCHAN, 'l')
			
		# Shift and scale both sets of coefficients
		equalizer_coeffs = np.ones(4096, 'l') * ((1<<shift_factor) - 1) * scale_factor * equalizer_coeffs
		lightning_coeffs = np.ones(4096, 'l') * ((1<<shift_factor) - 1) * scale_factor * lightning_coeffs
		
		# Connect to the ROACH2s
		roaches = []
		for roach in roach_config['host']:
			roaches.append(corr.katcp_wrapper.FpgaClient(roach, roach_config['port'], timeout=1.0)
			
		if config['enable'] and not active:
			## Set the new equalizer coefficients
			cstr = struct.pack('>4096l', *lightning_coeffs)
			for r,roach in enumerate(roaches):
				success = False
				for attempt in range(3):
					try:
						roach.write('fft_f1_cg_bpass_bram', cstr)
			            roach.write('fft_f2_cg_bpass_bram', cstr)
			            roach.write('fft_f3_cg_bpass_bram', cstr)
			            roach.write('fft_f4_cg_bpass_bram', cstr)
						success = True
						break
					except Exception as e:
						time.sleep(0.1)
				if not success:
					print("WARNING: Failed to update equalizer coefficients on '%s'" % roach_config['host'][r])
					
			## Set the file state
			try:
				with open(TRIGGERING_ACTIVE_FILE, 'w') as fh:
					fh.write("%s" % time.time())
				active = True
			except IOError:
				pass
				
		elif not config['enable'] and active:
			## Remove the triggering file
			try:
				os.unlink(TRIGGERING_ACTIVE_FILE)
				active = False
			except OSError:
				pass
				
			## Reset to the default equalizer coefficients
			cstr = struct.pack('>4096l', *equalizer_coeffs)
			for r,roach in enumerate(roaches):
				success = False
				for attempt in range(5):
					try:
						roach.write('fft_f1_cg_bpass_bram', cstr)
			            roach.write('fft_f2_cg_bpass_bram', cstr)
			            roach.write('fft_f3_cg_bpass_bram', cstr)
			            roach.write('fft_f4_cg_bpass_bram', cstr)
						success = True
						break
					except Exception as e:
						time.sleep(0.1)
				if not success:
					print("WARNING: Failed to update equalizer coefficients on '%s'" % roach_config['host'][r])
					
		# Close out the roach connections
		for roach in roaches:
			roach.close()
			
	print("ADP Triggering: %s" % ('enabled' if active else 'disabled',))


if __name__ == '__main__':
	main(sys.argv[1:])
