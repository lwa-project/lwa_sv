#!/usr/bin/env python

import valon_synth as valon

if __name__ == "__main__":
	import sys
	device = "/dev/ttyUSB0"
	if len(sys.argv) > 1:
		device = sys.argv[1]
	synth = valon.Synthesizer(device)
	#synth.set_label(valon.SYNTH_A, "Sampling clock") # Note: 16 char limit
	#synth.set_label(valon.SYNTH_B, "Tone injection") # Note: 16 char limit
	print "Old synth A freq:", synth.get_frequency(valon.SYNTH_A)
	synth.set_frequency(valon.SYNTH_A, 196.608, 0.008)
	print "New synth A freq:", synth.get_frequency(valon.SYNTH_A)
	synth.set_rf_level(valon.SYNTH_A, 5)
	synth.flash()

