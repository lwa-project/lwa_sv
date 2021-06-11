#!/usr/bin/env python

from __future__ import print_function, absolute_import

import valon_synth as valon

if __name__ == "__main__":
	import sys
	device = "/dev/ttyUSB0"
	if len(sys.argv) > 1:
		device = sys.argv[1]
	synth = valon.Synthesizer(device)
	#synth.set_label(valon.SYNTH_A, "Sampling clock") # Note: 16 char limit
	#synth.set_label(valon.SYNTH_B, "Tone injection") # Note: 16 char limit
	print("Old synth A freq:", synth.get_frequency(valon.SYNTH_A))
	synth.set_frequency(valon.SYNTH_A, 204.8, 0.8)
	print("New synth A freq:", synth.get_frequency(valon.SYNTH_A))
	synth.flash()

