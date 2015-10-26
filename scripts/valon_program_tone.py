#!/usr/bin/env python

import valon_synth as valon

if __name__ == "__main__":
	import sys
	device = "/dev/ttyUSB0"
	synth = valon.Synthesizer(device)
	if len(sys.argv) <= 1:
		print "Usage: Specify freq in MHz to enable, or nothing to disable"
		print "Disabling SYNTH B RF output"
		synth.set_rf_output_enabled(valon.SYNTH_B, False)
		synth.set_rf_level(         valon.SYNTH_B,    -4)
	else:
		freq_mhz = float(sys.argv[1])
		freq_mhz *= 2  # Correct for Nyquist sampling
		freq_mhz *= 24 # Correct for clock divider in injection path
		print "Enabling SYNTH B RF output @", freq_mhz, "MHz"
		synth.set_frequency(valon.SYNTH_B, freq_mhz, 0.001)
		print "  New frequency:", synth.get_frequency(valon.SYNTH_B), "MHz"
		synth.set_rf_output_enabled(valon.SYNTH_B, True)
		synth.set_rf_level(         valon.SYNTH_B,   +5)
	synth.flash()
	print "Done"
