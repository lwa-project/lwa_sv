#!/usr/bin/env python

import valon_synth as valon

def print_synth(synth, which):
	print "Synth A" if which == valon.SYNTH_A else "Synth B"
	print "  Label:       ", synth.get_label(     which)
	print "  Freq:        ", synth.get_frequency( which), "MHz"
	print "  Phase locked:", synth.get_phase_lock(which)
	print "  RF level:    ", synth.get_rf_level(  which), "dBm"
	print "  VCO range:   ", synth.get_vco_range( which), "MHz"
	options = synth.get_options(which)
	doubled = options[0]
	halved  = options[1]
	print "  Scale:       ", ("unity"   if doubled==halved else
	                          "doubled" if doubled else
	                          "halved")
	print "  Divisor:     ", options[2]
	print "  Mode:        ", ("minimize PLL spurs" if options[3] else
	                          "minimize phase noise")

if __name__ == "__main__":
	import sys
	device = "/dev/ttyUSB0"
	if len(sys.argv) > 1:
		device = sys.argv[1]
	synth = valon.Synthesizer(device)
	ref_names = {valon.INT_REF: "internal",
	             valon.EXT_REF: "external"}
	print "Ref source:", ref_names[synth.get_ref_select()]
	print "Ref freq:  ", synth.get_reference()/1e6, "MHz"
	print_synth(synth, valon.SYNTH_A)
	print_synth(synth, valon.SYNTH_B)
