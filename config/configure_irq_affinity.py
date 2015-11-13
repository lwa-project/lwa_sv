#!/usr/bin/env python

def to_int_list(vals):
	intvals = []
	for val in vals:
		try:
			intvals.append(int(val))
		except ValueError:
			pass
	return intvals

def read_irq_smp_affinity(irq):
	filename = "/proc/irq/%i/smp_affinity" % irq
	with open(filename, 'r') as f:
		contents = f.read()
	return int(contents, 16)
def write_irq_smp_affinity(irq, mask):
	mask_str = "%08x" % mask
	filename = "/proc/irq/%i/smp_affinity" % irq
	print "Writing %s to %s" % (mask_str, filename)
	with open(filename, 'w') as f:
		f.write(mask_str+"\n")

def main(argv):
	try:
		iface = sys.argv[1]
	except IndexError:
		iface = 'p5p1'
	interrupts_file = "/proc/interrupts"
	irqs = []
	with open(interrupts_file, 'r') as f:
		for line in f:
			cols = line.split()
			if cols[-1].startswith(iface):
				irq  = int(cols[0].replace(':',''))
				idx  = int(cols[-1].split('-')[1])
				mask = read_irq_smp_affinity(irq)
				#newmask = 1<<0                        # All on CPU 0 core 0 (single Hyperthread)
				#newmask = 1<<(0 + idx / 8 * 16)       # All on CPU 0 core 0 (divided b/w two Hyperthreads)
				#newmask = 1<<(0 + idx / 8 * 8)        # Split b/w 1st core of both CPUs
				#newmask = [0,6,7][idx%3]              # Split b/w 3 cores
				newmask = [6,7][idx%2]                 # Split b/w 2 cores on CPU 0
				#newmask = 1<<(idx % 8 + idx / 8 * 16) # Evenly distribute over 1st CPU
				counts = to_int_list(cols)
				peak_core = counts.index(max(counts))
				irqs.append((irq,idx,mask,newmask,peak_core))
	if len(irqs) == 0:
		raise KeyError("No IRQs found for interface %s" % iface)
	print "Current config for %s:" % iface
	print "Idx\tIRQ\tOld-mask\tNew-mask\tMost-used-core"
	for irq,idx,mask,newmask,core in irqs:
		print "%i\t%i\t%08x\t%08x\t%i" % (idx,irq,mask,newmask,core)
	for irq,idx,mask,newmask,core in irqs:
		write_irq_smp_affinity(irq, newmask)
	print "All done"
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
