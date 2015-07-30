#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def pad(x, n):
	return np.resize(x, max(n, x.size))
def plot_time(xy, y=None, **kwargs):
	if y is None:
		y = xy
		x = np.arange(y.size)
	else:
		x = xy
	plt.vlines(x, 0, y)
	plt.plot(x, y, 'o', **kwargs)
def plot_freq(y, symmetric=False, **kwargs):
	if symmetric:
		x = np.fft.rfftfreq(2*y.size-1)
	else:
		x = np.fft.fftfreq(y.size)
	plt.vlines(x, 0, y)
	plt.plot(x, y, 'o', **kwargs)

def test2():
	nfir = 28   # Coefficients
	novr = 16   # Oversampling
	nfft = 128  #1024#8192 # FFT length (real)
	t  = np.linspace(0, nfir, nfir)
	ot = np.linspace(0, nfir, nfir*novr)
	#c  = np.sinc(0.3*t)
	c = np.zeros_like(t)
	# Zero shift
	#c[c.size/2-1] = 1
	# Shift by -13 (i.e., i_new = i_old - 13; aka delay -13)
	#   Last 13 output samples must be discarded
	#c[0] = 1
	# Shift by +14 (i.e., i_new = i_old + 14; aka delay +14)
	#   First 14 output samples must be discarded
	c[c.size-1] = 1
	oc = np.sinc(0.3*ot)
	inp = np.random.normal(size=nfft)
	#out = fftconvolve(inp, c, mode='full')
	out = fftconvolve(inp, c, mode='same')
	#plot_time(t, out)
	#plot_time(inp)
	#plot_time(out)
	plt.plot(inp, label="Input")
	plt.plot(out, label="Convolved")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	import sys
	test2()
	sys.exit(0)
	
	nfir = 28   # Coefficients
	novr = 16   # Oversampling
	nfft = 1024#8192 # FFT length (real)
	#t  = np.linspace(-(nfir)/2-1, (nfir)/2+1, nfir)
	#ot = np.linspace(-(nfir)/2-1, (nfir)/2+1, nfir*novr)
	t = np.linspace(0, nfir, nfir)
	ot = np.linspace(0, nfir, nfir*novr)
	#c = np.sin(t) # Filter
	#oc = np.sin(ot)
	c = np.sinc(0.6*t)
	oc = np.sinc(0.6*ot)
	plot_time(t, c)
	plot_time(ot, oc)
	plt.show()
	
	print c.size
	#plot_time(np.fft.rfft(c, nfft))
	plot_freq(np.abs(np.fft.rfft(c, nfft)), True)
	plt.show()
	
	#pc = pad(c, nfft)
	#print pc.size
	#pf = np.fft.rfft(pc)
	pf = np.fft.rfft(c, nfft)
	print pf.size
	f = pf[:nfft/2]
	print f.size
	plot_freq(np.abs(f), True, label="Critical")
	
	print
	print oc.size
	#opc = pad(oc, novr*nfft)
	#print opc.size
	#opf = np.fft.rfft(opc) / novr
	opf = np.fft.rfft(oc, novr*nfft) / novr
	print opf.size
	of = opf[:nfft/2]
	print of.size
	plot_freq(np.abs(of), True, label="Oversampled")
	plt.legend()
	plt.show()
