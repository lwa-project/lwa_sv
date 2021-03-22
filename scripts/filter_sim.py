#!/usr/bin/env python

"""
** TODO: Simulate and compare these:
           Convolve time-domain signal with delay filter (approx. with FIR of some length)
           Break time-domain signal into N 
convolve(t, filter)
istft(merge(fft(convolve(ifft(split(stft(t, nchan), nsub)), filter_sb))))

A smooth bandpass => compact time-domain filter
A small  delay    => compact time-domain filter

Important: [MAYBE NOT TRUE]Building large FFTs from sub-bands of small ones
             without pre-planning is impossible.
             Must first shuffle (via bitrev) using the full length
             I.e., there is no way to construct overlapping FFTs from a STFT
             ACTUALLY, fft12.even = fft1 + fft2
                       fft12.odd  = shift(fft1, +0.5) + shift(fft2, +0.5)
                         where shift(x, +0.5) means shift up in freq by
                         half a bin (sinc interp in freq domain).

FIR filtering sub-bands
-----------------------
Each sub-band contains information on some Fourier components of the impulse
IFFT the sub-band to get a time stream
Use FFT+IFFT overlap-add or overlap-save to convolve the FIR filter
FFT back to sub-band channels
** Questions: How does the number of channels impact the quality of a
                STFT-based boxcar-windowed bandpass filter? (I.e., FFT, subselect, IFFT)
              How does the quality of a sub-band FIR filter operation (+recombine) compare
                with that of a larger-band FIR filter operation?
              More generally, what is the difference between applying an FIR to the
                original time stream vs. to multiple bandpass-filtered time streams (+recombine)?
                E.g., FIR(t) vs. MERGE(FIR(t_lo),FIR(t_hi))
              And how does this compare to not doing any overlapping at all?
                I.e., just multiply f-domain subbands by weights and then recombine at the end
                This will result in the wrapping problem.
                  Does the above sub-band FIR approach solve this problem (completely)?
                    Does it solve it, but introduce other artifacts due to sub-bands?
                      (Are the arfifacts acceptable?)

88 channels @ 40us = 227.27ns samples

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def db(x_):
	x = np.abs(x_)
	minval = x.max() / 1e6
	x[x<=minval] = minval
	ret = 10*np.log10(x)
	return ret - ret.max()
"""
ndisplay = 2048
#plt.plot(signal.get_window(4, 128, fftbins=True))
#plt.plot(signal.get_window("boxcar", 128, fftbins=True))
plt.plot(db(np.fft.rfft(signal.get_window("boxcar", 128), ndisplay)))
plt.plot(db(np.fft.rfft(signal.get_window("hann", 128), ndisplay)))
#plt.plot(signal.get_window("hamming", 128))
plt.plot(db(np.fft.rfft(signal.get_window(("kaiser", 14), 128), ndisplay)))
#plt.plot(signal.get_window("flattop", 128))
plt.plot(db(np.fft.rfft(signal.get_window("hamming", 128), ndisplay)))
#plt.yscale('log')
plt.show()
"""
def channelise_real(a, nchan):
	"""Real short-time Fourier transform"""
	print a.size, 2*nchan
	wrapped = a.reshape(a.size//(2*nchan),2*nchan)
	#return np.fft.rfft(wrapped, axis=1)
	return np.fft.rfft(wrapped, axis=1)[:,:nchan] # Note: Crops off Nyquist sample
	#return np.fft.rfft(wrapped, axis=1)[:,1:] # Note: Crops off DC sample
	#return np.fft.fft(wrapped, axis=1)
def channelise(a, nchan, bitrev=False):
	"""Complex short-time Fourier transform"""
	wrapped = a.reshape(a.size//nchan,nchan)
	"""
	if bitrev:
		inds = np.arange(nchan, dtype=np.uint32)
		inds = bitreverse(inds)
		#wrapped = wrapped[:,inds]
		wrapped[:,inds] = wrapped
	"""
	return np.fft.fft(wrapped, axis=1)

def bitreverse(x):
    rev = np.zeros_like(x)
    while np.any(x):
        rev <<= 1
        rev |= x & 1
        x >>= 1
    return rev

"""
def bitreverse_axis(x, axis):
	inds = np.arange(x.shape[axis])
"""	

def test_overlap_fft():
	#print bitreverse(np.arange(8))
	#return
	#print bitreverse(np.arange(5, dtype=np.uint32))
	#return
	#ncourse      = 7
	nchan_course = 32#128
	nchan_fine   = 4
	nchan        = nchan_course*nchan_fine
	overlap      = 1
	ncourse      = nchan_fine#2*nchan_fine-overlap # One for 'a' and one for 'b'
	np.random.seed(1234)
	#t  = np.random.normal(size=ncourse*2*nchan_course)
	#t = np.zeros(ncourse*2*nchan_course)
	#t[t.size//2] = 1
	#t = np.sin(np.linspace(0, 100, ncourse*2*nchan_course))
	#t = np.sin(3*np.linspace(0, 100, ncourse*nchan_course))
	t = np.sin(0.2*np.linspace(0, 100, ncourse*nchan_course))
	#t = np.sin(0.1*np.linspace(0, 100, ncourse*nchan_course))
	plt.plot(t)
	plt.show()
	#f  = channelise(t, nchan_course, bitrev=True)
	f  = channelise(t, nchan_course)
	idx_c = np.arange(nchan_course)[None,:]
	idx_f = np.arange(nchan_fine)[:,None]
	#*twiddle = np.exp(-2j*np.pi/nchan*idx_c*idx_f)
	twiddle = 1.
	f = f[:nchan_fine,:]
	
	#inds = np.arange(nchan, dtype=np.uint32)
	inds = np.arange(nchan_course, dtype=np.uint32)
	#inds = np.arange(nchan_fine, dtype=np.uint32)
	revinds = bitreverse(inds)
	#f[...] = f[:,revinds]
	#f[:,revinds] = f[...]
	#f[...] = f[revinds,:]
	#f[...] = f[revinds,:]
	#f[revinds,:] = f[...]
	
	fa = (np.fft.fft(twiddle*f,        axis=0)).reshape(1,nchan)
	#fa = (np.fft.fft(np.fft.fftshift(twiddle*f),        axis=0)).reshape(1,nchan)
	#fb = np.fft.fft(f[nchan_fine-overlap:,:], axis=0)
	#*fa_gold = channelise(t[:nchan*2], nchan)#.flatten()
	fa_gold = channelise(t[:nchan], nchan)#.flatten()
	
	inds = np.arange(nchan, dtype=np.uint32)
	#fa = np.fft.fftshift(fa)
	#fa = np.abs(fa)
	#fa += fa[0,inds//2]# + fa[0,1*inds//4] + fa[0,3*inds//4]
	
	#fa_gold = fa_gold[:,revinds]
	#fa_gold[:,revinds] = fa_gold
	
	#fb_gold = channelise(t[(nchan_course*(nchan_fine-overlap)):], nchan_course*nchan_fine)
	plt.plot(np.abs(fa_gold[0]), color="blue", label="Single")
	plt.plot(np.abs(fa[0]),      color="red",  label="2-stage")
	plt.legend()
	plt.show()
	print fa.shape, fa_gold.shape
	print np.allclose(fa, fa_gold)
	print fa[:,:10]
	print fa_gold[:,:10]
	#print np.allclose(fb, fb_gold)

def test_filter():
	nchan  = 128#4096
	ntime  = nchan*2 * 2
	npulse = 1
	oversample = 32
	t = np.zeros(ntime*oversample)
	t[(nchan//2-npulse//2)*oversample:(nchan//2+(npulse-1)//2+1)*oversample] = 1
	#plt.plot(db(t))
	#plt.show()
	f = np.fft.rfft(t.reshape(ntime//(2*nchan),2*nchan*oversample), axis=1)
	#plt.plot(db(f[0]))
	#plt.show()
	npass = nchan//2
	window = np.zeros(f.shape[1])
	window[(nchan//2-npass//2)*oversample:(nchan//2+(npass-1)//2+1)*oversample] = 1 * 2*nchan/npass
	#window /= window.sum()
	tnew1 = np.fft.irfft(f*window, axis=1).flatten()
	
	smooth  = np.kaiser(32*oversample, 14)
	window2 = np.convolve(window, smooth, mode='same') / smooth.sum()
	tnew2 = np.fft.irfft(f*window2, axis=1).flatten()
	
	smooth  = np.kaiser(32*oversample, 5)
	window3 = np.convolve(window, smooth, mode='same') / smooth.sum()
	tnew3 = np.fft.irfft(f*window3, axis=1).flatten()
	
	plt.plot(window,  color='red',   label="Boxcar")
	plt.plot(window2, color='green', label="Kaiser")
	plt.plot(window3, color='grey',  label="Hamming")
	plt.legend()
	plt.yscale('log')
	plt.xlabel("Frequency")
	plt.show()
	
	#f[:,:(nchan//2-npass//2)*oversample] = 0
	#f[:,(nchan//2+npass//2)*oversample:] = 0
	#tnew = np.fft.irfft(f, axis=1).flatten()# * nchan/npass
	plt.plot(db(t),     color='blue', label="Original")
	plt.plot(db(tnew1), color='red', label="Boxcar")
	plt.plot(db(tnew2), color='green', label="Kaiser")
	#plt.plot(db(tnew3), color='grey', label="Hamming")
	plt.legend()
	plt.xlabel("Frequency")
	plt.ylabel("Power [dB]")
	plt.show()

if __name__ == "__main__":
	import sys
	
	test_overlap_fft()
	sys.exit(0)
	test_filter()
	sys.exit(0)
	
	nwin = 128
	beta = 14
	oversample = 1#8
	
	ntime = 8192 * 2
	nchan = 4096
	nfft  = nchan*2
	assert(ntime % nfft == 0)
	
	window = np.kaiser(nchan+1, beta)
	#plt.plot(10*np.log10(window))
	#plt.ylim(-0,+1)
	window_f = np.fft.fft(window, window.size*oversample)
	print window
	print window_f.shape
	window_f /= window_f.max()
	plt.plot(10*np.log10(np.abs(np.fft.fftshift(window_f))))
	plt.show()
	
	t = np.zeros(ntime, dtype=np.float32)
	t[t.size//2+nfft//2] = 1
	#t[t.size//2] = 1
	#*f = np.fft.rfft(t.reshape(ntime//nfft,nfft), axis=1)[:,:nchan]
	f = np.fft.rfft(t.reshape(ntime//nfft,nfft), axis=1)
	f *= window_f
	#print f.shape
	#plt.plot(t)
	#plt.plot(np.abs(f))
	#plt.show()
	npass = nchan//2#2048#1024
	f[:,:nchan//2-npass//2] = 0
	f[:,nchan//2+npass//2:] = 0
	#tnew = np.fft.irfft(f, 2*nchan*oversample, axis=1).flatten()# * nchan/npass
	tnew = np.fft.irfft(f, axis=1).flatten()# * nchan/npass
	plt.plot(db(t),    color='blue')
	plt.plot(db(tnew), color='red')
	plt.show()
	
