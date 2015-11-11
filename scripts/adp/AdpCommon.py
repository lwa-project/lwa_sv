
import numpy as np

SUBSYSTEM        = "ADP"
FS               = 196.0e6
CLOCK            = 204.8e6
NCHAN            = 4096
# Note: The very last ('Nyquist') channel is discarded by the F-engine
FREQS            = np.fft.rfftfreq(2*NCHAN, 1./CLOCK)[:-1]
CHAN_BW          = FREQS[1] - FREQS[0]
NCHAN_GUARD      = 4
NCHAN_SELECT_MAX = 1920 # 48 MHz ** TODO: Check what the pipeline limit is!
NTUNING_MAX      = 32
NSTAND           = 260 # TODO: 256?
NPOL             = 2
NINPUT           = NSTAND*NPOL
FIR_NFINE        = 16
FIR_NCOEF        = 32
#NPIPELINE        = 12
NSERVER          = 6
NBOARD           = 16
NINPUT_PER_BOARD = 32
STAT_SAMP_SIZE   = 1024 # The ADC limit is 1024 (TODO: Allow > via multiples)
MAX_MSGS_PER_SEC = 80
ADC_BITS         = 8
ADC_MAXVAL       = (1<<(ADC_BITS-1))-1
#PIPELINE_HOSTS   = ['adp%i' % (i/2+1) for i in xrange(NPIPELINE)]
#SERVER_HOSTS     = ['adp%i' % (i+1) for i in xrange(NSERVER)]

def input2standpol(i):
	stand = i // NPOL
	pol   = i % NPOL
	return stand, pol
def input2boardstandpol(i):
	board = i // NINPUT_PER_BOARD
	i -= board*NINPUT_PER_BOARD
	stand = i // NPOL
	i -= stand*NPOL
	pol = i
	return board, stand, pol

def get_delay(nsamples):
	return nsamples * (1./FS)
def get_chan_idx(freqs):
	assert( np.all(freqs >= 0) )
	chans = int(freqs / CHAN_BW + 0.5)
	assert( chan < NCHAN )
	return chan
def get_freq_domain_delay(sample_delay):
	delay = get_delay(sample_delay)
	# TODO: Check the sign of the exponent!
	weights = np.exp(-2j*np.pi*FREQS*delay)
	return weights
def get_freq_domain_filter(fir_coefs):
	"""fir_coefs: [..., fine_delay, coef]"""
	weights = np.fft.rfft(fir_coefs.astype(np.float32),
	                      n=2*NCHAN, axis=-1)
	nfine = fir_coefs.shape[-2]
	k = np.arange(nfine)[:,None]
	sample_fine_delay = k / float(nfine)
	weights *= get_freq_domain_delay(sample_fine_delay)
	# Average over fine delays
	weights = weights.mean(axis=-2)
	# Crop off Nyquist sample to match F-engine
	weights = weights[...,:-1]
	# weights: [..., chan] complex64
	return weights

def test_adp_common():
	assert(FREQS[0] == 0)
	assert(np.isclose(FREQS[1], CLOCK/2/NCHAN, rtol=1e-9))
	assert(CHAN_BW == FREQS[1] - FREQS[0])
	assert(get_chan_idx(CHAN_BW * 0.5 - 1e-9) == 0)
	assert(get_chan_idx(CHAN_BW * 0.5)        == 1)
	assert(get_chan_idx(CHAN_BW * 0.5 + 1e-9) == 1)
	assert(get_chan_idx(CHAN_BW * 1.5 - 1e-9) == 1)
	assert(get_chan_idx(CHAN_BW * 1.5)        == 2)
	assert(get_chan_idx(CHAN_BW * 1.5 + 1e-9) == 2)
	return True

if __name__ == "__main__":
	test_adp_common()
	print "All tests PASSED"
