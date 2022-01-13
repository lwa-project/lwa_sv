# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import numpy as np
import datetime

SUBSYSTEM        = "ADP"
FS               = 196.0e6
CLOCK            = 204.8e6
NCHAN            = 4096
# Note: The very last ('Nyquist') channel is discarded by the F-engine
# Note: Rounded to mHz to avoid floating-point precision errors
FREQS            = np.around(np.fft.fftfreq(2*NCHAN, 1./CLOCK)[:NCHAN][:-1], 3)
CHAN_BW          = FREQS[1] - FREQS[0]
NCHAN_GUARD      = 4
NCHAN_SELECT_MAX = 1920 # 48 MHz ** TODO: Check what the pipeline limit is!
NTUNING_MAX      = 32
NSTAND           = 256
NPOL             = 2
NINPUT           = NSTAND*NPOL
NSUBSLOT_PER_SEC = 100
FIR_NFINE        = 16
FIR_NCOEF        = 32
#NPIPELINE        = 12
NSERVER          = 6
NBOARD           = 16
NINPUT_PER_BOARD = 32
STAT_SAMP_SIZE   = 1024 # The ADC limit is 1024 (TODO: Allow > via multiples)
MAX_MSGS_PER_SEC = 20
ADC_BITS         = 8
ADC_MAXVAL       = (1<<(ADC_BITS-1))-1
TBN_BITS         = 16
DATE_FORMAT      = "%Y_%m_%dT%H_%M_%S"
ADP_EPOCH        = datetime.datetime(1970, 1, 1)
M5C_EPOCH        = datetime.datetime(1990, 1, 1)
#M5C_OFFSET = int((M5C_EPOCH - ADP_EPOCH).total_seconds())
M5C_OFFSET = 0 # LWA convention re-defines this to use the 1970 epoch too
DRX_NSAMPLE_PER_PKT = 4096
TBN_NSAMPLE_PER_PKT = 512
TBF_NCHAN_PER_PKT   = 12
NFRAME_PER_SPECTRUM = int(FS) // int(CHAN_BW) # 7840
#PIPELINE_HOSTS   = ['adp%i' % (i/2+1) for i in xrange(NPIPELINE)]
#SERVER_HOSTS     = ['adp%i' % (i+1) for i in xrange(NSERVER)]
TRIGGERING_ACTIVE_FILE = '/home/adp/triggering_active'

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

"""
This module is used to fork the current process into a daemon.
Almost none of this is necessary (or advisable) if your daemon
is being started by inetd. In that case, stdin, stdout and stderr are
all set up for you to refer to the network connection, and the fork()s
and session manipulation should not be done (to avoid confusing inetd).
Only the chdir() and umask() steps remain as useful.
From:
http://code.activestate.com/recipes/66012-fork-a-daemon-process-on-unix/
References:
UNIX Programming FAQ
    1.7 How do I get my program to act like a daemon?
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        
    Advanced Programming in the Unix Environment
    W. Richard Stevens, 1992, Addison-Wesley, ISBN 0-201-56317-7.
"""

def daemonize(stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
    """
    This forks the current process into a daemon.
    The stdin, stdout, and stderr arguments are file names that
    will be opened and be used to replace the standard file descriptors
    in sys.stdin, sys.stdout, and sys.stderr.
    These arguments are optional and default to /dev/null.
    Note that stderr is opened unbuffered, so
    if it shares a file with stdout then interleaved output
    may not appear in the order that you expect.
    """
    
    # Do first fork.
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit first parent.
    except OSError as e:
        sys.stderr.write("fork #1 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)
        
    # Decouple from parent environment.
    os.chdir("/")
    os.umask(0)
    os.setsid()
    
    # Do second fork.
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit second parent.
    except OSError as e:
        sys.stderr.write("fork #2 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)
        
    # Now I am a daemon!
    
    # Redirect standard file descriptors.
    si = file(stdin, 'r')
    so = file(stdout, 'a+')
    se = file(stderr, 'a+', 0)
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

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
    print("All tests PASSED")
