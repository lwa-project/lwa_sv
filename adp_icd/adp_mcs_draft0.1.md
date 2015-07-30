
# ADP ICD

## MIB entries

Index    | Label                  | Type      | Bytes | Value(s) | Description
---      | ---                    | ---       | ---:  | ---      | ---
2        | `TBW_STATUS`           | `uint8`   | 1 | <ul><li>0: Idle.</li><li>4: Actively recording or writing out.</li></ul> | Current status of TBW.
3        | `NUM_TBN_BITS`         | `uint8`   | 1 | Always 16 (8 real + 8 imag) | No. bits per sample in TBN output. Currently always 16 (8 real + 8 imag).
4.1      | `NUM_DRX_TUNINGS`      | `uint8`   | 1 | <math>2<sup>n</sup></math> for <math>n</math> in <math>[0:5]</math> | No. frequency tunings available. Currently constrained to powers of 2 up to 32.
4.2      | `NUM_BEAMS`            | `uint8`   | 1 | [1-4] | No. beams.
4.3      | `NUM_STANDS`           | `uint16`  | 2 | Always 256 | No. stands.
4.4.1    | `NUM_BOARDS`           | `uint8`   | 1 | Always 16 | No. ROACH (FPGA) boards.
4.4.2    | `NUM_SERVERS`          | `uint8`   | 1 | Always 6 | No. servers.
4.5      | `BEAM_FIR_COEFFS`      | `uint8`   | 1 | Always 32 | No. FIR coeffs implemented.
4.6.n    | `T_NOMn`               | `uint16`  | 2 | Full range | T<sub>nom</sub>=L from LWA Memo 151, in units of samples at <math>f<sub>s</sub></math> for beam `n`.
5.1      | `FIR`                  | `sint16[16,32]` | 1024 | Full range | FIR coeffs for input specified by `FIR_CHAN_INDEX`.
5.5      | `FIR_CHAN_INDEX`       | `uint16`  | 2 | [1-512] | Returns and increments index of the input whose FIR coeffs are returned by `FIR`.
6        | `CLK_VAL`              | `uint32`  | 4 | [0:86401000) | Time at start of previous slot, in ms past station time midnight (MPM).
7.n.1    | `ANTn_RMS`             | `float32` | 4 | Full range | RMS power of `STAT_SAMP_SIZE` current samples for input `n`.
7.n.2    | `ANTn_DCOFFSET`        | `float32` | 4 | Full range | Mean of `STAT_SAMP_SIZE` current samples for input `n`.
7.n.3    | `ANTn_SAT`             | `float32` | 4 | Full range | No. saturated values (+-127) in `STAT_SAMP_SIZE` current samples for input `n`.
7.n.4    | `ANTn_PEAK`            | `float32` | 4 | Full range | Max of `STAT_SAMP_SIZE` current samples for input `n`.
7.0      | `STAT_SAMP_SIZE`       | `uint32`  | 4 | Typically 1024 | No. samples used to compute statistics.

## Control commands

### BAM command
#### Description
Configures a beam with new delays and gains.

#### Arguments
Name           | Type                   | Value(s)   | Description
---            | ---                    | ---        | ---
`BEAM_ID`      | `sint16`               | [1-NUM_BEAMS] | Beam to be changed.
`BEAM_DELAY`   | `fixed16.4[512]`       | [0-256)    | Sample delay for each input.
`BEAM_GAIN`    | `fixed16.1[256][2][2]` | Full range | 2x2 polarisation mixing matrix for each stand.
<b>`DRX_TUNING`</b> | `uint8`     | [1-NUM_DRX_TUNINGS] | Frequency tuning to be used.
`sub_slot`     | `uint8`                | [0-99]     | Sub-slot at which to take effect.

### DRX command
#### Description
Configures a frequency tuning, which can be used by the beamformer, correlator and TBW.

#### Arguments
Name           | Type                   | Value(s)   | Description
---            | ---                    | ---        | ---
~~`DRX_BEAM`~~ | `uint8`                | [1-NUM_BEAMS] | Beam to be changed.
`DRX_TUNING`   | `uint8`                | [1-NUM_DRX_TUNINGS] | Frequency tuning to be changed.
`DRX_FREQ`     | `float32`              | [0-102.1875e6] Hz | Center freq. in Hz.
`DRX_BW`       | `uint8`                | [1-7]      | Filter no. indicating sample rate. See Table ??.
`DRX_GAIN`     | `sint16`               | [0-15]     | Right-bitshift to compensate for BW reduction.
`sub_slot`     | `uint8`                | [0-99]     | Sub-slot at which to take effect.

### FST command
#### Description
...

TBW
TBN
DRX
FST
BAM
INI
STP
SHT

## Correlator output interface

TODO: Packet header, containing which baseline, which freq chans,
        integration time etc.

Each packet payload shall contain 144 frequency channels (each
representing 25.0 kHz of bandwidth) and 4 polarisation products for
one baseline (unique pair of stands). The data shall be ordered with
frequency channel changing slowest, followed by the polarisation of
the first stand, the polarisation of the second stand, and finally a
packed value of 8 bytes, for a total payload size of 4608 bytes.

    Slowest-changing                           Fastest-changing
    [144 chans][2 pol_i (X,Y)][2 pol_j (X,Y)][8 byte structure] = 4608 bytes

Polarisations are ordered X then Y, giving a combined order of XiXj,
XiYj, YiXj, YiYj. The 8-byte value structure shall contain the real
and imaginary components of a complex number, each 21 bits, followed
by a weight value of 22 bits. All three values are signed and in two's
complement format, with the structure packed MSB first.

    MSB                               LSB
    0        8        16       24     31
    ======== ======== ======== ========
    <------- --REAL-- ----><-- --------
    IMAG---- -><----- -WEIGHT- ------->
    ======== ======== ======== ========
