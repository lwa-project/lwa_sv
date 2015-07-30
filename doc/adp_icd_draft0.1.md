
# ADP ICD

## Hardware

- `headnode`: The ADP cluster head machine, which is named `adp` (aka
  `adp0`). This machine runs the `adp-control` service and interfaces
  between the cluster and the outside (of ADP) world.

- `roaches`: The ADP cluster FPGA boards, of which there are 16, named
  'roach1-16'. These boards run the ADP `F-engine` firmware to
  frequency-channelise data from the ADCs and send it through the data
  switch to the servers.

- `servers`: The ADP cluster processing machines, of which there are
  six, named `adp1-6`. These machines run the `adp-pipeline` service
  (aka the `X-engine`), which captures and processes data streams from
  the roaches.

## Observing modes

### TBN

### BAM

The beamformer. Supports up to 4 dual-pol beams, each using any tuning
set by the DRX command.

### COR

The correlator. Cross-correlates all 512 inputs with 25 kHz frequency
channels.

### TBW

## Constants

Note that these are defined in `AdpCommon.py`.

Symbol    | Value | Description
---       | ---   | ---
SUBSYSTEM | ADP   | Name of the subsystem
FS (<math>f<sub>s</sub></math>) | 196.0 MHz | Signal sampling frequency unit*.
FC (<math>f<sub>c</sub></math>) | 25.0 kHz  | Width of each correlator (COR) frequency channel.
NSTAND    | 256   | No. stands.
NSERVER   |   6   | No. ADP processing servers.
NBOARD    | 16    | No. roaches.

*<math>f<sub>s</sub></math> is not actually the sampling frequency
 used internally by ADP, but this definition is maintained for
 backwards compatibility with DP.

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

### TBN command

#### Description
Configures and starts TBN mode. All other modes are stopped.

#### Arguments

#### Filter codes

Filter | Sample rate (kHz)
---    | ---:
1-4    |      -
5      |     25
6      |     50
7      |    100
8      |    200
9      |    400
10     |    800
11     |   1600

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

Configures a frequency tuning, which can be used by the beamformer,
correlator and TBW. Any combination of up to NUM_DRX_TUNINGS tunings
totalling up to 39.2 MHz of bandwidth may be specified.

#### Arguments

Name           | Type                   | Value(s)   | Description
---            | ---                    | ---        | ---
~~`DRX_BEAM`~~ | `uint8`                | [1-NUM_BEAMS] | Beam to be changed.
`DRX_TUNING`   | `uint8`                | [1-NUM_DRX_TUNINGS] | Frequency tuning to be changed.
`DRX_FREQ`     | `float32`              | [0-102.1875e6] Hz | Center freq. in Hz.
`DRX_BW`       | `uint8`                | [1-7]      | Filter no. indicating sample rate. See Table ??.
`DRX_GAIN`     | `sint16`               | [0-15]     | Right-bitshift to compensate for BW reduction.
`sub_slot`     | `uint8`                | [0-99]     | Sub-slot at which to take effect.

#### Filter codes

Filter | Sample rate (kHz)
---    | ---:
1-2    |     -
3      |  1225
4      |  2450
5      |  4900
6      |  9800
7      | 19600
8      | 39200

### FST command
#### Description
TODO: ...

### COR command
#### Description

Configures and enables the correlator. The correlator generates
visibilities for all active frequency channels defined via the DRX
command.

#### Arguments

Name           | Type                   | Value(s)   | Description
---            | ---                    | ---        | ---
`COR_NAVG`     | `sint32`               | Full range > 0 | Number of visibility spectra to integrate, in units of 1/<math>f<sub>c</sub></math>.
`COR_GAIN`     | `sint16`               | [0-15]     | Right-bitshift to compensate for BW reduction.
`sub_slot`     | `uint8`                | [0-99]     | Sub-slot at which to take effect.

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
by a weight value with format `fixed22.21`. All three values are
signed and in two's complement format, with the structure packed MSB
first. Negative weight values indicate that samples were flagged.

    MSB                               LSB
    0        8        16       24     31
    ======== ======== ======== ========
    <------- --REAL-- ----><-- --------
    IMAG---- -><----- -WEIGHT- ------->
    ======== ======== ======== ========
