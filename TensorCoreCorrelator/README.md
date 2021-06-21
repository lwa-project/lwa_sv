# The Tensor-Core Correlator

TODO: improve documentation.

## Brief overview on how to use the Tensor-Core Correlator library:

Build the library (just type `make`)

Include `libtcc/Correlator.h`, and link with `libtcc/libtcc.so`.
Create a `tcc::Correlator` object with the number of receivers, channels, etc.
as arguments; this will automatically compile the CUDA code (at runtime).
Use the launchAsync method to correlate a block of samples; you must make
sure that the samples data is already in device memory.
The TCC adheres to RAII: any error will result in the failure to create
an `tcc::Correlator()` object (and throw some explanatory exception).
`test/SimpleExample/SimpleExample.cu` illustrates how the TCC library can be used.

The TCC internally uses wrappers around the CUDA driver API (`util/cu.h`) and
the NVRTC library (`util/nvrth.h`).  The rest of the correlator code can use
these wrappers as well, use the CUDA driver API directly, use the CUDA
runtime API, or the OpenCL environment.  See: `test/SimpleExample/SimpleExample.cu`
on how to use the CUDA runtime API; `test/CorrelatorTest/CorrelatorTest.cc` on
how to use the CUDA driver API (wrappers); and
`test/OpenCLCorrelatorTest/OpenCLCorrelatorTest.cc` on how to use TCC in an
OpenCL program.  `test/CorrelatorTest/CorrelatorTest.cc` is a much more versatile,
robust (and complex) example than `test/SimpleExample/SimpleExample.cu`.

Input and output data types are defined as follows:

```
#if NR_BITS == 4
typedef complex_int4_t        Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 8
typedef std::complex<int8_t>  Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 16
typedef std::complex<__half>  Sample;
typedef std::complex<float>   Visibility;
#endif

#define NR_TIMES_PER_BLOCK (128 / NR_BITS)

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
```

The input data type (`Samples`) is a weird format, but this seemed to be the only
format that yields good performance (tensor cores are very unforgiving).

Limitations:
- `NR_POLARIZATIONS` must be 2
- `NR_BITS` must be 4, 8, or 16
- the amount of samples over which is integrated) must be a multiple of 128 / `NR_BITS`
  (i.e., 32, 16, or 8 for 4-bit, 8-bit, or 16-bit input, respectively).

Contact John Romein (romein@astron.nl) to report bugs/feedback
