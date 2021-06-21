#define NR_BITS 8
#define NR_CHANNELS 480
#define NR_POLARIZATIONS 2
#define NR_SAMPLES_PER_CHANNEL 3072
#define NR_RECEIVERS 576
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 64
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))


#include "test/Common/ComplexInt4.h"
#include "libtcc/Correlator.h"

#include <complex>
#include <iostream>

#include <cuda.h>
#if NR_BITS == 16
#include <cuda_fp16.h>
#endif


inline void checkCudaCall(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}


#if NR_BITS == 4
typedef complex_int4_t	      Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 8
typedef std::complex<int8_t>  Sample;
typedef std::complex<int32_t> Visibility;
#elif NR_BITS == 16
typedef std::complex<__half>  Sample;
typedef std::complex<float>   Visibility;
#endif

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];


int main()
{
  try {
    checkCudaCall(cudaSetDevice(0)); // combine the CUDA runtime API and CUDA driver API
    checkCudaCall(cudaFree(0));

    tcc::Correlator correlator(NR_BITS, NR_RECEIVERS, NR_CHANNELS, NR_SAMPLES_PER_CHANNEL, NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);

    cudaStream_t stream;
    Samples *samples;
    Visibilities *visibilities;

    checkCudaCall(cudaStreamCreate(&stream));
    checkCudaCall(cudaMallocManaged(&samples, sizeof(Samples)));
    checkCudaCall(cudaMallocManaged(&visibilities, sizeof(Visibilities)));

    (*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][174][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(2, 3);
    (*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][418][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(4, 5);

    correlator.launchAsync((CUstream) stream, (CUdeviceptr) visibilities, (CUdeviceptr) samples);
    checkCudaCall(cudaDeviceSynchronize());

    std::cout << ((*visibilities)[160][87745][0][0] == Visibility(23, -2) ? "success" : "failed") << std::endl;

    checkCudaCall(cudaFree(visibilities));
    checkCudaCall(cudaFree(samples));
    checkCudaCall(cudaStreamDestroy(stream));
  } catch (std::exception &error) {
    std::cerr << error.what() << std::endl;
  }
}
