#if !defined CORRELATOR_H
#define CORRELATOR_H

#include "libtcc/CorrelatorKernel.h"
#include "util/cu.h"
#include "util/nvrtc.h"


namespace tcc {
  class Correlator {
    public:
      Correlator(unsigned nrBits,
		 unsigned nrReceivers,
		 unsigned nrChannels,
		 unsigned nrSamplesPerChannel,
		 unsigned nrPolarizations = 2,
		 unsigned nrReceiversPerBlock = 64
		); // throw (cu::Error, nvrtc::Error)

      void launchAsync(cu::Stream &, cu::DeviceMemory &isibilities, cu::DeviceMemory &samples); // throw (cu::Error)
      void launchAsync(CUstream, CUdeviceptr visibilities, CUdeviceptr samples); // throw (cu::Error)

      uint64_t FLOPS() const;

    private:
      std::string      findNVRTCincludePath() const;
      cu::Module       compileModule(unsigned nrBits,
				     unsigned nrReceivers,
				     unsigned nrChannels,
				     unsigned nrSamplesPerChannel,
				     unsigned nrPolarizations,
				     unsigned nrReceiversPerBlock
				    );

      cu::Module       correlatorModule;
      CorrelatorKernel correlatorKernel;
  };
}

#endif
