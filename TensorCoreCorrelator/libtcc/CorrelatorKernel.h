#if !defined CORRELATOR_KERNEL_H
#define CORRELATOR_KERNEL_H

#include "libtcc/Kernel.h"


namespace tcc {
  class CorrelatorKernel : public Kernel
  {
    public:
      CorrelatorKernel(cu::Module &module,
		       unsigned nrBits,
		       unsigned nrReceivers,
		       unsigned nrChannels,
		       unsigned nrSamplesPerChannel,
		       unsigned nrPolarizations = 2,
		       unsigned nrReceiversPerBlock = 64
		      );

      void launchAsync(cu::Stream &, cu::DeviceMemory &deviceVisibilities, cu::DeviceMemory &deviceSamples);

      virtual uint64_t FLOPS() const;

    private:
      const unsigned nrBits;
      const unsigned nrReceivers;
      const unsigned nrChannels;
      const unsigned nrSamplesPerChannel;
      const unsigned nrPolarizations;
      const unsigned nrReceiversPerBlock;
      unsigned       nrThreadBlocksPerChannel;
  };
}

#endif
