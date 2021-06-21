#include "libtcc/CorrelatorKernel.h"


namespace tcc {

CorrelatorKernel::CorrelatorKernel(cu::Module &module,
				   unsigned nrBits,
				   unsigned nrReceivers,
				   unsigned nrChannels,
				   unsigned nrSamplesPerChannel,
				   unsigned nrPolarizations,
				   unsigned nrReceiversPerBlock
				  )
:
  Kernel(module, "correlate"),
  nrBits(nrBits),
  nrReceivers(nrReceivers),
  nrChannels(nrChannels),
  nrSamplesPerChannel(nrSamplesPerChannel),
  nrPolarizations(nrPolarizations),
  nrReceiversPerBlock(nrReceiversPerBlock)
{
  unsigned blocksPerDim = (nrReceivers + nrReceiversPerBlock - 1) / nrReceiversPerBlock;
  nrThreadBlocksPerChannel = nrReceiversPerBlock == 64 ? blocksPerDim * blocksPerDim : blocksPerDim * (blocksPerDim + 1) / 2;
}


void CorrelatorKernel::launchAsync(cu::Stream &stream, cu::DeviceMemory &deviceVisibilities, cu::DeviceMemory &deviceSamples)
{
  std::vector<const void *> parameters = { deviceVisibilities.parameter(), deviceSamples.parameter() };
  stream.launchKernel(function,
		      nrThreadBlocksPerChannel, nrChannels, 1,
		      32, 2, 2,
		      0, parameters);
}


uint64_t CorrelatorKernel::FLOPS() const
{
  return 8ULL * nrReceivers * nrReceivers / 2 * nrPolarizations * nrPolarizations * nrChannels * nrSamplesPerChannel;
}

}
