#include "libtcc/Correlator.h"

#include <iostream>

#define GNU_SOURCE
#include <link.h>


extern const char _binary_libtcc_TCCorrelator_cu_start, _binary_libtcc_TCCorrelator_cu_end;

namespace tcc {

std::string Correlator::findNVRTCincludePath() const
{
  std::string path;

  if (dl_iterate_phdr([] (struct dl_phdr_info *info, size_t, void *arg) -> int
		      {
			std::string &path = *static_cast<std::string *>(arg);
			path = info->dlpi_name;
			return path.find("libnvrtc.so") != std::string::npos;
		      }, &path))
  {
    path.erase(path.find_last_of("/")); // remove library name
    path.erase(path.find_last_of("/")); // remove /lib64
    path += "/include";
  }

  return path;
}


Correlator::Correlator(unsigned nrBits,
		       unsigned nrReceivers,
		       unsigned nrChannels,
		       unsigned nrSamplesPerChannel,
		       unsigned nrPolarizations,
		       unsigned nrReceiversPerBlock
		      )
:
  correlatorModule(compileModule(nrBits, nrReceivers, nrChannels, nrSamplesPerChannel, nrPolarizations, nrReceiversPerBlock)),
  correlatorKernel(correlatorModule, nrBits, nrReceivers, nrChannels, nrSamplesPerChannel, nrPolarizations, nrReceiversPerBlock)
{
}


cu::Module Correlator::compileModule(unsigned nrBits,
				     unsigned nrReceivers,
				     unsigned nrChannels,
				     unsigned nrSamplesPerChannel,
				     unsigned nrPolarizations,
				     unsigned nrReceiversPerBlock
				    )
{
  cu::Device device(cu::Context::getCurrent().getDevice());
  int capability = 10 * device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>() + device.getAttribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>();

  std::vector<std::string> options =
  {
    "-I" + findNVRTCincludePath(),
    "-std=c++11",
    "-arch=compute_" + std::to_string(capability),
    "-lineinfo",
    "-DNR_BITS=" + std::to_string(nrBits),
    "-DNR_RECEIVERS=" + std::to_string(nrReceivers),
    "-DNR_CHANNELS=" + std::to_string(nrChannels),
    "-DNR_SAMPLES_PER_CHANNEL=" + std::to_string(nrSamplesPerChannel),
    "-DNR_POLARIZATIONS=" + std::to_string(nrPolarizations),
    "-DNR_RECEIVERS_PER_BLOCK=" + std::to_string(nrReceiversPerBlock),
  };

  //std::for_each(options.begin(), options.end(), [] (const std::string &e) { std::cout << e << ' '; }); std::cout << std::endl;

#if 0
  nvrtc::Program program("tcc/TCCorrelator.cu");
#else
  // embed the CUDA source code in libtcc.so, so that it need not be installed separately
  // for runtime compilation
  // copy into std::string for '\0' termination
  std::string source(&_binary_libtcc_TCCorrelator_cu_start, &_binary_libtcc_TCCorrelator_cu_end);
  nvrtc::Program program(source, "TCCorrelator.cu");
#endif

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  return cu::Module((void *) program.getPTX().data());
}


void Correlator::launchAsync(cu::Stream &stream, cu::DeviceMemory &visibilities, cu::DeviceMemory &samples)
{
  correlatorKernel.launchAsync(stream, visibilities, samples);
}


void Correlator::launchAsync(CUstream stream, CUdeviceptr visibilities, CUdeviceptr samples)
{
  cu::Stream _stream(stream);
  cu::DeviceMemory _visibilities(visibilities);
  cu::DeviceMemory _samples(samples);
  correlatorKernel.launchAsync(_stream, _visibilities, _samples);
}


uint64_t Correlator::FLOPS() const
{
  return correlatorKernel.FLOPS();
}

}
