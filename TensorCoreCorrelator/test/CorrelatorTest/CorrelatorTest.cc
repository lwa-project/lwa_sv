#include "test/Common/ComplexInt4.h"
#include "test/Common/Record.h"
#include "test/CorrelatorTest/CorrelatorTest.h"
#include "util/ExceptionPropagator.h"
#include "util/nvrtc.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#define GNU_SOURCE
#include <link.h>
#include <omp.h>


CorrelatorTest::CorrelatorTest(const Options &options)
:
  UnitTest(options.deviceNumber),
  options(options),
  correlator(options.nrBits, options.nrReceivers, options.nrChannels, options.nrSamplesPerChannel, options.nrPolarizations, options.nrReceiversPerBlock)
{
#if defined MEASURE_POWER
  Record start(*powerSensor), stop(*powerSensor);
#else
  Record start, stop;
#endif

  start.enqueue(stream);

  switch (options.nrBits) {
    case  4 : doTest<complex_int4_t, std::complex<int32_t>>();
	      break;

    case  8 : doTest<std::complex<int8_t>, std::complex<int32_t>>();
	      break;

    case 16 : doTest<std::complex<__half>, std::complex<float>>();
	      break;
  }

  stop.enqueue(stream);
  stream.synchronize();

  report("total              ", start, stop, options.innerRepeatCount * options.outerRepeatCount * correlator.FLOPS());
}


template <typename SampleType, typename VisibilityType> void CorrelatorTest::doTest()
{
  omp_set_nested(1);

  ExceptionPropagator ep;

#pragma omp parallel num_threads(2)
  ep([&] () {  
    context.setCurrent();

    multi_array::extent<5> samplesExtent(multi_array::extents[options.nrChannels][options.nrSamplesPerChannel / options.nrTimesPerBlock][options.nrReceivers][options.nrPolarizations][options.nrTimesPerBlock]);
    multi_array::extent<4> visibilitiesExtent(multi_array::extents[options.nrChannels][options.nrBaselines()][options.nrPolarizations][options.nrPolarizations]);

    cu::HostMemory hostSamples(sizeof(SampleType) * samplesExtent.size, CU_MEMHOSTALLOC_WRITECOMBINED);
    cu::HostMemory hostVisibilities(sizeof(VisibilityType) * visibilitiesExtent.size);

#if defined UNIFIED_MEMORY
    cu::DeviceMemory deviceSamples(hostSamples);
    cu::DeviceMemory deviceVisibilities(hostVisibilities);
#else
    cu::DeviceMemory deviceSamples(sizeof(SampleType) * samplesExtent.size);
    cu::DeviceMemory deviceVisibilities(sizeof(VisibilityType) * visibilitiesExtent.size);

    cu::Stream hostToDeviceStream, deviceToHostStream;
#endif

    multi_array::array_ref<SampleType, 5> samplesRef(* (SampleType *) hostSamples, samplesExtent);
    multi_array::array_ref<VisibilityType, 4> visibilitiesRef(* (VisibilityType *) hostVisibilities, visibilitiesExtent);

    setTestPattern<SampleType>(samplesRef);

#pragma omp for schedule(dynamic), ordered
    for (int i = 0; i < options.outerRepeatCount; i ++)
      if (!ep)
	ep([&] () {
#if !defined UNIFIED_MEMORY
	  cu::Event inputDataTransferred, executeFinished, executeFinished2;
#endif

#if defined MEASURE_POWER
	  Record hostToDeviceRecordStart(*powerSensor), hostToDeviceRecordStop(*powerSensor);
	  Record computeRecordStart(*powerSensor), computeRecordStop(*powerSensor);
	  Record deviceToHostRecordStart(*powerSensor), deviceToHostRecordStop(*powerSensor);
#else
	  Record hostToDeviceRecordStart, hostToDeviceRecordStop;
	  Record computeRecordStart, computeRecordStop;
	  Record deviceToHostRecordStart, deviceToHostRecordStop;
#endif

#pragma omp critical (GPU) // TODO: use multiple locks when using multiple GPUs
	  ep([&] () {
#if !defined UNIFIED_MEMORY
	    hostToDeviceRecordStart.enqueue(hostToDeviceStream);
	    hostToDeviceStream.memcpyHtoDAsync(deviceSamples, hostSamples, samplesRef.bytesize());
	    hostToDeviceRecordStop.enqueue(hostToDeviceStream);

	    stream.waitEvent(hostToDeviceRecordStop.event);
#endif

	    computeRecordStart.enqueue(stream);

	    for (unsigned j = 0; j < options.innerRepeatCount; j ++)
	      correlator.launchAsync(stream, deviceVisibilities, deviceSamples);

	    computeRecordStop.enqueue(stream);

#if !defined UNIFIED_MEMORY
	    deviceToHostStream.waitEvent(computeRecordStop.event);
	    deviceToHostRecordStart.enqueue(deviceToHostStream);
	    deviceToHostStream.memcpyDtoHAsync(hostVisibilities, deviceVisibilities, visibilitiesRef.bytesize());
	    deviceToHostRecordStop.enqueue(deviceToHostStream);
#endif
	  });

#if !defined UNIFIED_MEMORY
	  deviceToHostStream.synchronize();
#else
	  stream.synchronize();
#endif

	  if (i == options.outerRepeatCount - 1 && options.verifyOutput)
	    verifyOutput<SampleType, VisibilityType>(samplesRef, visibilitiesRef);

#if !defined UNIFIED_MEMORY
	  report("host-to-device     ", hostToDeviceRecordStart, hostToDeviceRecordStop, 0, samplesRef.bytesize());
#endif

	  report("correlate-total    ", computeRecordStart, computeRecordStop, options.innerRepeatCount * correlator.FLOPS());

#if !defined UNIFIED_MEMORY
	  report("device-to-host     ", deviceToHostRecordStart, deviceToHostRecordStop, 0, visibilitiesRef.bytesize());
#endif
	});
  });
}


template<typename SampleType> void CorrelatorTest::setTestPattern(const multi_array::array_ref<SampleType, 5> &samples)
{
#if 0
  memset(samples.begin(), 0, samples.bytesize());

  unsigned channel = options.nrChannels / 3;
  unsigned time    = options.nrSamplesPerChannel / 5;
  unsigned recv0   = options.nrReceivers > 174 ? 174 : options.nrReceivers / 3;
  unsigned recv1   = options.nrReceivers > 418 ? 418 : options.nrReceivers / 2;

  samples[channel][time / options.nrTimesPerBlock][recv0][POL_X][time % options.nrTimesPerBlock] = SampleType(2.0, 3.0);
  samples[channel][time / options.nrTimesPerBlock][recv1][POL_X][time % options.nrTimesPerBlock] = SampleType(4.0, 5.0);
#else
  SampleType randomValues[7777]; // use a limited set of random numbers to save time

  for (unsigned i = 0; i < 7777; i ++)
    randomValues[i] = randomValue<SampleType>();

  unsigned i = 0;

  for (SampleType &sample : samples)
    sample = randomValues[i ++ % 7777U];
#endif
}


template<typename SampleType, typename VisibilityType> void CorrelatorTest::verifyOutput(const multi_array::array_ref<SampleType, 5> &samples, const multi_array::array_ref<VisibilityType, 4> &visibilities) const
{
  std::atomic<int> count(0);
  ExceptionPropagator ep;

#if 1
  std::cout << "verifying ..." << std::endl;

#pragma omp parallel for schedule (dynamic)
  for (unsigned channel = 0; channel < options.nrChannels; channel ++)
    ep([&] () {
      multi_array::array<VisibilityType, 3> sum(multi_array::extents[options.nrBaselines()][options.nrPolarizations][options.nrPolarizations]);

      memset(sum.begin(), 0, sum.bytesize());

      for (unsigned major_time = 0; major_time < options.nrSamplesPerChannel / options.nrTimesPerBlock; major_time ++) {
	multi_array::array_ref<SampleType, 3> ref = samples[channel][major_time];

	for (unsigned recv1 = 0, baseline = 0; recv1 < options.nrReceivers; recv1 ++)
	  for (unsigned recv0 = 0; recv0 <= recv1; recv0 ++, baseline ++)
	    for (unsigned minor_time = 0; minor_time < options.nrTimesPerBlock; minor_time ++)
	      for (unsigned pol0 = 0; pol0 < options.nrPolarizations; pol0 ++)
		for (unsigned pol1 = 0; pol1 < options.nrPolarizations; pol1 ++) {
		  SampleType sample0 = ref[recv0][pol0][minor_time];
		  SampleType sample1 = ref[recv1][pol1][minor_time];

		  sum[baseline][pol1][pol0] += VisibilityType(sample1.real(), sample1.imag()) * conj(VisibilityType(sample0.real(), sample0.imag()));
		}
      }

      for (unsigned baseline = 0; baseline < options.nrBaselines(); baseline ++)
	for (unsigned pol0 = 0; pol0 < options.nrPolarizations; pol0 ++)
	  for (unsigned pol1 = 0; pol1 < options.nrPolarizations; pol1 ++)
	    if (!approximates(visibilities[channel][baseline][pol1][pol0], sum[baseline][pol1][pol0]) && ++ count < 100)
#pragma omp critical (cout)
	      ep([&] () {
		std::cout << "visibilities[" << channel << "][" << baseline << "][" << pol1 << "][" << pol0 << "], expected " << sum[baseline][pol1][pol0] << ", got " << visibilities[channel][baseline][pol1][pol0] << std::endl;
	      });
    });

  std::cout << "#errors = " << count << std::endl;
#else
  for (unsigned channel = 0; channel < options.nrChannels; channel ++)
    for (unsigned baseline = 0; baseline < options.nrBaselines(); baseline ++)
      for (unsigned pol0 = 0; pol0 < options.nrPolarizations; pol0 ++)
	for (unsigned pol1 = 0; pol1 < options.nrPolarizations; pol1 ++)
	    if (visibilities[channel][baseline][pol0][pol1] != (VisibilityType)(0, 0))
	      if (count ++ < 10)
		std::cout << "visibilities[" << channel << "][" << baseline << "][" << pol0 << "][" << pol1 << "] = " << visibilities[channel][baseline][pol0][pol1] << std::endl;
#endif
}


int main(int argc, char *argv[])
{
  try {
    cu::init();
    Options options(argc, argv);
    CorrelatorTest test(options);
  } catch (cu::Error &error) {
    std::cerr << "cu::Error: " << error.what() << std::endl;
  } catch (nvrtc::Error &error) {
    std::cerr << "nvrtc::Error: " << error.what() << std::endl;
  } catch (Options::Error &error) {
    std::cerr << "Options::Error: " << error.what() << std::endl;
  }

  return 0;
}
