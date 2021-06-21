#include "test/Common/UnitTest.h"

#include <iostream>


UnitTest::UnitTest(unsigned deviceNumber)
:
  device(deviceNumber),
  context(CU_CTX_SCHED_BLOCKING_SYNC, device)
#if defined MEASURE_POWER
, powerSensor(powersensor::nvml::NVMLPowerSensor::create(0))
#endif
{
#pragma omp critical (clog)
  std::clog << "running test on " << device.getName() << std::endl;

#if 0 && defined MEASURE_POWER
  powerSensor->dump("/tmp/sensor_readings");
#endif
}


UnitTest::~UnitTest()
{
#if defined MEASURE_POWER
  delete powerSensor;
#endif
}


void UnitTest::report(const char *name, const Record &startRecord, const Record &stopRecord, uint64_t FLOPS, uint64_t bytes)
{
#if defined MEASURE_POWER
  //powerSensor->mark(startRecord.state, name);

  double Watt    = powersensor::PowerSensor::Watt(startRecord.state, stopRecord.state);
#endif
  
  double runtime = stopRecord.event.elapsedTime(startRecord.event) * 1e-3;

#pragma omp critical (cout)
  { 
    std::cout << name << ": " << runtime << " s";

    if (FLOPS != 0)
      std::cout << ", " << FLOPS / runtime * 1e-12 << " TOPS";
       
    if (bytes != 0)
      std::cout << ", " << bytes / runtime * 1e-9 << " GB/s";

#if defined MEASURE_POWER
    std::cout << ", " << Watt << " W";

    if (FLOPS != 0)
      std::cout << ", " << FLOPS / runtime / Watt * 1e-9 << " GOPS/W";
#endif

    std::cout << std::endl;
  }
}
