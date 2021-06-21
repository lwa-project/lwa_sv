#include "test/Common/Record.h"


#if defined MEASURE_POWER
Record::Record(powersensor::PowerSensor &powerSensor)
:
  powerSensor(powerSensor)
{
}
#endif


#if defined MEASURE_POWER
void Record::getPower(CUstream, CUresult, void *userData)
{
  Record *record = (Record *) userData;
  record->state = record->powerSensor.read();
}

#endif


void Record::enqueue(cu::Stream &stream)
{
  stream.record(event); // if this is omitted, the callback takes ~100 ms ?!

#if defined MEASURE_POWER
#if !defined TEGRA_QUIRKS
  stream.addCallback(&Record::getPower, this); // does not work well on Tegra
#else
  stream.synchronize();
  state = powerSensor.read();
#endif
#endif
}
