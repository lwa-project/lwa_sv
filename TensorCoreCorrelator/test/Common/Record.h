#if !defined RECORD_H
#define RECORD_H

#include "test/Common/Config.h"

#include "util/cu.h"

#if defined MEASURE_POWER
#include <powersensor/NVMLPowerSensor.h>
#endif


struct Record
{
  public:
#if defined MEASURE_POWER
    Record(powersensor::PowerSensor &);
#endif

    void enqueue(cu::Stream &);

    mutable cu::Event event;

#if defined MEASURE_POWER
    powersensor::PowerSensor &powerSensor;
    powersensor::State		   state;

  private:
    static void getPower(CUstream, CUresult, void *userData);
#endif
};

#endif
