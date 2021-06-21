#if !defined CORRELATOR_TEST_H
#define CORRELATOR_TEST_H

#include "test/Common/ComplexInt4.h"
#include "test/Common/UnitTest.h"
#include "test/CorrelatorTest/Options.h"
#include "libtcc/Correlator.h"
#include "util/multi_array.h"

#include <cuda_fp16.h>


class CorrelatorTest : public UnitTest
{
  public:
    CorrelatorTest(const Options &);

  private:
    template<typename SampleType, typename VisibilityType> void doTest();
    template<typename SampleType>                          void setTestPattern(const multi_array::array_ref<SampleType, 5> &samples);
    template<typename SampleType, typename VisibilityType> void verifyOutput(const multi_array::array_ref<SampleType, 5> &samples, const multi_array::array_ref<VisibilityType, 4> &visibilities) const;

    template<typename SampleType> static SampleType randomValue();
    template<typename VisibilityType> bool approximates(const VisibilityType &a, const VisibilityType &b) const;

    tcc::Correlator correlator;
    Options         options;
};


template<> complex_int4_t CorrelatorTest::randomValue<complex_int4_t>()
{
  return complex_int4_t(16 * drand48() - 8, 16 * drand48() - 8);
}


template<> std::complex<int8_t> CorrelatorTest::randomValue<std::complex<int8_t>>()
{
  return std::complex<int8_t>(256 * drand48() - 128, 256 * drand48() - 128);
}


template<> std::complex<__half> CorrelatorTest::randomValue<std::complex<__half>>()
{
  return std::complex<__half>(drand48() - .5, drand48() - .5);
}

template <typename VisibilityType> bool CorrelatorTest::approximates(const VisibilityType &a, const VisibilityType &b) const
{
  return a == b;
}


template <> bool CorrelatorTest::approximates(const std::complex<float> &a, const std::complex<float> &b) const
{
  float absolute = abs(a - b), relative = abs(a / b);

  return (relative > .999 && relative < 1.001) || absolute < .0001 * options.nrSamplesPerChannel;
}

#endif
