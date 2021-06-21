#if !defined COMPLEX_INT4_T
#define COMPLEX_INT4_T

#include <complex>
#include <limits>

class complex_int4_t
{
  public:
    complex_int4_t() {}
    complex_int4_t(int real, int imag) { value = (imag << 4) | (real & 0xF); }
    complex_int4_t operator = (const complex_int4_t &other) { value = other.value; return *this; }
    int real() const { return value << (std::numeric_limits<int>::digits - 4) >> (std::numeric_limits<int>::digits - 4); }
    int imag() const { return value << (std::numeric_limits<int>::digits - 8) >> (std::numeric_limits<int>::digits - 4); }
    operator std::complex<int> () const { return std::complex<int>(real(), imag()); }

  private:
    char value;
};

#endif
