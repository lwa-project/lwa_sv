#if !defined KERNEL_H
#define KERNEL_H

#include "util/cu.h"

#include <stdint.h>


namespace tcc {
  class Kernel
  {
    public:
      Kernel(cu::Module &, const char *name);

      virtual uint64_t FLOPS() const = 0;

    protected:
      cu::Module   &module;
      cu::Function function;
  };
}

#endif
