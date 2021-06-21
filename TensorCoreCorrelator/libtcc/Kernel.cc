#include "libtcc/Kernel.h"


namespace tcc {

Kernel::Kernel(cu::Module &module, const char *name)
:
  module(module),
  function(module, name)
{
}

}
