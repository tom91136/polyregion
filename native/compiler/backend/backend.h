#pragma once

#include "compiler.h"
namespace polyregion::backend {

using namespace polyregion;

class Backend {
public:
  virtual compiler::Compilation run(const polyast::Program &program) = 0;
  virtual ~Backend() = default;
};

} // namespace polyregion::backend