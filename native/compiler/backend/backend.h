#pragma once

#include "compiler.h"
namespace polyregion::backend {

using namespace polyregion;

class Backend {
  virtual compiler::Compilation run(const polyast::Function &fnTree) = 0;
};

} // namespace polyregion::backend