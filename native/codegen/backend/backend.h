#pragma once

namespace polyregion::backend {

class Backend {
  virtual void run(const polyast::Function &fnTree) = 0;
};

} // namespace polyregion::codegen