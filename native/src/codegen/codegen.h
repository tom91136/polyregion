#pragma once

namespace polyregion::codegen {

class CodeGen {
  virtual void run(const polyast::Function &fnTree) = 0;
};

} // namespace polyregion::codegen