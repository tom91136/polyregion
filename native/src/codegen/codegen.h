#pragma once

namespace polyregion::codegen{

  class CodeGen{
    virtual void run(const Tree_Function &fnTree) = 0;
  };

}