#pragma once

#include "compiler.h"
namespace polyregion::backend {

using namespace polyregion;

class Backend {
public:
  [[nodiscard]] virtual polyast::CompileResult compileProgram( //
      const polyast::Program &program,                        //
      const polyast::OptLevel &opt                                //
      ) = 0;
  [[nodiscard]] virtual std::vector<polyast::CompileLayout> resolveLayouts( //
      const std::vector<polyast::StructDef> &defs,                    //
      const polyast::OptLevel &opt                                        //
      ) = 0;
  virtual ~Backend() = default;
};

} // namespace polyregion::backend