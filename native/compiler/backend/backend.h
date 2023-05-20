#pragma once

#include "compiler.h"
namespace polyregion::backend {

using namespace polyregion;

class Backend {
public:
  [[nodiscard]] virtual compiler::Compilation compileProgram( //
      const polyast::Program &program,                        //
      const compiler::Opt &opt                                //
      ) = 0;
  [[nodiscard]] virtual std::vector<compiler::Layout> resolveLayouts( //
      const std::vector<polyast::StructDef> &defs,                    //
      const compiler::Opt &opt                                        //
      ) = 0;
  virtual ~Backend() = default;
};

} // namespace polyregion::backend