#pragma once

#include <stdexcept>

#include "compiler.h"

namespace polyregion::backend {

struct BackendException : public std::logic_error {
  explicit BackendException(const char *what,                      //
                   const char *file = __builtin_FILE(),   //
                   const char *fn = __builtin_FUNCTION(), //
                   int line = __builtin_LINE());
  explicit BackendException(const std::string &what,               //
                   const char *file = __builtin_FILE(),   //
                   const char *fn = __builtin_FUNCTION(), //
                   int line = __builtin_LINE());
};

using namespace polyregion;

class Backend {
public:
  [[nodiscard]] virtual polyast::CompileResult compileProgram( //
      const polyast::Program &program,                         //
      const polyast::OptLevel &opt                             //
      ) = 0;
  [[nodiscard]] virtual std::vector<polyast::CompileLayout> resolveLayouts( //
      const std::vector<polyast::StructDef> &defs,                          //
      const polyast::OptLevel &opt                                          //
      ) = 0;
  virtual ~Backend() = default;
};

} // namespace polyregion::backend