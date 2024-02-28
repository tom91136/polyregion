#pragma once

#include "ast.h"
#include "backend.h"
#include "generated/polyast.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class CSource : public polyregion::backend::Backend {

public:
  enum class Dialect {
    C11,
    OpenCL1_1,
    MSL1_0,
  };

private:
  Dialect dialect;
  std::string mkTpe(const Type::Any &tpe);
  std::string mkRef(const Term::Any &ref);
  std::string mkExpr(const Expr::Any &expr, const std::string &key);
  std::string mkStmt(const Stmt::Any &stmt);
  std::string mkFn(const Function &);

public:
  explicit CSource(const Dialect &dialect) : dialect(dialect) {}

  polyast::CompileResult compileProgram( //
      const Program &,                  //
      const compiletime::OptLevel &             //
      ) override;
  std::vector<polyast::CompileLayout> resolveLayouts( //
      const std::vector<polyast::StructDef> &,  //
      const compiletime::OptLevel &                     //
      ) override;
};

} // namespace polyregion::backend
