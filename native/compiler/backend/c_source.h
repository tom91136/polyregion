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
  };

private:
  Dialect dialect;
  std::string mkTpe(const Type::Any &tpe);
  std::string mkRef(const Term::Any &ref);
  std::string mkExpr(const Expr::Any &expr, const std::string &key);
  std::string mkStmt(const Stmt::Any &stmt);

public:
  explicit CSource(const Dialect &dialect) : dialect(dialect) {}

  compiler::Compilation run(const Program &, const compiler::Opt& opt) override;
};

} // namespace polyregion::backend
