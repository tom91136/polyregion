#pragma once

#include "ast.h"
#include "backend.h"
#include "generated/polyast.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class OpenCL : public polyregion::backend::Backend {

private:
  std::string mkTpe(const Type::Any &tpe);
  std::string mkRef(const Term::Any &ref);
  std::string mkExpr(const Expr::Any &expr, const std::string &key);
  std::string mkStmt(const Stmt::Any &stmt);

public:
  compiler::Compilation run(const Program &) override;
};

} // namespace polyregion::codegen
