#pragma once

#include "ast.h"
#include "codegen.h"
#include "generated/polyast.h"

namespace polyregion::codegen {

using namespace polyregion::polyast;

class OpenCLCodeGen : public polyregion::codegen::CodeGen {

private:
  std::string mkTpe(const Type::Any &tpe);
  std::string mkRef(const Term::Any &ref);
  std::string mkExpr(const Expr::Any &expr, const std::string &key);
  std::string mkStmt(const Stmt::Any &stmt);

public:
  void run(const Function &arg) override;
};

} // namespace polyregion::codegen
