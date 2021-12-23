#pragma once

#include "ast.h"
#include "codegen.h"
namespace polyregion::codegen {

class OpenCLCodeGen : public polyregion::codegen::CodeGen {

private:
  std::string mkTpe(const Types_Type &tpe);
  std::string mkRef(const Refs_Ref &ref);

  std::string mkExpr(const Tree_Expr &expr, const std::string &key);
  std::string mkStmt(const Tree_Stmt &stmt);

public:

  void run(const Tree_Function &arg) override;
};

} // namespace polyregion::codegen
