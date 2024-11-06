#pragma once

#include "ast.h"
#include "backend.h"
#include "generated/polyast.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class CSource : public Backend {

public:
  enum class Dialect : uint8_t {
    C11,
    OpenCL1_1,
    MSL1_0,
  };

private:
  Dialect dialect;
  std::string mkTpe(const Type::Any &tpe);
  std::string mkExpr(const Expr::Any &expr);
  std::string mkStmt(const Stmt::Any &stmt);
  std::string mkFnProto(const Function &);
  std::string mkFn(const Function &);

public:
  explicit CSource(const Dialect &dialect) : dialect(dialect) {}

  CompileResult compileProgram(const Program &, const compiletime::OptLevel &) override;
  std::vector<StructLayout> resolveLayouts(const std::vector<StructDef> &) override;
};

} // namespace polyregion::backend
