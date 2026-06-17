#pragma once

#include "polyregion/aliases.h"

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
  Map<std::string, std::vector<std::pair<std::string, Type::Any>>> structDefsByName;
  Type::Any resolveFieldType(const Type::Any &owner, const std::string &fieldName) const;
  std::string mkTpe(const Type::Any &tpe);
  std::string mkDecl(const Type::Any &tpe, const std::string &name);
  std::string mkTerm(const Term::Any &term);
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
