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
    OpenCL2_0,
    MSL1_0,
  };

private:
  Dialect dialect;
  [[nodiscard]] bool openCL() const { return dialect == Dialect::OpenCL1_1 || dialect == Dialect::OpenCL2_0; }
  // OpenCL C 2.0 keeps every pointer generic except where a named space is mandatory: kernel parameters and constant data
  bool kernelBoundary = false;
  [[nodiscard]] bool genericPointer(const TypeSpace::Any &space) const {
    return dialect == Dialect::OpenCL2_0 && !kernelBoundary && !space.template is<TypeSpace::Constant>();
  }
  uint32_t workgroupMemoryBytes;
  Map<std::string, std::vector<std::pair<std::string, Type::Any>>> structDefsByName;
  Set<std::string> unionDefNames;
  Map<std::string, std::string> stringConstNames;
  Map<std::string, std::string> localNames;
  Set<std::string> fileScopeNames;
  size_t localNameCounter = 0;
  void bindLocalNames(const Function &fn);
  std::string localName(const std::string &symbol);
  Type::Any resolveFieldType(const Type::Any &owner, const std::string &fieldName) const;
  std::string mkValueCopy(const std::string &lhs, const std::string &rhs, const Type::Any &tpe, int depth) const;
  std::string mkVolatileCopy(const std::string &lhs, const std::string &rhs, const Type::Any &tpe, int depth) const;
  std::string mkVolatileHelper(bool load, const Type::Any &tpe, const std::string &space);
  std::optional<std::string> mkZeroInit(const Type::Any &tpe) const;
  std::string mslPtrSpace(const Term::Any &ptr) const;
  std::string mkTpe(const Type::Any &tpe);
  std::string mkArrayDecl(const Type::Any &element, const TypeSpace::Any &space, const std::string &name, const std::string &extent);
  std::string mkDecl(const Type::Any &tpe, const std::string &name);
  std::optional<std::string> mkArrayAliasDecl(const Type::Any &tpe, const Term::Any &source, const std::string &name);
  std::string reinterpretScalar(const std::string &value, const Type::Any &from, const Type::Any &to);
  std::string mkTerm(const Term::Any &term);
  std::string workGroupCollective(const std::string &spec, const std::string &kind, const AtomicOp::Any &op, const Term::Any &value,
                                  const Type::Any &rtn);
  std::string mkExpr(const Expr::Any &expr);
  std::string mkStmt(const Stmt::Any &stmt);
  std::string mkFnProto(const Function &);
  std::string mkFn(const Function &);

public:
  explicit CSource(const Dialect &dialect, uint32_t workgroupMemoryBytes) : dialect(dialect), workgroupMemoryBytes(workgroupMemoryBytes) {}

  CompileResult compileProgram(const Program &, const compiletime::OptLevel &) override;
  std::vector<StructLayout> resolveLayouts(const std::vector<StructDef> &) override;
};

} // namespace polyregion::backend
