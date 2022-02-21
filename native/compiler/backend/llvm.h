#pragma once

#include <unordered_map>
#include <utility>

#include "ast.h"
#include "backend.h"

#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class LLVMAstTransformer {
  llvm::LLVMContext &C;

private:
  using StructMemberTable = std::unordered_map<std::string, size_t>;
  std::unordered_map<std::string, std::pair<Type::Any, llvm::Value *>> lut;
  std::unordered_map<Sym, std::pair<llvm::StructType *, StructMemberTable>> structTypes;
  llvm::IRBuilder<> B;

  llvm::Value *mkSelectPtr(const Term::Select &select);
  llvm::Value *mkRef(const Term::Any &ref);
  llvm::Value *mkExprValue(const Expr::Any &expr, llvm::Function *overload, const std::string &key);
  void mkStmt(const Stmt::Any &stmt, llvm::Function *fn);

public:
  std::pair<llvm::StructType *, StructMemberTable> mkStruct(const StructDef &def);
  llvm::Type *mkTpe(const Type::Any &tpe);
  std::optional<llvm::StructType *> lookup(const Sym &s);

  explicit LLVMAstTransformer(llvm::LLVMContext &c) : C(c), lut(), structTypes(), B(C) {}

  std::pair<std::optional<std::string>, std::string> transform(const std::unique_ptr<llvm::Module> &module,
                                                               const Program &);
  std::pair<std::optional<std::string>, std::string> optimise(const std::unique_ptr<llvm::Module> &module);
  llvm::Value *conditionalLoad(llvm::Value *rhs);
};

class LLVM : public Backend {
public:
  explicit LLVM();
  compiler::Compilation run(const Program &) override;
};

} // namespace polyregion::backend