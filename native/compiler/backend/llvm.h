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
  std::unordered_map<std::string, llvm::Value *> lut;
  std::unordered_map<Sym, std::pair<llvm::StructType *, StructMemberTable>> structTypes;
  llvm::IRBuilder<> B;

  llvm::Value *mkSelect(const Term::Select &s);
  llvm::Value *mkRef(const Term::Any &ref);
  llvm::Value *mkExpr(const Expr::Any &expr, llvm::Function *fn, const std::string &key);
  void mkStmt(const Stmt::Any &stmt, llvm::Function *fn);

public:
  std::pair<llvm::StructType *, StructMemberTable> mkStruct(const StructDef &def);
  llvm::Type *mkTpe(const Type::Any &tpe);

  explicit LLVMAstTransformer(llvm::LLVMContext &c) : C(c), lut(), structTypes(), B(C) {}
  void transform(const std::unique_ptr<llvm::Module> &module, const Function &arg);
};

class LLVM : public Backend {
public:
  explicit LLVM();
  compiler::Compilation run(const Function &fn) override;
};

} // namespace polyregion::backend