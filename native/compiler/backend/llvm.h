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

struct WhileCtx {
  llvm::BasicBlock* exit;
  llvm::BasicBlock* test;
};

class LLVMAstTransformer {

  llvm::LLVMContext &C;

private:
  using StructMemberTable = Map<std::string, size_t>;
  Map<std::string, Pair<Type::Any, llvm::Value *>> lut;
  Map<Sym, Pair<llvm::StructType *, StructMemberTable>> structTypes;
  Map<Signature, llvm::Function *> functions;
  llvm::IRBuilder<> B;

  llvm::Value *mkSelectPtr(const Term::Select &select);
  llvm::Value *mkRef(const Term::Any &ref);
  llvm::Value *mkExprValue(const Expr::Any &expr, llvm::Function *overload, const std::string &key);
  void mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx);

  llvm::Function *mkExternalFn(llvm::Function *parent, const Type::Any &rtn, const std::string &name,
                               const std::vector<Type::Any> &args);
  llvm::Value *invokeMalloc(llvm::Function *parent, llvm::Value *size);

  llvm::Value *conditionalLoad(llvm::Value *rhs);

public:
  Pair<llvm::StructType *, StructMemberTable> mkStruct(const StructDef &def);
  llvm::Type *mkTpe(const Type::Any &tpe);
  Opt<llvm::StructType *> lookup(const Sym &s);

  explicit LLVMAstTransformer(llvm::LLVMContext &c) : C(c), lut(), structTypes(), functions(), B(C) {}

  Pair<Opt<std::string>, std::string> transform(const std::unique_ptr<llvm::Module> &module, const Program &);
  Pair<Opt<std::string>, std::string> optimise(const std::unique_ptr<llvm::Module> &module);
};

class LLVM : public Backend {
public:
  explicit LLVM();
  compiler::Compilation run(const Program &) override;
};

} // namespace polyregion::backend