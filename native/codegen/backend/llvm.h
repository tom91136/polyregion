#pragma once

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

class AstTransformer {
   llvm::LLVMContext &C;

private:
  std::unordered_map<std::string, llvm::Value *> lut;
  llvm::IRBuilder<> B;

  llvm::Type *mkTpe(const Type::Any &tpe);
  llvm::Value *mkSelect(const Term::Select &select);
  llvm::Value *mkRef(const Term::Any &ref);
  llvm::Value *mkExpr(const Expr::Any &expr, const std::string &key);
  void mkStmt(const Stmt::Any &stmt, llvm::Function *fn);

public:
  explicit AstTransformer( llvm::LLVMContext &c) : C(c), lut(), B(C) {}
  void transform(const std::unique_ptr<llvm::Module> &module, const Function &arg);
};

//class JitObjectCache : public llvm::ObjectCache {
//private:
//  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> CachedObjects;
//
//public:
//
//  JitObjectCache()  ;
//  void notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef ObjBuffer) override;
//  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *M) override;
//  ~JitObjectCache() override ;
//  void anchor() override  ;
//
//};

class LLVM : public Backend {

private:
//  JitObjectCache cache;
//  std::unique_ptr<llvm::orc::LLJIT> jit;

public:
  explicit LLVM();
  void run(const Function &fn) override;
};

} // namespace polyregion::backend