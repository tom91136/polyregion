#pragma once

#include <unordered_map>
#include <utility>

#include "ast.h"
#include "backend.h"

#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include "llvmc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class LLVM : public Backend {

public:
  enum class Target { x86_64, AArch64, ARM, NVPTX64, AMDGCN, SPIRV64 };
  struct Options {
    Target target;
    std::string arch;
    [[nodiscard]] llvmc::TargetInfo toTargetInfo() const;
  };

private:
  struct WhileCtx {
    llvm::BasicBlock *exit;
    llvm::BasicBlock *test;
  };

  enum class BlockKind { Terminal, Normal };

public:
  class AstTransformer {

    Options options;
    llvm::LLVMContext &C;
    unsigned int AllocaAS = 0;

  private:
    using StructMemberTable = Map<std::string, size_t>;
    Map<std::string, Pair<Type::Any, llvm::Value *>> stackVarPtrs;
    Map<Sym, Pair<llvm::StructType *, StructMemberTable>> structTypes;
    Map<Signature, llvm::Function *> functions;
    llvm::IRBuilder<> B;

    llvm::Value *findStackVar(const Named &named);

    llvm::Value *mkSelectPtr(const Term::Select &select);
    llvm::Value *mkTermVal(const Term::Any &ref);
    llvm::Value *mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key);
    BlockKind mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx);

    llvm::Function *mkExternalFn(llvm::Function *parent, const Type::Any &rtn, const std::string &name,
                                 const std::vector<Type::Any> &args);
    llvm::Value *invokeMalloc(llvm::Function *parent, llvm::Value *size);

    llvm::Value *conditionalLoad(llvm::Value *rhs);

  public:
    AstTransformer(Options options, llvm::LLVMContext &c)
        : options(std::move(options)), C(c), stackVarPtrs(), structTypes(), functions(), B(C) {}

    Pair<llvm::StructType *, StructMemberTable> mkStruct(const StructDef &def);
    llvm::Type *mkTpe(const Type::Any &tpe, unsigned AS = 0, bool functionBoundary = false);
    Opt<llvm::StructType *> lookup(const Sym &s);
    Pair<Opt<std::string>, std::string> transform(const std::unique_ptr<llvm::Module> &module, const Program &);
    Pair<Opt<std::string>, std::string> optimise(const std::unique_ptr<llvm::Module> &module);
  };

public:
  Options options;
  explicit LLVM(Options options) : options(std::move(options)){};
  compiler::Compilation run(const Program &, const compiler::Opt& opt) override;
};

} // namespace polyregion::backend