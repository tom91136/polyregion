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
    llvm::BasicBlock *exit, *test;
  };

  enum class BlockKind { Terminal, Normal };

public:
  class AstTransformer {

    Options options;
    llvm::LLVMContext &C;
    unsigned int AllocaAS = 0;

  private:
    using StructMemberIndexTable = Map<std::string, size_t>;
    Map<std::string, Pair<Type::Any, llvm::Value *>> stackVarPtrs;
    Map<Sym, Pair<llvm::StructType *, StructMemberIndexTable>> structTypes;
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
    llvm::Type *mkTpe(const Type::Any &tpe, unsigned AS = 0, bool functionBoundary = false);

    Pair<llvm::StructType *, StructMemberIndexTable> mkStruct(const StructDef &def);

    //    Opt<llvm::StructType *> lookup(const Sym &s);
    //    llvm::Value *conditionalLoad(llvm::Value *rhs);

  public:
    AstTransformer(Options options, llvm::LLVMContext &c)
        : options(std::move(options)), C(c), stackVarPtrs(), structTypes(), functions(), B(C) {}

    void addDefs(const std::vector<StructDef> &);

    std::vector<Pair<Sym, llvm::StructType *>> getStructTypes() const;

    Pair<Opt<std::string>, std::string> transform(llvm::Module &, const Function &);
  };

  std::vector<compiler::Layout> resolveLayouts(const std::vector<StructDef> &defs, const AstTransformer &xform) const;

public:
  Options options;
  explicit LLVM(Options options) : options(std::move(options)){};
  std::vector<compiler::Layout> resolveLayouts(const std::vector<StructDef> &defs, const compiler::Opt &opt) override;
  compiler::Compilation compileProgram(const Program &, const compiler::Opt &opt) override;
};

} // namespace polyregion::backend