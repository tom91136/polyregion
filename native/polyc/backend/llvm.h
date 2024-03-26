#pragma once

#include <unordered_map>
#include <utility>

#include "ast.h"
#include "backend.h"

#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include "llvmc.h"
// #include "utils.hpp"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

using ValPtr = llvm::Value *;
using ValPtrFn2 = std::function<ValPtr(ValPtr, ValPtr)>;
using ValPtrFn1 = std::function<ValPtr(ValPtr)>;
using AnyTerm = Term::Any;
using AnyType = Type::Any;
using AnyExpr = Expr::Any;
using AnyStmt = Stmt::Any;

class AMDGPUTargetSpecificHandler;
class NVPTXTargetSpecificHandler;
class CPUTargetSpecificHandler;
class OpenCLTargetSpecificHandler;

class LLVMBackend : public Backend {

public:
  enum class Target { x86_64, AArch64, ARM, NVPTX64, AMDGCN, SPIRV32, SPIRV64 };
  struct Options {
    Target target;
    std::string arch;
    [[nodiscard]] llvmc::TargetInfo targetInfo() const;
  };

private:
  struct WhileCtx {
    llvm::BasicBlock *exit, *test;
  };

  enum class BlockKind { Terminal, Normal };

public:
  static ValPtr sizeOf(llvm::IRBuilder<> &B, llvm::LLVMContext &C, llvm::Type *ptrTpe);

  static ValPtr allocaAS(llvm::IRBuilder<> &B, llvm::Type *ty, unsigned int AS, const std::string &key);

  class AstTransformer;

  struct TargetSpecificHandler {
    virtual void witnessEntry(AstTransformer &xform, llvm::Module &mod, llvm::Function &fn) = 0;
    virtual ValPtr mkSpecVal(AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &op) = 0;
    virtual ValPtr mkMathVal(AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &op) = 0;
    virtual ~TargetSpecificHandler();
    static std::unique_ptr<TargetSpecificHandler> from(Target target);
  };

  class AstTransformer {

    friend AMDGPUTargetSpecificHandler;
    friend NVPTXTargetSpecificHandler;
    friend CPUTargetSpecificHandler;
    friend OpenCLTargetSpecificHandler;

    Options options;
    llvm::LLVMContext &C;
    std::unique_ptr<TargetSpecificHandler> targetHandler;
    unsigned int AllocaAS = 0;
    unsigned int GlobalAS = 0;
    unsigned int LocalAS = 0;

    using StructMemberIndexTable = Map<std::string, size_t>;
    struct StructInfo {
      StructDef def;
      llvm::StructType *tpe;
      StructMemberIndexTable memberIndices;
    };
    Map<std::string, Pair<Type::Any, llvm::Value *>> stackVarPtrs;
    Map<Sym, StructInfo> structTypes;
    Map<InvokeSignature, llvm::Function *> functions;
    llvm::IRBuilder<> B;

    template <typename T>
    Opt<Pair<std::vector<llvm::StructType *>, T>> findSymbolInHeirachy( //
        const Sym &structName, std::function<Opt<T>(StructDef, llvm::StructType *, StructMemberIndexTable)> f,
        const std::vector<llvm::StructType *> &xs = {}) const;

    ValPtr load(ValPtr rhs, llvm::Type *ty);
    ValPtr store(ValPtr rhsVal, ValPtr lhsPtr);

    ValPtr findStackVar(const Named &named);
    ValPtr mkSelectPtr(const Term::Select &select);
    ValPtr mkTermVal(const Term::Any &ref);
    ValPtr mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key);
    BlockKind mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx);
    llvm::Function *mkExternalFn(llvm::Function *parent, const Type::Any &rtn, const std::string &name, const std::vector<Type::Any> &args);
    ValPtr invokeMalloc(llvm::Function *parent, ValPtr size);
    ValPtr invokeAbort(llvm::Function *parent);

    llvm::Type *mkTpe(const Type::Any &tpe, bool functionBoundary = false);

    StructInfo mkStruct(const StructDef &def);

    ValPtr unaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyType &rtn, const ValPtrFn1 &fn);
    ValPtr binaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, const ValPtrFn2 &fn);

    ValPtr unaryNumOp(const AnyExpr &expr, const AnyTerm &arg, const AnyType &rtn, const ValPtrFn1 &integralFn,
                      const ValPtrFn1 &fractionalFn);
    ValPtr binaryNumOp(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, const ValPtrFn2 &integralFn,
                       const ValPtrFn2 &fractionalFn);

    ValPtr extFn1(llvm::Function *fn, const std::string &name, const AnyType &tpe, const AnyTerm &arg);
    ValPtr extFn2(llvm::Function *fn, const std::string &name, const AnyType &tpe, const AnyTerm &lhs, const AnyTerm &rhs);
    ValPtr intr0(llvm::Function *fn, llvm::Intrinsic::ID id);
    ValPtr intr1(llvm::Function *fn, llvm::Intrinsic::ID id, const AnyType &overload, const AnyTerm &arg);
    ValPtr intr2(llvm::Function *fn, llvm::Intrinsic::ID id, const AnyType &overload, const AnyTerm &lhs, const AnyTerm &rhs);

  public:
    AstTransformer(const Options &options, llvm::LLVMContext &c);

    void addDefs(const std::vector<StructDef> &);
    void addFn(llvm::Module &mod, const Function &f, bool entry);

    std::vector<Pair<Sym, llvm::StructType *>> getStructTypes() const;

    void transform(llvm::Module &mod, const Function &program);

    Pair<Opt<std::string>, std::string> transform(llvm::Module &mod, const Program &program);
  };

  [[nodiscard]] std::vector<polyast::CompileLayout> resolveLayouts(const std::vector<StructDef> &defs, const AstTransformer &xform) const;

public:
  Options options;
  explicit LLVMBackend(Options options) : options(std::move(options)){};
  [[nodiscard]] std::vector<polyast::CompileLayout> resolveLayouts(const std::vector<StructDef> &defs,
                                                                   const compiletime::OptLevel &opt) override;
  [[nodiscard]] polyast::CompileResult compileProgram(const Program &, const compiletime::OptLevel &opt) override;
};

} // namespace polyregion::backend