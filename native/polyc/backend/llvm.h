#pragma once

#include <unordered_map>
#include <utility>

#include "ast.h"
#include "backend.h"
#include "llvmc.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace polyregion::backend {

using namespace polyregion::polyast;

class AMDGPUTargetSpecificHandler;
class NVPTXTargetSpecificHandler;
class CPUTargetSpecificHandler;
class OpenCLTargetSpecificHandler;

class LLVMBackend final : public Backend {
public:
  enum class Target { x86_64, AArch64, ARM, NVPTX64, AMDGCN, SPIRV32, SPIRV64 };
  struct Options {
    Target target;
    std::string arch;
    [[nodiscard]] llvmc::TargetInfo targetInfo() const;
  };

  Options options;
  explicit LLVMBackend(const Options &options);
  [[nodiscard]] std::vector<StructLayout> resolveLayouts(const std::vector<StructDef> &structs) override;
  [[nodiscard]] CompileResult compileProgram(const Program &program, const compiletime::OptLevel &opt) override;
};

namespace details {

using ValPtr = llvm::Value *;
using ValPtrFn2 = std::function<ValPtr(ValPtr, ValPtr)>;
using ValPtrFn1 = std::function<ValPtr(ValPtr)>;
using AnyType = Type::Any;
using AnyExpr = Expr::Any;
using AnyStmt = Stmt::Any;

struct StructInfo {
  StructDef def;
  StructLayout layout;
  llvm::StructType *tpe;
  Map<std::string, size_t> memberIndices;
};

struct TargetedContext {
  using AS = unsigned int;
  LLVMBackend::Options options;
  llvm::LLVMContext actual;
  AS AllocaAS = 0, GlobalAS = 0, LocalAS = 0;
  explicit TargetedContext(const LLVMBackend::Options &options);

  [[nodiscard]] llvm::Type *i32Ty();

  [[nodiscard]] AS addressSpace(const TypeSpace::Any &s) const;
  [[nodiscard]] ValPtr allocaAS(llvm::IRBuilder<> &B, llvm::Type *ty, unsigned int AS, const std::string &key) const;
  [[nodiscard]] ValPtr load(llvm::IRBuilder<> &B, ValPtr rhs, llvm::Type *ty) const;
  [[nodiscard]] ValPtr store(llvm::IRBuilder<> &B, ValPtr rhsVal, ValPtr lhsPtr) const;
  [[nodiscard]] ValPtr sizeOf(llvm::IRBuilder<> &B, llvm::Type *ptrTpe);
  [[nodiscard]] llvm::Type *resolveType(const AnyType &tpe, const Map<std::string, StructInfo> &structs, bool functionBoundary = false);
  [[nodiscard]] StructInfo resolveStruct(const StructDef &def, const Map<std::string, StructInfo> &structs);
  [[nodiscard]] Map<std::string, StructInfo> resolveLayouts(const std::vector<StructDef> &structs);
};

struct CodeGen;

struct TargetSpecificHandler {
  virtual void witnessEntry(CodeGen &gen, llvm::Function &fn) = 0;
  virtual ValPtr mkSpecVal(CodeGen &gen, const Expr::SpecOp &op) = 0;
  virtual ValPtr mkMathVal(CodeGen &gen, const Expr::MathOp &op) = 0;
  virtual ~TargetSpecificHandler();
  static std::unique_ptr<TargetSpecificHandler> from(LLVMBackend::Target target);
};

struct CodeGen {

  enum class BlockKind : uint8_t { Terminal, Normal };

  struct WhileCtx {
    llvm::BasicBlock *exit, *test;
  };

  TargetedContext C;
  std::unique_ptr<TargetSpecificHandler> targetHandler;
  llvm::IRBuilder<> B;
  llvm::Module M;

  Map<std::string, Pair<AnyType, llvm::Value *>> stackVarPtrs{};
  Map<std::string, StructInfo> structTypes{};
  Map<Signature, llvm::Function *> functions{};

  explicit CodeGen(const LLVMBackend::Options &options, const std::string &moduleName);

  [[nodiscard]] llvm::Type *resolveType(const AnyType &tpe, bool functionBoundary = false);
  [[nodiscard]] llvm::Function *resolveExtFn(const AnyType &rtn, const std::string &name, const std::vector<AnyType> &args);

  [[nodiscard]] ValPtr extFn1(const std::string &name, const AnyType &rtn, const AnyExpr &arg);
  [[nodiscard]] ValPtr extFn2(const std::string &name, const AnyType &rtn, const AnyExpr &lhs, const AnyExpr &rhs);
  [[nodiscard]] ValPtr invokeMalloc(ValPtr size);
  [[nodiscard]] ValPtr invokeAbort();
  [[nodiscard]] ValPtr intr0(llvm::Intrinsic::ID id);
  [[nodiscard]] ValPtr intr1(llvm::Intrinsic::ID id, const AnyType &overload, const AnyExpr &arg);
  [[nodiscard]] ValPtr intr2(llvm::Intrinsic::ID id, const AnyType &overload, const AnyExpr &lhs, const AnyExpr &rhs);

  [[nodiscard]] ValPtr findStackVar(const Named &named);
  [[nodiscard]] ValPtr mkSelectPtr(const Expr::Select &select);

  [[nodiscard]] ValPtr mkExprVal(const Expr::Any &expr, const std::string &key = "");
  [[nodiscard]] BlockKind mkStmt(const Stmt::Any &stmt, llvm::Function &fn, const Opt<WhileCtx> &whileCtx);

  [[nodiscard]] ValPtr unaryExpr(const AnyExpr &expr, const AnyExpr &l, const AnyType &rtn, const ValPtrFn1 &fn);
  [[nodiscard]] ValPtr binaryExpr(const AnyExpr &expr, const AnyExpr &l, const AnyExpr &r, const AnyType &rtn, const ValPtrFn2 &fn);

  [[nodiscard]] ValPtr unaryNumOp(const AnyExpr &expr, const AnyExpr &arg, const AnyType &rtn, const ValPtrFn1 &integralFn,
                                  const ValPtrFn1 &fractionalFn);
  [[nodiscard]] ValPtr binaryNumOp(const AnyExpr &expr, const AnyExpr &l, const AnyExpr &r, const AnyType &rtn, const ValPtrFn2 &integralFn,
                                   const ValPtrFn2 &fractionalFn);

  void addFn(llvm::Module &mod, const Function &f, bool entry);

  std::vector<Pair<std::string, llvm::StructType *>> getStructTypes() const;

  void transform(const Function &program);

  Pair<Opt<std::string>, std::string> transform(const Program &program);
};

} // namespace details

} // namespace polyregion::backend