#include "llvm.h"

#include <unordered_set>

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Host.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "ast.h"
#include "llvm_amdgpu.h"
#include "llvm_cpu.h"
#include "llvm_nvptx.h"
#include "llvm_spirv_cl.h"
#include "llvmc.h"

using namespace aspartame;
using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;
using namespace polyregion::backend::details;

template <typename T> static std::string llvm_tostring(const T *t) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  t->print(rso);
  return rso.str();
}

std::unique_ptr<TargetSpecificHandler> TargetSpecificHandler::from(LLVMBackend::Target target) {
  switch (target) {
    case LLVMBackend::Target::x86_64: [[fallthrough]];
    case LLVMBackend::Target::AArch64: [[fallthrough]];
    case LLVMBackend::Target::ARM: return std::make_unique<CPUTargetSpecificHandler>();
    case LLVMBackend::Target::NVPTX64: return std::make_unique<NVPTXTargetSpecificHandler>();
    case LLVMBackend::Target::AMDGCN: return std::make_unique<AMDGPUTargetSpecificHandler>();
    case LLVMBackend::Target::SPIRV32_Kernel: [[fallthrough]];
    case LLVMBackend::Target::SPIRV64_Kernel: [[fallthrough]];
    case LLVMBackend::Target::SPIRV_GLCompute: return std::make_unique<SPIRVOpenCLTargetSpecificHandler>();
    default: throw BackendException(fmt::format("Unknown target {}", magic_enum::enum_name(target)));
  }
}

TargetSpecificHandler::~TargetSpecificHandler() = default;

static bool isUnsigned(const Type::Any &tpe) { // the only unsigned type in PolyAst
  return tpe.is<Type::IntU8>() || tpe.is<Type::IntU16>() || tpe.is<Type::IntU32>() || tpe.is<Type::IntU64>();
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

CodeGen::CodeGen(const LLVMBackend::Options &options, const std::string &moduleName)
    : C(options), targetHandler(TargetSpecificHandler::from(options.target)), B(C.actual), M(moduleName, C.actual) {}

llvm::Type *CodeGen::resolveType(const AnyType &tpe, const bool functionBoundary, const bool kernelEntryArg) {
  return C.resolveType(tpe, structTypes, functionBoundary, kernelEntryArg);
}

llvm::Value *CodeGen::byteOffsetPtr(llvm::Value *base, llvm::Value *byteOff, const std::string &name) {
  auto *baseInt = B.CreatePtrToInt(base, C.i64Ty());
  auto *newInt = B.CreateAdd(baseInt, byteOff);
  return B.CreateIntToPtr(newInt, base->getType(), name);
}

llvm::Value *CodeGen::i64SExt(llvm::Value *v) { return B.CreateSExtOrTrunc(v, C.i64Ty()); }

llvm::Function *CodeGen::resolveExtFn(const Type::Any &rtn, const std::string &name, const std::vector<Type::Any> &args) {
  return functions ^= get_or_emplace(Signature(Sym({name}), {}, {}, args, {}, {}, rtn), [&](auto &sig) -> llvm::Function * {
           auto tpe = llvm::FunctionType::get(
               /*Result*/ resolveType(rtn, true),
               /*Params*/ args ^ map([&](auto &t) { return resolveType(t, true); }),
               /*isVarArg*/ false);
           auto fn = llvm::Function::Create(tpe, llvm::Function::ExternalLinkage, name, M);
           return fn;
         });
}

ValPtr CodeGen::invokeMalloc(ValPtr size) {
  return B.CreateCall(resolveExtFn(Type::Ptr(Type::IntS8(), TypeSpace::Global()), "malloc", {Type::IntS64()}), size);
}
ValPtr CodeGen::invokeAbort() { return B.CreateCall(resolveExtFn(Type::Nothing(), "abort", {})); }

ValPtr CodeGen::extFn1(const std::string &name, const AnyType &rtn, const AnyTerm &arg) { //
  const auto fn = resolveExtFn(rtn, name, {arg.tpe()});
  if (C.isSpirv()) fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  if (!rtn.is<Type::Unit0>()) fn->addFnAttr(llvm::Attribute::WillReturn);
  const auto call = B.CreateCall(fn, mkTermVal(arg));
  call->setCallingConv(fn->getCallingConv());
  return call;
}
ValPtr CodeGen::extFn2(const std::string &name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
  const auto fn = resolveExtFn(rtn, name, {lhs.tpe(), rhs.tpe()});
  if (C.isSpirv()) {
    fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    fn->addFnAttr(llvm::Attribute::NoBuiltin);
    fn->addFnAttr(llvm::Attribute::Convergent);
  }
  const auto call = B.CreateCall(fn, {mkTermVal(lhs), mkTermVal(rhs)});
  call->setCallingConv(fn->getCallingConv());
  return call;
}
ValPtr CodeGen::intr0(const llvm::Intrinsic::ID id) { //
  const auto callee = llvm::Intrinsic::getOrInsertDeclaration(&M, id, {});
  return B.CreateCall(callee);
}
ValPtr CodeGen::intr1(const llvm::Intrinsic::ID id, const AnyType &overload, const AnyTerm &arg) { //
  const auto callee = llvm::Intrinsic::getOrInsertDeclaration(&M, id, resolveType(overload));
  return B.CreateCall(callee, mkTermVal(arg));
}
ValPtr CodeGen::intr2(const llvm::Intrinsic::ID id, const AnyType &overload, //
                      const AnyTerm &lhs, const AnyTerm &rhs) {              //
  const auto callee = llvm::Intrinsic::getOrInsertDeclaration(&M, id, resolveType(overload));
  return B.CreateCall(callee, {mkTermVal(lhs), mkTermVal(rhs)});
}

ValPtr CodeGen::findStackVar(const Named &named) {
  if (named.tpe.is<Type::Unit0>()) return mkTermVal(Term::Unit0Const());
  // Nothing-typed names are absent from stackVarPtrs (FunctionType::get rejects void params).
  // Return a pointer-typed poison so synthetic refs from the rewriter compile - downstream
  // GEP/load expect a pointer slot.
  if (named.tpe.is<Type::Nothing>()) return llvm::PoisonValue::get(llvm::PointerType::getUnqual(C.actual));
  //  check the LUT table for known variables defined by var or brought in scope by parameters
  return stackVarPtrs ^ get_maybe(named.symbol) ^
         fold(
             [&](auto &tpe, auto &value) {
               if (named.tpe != tpe)
                 throw BackendException("Named local variable (" + to_string(named) + ") has different type from LUT (" + to_string(tpe) +
                                        ")");
               return value;
             },
             [&]() -> ValPtr {
               auto pool = stackVarPtrs | mk_string("\n", "\n", "\n", [](auto &k, auto &v) {
                             auto &[tpe, ir] = v;
                             return " -> `" + k + "` = " + to_string(tpe) + "(IR=" + llvm_tostring(ir) + ")";
                           });
               throw BackendException("Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
             });
}

ValPtr CodeGen::mkSelectPtr(const Term::Select &select) {

  auto fail = [&] { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> StructInfo {
    auto findTy = [&](const Type::Struct &s) -> StructInfo {
      return structTypes ^ get_maybe(repr(s.name)) ^
             fold([&]() -> StructInfo { throw BackendException("Unseen struct type " + to_string(s.name) + " in select path" + fail()); });
    };

    if (auto s = tpe.get<Type::Struct>(); s) {
      return findTy(*s);
    } else if (auto p = tpe.get<Type::Ptr>(); p) {
      if (auto _s = p->comp.get<Type::Struct>(); _s) return findTy(*_s);
      else
        throw BackendException("Illegal select path involving pointer to non-struct type " + to_string(s->name) + " in select path" +
                               fail());
    } else throw BackendException("Illegal select path involving non-struct type " + to_string(tpe) + fail());
  };

  if (select.steps.empty()) return findStackVar(select.root); // local var lookup
  auto tpe = select.root.tpe;
  auto root = findStackVar(select.root);

  for (auto &step : select.steps) {
    if (step.template is<PathStep::Deref>()) {
      if (auto p = tpe.template get<Type::Ptr>()) {
        root = C.load(B, root, C.loadedPtrTy(B));
        tpe = p->comp;
        continue;
      }
      throw BackendException("Deref step on non-pointer type " + to_string(tpe) + fail());
    }
    const auto fieldStep = step.template get<PathStep::Field>();
    if (!fieldStep) throw BackendException("Unhandled PathStep variant" + fail());
    // A Field on a Ptr type means implicit deref (load) then GEP.
    if (tpe.template is<Type::Ptr>()) {
      root = C.load(B, root, C.loadedPtrTy(B));
      tpe = tpe.template get<Type::Ptr>()->comp;
    }
    const auto info = structTypeOf(tpe);
    const auto idxOpt = info.memberIndices ^ get_maybe(fieldStep->name);
    if (!idxOpt) {
      auto pool =
          info.memberIndices | mk_string("\n", "\n", "\n", [](auto &k, auto &v) { return " -> `" + k + "` = " + std::to_string(v) + ")"; });
      throw BackendException("Illegal select path with unknown struct member index of name `" + fieldStep->name + "`, pool=" + pool +
                             fail());
    }
    const auto idx = *idxOpt;
    if (spirvStructByMemcpy()) {
      const auto offsetBytes = static_cast<size_t>(info.layout.members[idx].offsetInBytes);
      auto *off = llvm::ConstantInt::get(C.i64Ty(), offsetBytes);
      root = byteOffsetPtr(root, off, qualified(select) + "_select_ptr");
    } else {
      root = B.CreateInBoundsGEP(info.tpe, root, {llvm::ConstantInt::get(C.i32Ty(), 0), llvm::ConstantInt::get(C.i32Ty(), idx)},
                                 qualified(select) + "_select_ptr");
    }
    // Advance the type cursor to the member's declared type (looked up via StructDef).
    if (idx < info.def.members.size()) {
      tpe = info.def.members[idx].tpe;
    }
    // If the lowered field is a pointer wrapper of a Struct (functionBoundary lowering), deref.
    const auto fieldLlvmType = info.tpe->getElementType(idx);
    if (fieldLlvmType->isPointerTy() && tpe.template is<Type::Struct>()) {
      root = C.load(B, root, C.loadedPtrTy(B));
    }
  }
  return root;
}

ValPtr CodeGen::mkTermVal(const Term::Any &term, const std::string &key) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return term.match_total( //
      [&](const Term::Float16Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getHalfTy(C.actual), x.value); },
      [&](const Term::Float32Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getFloatTy(C.actual), x.value); },
      [&](const Term::Float64Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getDoubleTy(C.actual), x.value); },

      [&](const Term::IntU8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C.actual), x.value); },
      [&](const Term::IntU16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C.actual), x.value); },
      [&](const Term::IntU32Const &x) -> ValPtr { return ConstantInt::get(C.i32Ty(), x.value); },
      [&](const Term::IntU64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C.actual), x.value); },

      [&](const Term::IntS8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C.actual), x.value); },
      [&](const Term::IntS16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C.actual), x.value); },
      [&](const Term::IntS32Const &x) -> ValPtr { return ConstantInt::get(C.i32Ty(), x.value); },
      [&](const Term::IntS64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C.actual), x.value); },

      [&](const Term::Unit0Const &) -> ValPtr { return ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0); },
      [&](const Term::Bool1Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt1Ty(C.actual), x.value); },
      [&](const Term::NullPtrConst &x) -> ValPtr {
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(C.actual, C.addressSpace(x.space)));
      },
      [&](const Term::Poison &x) -> ValPtr {
        // Pointer poison maps to null (analyses treat it as poison-equivalent); other types use
        // PoisonValue so non-pointer Poison nodes from the rewriter do not abort codegen.
        auto tpe = resolveType(x.t);
        if (llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        }
        return llvm::PoisonValue::get(tpe);
      },
      [&](const Term::Select &x) -> ValPtr {
        if (x.tpe.is<Type::Unit0>()) return mkTermVal(Term::Unit0Const());
        // Arr lowering is location-dependent: as a function arg / local stack var (no path
        // steps) the alloca slot holds a `ptr` we must load; as a struct field it stays inline
        // as `[N x T]` and mkSelectPtr already returns the array address.
        if (x.tpe.template is<Type::Arr>()) {
          if (x.steps.empty()) {
            // XXX Stmt::Var stores a FLAT ptr; load as FLAT to round-trip (Generic on SPIR-V).
            const auto loadAS = C.GenericAS != 0 ? C.GenericAS : 0;
            return C.load(B, mkSelectPtr(x), B.getPtrTy(loadAS));
          }
          return mkSelectPtr(x);
        }
        if (spirvStructByMemcpy() && x.tpe.template is<Type::Struct>()) return mkSelectPtr(x);
        return C.load(B, mkSelectPtr(x), resolveType(x.tpe));
      });
}

ValPtr CodeGen::mkExprVal(const Expr::Any &expr, const std::string &key) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return expr.match_total( //
      [&](const Expr::Alias &x) -> ValPtr { return mkTermVal(x.ref, key); },
      [&](const Expr::SpecOp &x) -> ValPtr { return targetHandler->mkSpecVal(*this, x); },
      [&](const Expr::MathOp &x) -> ValPtr { return targetHandler->mkMathVal(*this, x); },
      [&](const Expr::IntrOp &x) -> ValPtr {
        auto intr = x.op;
        return intr.match_total( //
            [&](const Intr::BNot &v) -> ValPtr { return unaryExpr(expr, v.x, v.tpe, [&](auto x) { return B.CreateNot(x); }); },
            [&](const Intr::LogicNot &v) -> ValPtr { return B.CreateNot(mkTermVal(v.x)); },
            [&](const Intr::Pos &v) -> ValPtr {
              return unaryNumOp(expr, v.x, v.tpe, [&](auto x) { return x; }, [&](auto x) { return x; });
            },
            [&](const Intr::Neg &v) -> ValPtr {
              return unaryNumOp(expr, v.x, v.tpe, [&](auto x) { return B.CreateNeg(x); }, [&](auto x) { return B.CreateFNeg(x); });
            },
            [&](const Intr::Add &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateAdd(l, r); }, [&](auto l, auto r) { return B.CreateFAdd(l, r); });
            },
            [&](const Intr::Sub &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateSub(l, r); }, [&](auto l, auto r) { return B.CreateFSub(l, r); });
            },
            [&](const Intr::Mul &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateMul(l, r); }, [&](auto l, auto r) { return B.CreateFMul(l, r); });
            },
            [&](const Intr::Div &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateSDiv(l, r); }, [&](auto l, auto r) { return B.CreateFDiv(l, r); });
            },
            [&](const Intr::Rem &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateSRem(l, r); }, [&](auto l, auto r) { return B.CreateFRem(l, r); });
            },
            [&](const Intr::Min &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateSelect(B.CreateICmpSLT(l, r), l, r); },
                  [&](auto l, auto r) { return B.CreateMinimum(l, r); });
            },
            [&](const Intr::Max &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](auto l, auto r) { return B.CreateSelect(B.CreateICmpSLT(l, r), r, l); },
                  [&](auto l, auto r) { return B.CreateMaximum(l, r); });
            }, //
            [&](const Intr::BAnd &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateAnd(l, r); });
            },
            [&](const Intr::BOr &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateOr(l, r); });
            },
            [&](const Intr::BXor &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateXor(l, r); });
            },
            [&](const Intr::BSL &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateShl(l, r); });
            },
            [&](const Intr::BSR &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateAShr(l, r); });
            },
            [&](const Intr::BZSR &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](auto l, auto r) { return B.CreateLShr(l, r); });
            },                                                                                                     //
            [&](const Intr::LogicAnd &v) -> ValPtr { return B.CreateLogicalAnd(mkTermVal(v.x), mkTermVal(v.y)); }, //
            [&](const Intr::LogicOr &v) -> ValPtr { return B.CreateLogicalOr(mkTermVal(v.x), mkTermVal(v.y)); },   //
            [&](const Intr::LogicEq &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpEQ(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOEQ(l, r); });
            },
            [&](const Intr::LogicNeq &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpNE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpONE(l, r); });
            },
            [&](const Intr::LogicLte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpSLE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOLE(l, r); });
            },
            [&](const Intr::LogicGte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpSGE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOGE(l, r); });
            },
            [&](const Intr::LogicLt &v) -> ValPtr {
              // Signed less-than: matches LogicLte/LogicGte/LogicGt which all use signed
              // comparison. ICmpULT here would treat negative ints as huge unsigned values
              // (e.g. `-1 < 10` reads as `0xFFFFFFFF < 10 == false`), breaking any kernel
              // loop whose induction variable goes negative.
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpSLT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOLT(l, r); });
            },
            [&](const Intr::LogicGt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpSGT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOGT(l, r); });
            });
      },

      [&](const Expr::Cast &x) -> ValPtr {
        // we only allow widening or narrowing of integral and fractional types
        // pointers are not allowed to participate on either end
        auto from = mkTermVal(x.from);
        auto fromTpe = resolveType(x.from.tpe());
        auto toTpe = resolveType(x.as);
        enum class NumKind { Fractional, Integral };

        // Same type
        if (x.as == x.from.tpe()) return from;

        // Allow any pointer casts of struct
        if (const auto rhsPtr = x.from.tpe().get<Type::Ptr>()) {
          if (const auto lhsPtr = x.as.get<Type::Ptr>()) {
            // TODO check layout and loss of information
            // Cross-AS pointer casts need an explicit addrspacecast (e.g. NVPTX `addrspace(3)`
            // shared -> generic), otherwise the AS is silently dropped on the next load and
            // shared accesses degrade to generic stores.
            const auto fromAS = C.addressSpace(rhsPtr->space);
            const auto toAS = C.addressSpace(lhsPtr->space);
            if (fromAS != toAS) return B.CreateAddrSpaceCast(from, toTpe);
            return from;
          }
        }

        // Struct-to-struct casts (e.g. dispatch upcast `Monoid -> anon$1`): both sides are
        // opaque pointers at the boundary, so we just pass the pointer through.
        if (x.from.tpe().is<Type::Struct>() && x.as.is<Type::Struct>()) {
          // The expression is materialised as a select-pointer (the struct's stack address);
          // grab that instead of the loaded struct value.
          if (const auto sel = x.from.template get<Term::Select>()) {
            return mkSelectPtr(*sel);
          }
          return from;
        }

        // Casts to/from a None-kind type (Nothing/Unit0/Exec) are no-ops: void-shaped types carry no value.
        if (x.from.tpe().kind().is<TypeKind::None>() || x.as.kind().is<TypeKind::None>()) {
          return from;
        }

        // Disallowed on Logical SPIR-V; permitted elsewhere.
        if (x.from.tpe().is<Type::Ptr>() && x.as.kind().is<TypeKind::Integral>()) {
          return B.CreatePtrToInt(from, toTpe);
        }

        auto fromKind = x.from.tpe().kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException("Semantic error: conversion from ref type (" + llvm_tostring(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw BackendException("none!?"); });

        auto toKind = x.as.kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException("Semantic error: conversion to ref type (" + llvm_tostring(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw BackendException("none!?"); });

        if (fromKind == NumKind::Fractional && toKind == NumKind::Integral) {

          // to the equally sized integral type first if narrowing; XXX narrowing directly produces a poison value

          auto min32BitIntBits = std::max<llvm::TypeSize::ScalarTy>(32, toTpe->getPrimitiveSizeInBits());
          auto toTpeMaxInFp = llvm::ConstantFP::get(fromTpe, double(nIntMax(min32BitIntBits)));
          auto toTpeMinInFp = llvm::ConstantFP::get(fromTpe, double(nIntMin(min32BitIntBits)));
          auto min32BitIntTy = llvm::Type::getIntNTy(C.actual, min32BitIntBits);
          auto toTpeMaxInInt = llvm::ConstantInt::get(min32BitIntTy, nIntMax(min32BitIntBits));
          auto toTpeMinInInt = llvm::ConstantInt::get(min32BitIntTy, nIntMin(min32BitIntBits));

          ValPtr c = B.CreateSelect(B.CreateFCmpOGE(from, toTpeMaxInFp), toTpeMaxInInt,                //
                                    B.CreateSelect(B.CreateFCmpOLE(from, toTpeMinInFp), toTpeMinInInt, //
                                                   B.CreateFPToSI(from, min32BitIntTy)));
          c = B.CreateIntCast(c, toTpe, !isUnsigned(x.as));

          auto zero = llvm::ConstantInt::get(toTpe, 0);
          return B.CreateSelect(B.CreateFCmpUNO(from, from), zero, c);
        } else if (fromKind == NumKind::Integral && toKind == NumKind::Fractional) {
          // XXX this is a *widening* conversion, even though we may lose precision
          // XXX here the result is rounded using the default rounding mode so the dest bit width doesn't matter
          return isUnsigned(x.from.tpe()) ? B.CreateUIToFP(from, toTpe) : B.CreateSIToFP(from, toTpe);
        } else if (fromKind == NumKind::Integral && toKind == NumKind::Integral) {
          return B.CreateIntCast(from, toTpe, !isUnsigned(x.from.tpe()), "integral_cast");
        } else if (fromKind == NumKind::Fractional && toKind == NumKind::Fractional) {
          return B.CreateFPCast(from, toTpe, "fractional_cast");
        } else throw BackendException("unhandled cast");
      },
      [&](const Expr::Invoke &x) -> ValPtr {
        Vector<AnyTerm> allArgs;
        if (x.receiver) allArgs.push_back(*x.receiver);
        for (auto &a : x.args)
          allArgs.push_back(a);
        // Mirror the declaration filter: drop Unit0/Nothing args; both lower to LLVM void at the boundary.
        const auto argNoUnit = allArgs ^ filter([](auto &arg) {
                                 return !arg.tpe().template is<Type::Unit0>() //
                                        && !arg.tpe().template is<Type::Nothing>();
                               });
        const auto sig = Signature(x.name, /*tpeVars*/ {}, /*receiver*/ {}, argNoUnit ^ map([](auto &arg) { return arg.tpe(); }),
                                   /*moduleCaptures*/ {}, /*termCaptures*/ {}, x.rtn);
        return functions ^ get_maybe(sig) ^
               fold(
                   [&](auto &fn) -> ValPtr {
                     auto params =
                         argNoUnit ^ map([&](auto &term) -> ValPtr {
                           if (term.tpe().template is<Type::Struct>()) {
                             if (auto sel = term.template get<Term::Select>()) return mkSelectPtr(*sel);
                           }
                           const auto val = mkTermVal(term);
                           return term.tpe().template is<Type::Bool1>() ? B.CreateZExt(val, resolveType(Type::Bool1(), true)) : val;
                         });
                     const bool calleeUsesSret = fn->arg_size() > 0 && fn->getArg(0)->hasStructRetAttr();
                     llvm::Value *sretSlot = nullptr;
                     if (calleeUsesSret) {
                       auto *sretSlotTy = resolveType(x.rtn, /*functionBoundary*/ false);
                       sretSlot = C.allocaAS(B, sretSlotTy, C.AllocaAS, "sret_slot");
                       params.insert(params.begin(), sretSlot);
                     }
                     // SPIR-V: widen Function/CrossWorkgroup pointers to the formal's Generic AS;
                     // the reverse direction is UB. AMDGCN/NVPTX have no Generic AS.
                     if (C.GenericAS != 0) {
                       for (size_t i = 0; i < params.size(); ++i) {
                         auto *formal = fn->getFunctionType()->getParamType(i);
                         auto *actual = params[i]->getType();
                         if (formal != actual && formal->isPointerTy() && actual->isPointerTy())
                           params[i] = B.CreateAddrSpaceCast(params[i], formal);
                       }
                     }
                     const auto call = B.CreateCall(fn, params);
                     if (calleeUsesSret) return sretSlot;
                     return x.rtn.is<Type::Unit0>() ? mkTermVal(Term::Unit0Const()) : call;
                   },
                   [&]() -> ValPtr {
                     throw BackendException(fmt::format("Unhandled invocation {}, known functions are:\n{}", repr(sig),
                                                        functions | keys() | mk_string("\n -> ", show_repr)));
                   });
      },
      [&](const Expr::Index &x) -> ValPtr {
        // SPIR-V backend treats access-chain Element as unsigned, so a 32-bit -1 becomes a
        // ~16 GB forward step. Sign-extending to i64 forces a true signed -1 bit pattern.
        if (const auto lhs = x.lhs.template get<Term::Select>()) {
          if (const auto arrTpe = lhs->tpe.template get<Type::Ptr>()) {
            if (arrTpe->comp.is<Type::Unit0>()) {
              const auto val = mkTermVal(Term::Unit0Const());
              B.CreateInBoundsGEP(val->getType(), mkTermVal(*lhs), i64SExt(mkTermVal(x.idx)), key + "_ptr");
              return val;
            } else if (auto innerArr = arrTpe->comp.get<Type::Arr>()) {
              const auto arrLlvmTy = resolveType(*innerArr);
              const auto compLlvmTy = resolveType(innerArr->comp);
              const auto basePtr = mkTermVal(*lhs);
              const auto ptr =
                  B.CreateInBoundsGEP(arrLlvmTy, basePtr, {ConstantInt::get(C.i32Ty(), 0), i64SExt(mkTermVal(x.idx))}, key + "_idx_ptr");
              if (innerArr->comp.is<Type::Bool1>()) {
                return B.CreateICmpNE(C.load(B, ptr, compLlvmTy), ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0, true));
              }
              return C.load(B, ptr, compLlvmTy);
            } else {
              const auto ty = resolveType(arrTpe->comp);
              auto *basePtr = mkTermVal(*lhs);
              llvm::Value *ptr;
              if (spirvStructByMemcpy()) {
                auto *elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(ty));
                auto *byteOff = B.CreateMul(i64SExt(mkTermVal(x.idx)), elemSize);
                ptr = byteOffsetPtr(basePtr, byteOff, key + "_idx_ptr");
              } else {
                ptr = B.CreateInBoundsGEP(ty, basePtr, i64SExt(mkTermVal(x.idx)), key + "_idx_ptr");
              }
              if (arrTpe->comp.is<Type::Bool1>()) {
                return B.CreateICmpNE(C.load(B, ptr, ty), ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0, true));
              }
              if (spirvStructByMemcpy() && arrTpe->comp.template is<Type::Struct>()) return ptr;
              return C.load(B, ptr, ty);
            }
          } else if (const auto arrTpe = lhs->tpe.template get<Type::Arr>()) {
            const auto ty = resolveType(*arrTpe);
            const auto ptr =
                B.CreateInBoundsGEP(ty, mkTermVal(*lhs), {ConstantInt::get(C.i32Ty(), 0), i64SExt(mkTermVal(x.idx))}, key + "_idx_ptr");
            if (spirvStructByMemcpy() && arrTpe->comp.template is<Type::Struct>()) return ptr;
            return C.load(B, ptr, resolveType(arrTpe->comp));
          } else {
            throw BackendException("Semantic error: array index not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else throw BackendException("Semantic error: LHS of " + to_string(x) + " (index) is not a select");
      },
      [&](const Expr::RefTo &x) -> ValPtr {
        if (auto lhs = x.lhs.template get<Term::Select>()) {
          if (auto arrTpe = lhs->tpe.template get<Type::Ptr>(); arrTpe) {
            auto offset = x.idx ? i64SExt(mkTermVal(*x.idx)) : llvm::ConstantInt::get(C.i64Ty(), 0, true);
            if (auto innerArr = arrTpe->comp.get<Type::Arr>()) {
              auto arrLlvmTy = resolveType(*innerArr);
              return B.CreateGEP(arrLlvmTy, mkTermVal(*lhs), {llvm::ConstantInt::get(C.i32Ty(), 0), offset}, key + "_ref_to_ptr_arr");
            }
            auto ty = arrTpe->comp.is<Type::Unit0>() ? llvm::Type::getInt8Ty(C.actual) : resolveType(arrTpe->comp);
            auto *base = mkTermVal(*lhs);
            // XXX Kernel SPIR-V only: Intel Arc OpenCL mis-handles negative OpPtrAccessChain
            // elements; the ptrtoint round-trip is the workaround. Logical SPIR-V forbids
            // OpConvertPtrToU/UToPtr; other targets don't need the workaround.
            if (C.isSpirvKernel()) {
              auto elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(ty));
              auto *byteOffset = B.CreateMul(offset, elemSize);
              return byteOffsetPtr(base, byteOffset, key + "_ref_to_ptr");
            }
            return B.CreateInBoundsGEP(ty, base, offset, key + "_ref_to_ptr");
          } else if (auto arrTpe = lhs->tpe.template get<Type::Arr>(); arrTpe) {
            auto offset = x.idx ? i64SExt(mkTermVal(*x.idx)) : llvm::ConstantInt::get(C.i64Ty(), 0, true);
            auto arrLlvmTy = resolveType(*arrTpe);
            return B.CreateInBoundsGEP(arrLlvmTy, mkTermVal(*lhs), {llvm::ConstantInt::get(C.i32Ty(), 0), offset},
                                       key + "_ref_to_" + llvm_tostring(arrLlvmTy));
          } else {
            if (x.idx) throw BackendException("Semantic error: Cannot take reference of scalar with index in " + to_string(x));
            if (lhs->tpe.is<Type::Unit0>())
              throw BackendException("Semantic error: Cannot take reference of an select with unit type in " + to_string(x));
            return mkSelectPtr(*lhs);
          }
        } else
          throw BackendException("Semantic error: LHS of " + to_string(x) + " (index) is not a select, can't take reference of a constant");
      },
      [&](const Expr::Alloc &x) -> ValPtr { //
        const auto componentTpe = B.getPtrTy(0);
        const auto size = mkTermVal(x.size);
        const auto elemSize = C.sizeOf(B, componentTpe);
        const auto ptr = invokeMalloc(B.CreateMul(B.CreateIntCast(size, resolveType(Type::IntS64()), true), elemSize));
        return B.CreateBitCast(ptr, componentTpe);
      });
}

CodeGen::BlockKind CodeGen::mkStmt(const Stmt::Any &stmt, llvm::Function &fn, const Opt<WhileCtx> &whileCtx = {}) {
  return stmt.match_total(
      [&](const Stmt::Var &x) -> BlockKind {
        // [T : ref] =>> t:T  = _        ; lut += &t
        // [T : ref] =>> t:T* = &(rhs:T) ; lut += t
        // [T : val] =>> t:T  =   rhs:T  ; lut += &t
        if (x.expr && x.expr->tpe() != x.name.tpe) {
          throw BackendException("Semantic error: name type " + to_string(x.name.tpe) + " and rhs expr type " + to_string(x.expr->tpe()) +
                                 " mismatch (" + repr(x) + ")");
        }

        if (x.name.tpe.is<Type::Unit0>()) {
          // Unit0 declaration, discard declaration but keep RHS effect.
          if (x.expr) auto _ = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
        } else {
          const auto tpe = resolveType(x.name.tpe);
          auto stackPtr = C.allocaAS(B, tpe, C.AllocaAS, x.name.symbol + "_stack_ptr");
          // XXX Inline Type::Arr needs a FLAT ptr slot (AllocaAS is 32-bit on AMDGCN so the 64-bit FLAT store overflows).
          if (x.name.tpe.template is<Type::Arr>()) {
            auto refSlot = C.allocaAS(B, B.getPtrTy(), C.AllocaAS, x.name.symbol + "_ref_ptr");
            const auto _ = C.store(B, stackPtr, refSlot);
            stackPtr = refSlot;
          }
          // Rebind on same-name redeclaration (adjacent `for (int l = 0; ...)` loops);
          // `emplace` would keep the prior slot and the second loop would see the stale value.
          if (auto it = stackVarPtrs.find(x.name.symbol); it != stackVarPtrs.end() && it->second.first != x.name.tpe) {
            throw BackendException("Re-declaration of " + x.name.symbol + " changes type from " + to_string(it->second.first) + " to " +
                                   to_string(x.name.tpe));
          }
          stackVarPtrs.insert_or_assign(x.name.symbol, Pair<Type::Any, llvm::Value *>{x.name.tpe, stackPtr});
          if (x.expr) {
            auto rhs = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
            if (spirvStructByMemcpy() && x.name.tpe.template is<Type::Struct>()) {
              const auto &info = structTypes.at(repr(x.name.tpe.template get<Type::Struct>()->name));
              B.CreateMemCpy(stackPtr, llvm::MaybeAlign(info.layout.alignment), rhs, llvm::MaybeAlign(info.layout.alignment),
                             info.layout.sizeInBytes);
            } else {
              if (tpe->isPointerTy() && rhs->getType()->isPointerTy() && rhs->getType() != tpe) rhs = B.CreateAddrSpaceCast(rhs, tpe);
              const auto _ = C.store(B, rhs, stackPtr); //
            }
          }
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::Mut &x) -> BlockKind {
        // [T : ref]        =>> t   := &(rhs:T) ; lut += t
        // [T : ref {u: U}] =>> t.u := &(rhs:U)
        // [T : val]        =>> t   :=   rhs:T
        const auto &lhs = x.name;
        if (x.expr.tpe() != lhs.tpe) {
          throw BackendException("Semantic error: name type (" + to_string(x.expr.tpe()) + ") and rhs expr (" + to_string(lhs.tpe) +
                                 ") mismatch (" + repr(x) + ")");
        }
        if (lhs.tpe.is<Type::Unit0>()) return BlockKind::Normal;
        auto rhs = mkExprVal(x.expr, qualified(lhs) + "_mut");
        const auto dst = lhs.steps.empty() ? findStackVar(lhs.root) : mkSelectPtr(lhs);
        if (spirvStructByMemcpy() && lhs.tpe.template is<Type::Struct>()) {
          const auto &info = structTypes.at(repr(lhs.tpe.template get<Type::Struct>()->name));
          B.CreateMemCpy(dst, llvm::MaybeAlign(info.layout.alignment), rhs, llvm::MaybeAlign(info.layout.alignment),
                         info.layout.sizeInBytes);
          return BlockKind::Normal;
        }
        const auto slotTpe = resolveType(lhs.tpe);
        if (slotTpe->isPointerTy() && rhs->getType()->isPointerTy() && rhs->getType() != slotTpe) rhs = B.CreateAddrSpaceCast(rhs, slotTpe);
        const auto _ = C.store(B, rhs, dst);
        return BlockKind::Normal;
      },
      [&](const Stmt::Update &x) -> BlockKind {
        const auto &lhs = x.lhs;
        const auto compTpe = [&]() -> Opt<Type::Any> {
          if (auto p = lhs.tpe.template get<Type::Ptr>()) return p->comp;
          if (auto a = lhs.tpe.template get<Type::Arr>()) return a->comp;
          return {};
        }();
        if (!compTpe) {
          throw BackendException("Semantic error: array update not called on array type (" + to_string(lhs.tpe) + ")(" + repr(x) + ")");
        }
        if (*compTpe != x.value.tpe()) {
          throw BackendException("Semantic error: array comp type (" + to_string(*compTpe) + ") and rhs term (" + to_string(x.value.tpe()) +
                                 ") mismatch (" + repr(x) + ")");
        }
        // XXX Unit0 store: no-op. Host storage may be a JVM Object[]; a byte write clobbers the first ref.
        if (x.value.tpe().template is<Type::Unit0>()) return BlockKind::Normal;
        const bool componentIsSizedArray = lhs.tpe.template is<Type::Arr>();
        const auto dest = mkTermVal(lhs);
        // Arr GEP indexes the array type itself with [0, idx]; Ptr GEP indexes the comp type with [idx].
        const auto valTy = x.value.tpe().template is<Type::Bool1>() ? llvm::Type::getInt8Ty(C.actual) : resolveType(x.value.tpe());
        const auto gepTy = componentIsSizedArray ? resolveType(lhs.tpe) : valTy;
        const auto getPtr = [&]() -> llvm::Value * {
          if (componentIsSizedArray) {
            return B.CreateInBoundsGEP(gepTy, dest, {llvm::ConstantInt::get(C.i32Ty(), 0), mkTermVal(x.idx)},
                                       qualified(lhs) + "_update_ptr");
          }
          if (spirvStructByMemcpy()) {
            auto *elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(valTy));
            auto *byteOff = B.CreateMul(i64SExt(mkTermVal(x.idx)), elemSize);
            return byteOffsetPtr(dest, byteOff, qualified(lhs) + "_update_ptr");
          }
          return B.CreateInBoundsGEP(gepTy, dest, {mkTermVal(x.idx)}, qualified(lhs) + "_update_ptr");
        };
        const auto ptr = getPtr();
        if (x.value.tpe().template is<Type::Bool1>()) {
          const auto _ = C.store(B, B.CreateIntCast(mkTermVal(x.value), valTy, true), ptr);
        } else {
          const auto _ = C.store(B, mkTermVal(x.value), ptr);
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::While &x) -> BlockKind {
        const auto loopTest = llvm::BasicBlock::Create(C.actual, "loop_test", &fn);
        const auto loopBody = llvm::BasicBlock::Create(C.actual, "loop_body", &fn);
        const auto loopExit = llvm::BasicBlock::Create(C.actual, "loop_exit", &fn);
        WhileCtx ctx{.exit = loopExit, .test = loopTest};
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          const auto continue_ = mkTermVal(x.cond);
          B.CreateCondBr(continue_, loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          auto kind = BlockKind::Normal;
          for (auto &body : x.body)
            kind = mkStmt(body, fn, {ctx});
          if (kind != BlockKind::Terminal) B.CreateBr(loopTest);
        }
        // The loopExit block is a normal continuation point — `loop_test` falls through to it
        // when the condition first turns false. Return `Normal` (not `Terminal`) so that the
        // caller knows the current block isn't yet closed; otherwise we may emerge from an
        // enclosing Cond branch with `kind == Terminal` and skip emitting a branch into the
        // surrounding cond_exit, leaving loopExit dangling without a terminator.
        B.SetInsertPoint(loopExit);
        return BlockKind::Normal;
      },
      [&](const Stmt::ForRange &x) -> BlockKind {
        const auto loopTest = llvm::BasicBlock::Create(C.actual, "loop_test", &fn);
        const auto loopBody = llvm::BasicBlock::Create(C.actual, "loop_body", &fn);
        const auto loopExit = llvm::BasicBlock::Create(C.actual, "loop_exit", &fn);
        const auto inductionSelect = Term::Select(x.induction, {}, x.induction.tpe);
        const auto inductionTerm = Term::Any(inductionSelect);
        auto _alloc = mkStmt(Stmt::Var(x.induction, std::optional<Expr::Any>{}, /*isMutable*/ true), fn, whileCtx);
        auto _ = mkStmt(Stmt::Mut(inductionSelect, Expr::Alias(x.lbIncl)), fn, whileCtx);
        WhileCtx ctx{.exit = loopExit, .test = loopTest};
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          B.CreateCondBr(mkExprVal(Expr::IntrOp(Intr::LogicLt(inductionTerm, x.ubExcl))), loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          auto kind = BlockKind::Normal;
          for (auto &body : x.body)
            kind = mkStmt(body, fn, {ctx});
          if (kind != BlockKind::Terminal) {
            [[maybe_unused]] auto _0 =
                mkStmt(Stmt::Mut(inductionSelect, Expr::IntrOp(Intr::Add(inductionTerm, x.step, x.induction.tpe))), fn, {ctx});
            B.CreateBr(loopTest);
          }
        }
        B.SetInsertPoint(loopExit);
        return BlockKind::Terminal;
      },
      [&](const Stmt::Break &) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->exit);
        else throw BackendException("orphaned break!");
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cont &) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->test);
        else throw BackendException("orphaned cont!");
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cond &x) -> BlockKind {
        const auto condTrue = llvm::BasicBlock::Create(C.actual, "cond_true", &fn);
        const auto condFalse = llvm::BasicBlock::Create(C.actual, "cond_false", &fn);
        const auto condExit = llvm::BasicBlock::Create(C.actual, "cond_exit", &fn);
        B.CreateCondBr(mkTermVal(x.cond, "cond"), condTrue, condFalse);
        {
          B.SetInsertPoint(condTrue);
          auto kind = BlockKind::Normal;
          for (auto &body : x.trueBr)
            kind = mkStmt(body, fn, whileCtx);
          if (kind != BlockKind::Terminal) B.CreateBr(condExit);
        }
        {
          B.SetInsertPoint(condFalse);
          auto kind = BlockKind::Normal;
          for (auto &body : x.falseBr)
            kind = mkStmt(body, fn, whileCtx);
          if (kind != BlockKind::Terminal) B.CreateBr(condExit);
        }
        if (condExit->getNumUses() > 0) {
          B.SetInsertPoint(condExit);
          return BlockKind::Normal;
        } else {
          condExit->removeFromParent();
          return BlockKind::Terminal;
        }
      },
      [&](const Stmt::Return &x) -> BlockKind {
        if (auto rtnTpe = x.value.tpe(); rtnTpe.is<Type::Unit0>()) {
          B.CreateRetVoid();
        } else if (rtnTpe.is<Type::Nothing>()) {
          B.CreateUnreachable();
        } else if (currentSretParam && rtnTpe.is<Type::Struct>()) {
          const auto val = mkExprVal(x.value, "return_sret_val");
          const auto structInfo = structTypes.at(repr(rtnTpe.get<Type::Struct>()->name));
          auto spill = C.allocaAS(B, structInfo.tpe, C.AllocaAS, "return_sret_spill");
          auto _ = C.store(B, val, spill);
          const auto size = structInfo.layout.sizeInBytes;
          const auto align = structInfo.layout.alignment;
          B.CreateMemCpy(currentSretParam, llvm::MaybeAlign(align), spill, llvm::MaybeAlign(align), size);
          B.CreateRetVoid();
        } else {
          const auto expr = mkExprVal(x.value, "return");
          if (rtnTpe.is<Type::Bool1>()) {
            // Extend from i1 to i8
            B.CreateRet(B.CreateIntCast(expr, llvm::Type::getInt8Ty(C.actual), true));
          } else if (auto ptr = rtnTpe.get<Type::Ptr>(); false) {
            B.CreateRet(C.load(B, expr, resolveType(rtnTpe)));
          } else {
            B.CreateRet(expr);
          }
        }
        return BlockKind::Terminal;
      },
      [&](const Stmt::Annotated &x) -> BlockKind { return mkStmt(x.inner, fn, whileCtx); });
}

// SPIR-V: struct-by-value returns get coerced to a single i32 by the pre-legaliser. Convert
// to sret form (leading out-pointer, void return) so no struct crosses a function boundary.
static bool shouldUseSret(const CodeGen &cg, const Function &fn) { return fn.rtn.is<Type::Struct>() && cg.C.isSpirv(); }

static auto createPrototype(CodeGen &cg, llvm::Module &mod, const Function &fn) {

  Vector<Arg> allArgs;
  // CPU HostThreaded kernels receive `tid` as a leading arg from the runtime; GPU launches
  // provide it via intrinsics, so adding `__tid` there would off-by-one the kernel ABI.
  const auto cpuTarget = cg.C.options.target == LLVMBackend::Target::x86_64 ||  //
                         cg.C.options.target == LLVMBackend::Target::AArch64 || //
                         cg.C.options.target == LLVMBackend::Target::ARM;
  if (fn.isEntry && cpuTarget) {
    allArgs.push_back(Arg(Named("__tid", Type::IntS64()), {}));
  }
  if (fn.receiver) allArgs.push_back(*fn.receiver);
  for (auto &a : fn.moduleCaptures)
    allArgs.push_back(a);
  for (auto &a : fn.termCaptures)
    allArgs.push_back(a);
  for (auto &a : fn.args)
    allArgs.push_back(a);

  // Drop Unit0/Nothing args: both lower to void, which FunctionType::get's isValidArgumentType asserts.
  const auto argsNoUnit = allArgs | filter([](auto &arg) {
                            return !arg.named.tpe.template is<Type::Unit0>() //
                                   && !arg.named.tpe.template is<Type::Nothing>();
                          }) //
                          | to_vector();

  const bool useSret = shouldUseSret(cg, fn);

  // Structs are returned by-value (functionBoundary=false); other args travel as opaque pointers.
  const auto rtnTpe = (fn.rtn.is<Type::Unit0>() || useSret) ? llvm::Type::getVoidTy(cg.C.actual)
                      : fn.rtn.is<Type::Struct>()           ? cg.resolveType(fn.rtn, false)
                                                            : cg.resolveType(fn.rtn, true);

  auto argTys = argsNoUnit                                                                        //
                | map([&](auto &arg) { return cg.resolveType(arg.named.tpe, true, fn.isEntry); }) //
                | to_vector();

  if (useSret) argTys.insert(argTys.begin(), llvm::PointerType::get(cg.C.actual, cg.C.AllocaAS));
  llvm::Type *sretStructTy = useSret ? cg.resolveType(fn.rtn, /*functionBoundary*/ false) : nullptr;

  // XXX Normalise names as NVPTX has a relatively limiting range of supported characters in symbols
  const auto normalisedName = repr(fn.name) ^ map([](const char c) { return !std::isalnum(c) && c != '_' ? '_' : c; });

  Signature sig(fn.name, /*tpeVars*/ {}, /*receiver*/ {}, argsNoUnit ^ map([](auto &x) { return x.named.tpe; }),
                /*moduleCaptures*/ {}, /*termCaptures*/ {}, fn.rtn);
  llvm::Function *llvmFn = llvm::Function::Create(llvm::FunctionType::get(/*Result*/ rtnTpe, /*Params*/ argTys, /*isVarArg*/ false), //
                                                  fn.visibility.is<FunctionVisibility::Exported>()                                   //
                                                      ? llvm::Function::ExternalLinkage
                                                      : llvm::Function::InternalLinkage,
                                                  normalisedName, //
                                                  mod);

  // Attach sret attributes here, before any other prototype's body emission can look this
  // function up via Expr::Invoke. Deferring until the body loop would let the first caller see
  // the function without the sret marker and trip an LLVM signature mismatch.
  if (useSret) {
    auto *sretArg = llvmFn->getArg(0);
    sretArg->setName("sret");
    sretArg->addAttr(llvm::Attribute::get(cg.C.actual, llvm::Attribute::StructRet, sretStructTy));
    sretArg->addAttr(llvm::Attribute::NoAlias);
  }

  llvm::MDBuilder mdbuilder(cg.C.actual);
  llvm::MDNode *root = mdbuilder.createTBAARoot("TBAA root");

  cg.targetHandler->witnessFn(cg, *llvmFn, fn);

  cg.functions.emplace(sig, llvmFn);
  return std::tuple{llvmFn, fn, argsNoUnit};
}

Pair<Opt<std::string>, std::string> CodeGen::transform(const Program &program) {
  structTypes = C.resolveLayouts(program.defs);

  Vector<Function> allFns;
  allFns.reserve(program.functions.size() + 1);
  allFns.push_back(program.entry);
  for (auto &f : program.functions)
    allFns.push_back(f);
  const auto prototypes = allFns ^ map([&](auto &fn) { return createPrototype(*this, M, fn); });

  prototypes | for_each([&](auto &llvmFn, auto &fn, auto &argsNoUnit) {
    B.SetInsertPoint(llvm::BasicBlock::Create(C.actual, "entry", llvmFn));
    const bool useSret = shouldUseSret(*this, fn);
    currentSretParam = useSret ? llvmFn->getArg(0) : nullptr;
    const size_t argOffset = useSret ? 1 : 0;
    stackVarPtrs = argsNoUnit | zip_with_index() | map([&](auto &arg, auto i) -> Pair<std::string, Pair<Type::Any, ValPtr>> { //
                     auto llvmArg = llvmFn->getArg(i + argOffset);

                     llvmArg->setName(arg.named.symbol);

                     // XXX Structs arrive at the boundary as pointers; use directly without a slot.
                     if (arg.named.tpe.template is<Type::Struct>()) {
                       return {arg.named.symbol, {arg.named.tpe, llvmArg}};
                     }

                     auto llvmArgValue = arg.named.tpe.template is<Type::Bool1>() || arg.named.tpe.template is<Type::Unit0>()
                                             ? B.CreateICmpNE(llvmArg, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C.actual), 0, true))
                                             : llvmArg;

                     // XXX SPIR-V kernel-entry pointers arrive in CrossWorkgroup; the slot wants
                     // Generic so loads see the typed pointer they expect. OpPtrCastToGeneric.
                     auto *slotTy = resolveType(arg.named.tpe);
                     if (llvmArgValue->getType() != slotTy && llvmArgValue->getType()->isPointerTy() && slotTy->isPointerTy()) {
                       llvmArgValue = B.CreateAddrSpaceCast(llvmArgValue, slotTy);
                     }
                     auto stackPtr = C.allocaAS(B, slotTy, C.AllocaAS, arg.named.symbol + "_stack_ptr");
                     auto _ = C.store(B, llvmArgValue, stackPtr);
                     return {arg.named.symbol, {arg.named.tpe, stackPtr}};
                   }) //
                   | to<Map>();
    for (auto &stmt : fn.body)
      auto _ = mkStmt(stmt, *llvmFn);
    // Abstract method bodies (e.g. typeclass methods like `Monoid.mempty`) emit no terminator.
    // Insert an `unreachable` so LLVM module verification is happy — the symbol should never
    // actually be invoked since DynamicDispatchPass routes calls through a vtable.
    if (auto *bb = B.GetInsertBlock(); bb && bb->getTerminator() == nullptr) {
      B.CreateUnreachable();
    }
    stackVarPtrs.clear();
    currentSretParam = nullptr;
  });

  targetHandler->postProcessModule(*this);

  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  M.print(irOut, nullptr);

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (verifyModule(M, &errOut)) {
    fmt::print(stderr, "Verification failed:\n{}\nIR=\n{}\n", errOut.str(), irOut.str());
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

ValPtr CodeGen::unaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyType &rtn, const ValPtrFn1 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException("Semantic error: lhs type " + to_string(l.tpe()) + " of unary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  return fn(mkTermVal(l));
}
ValPtr CodeGen::binaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn,
                           const ValPtrFn2 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException("Semantic error: lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  if (r.tpe() != rtn) {
    throw BackendException("Semantic error: rhs type " + to_string(r.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  return fn(mkTermVal(l), mkTermVal(r));
}
ValPtr CodeGen::unaryNumOp(const AnyExpr &expr, const AnyTerm &arg, const AnyType &rtn, //
                           const ValPtrFn1 &integralFn, const ValPtrFn1 &fractionalFn) {
  return unaryExpr(expr, arg, rtn, [&](auto lhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) return integralFn(lhs);
    if (rtn.kind().is<TypeKind::Fractional>()) return fractionalFn(lhs);
    // None-kind result (Nothing/Unit0/Exec) needs a sized poison; void poison is unrepresentable, so use i8.
    if (rtn.kind().is<TypeKind::None>()) return llvm::PoisonValue::get(llvm::Type::getInt8Ty(C.actual));
    throw BackendException("unimplemented");
  });
}
ValPtr CodeGen::binaryNumOp(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, //
                            const ValPtrFn2 &integralFn, const ValPtrFn2 &fractionalFn) {
  return binaryExpr(expr, l, r, rtn, [&](auto lhs, auto rhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) return integralFn(lhs, rhs);
    if (rtn.kind().is<TypeKind::Fractional>()) return fractionalFn(lhs, rhs);
    if (rtn.kind().is<TypeKind::None>()) return llvm::PoisonValue::get(llvm::Type::getInt8Ty(C.actual));
    throw BackendException("unimplemented");
  });
}
ValPtr CodeGen::mkSignumVal(const AnyExpr &expr, const AnyTerm &x, const AnyType &tpe) {
  return unaryNumOp(
      expr, x, tpe,
      [&](auto v) -> ValPtr {
        auto msb = v->getType()->getPrimitiveSizeInBits() - 1;
        return B.CreateOr(B.CreateAShr(v, msb), B.CreateLShr(B.CreateNeg(v), msb));
      },
      [&](auto v) -> ValPtr {
        auto isNaN = B.CreateFCmpUNO(v, v);
        auto isZero = B.CreateFCmpOEQ(v, llvm::ConstantFP::get(v->getType(), 0.0));
        return B.CreateSelect(B.CreateLogicalOr(isNaN, isZero), v, intr2(llvm::Intrinsic::copysign, tpe, dsl::fractional(tpe, 1.0L), x));
      });
}

LLVMBackend::LLVMBackend(const Options &options) : options(options) {}

std::vector<StructLayout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &structs) {
  return TargetedContext(options).resolveLayouts(structs) | values() | map([&](auto &i) { return i.layout; }) | to_vector();
}

CompileResult LLVMBackend::compileProgram(const Program &program, const compiletime::OptLevel &opt) {
  using namespace llvm;

  CodeGen cg(options, "program");
  auto transformStart = compiler::nowMono();
  auto [maybeTransformErr, transformMsg] = cg.transform(program);
  CompileEvent ast2IR(compiler::nowMs(), compiler::elapsedNs(transformStart), "ast_to_llvm_ir", transformMsg, {});

  auto verifyStart = compiler::nowMono();
  auto [maybeVerifyErr, verifyMsg] = llvmc::verifyModule(cg.M);
  CompileEvent astOpt(compiler::nowMs(), compiler::elapsedNs(verifyStart), "llvm_ir_verify", verifyMsg, {});

  if (maybeTransformErr || maybeVerifyErr) {
    std::vector<std::string> errors;
    if (maybeTransformErr) errors.push_back(*maybeTransformErr);
    if (maybeVerifyErr) errors.push_back(*maybeVerifyErr);
    return {{},
            {},               //
            {ast2IR, astOpt}, //
            {},               //
            errors ^ mk_string("\n")};
  }

  auto c = compileModule(options.targetInfo(), opt, true, cg.M);
  c.layouts = resolveLayouts(program.defs);
  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
