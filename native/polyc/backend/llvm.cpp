#include <iostream>
#include <unordered_set>

#include "aspartame/all.hpp"

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"

#include "fmt/core.h"
#include "magic_enum.hpp"

#include "llvm_amdgpu.h"
#include "llvm_cpu.h"
#include "llvm_nvptx.h"
#include "llvm_spirv_cl.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Host.h"

#include <magic_enum.hpp>

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
    case LLVMBackend::Target::SPIRV32: [[fallthrough]];
    case LLVMBackend::Target::SPIRV64: return std::make_unique<SPIRVOpenCLTargetSpecificHandler>();
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

llvm::Type *CodeGen::resolveType(const AnyType &tpe, const bool functionBoundary) {
  return C.resolveType(tpe, structTypes, functionBoundary);
}

llvm::Function *CodeGen::resolveExtFn(const Type::Any &rtn, const std::string &name, const std::vector<Type::Any> &args) {
  return functions ^= get_or_emplace(Signature(name, args, rtn), [&](auto &sig) -> llvm::Function * {
           auto tpe = llvm::FunctionType::get(
               /*Result*/ resolveType(rtn, true),
               /*Params*/ args ^ map([&](auto &t) { return resolveType(t, true); }),
               /*isVarArg*/ false);
           auto fn = llvm::Function::Create(tpe, llvm::Function::ExternalLinkage, name, M);
           return fn;
         });
}

ValPtr CodeGen::invokeMalloc(ValPtr size) {
  return B.CreateCall(resolveExtFn(Type::Ptr(Type::IntS8(), {}, TypeSpace::Global()), "malloc", {Type::IntS64()}), size);
}
ValPtr CodeGen::invokeAbort() { return B.CreateCall(resolveExtFn(Type::Nothing(), "abort", {})); }

ValPtr CodeGen::extFn1(const std::string &name, const AnyType &rtn, const AnyExpr &arg) { //
  const auto fn = resolveExtFn(rtn, name, {arg.tpe()});
  if (C.options.target == LLVMBackend::Target::SPIRV32 || C.options.target == LLVMBackend::Target::SPIRV64) {
    fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  }
  if (!rtn.is<Type::Unit0>()) {
    fn->addFnAttr(llvm::Attribute::WillReturn);
  }
  const auto call = B.CreateCall(fn, mkExprVal(arg));
  call->setCallingConv(fn->getCallingConv());
  return call;
}
ValPtr CodeGen::extFn2(const std::string &name, const AnyType &rtn, const AnyExpr &lhs, const AnyExpr &rhs) {
  const auto fn = resolveExtFn(rtn, name, {lhs.tpe(), rhs.tpe()});
  if (C.options.target == LLVMBackend::Target::SPIRV32 || C.options.target == LLVMBackend::Target::SPIRV64) {
    fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    fn->addFnAttr(llvm::Attribute::NoBuiltin);
    fn->addFnAttr(llvm::Attribute::Convergent);
  }
  const auto call = B.CreateCall(fn, {mkExprVal(lhs), mkExprVal(rhs)});
  call->setCallingConv(fn->getCallingConv());
  return call;
}
ValPtr CodeGen::intr0(const llvm::Intrinsic::ID id) { //
  const auto callee = llvm::Intrinsic::getDeclaration(&M, id, {});
  return B.CreateCall(callee);
}
ValPtr CodeGen::intr1(const llvm::Intrinsic::ID id, const AnyType &overload, const AnyExpr &arg) { //
  const auto callee = llvm::Intrinsic::getDeclaration(&M, id, resolveType(overload));
  return B.CreateCall(callee, mkExprVal(arg));
}
ValPtr CodeGen::intr2(const llvm::Intrinsic::ID id, const AnyType &overload, //
                      const AnyExpr &lhs, const AnyExpr &rhs) {              //
  // XXX the overload type here is about the overloading of intrinsic names, not about the parameter types
  // i.e., f32 is for foo.f32(float %a, float %b, float %c)
  const auto callee = llvm::Intrinsic::getDeclaration(&M, id, resolveType(overload));
  return B.CreateCall(callee, {mkExprVal(lhs), mkExprVal(rhs)});
}

ValPtr CodeGen::findStackVar(const Named &named) {
  if (named.tpe.is<Type::Unit0>()) return mkExprVal(Expr::Unit0Const());
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

ValPtr CodeGen::mkSelectPtr(const Expr::Select &select) {

  auto fail = [&] { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> StructInfo {
    auto findTy = [&](const Type::Struct &s) -> StructInfo {
      return structTypes ^ get_maybe(s.name) ^
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

  if (select.init.empty()) return findStackVar(select.last); // local var lookup
  else {
    // we're in a select chain, init elements must return struct type; the head must come from LUT
    // any nested struct *member* access (through pointer, the struct ) must be on a separate GEP instruction as
    // memory access via pointer indirections is not part of GEP
    // & _S -> e
    // & _S -> S
    // & _S -> S -> e
    // & _S -> S -> S
    auto [head, tail] = uncons(select);
    auto tpe = head.tpe;
    auto root = findStackVar(head);
    for (auto &path : tail) {
      const auto info = structTypeOf(tpe);
      if (auto idx = info.memberIndices ^ get_maybe(path.symbol)) {
        if (auto p = tpe.get<Type::Ptr>(); p && !p->length) {
          root = B.CreateInBoundsGEP(info.tpe, C.load(B, root, B.getPtrTy(C.AllocaAS)),
                                     {//
                                      llvm::ConstantInt::get(C.i32Ty(), 0), llvm::ConstantInt::get(C.i32Ty(), *idx)},
                                     qualified(select) + "_select_ptr");
        } else {
          root = B.CreateInBoundsGEP(info.tpe, root,
                                     {//
                                      llvm::ConstantInt::get(C.i32Ty(), 0), llvm::ConstantInt::get(C.i32Ty(), *idx)},
                                     qualified(select) + "_select_ptr");
        }
        tpe = path.tpe;
      } else {
        auto pool = info.memberIndices |
                    mk_string("\n", "\n", "\n", [](auto &k, auto &v) { return " -> `" + k + "` = " + std::to_string(v) + ")"; });
        throw BackendException("Illegal select path with unknown struct member index of name `" + to_string(path) + "`, pool=" + pool +
                               fail());
      }
    }
    return root;
  }
}

ValPtr CodeGen::mkExprVal(const Expr::Any &expr, const std::string &key) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return expr.match_total( //

      [&](const Expr::Float16Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getHalfTy(C.actual), x.value); },
      [&](const Expr::Float32Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getFloatTy(C.actual), x.value); },
      [&](const Expr::Float64Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getDoubleTy(C.actual), x.value); },

      [&](const Expr::IntU8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C.actual), x.value); },
      [&](const Expr::IntU16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C.actual), x.value); },
      [&](const Expr::IntU32Const &x) -> ValPtr { return ConstantInt::get(C.i32Ty(), x.value); },
      [&](const Expr::IntU64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C.actual), x.value); },

      [&](const Expr::IntS8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C.actual), x.value); },
      [&](const Expr::IntS16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C.actual), x.value); },
      [&](const Expr::IntS32Const &x) -> ValPtr { return ConstantInt::get(C.i32Ty(), x.value); },
      [&](const Expr::IntS64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C.actual), x.value); },

      [&](const Expr::Unit0Const &x) -> ValPtr {
        // this only exists to represent the singleton
        return ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0);
      },
      [&](const Expr::Bool1Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt1Ty(C.actual), x.value); },
      [&](const Expr::NullPtrConst &x) -> ValPtr {
        return llvm::ConstantPointerNull::get(llvm::PointerType::get(resolveType(x.tpe), C.addressSpace(x.space)));
      },

      [&](const Expr::SpecOp &x) -> ValPtr { return targetHandler->mkSpecVal(*this, x); },
      [&](const Expr::MathOp &x) -> ValPtr { return targetHandler->mkMathVal(*this, x); },
      [&](const Expr::IntrOp &x) -> ValPtr {
        auto intr = x.op;
        return intr.match_total( //
            [&](const Intr::BNot &v) -> ValPtr { return unaryExpr(expr, v.x, v.tpe, [&](auto x) { return B.CreateNot(x); }); },
            [&](const Intr::LogicNot &v) -> ValPtr { return B.CreateNot(mkExprVal(v.x)); },
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
            [&](const Intr::LogicAnd &v) -> ValPtr { return B.CreateLogicalAnd(mkExprVal(v.x), mkExprVal(v.y)); }, //
            [&](const Intr::LogicOr &v) -> ValPtr { return B.CreateLogicalOr(mkExprVal(v.x), mkExprVal(v.y)); },   //
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
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpULT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOLT(l, r); });
            },
            [&](const Intr::LogicGt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](auto l, auto r) { return B.CreateICmpSGT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOGT(l, r); });
            });
      },

      [&](const Expr::Select &x) -> ValPtr {
        if (x.tpe.is<Type::Unit0>()) return mkExprVal(Expr::Unit0Const());
        if (const auto ptr = x.tpe.get<Type::Ptr>(); ptr && ptr->length) return mkSelectPtr(x);
        else return C.load(B, mkSelectPtr(x), resolveType(x.tpe));
      },
      [&](const Expr::Poison &x) -> ValPtr {
        if (auto tpe = resolveType(x.tpe); llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        } else {
          throw BackendException("unimplemented");
        }
      },

      [&](const Expr::Cast &x) -> ValPtr {
        // we only allow widening or narrowing of integral and fractional types
        // pointers are not allowed to participate on either end
        auto from = mkExprVal(x.from);
        auto fromTpe = resolveType(x.from.tpe());
        auto toTpe = resolveType(x.as);
        enum class NumKind { Fractional, Integral };

        // Same type
        if (x.as == x.from.tpe()) return from;

        // Allow any pointer casts of struct
        if (const auto rhsPtr = x.from.tpe().get<Type::Ptr>()) {
          if (const auto lhsPtr = x.as.get<Type::Ptr>()) {
            // TODO check layout and loss of information
            if (lhsPtr->comp.is<Type::Struct>() && rhsPtr->comp.is<Type::Struct>()) {
              return from;
            }
          }
        }

        auto fromKind = x.from.tpe().kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException("Semantic error: conversion from ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw BackendException("none!?"); });

        auto toKind = x.as.kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException("Semantic error: conversion to ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw BackendException("none!?"); });

        if (fromKind == NumKind::Fractional && toKind == NumKind::Integral) {

          // to the equally sized integral type first if narrowing; XXX narrowing directly produces a poison value

          ValPtr c = nullptr;
          if (fromTpe->getPrimitiveSizeInBits() > toTpe->getPrimitiveSizeInBits() || true) {
            auto min32BitIntBits = std::max<llvm::TypeSize::ScalarTy>(32, toTpe->getPrimitiveSizeInBits());
            auto toTpeMaxInFp = llvm::ConstantFP::get(fromTpe, double(nIntMax(min32BitIntBits)));
            auto toTpeMinInFp = llvm::ConstantFP::get(fromTpe, double(nIntMin(min32BitIntBits)));
            auto min32BitIntTy = llvm::Type::getIntNTy(C.actual, min32BitIntBits);
            auto toTpeMaxInInt = llvm::ConstantInt::get(min32BitIntTy, nIntMax(min32BitIntBits));
            auto toTpeMinInInt = llvm::ConstantInt::get(min32BitIntTy, nIntMin(min32BitIntBits));

            c = B.CreateSelect(B.CreateFCmpOGE(from, toTpeMaxInFp), toTpeMaxInInt,                //
                               B.CreateSelect(B.CreateFCmpOLE(from, toTpeMinInFp), toTpeMinInInt, //
                                              B.CreateFPToSI(from, min32BitIntTy)));
            c = B.CreateIntCast(c, toTpe, !isUnsigned(x.as));

          } else {
            c = B.CreateFPToSI(from, toTpe, "fractional_to_integral_cast");
          }

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
        const auto argNoUnit = x.args ^ filter([](auto &arg) { return !arg.tpe().template is<Type::Unit0>(); });
        const auto sig = Signature(x.name, argNoUnit ^ map([](auto &arg) { return arg.tpe(); }), x.rtn);
        return functions ^ get_maybe(sig) ^
               fold(
                   [&](auto &fn) -> ValPtr {
                     const auto params =
                         argNoUnit ^ map([&](auto &term) {
                           const auto val = mkExprVal(term);
                           return term.tpe().template is<Type::Bool1>() ? B.CreateZExt(val, resolveType(Type::Bool1(), true)) : val;
                         });
                     const auto call = B.CreateCall(fn, params);
                     // in case the fn returns a unit (which is mapped to void), we just return the constant
                     return x.rtn.is<Type::Unit0>() ? mkExprVal(Expr::Unit0Const()) : call;
                   },
                   [&]() -> ValPtr {
                     throw BackendException(fmt::format("Unhandled invocation {}, known functions are:\n{}", repr(sig),
                                                        functions | keys() | mk_string("\n -> ", show_repr)));
                   });
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (const auto lhs = x.lhs.get<Expr::Select>()) {
          if (const auto arrTpe = lhs->tpe.get<Type::Ptr>()) {
            if (arrTpe->comp.is<Type::Unit0>()) { // Still call GEP so that memory access and OOB effects are still present.
              const auto val = mkExprVal(Expr::Unit0Const());
              B.CreateInBoundsGEP(val->getType(), mkExprVal(*lhs), mkExprVal(x.idx), key + "_ptr");
              return val;
            } else if (arrTpe->length) {
              const auto ty = resolveType(*arrTpe);
              const auto ptr =
                  B.CreateInBoundsGEP(ty, mkExprVal(*lhs), {ConstantInt::get(C.i32Ty(), 0), mkExprVal(x.idx)}, key + "_idx_ptr");
              return C.load(B, ptr, resolveType(arrTpe->comp));
            } else {
              const auto ty = resolveType(arrTpe->comp);
              const auto ptr = B.CreateInBoundsGEP(ty, mkExprVal(*lhs), mkExprVal(x.idx), key + "_idx_ptr");
              if (arrTpe->comp.is<Type::Bool1>()) { // Narrow from i8 to i1
                return B.CreateICmpNE(C.load(B, ptr, ty), ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0, true));
              } else {
                return C.load(B, ptr, ty);
              }
            }
          } else {
            throw BackendException("Semantic error: array index not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else throw BackendException("Semantic error: LHS of " + to_string(x) + " (index) is not a select");
      },
      [&](const Expr::RefTo &x) -> ValPtr {
        if (auto lhs = x.lhs.get<Expr::Select>()) {
          if (auto arrTpe = lhs->tpe.get<Type::Ptr>(); arrTpe) { // taking reference of an index in an array
            auto offset = x.idx ? mkExprVal(*x.idx) : llvm::ConstantInt::get(llvm::Type::getInt64Ty(C.actual), 0, true);
            if (auto nestedArrTpe = arrTpe->comp.get<Type::Ptr>(); nestedArrTpe && nestedArrTpe->length) {
              auto ty = arrTpe->comp.is<Type::Unit0>() ? llvm::Type::getInt8Ty(C.actual) : resolveType(arrTpe->comp);
              return B.CreateInBoundsGEP(ty,              //
                                         mkExprVal(*lhs), //
                                         {llvm::ConstantInt::get(C.i32Ty(), 0), offset}, key + "_ref_to_" + llvm_tostring(ty));

            } else {
              auto ty = arrTpe->comp.is<Type::Unit0>() ? llvm::Type::getInt8Ty(C.actual) : resolveType(arrTpe->comp);
              return B.CreateInBoundsGEP(ty,              //
                                         mkExprVal(*lhs), //
                                         offset, key + "_ref_to_ptr");
            }
          } else { // taking reference of a var
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
        const auto size = mkExprVal(x.size);
        const auto elemSize = C.sizeOf(B, componentTpe);
        const auto ptr = invokeMalloc(B.CreateMul(B.CreateIntCast(size, resolveType(Type::IntS64()), true), elemSize));
        return B.CreateBitCast(ptr, componentTpe);
      },
      [&](const Expr::Annotated &x) -> ValPtr { return mkExprVal(x.expr, key); });
}

CodeGen::BlockKind CodeGen::mkStmt(const Stmt::Any &stmt, llvm::Function &fn, const Opt<WhileCtx> &whileCtx = {}) {
  return stmt.match_total(
      [&](const Stmt::Block &x) -> BlockKind {
        auto kind = BlockKind::Normal;
        for (auto &body : x.stmts)
          kind = mkStmt(body, fn);
        return kind;
      },
      [&](const Stmt::Comment &) -> BlockKind { return BlockKind::Normal; }, // discard comments
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
          stackVarPtrs.emplace(x.name.symbol, Pair<Type::Any, llvm::Value *>{x.name.tpe, stackPtr});
          if (x.expr) {
            const auto rhs = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
            const auto _ = C.store(B, rhs, stackPtr); //
          }
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::Mut &x) -> BlockKind {
        // [T : ref]        =>> t   := &(rhs:T) ; lut += t
        // [T : ref {u: U}] =>> t.u := &(rhs:U)
        // [T : val]        =>> t   :=   rhs:T
        if (auto lhs = x.name.get<Expr::Select>(); lhs) {
          if (x.expr.tpe() != lhs->tpe) {
            throw BackendException("Semantic error: name type (" + to_string(x.expr.tpe()) + ") and rhs expr (" + to_string(lhs->tpe) +
                                   ") mismatch (" + repr(x) + ")");
          }
          if (lhs->tpe.is<Type::Unit0>()) return BlockKind::Normal;
          const auto rhs = mkExprVal(x.expr, qualified(*lhs) + "_mut");
          if (lhs->init.empty()) { // local var
            const auto stackPtr = findStackVar(lhs->last);
            const auto _ = C.store(B, rhs, stackPtr);
            // FIXME
          } else { // struct member select
            const auto _ = C.store(B, rhs, mkSelectPtr(*lhs));
          }
        } else throw BackendException("Semantic error: LHS of " + to_string(x) + " (mut) is not a select");
        return BlockKind::Normal;
      },
      [&](const Stmt::Update &x) -> BlockKind {
        if (auto lhs = x.lhs.get<Expr::Select>(); lhs) {
          if (auto arrTpe = lhs->tpe.get<Type::Ptr>(); arrTpe) {
            auto rhs = x.value;

            bool componentIsSizedArray = false;
            if (auto p = arrTpe->comp.get<Type::Ptr>(); p && p->length) {
              componentIsSizedArray = true;
            }

            if (arrTpe->comp != rhs.tpe()) {
              throw BackendException("Semantic error: array comp type (" + to_string(arrTpe->comp) + ") and rhs expr (" +
                                     to_string(rhs.tpe()) + ") mismatch (" + repr(x) + ")");
            } else {
              auto dest = mkExprVal(*lhs);
              if (rhs.tpe().is<Type::Bool1>() || rhs.tpe().is<Type::Unit0>()) { // Extend from i1 to i8
                auto ty = llvm::Type::getInt8Ty(C.actual);
                auto ptr = B.CreateInBoundsGEP( //
                    ty, dest,
                    componentIsSizedArray ? llvm::ArrayRef<ValPtr>{llvm::ConstantInt::get(C.i32Ty(), 0), mkExprVal(x.idx)}
                                          : llvm::ArrayRef<ValPtr>{mkExprVal(x.idx)},
                    qualified(*lhs) + "_update_ptr");
                const auto _ = C.store(B, B.CreateIntCast(mkExprVal(rhs), ty, true), ptr);
              } else {

                auto ptr = B.CreateInBoundsGEP(   //
                    resolveType(rhs.tpe()), dest, //
                    componentIsSizedArray ? llvm::ArrayRef<ValPtr>{llvm::ConstantInt::get(C.i32Ty(), 0), mkExprVal(x.idx)}
                                          : llvm::ArrayRef<ValPtr>{mkExprVal(x.idx)},
                    qualified(*lhs) + "_update_ptr" //
                );                                  //
                const auto _ = C.store(B, mkExprVal(rhs), ptr);
              }
            }
          } else {
            throw BackendException("Semantic error: array update not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else throw BackendException("Semantic error: LHS of " + to_string(x) + " (update) is not a select");

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
          auto kind = BlockKind::Normal;
          for (auto &test : x.tests)
            kind = mkStmt(test, fn, {ctx});
          if (kind != BlockKind::Terminal) {
            const auto continue_ = mkExprVal(x.cond);
            B.CreateCondBr(continue_, loopBody, loopExit);
          }
        }
        {
          B.SetInsertPoint(loopBody);
          auto kind = BlockKind::Normal;
          for (auto &body : x.body)
            kind = mkStmt(body, fn, {ctx});
          if (kind != BlockKind::Terminal) B.CreateBr(loopTest);
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
        B.CreateCondBr(mkExprVal(x.cond, "cond"), condTrue, condFalse);
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
        } else {
          const auto expr = mkExprVal(x.value, "return");
          if (rtnTpe.is<Type::Bool1>()) {
            // Extend from i1 to i8
            B.CreateRet(B.CreateIntCast(expr, llvm::Type::getInt8Ty(C.actual), true));
          } else if (auto ptr = rtnTpe.get<Type::Ptr>(); ptr && ptr->length) {
            B.CreateRet(C.load(B, expr, resolveType(rtnTpe)));
          } else {
            B.CreateRet(expr);
          }
        }
        return BlockKind::Terminal;
      },
      [&](const Stmt::Annotated &x) -> BlockKind { return mkStmt(x.stmt, fn); });
}

static auto createPrototype(CodeGen &cg, llvm::Module &mod, const Function &fn) {

  // Unit types in arguments are discarded
  const auto argsNoUnit = fn.args | filter([](auto &arg) { return !arg.named.tpe.template is<Type::Unit0>(); }) | to_vector();
  // Unit type at function return type position is void, any other location, Unit is a singleton value
  const auto rtnTpe = fn.rtn.is<Type::Unit0>() ? llvm::Type::getVoidTy(cg.C.actual) : cg.resolveType(fn.rtn, true);

  const auto argTys = argsNoUnit                                                            //
                      | map([&](auto &arg) { return cg.resolveType(arg.named.tpe, true); }) //
                      | to_vector();

  // XXX Normalise names as NVPTX has a relatively limiting range of supported characters in symbols
  const auto normalisedName = fn.name ^ map([](const char c) { return !std::isalnum(c) && c != '_' ? '_' : c; });

  Signature sig(fn.name, argsNoUnit ^ map([](auto &x) { return x.named.tpe; }), fn.rtn);
  llvm::Function *llvmFn = llvm::Function::Create(llvm::FunctionType::get(/*Result*/ rtnTpe, /*Params*/ argTys, /*isVarArg*/ false), //
                                                  fn.attrs.contains(FunctionAttr::Exported())                                        //
                                                      ? llvm::Function::ExternalLinkage
                                                      : llvm::Function::InternalLinkage,
                                                  normalisedName, //
                                                  mod);

  llvm::MDBuilder mdbuilder(cg.C.actual);
  llvm::MDNode *root = mdbuilder.createTBAARoot("TBAA root");

  cg.targetHandler->witnessFn(cg, *llvmFn, fn);

  cg.functions.emplace(sig, llvmFn);
  return std::tuple{llvmFn, fn, argsNoUnit};
}

Pair<Opt<std::string>, std::string> CodeGen::transform(const Program &program) {
  structTypes = C.resolveLayouts(program.structs);

  const auto prototypes = program.functions ^ map([&](auto &fn) { return createPrototype(*this, M, fn); });

  prototypes | for_each([&](auto &llvmFn, auto &fn, auto &argsNoUnit) {
    B.SetInsertPoint(llvm::BasicBlock::Create(C.actual, "entry", llvmFn));
    stackVarPtrs = argsNoUnit | zip_with_index() | map([&](auto &arg, auto i) -> Pair<std::string, Pair<Type::Any, ValPtr>> { //
                     auto llvmArg = llvmFn->getArg(i);

                     llvmArg->setName(arg.named.symbol);

                     auto llvmArgValue = arg.named.tpe.template is<Type::Bool1>() || arg.named.tpe.template is<Type::Unit0>()
                                             ? B.CreateICmpNE(llvmArg, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C.actual), 0, true))
                                             : llvmArg;

                     auto stackPtr = C.allocaAS(B, resolveType(arg.named.tpe), C.AllocaAS, arg.named.symbol + "_stack_ptr");
                     auto _ = C.store(B, llvmArgValue, stackPtr);
                     return {arg.named.symbol, {arg.named.tpe, stackPtr}};
                   }) //
                   | to<Map>();
    for (auto &stmt : fn.body)
      auto _ = mkStmt(stmt, *llvmFn);
    stackVarPtrs.clear();
  });

  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  M.print(irOut, nullptr);

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (verifyModule(M, &errOut)) {
    std::cerr << "Verification failed:\n" << errOut.str() << "\nIR=\n" << irOut.str() << std::endl;
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

ValPtr CodeGen::unaryExpr(const AnyExpr &expr, const AnyExpr &l, const AnyType &rtn, const ValPtrFn1 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException("Semantic error: lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkExprVal(l));
}
ValPtr CodeGen::binaryExpr(const AnyExpr &expr, const AnyExpr &l, const AnyExpr &r, const AnyType &rtn,
                           const ValPtrFn2 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException("Semantic error: lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  if (r.tpe() != rtn) {
    throw BackendException("Semantic error: rhs type " + to_string(r.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkExprVal(l), mkExprVal(r));
}
ValPtr CodeGen::unaryNumOp(const AnyExpr &expr, const AnyExpr &arg, const AnyType &rtn, //
                           const ValPtrFn1 &integralFn, const ValPtrFn1 &fractionalFn) {
  return unaryExpr(expr, arg, rtn, [&](auto lhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) {
      return integralFn(lhs);
    } else if (rtn.kind().is<TypeKind::Fractional>()) {
      return fractionalFn(lhs);
    } else {
      throw BackendException("unimplemented");
    }
  });
}
ValPtr CodeGen::binaryNumOp(const AnyExpr &expr, const AnyExpr &l, const AnyExpr &r, const AnyType &rtn, //
                            const ValPtrFn2 &integralFn, const ValPtrFn2 &fractionalFn) {
  return binaryExpr(expr, l, r, rtn, [&](auto lhs, auto rhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) {
      return integralFn(lhs, rhs);
    } else if (rtn.kind().is<TypeKind::Fractional>()) {
      return fractionalFn(lhs, rhs);
    } else {
      throw BackendException("unimplemented");
    }
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
  CompileEvent ast2IR(compiler::nowMs(), compiler::elapsedNs(transformStart), "ast_to_llvm_ir", transformMsg);

  auto verifyStart = compiler::nowMono();
  auto [maybeVerifyErr, verifyMsg] = llvmc::verifyModule(cg.M);
  CompileEvent astOpt(compiler::nowMs(), compiler::elapsedNs(verifyStart), "llvm_ir_verify", verifyMsg);

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
  c.layouts = resolveLayouts(program.structs);
  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
