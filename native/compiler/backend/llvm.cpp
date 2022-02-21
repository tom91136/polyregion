#include <iostream>

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"
#include "utils.hpp"
#include "variants.hpp"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;

template <typename T> static std::string llvm_tostring(const T *t) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  t->print(rso);
  return rso.str();
}

static llvm::Value *sizeOf(llvm::IRBuilder<> &B, llvm::LLVMContext &C, llvm::Type *ptrTpe) {
  // http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
  // we want
  // %SizePtr = getelementptr %T, %T* null, i32 1
  // %Size = ptrtoint %T* %SizePtr to i32
  auto sizePtr = B.CreateGEP(                                                    //
      ptrTpe->getPointerElementType(),                                           //
      llvm::ConstantPointerNull::get(llvm::dyn_cast<llvm::PointerType>(ptrTpe)), //
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 1),                      //
      "sizePtr"                                                                  //
  );
  auto sizeVal = B.CreatePtrToInt(sizePtr, llvm::Type::getInt32Ty(C));
  return sizeVal;
}

static llvm::Value *invokeMalloc(llvm::IRBuilder<> &B, llvm::LLVMContext &C, llvm::Module *m, llvm::Value *size) {
  auto ft = llvm::FunctionType::get(llvm::Type::getInt8Ty(C)->getPointerTo(), {llvm::Type::getInt32Ty(C)}, false);
  auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "malloc", m);
  return B.CreateCall(f, size);
}

static bool isUnsigned(const Type::Any &tpe) {
  // the only unsigned type in PolyAst
  return std::holds_alternative<Type::Char>(*tpe);
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

std::pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable>
LLVMAstTransformer::mkStruct(const StructDef &def) {
  std::vector<llvm::Type *> types(def.members.size());
  std::transform(def.members.begin(), def.members.end(), types.begin(),
                 [&](const polyast::Named &n) { return mkTpe(n.tpe); });
  LLVMAstTransformer::StructMemberTable table;
  for (size_t i = 0; i < def.members.size(); ++i) {
    table[def.members[i].symbol] = i;
  }
  return {llvm::StructType::create(C, types, qualified(def.name)), table};
}

llvm::Type *LLVMAstTransformer::mkTpe(const Type::Any &tpe) {                                    //
  return variants::total(                                                                        //
      *tpe,                                                                                      //
      [&](const Type::Float &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },           //
      [&](const Type::Double &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },         //
      [&](const Type::Bool &x) -> llvm::Type * { return llvm::Type::getInt1Ty(C); },             //
      [&](const Type::Byte &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },             //
      [&](const Type::Char &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },            //
      [&](const Type::Short &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },           //
      [&](const Type::Int &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); },             //
      [&](const Type::Long &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); },            //
      [&](const Type::String &x) -> llvm::Type * { return undefined(__FILE_NAME__, __LINE__); }, //
      [&](const Type::Unit &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); },             //
      [&](const Type::Struct &x) -> llvm::Type * {
        if (auto def = polyregion::get_opt(structTypes, x.name); def) {
          return def->first->getPointerTo();
        } else {
          return undefined(__FILE_NAME__, __LINE__, "Unseen struct def: " + to_string(x));
        }
      }, //
      [&](const Type::Array &x) -> llvm::Type * {
        auto comp = mkTpe(x.component);
        return comp->isPointerTy() ? comp : comp->getPointerTo();
      } //
  );
}

llvm::Value *LLVMAstTransformer::mkSelectPtr(const Term::Select &select) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> std::pair<llvm::StructType *, StructMemberTable> {
    if (auto s = polyast::get_opt<Type::Struct>(tpe); s) {
      if (auto def = polyregion::get_opt(structTypes, s->name); def) return *def;
      else
        error(__FILE_NAME__, __LINE__, "Unseen struct type " + to_string(s->name) + " in select path" + fail());
    } else
      error(__FILE_NAME__, __LINE__, "Illegal select path involving non-struct type " + to_string(tpe) + fail());
  };

  auto selectNamed = [&](const Named &named) -> llvm::Value * {
    //  check the LUT table for known variables brought in scope by parameters
    if (auto x = polyregion::get_opt(lut, named.symbol); x) {
      auto [tpe, value] = *x;
      if (named.tpe != tpe) {
        error(__FILE_NAME__, __LINE__,
              "Named local variable type (" + to_string(named.tpe) + ") is different from located variable type (" +
                  to_string(tpe) + ")");
      }
      return value;
    } else {
      auto pool = mk_string2<std::string, std::pair<Type::Any, llvm::Value *>>(
          lut,
          [](auto &&p) {
            return "`" + p.first + "` = " + to_string(p.second.first) + "(IR=" + llvm_tostring(p.second.second) + ")";
          },
          "\n->");
      return undefined(__FILE_NAME__, __LINE__,
                       "Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
    }
  };

  if (select.init.empty()) return selectNamed(select.last); // local var lookup
  else {
    // we're in a select chain, init elements must return struct type; the head must come from LUT
    auto [head, tail] = uncons(select);
    auto local = selectNamed(head);
    auto [structTy, _] = structTypeOf(head.tpe);

    std::vector<llvm::Value *> gepIndices;
    gepIndices.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0));

    auto tpe = head.tpe;
    for (auto &path : tail) {
      auto [ignore, table] = structTypeOf(tpe);
      if (auto idx = get_opt(table, path.symbol); idx) {
        gepIndices.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), *idx));
        tpe = path.tpe;
      } else {
        return undefined(__FILE_NAME__, __LINE__,
                         "Illegal select path with unknown struct member index of name `" + to_string(path) + "`" +
                             fail());
      }
    }

    return B.CreateInBoundsGEP(structTy, local, gepIndices, qualified(select) + "_ptr");
  }
}

llvm::Value *LLVMAstTransformer::mkRef(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> llvm::Value * { return mkSelectPtr(x); },
      [&](const Term::UnitConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), 0); },
      [&](const Term::BoolConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), x.value); },
      [&](const Term::ByteConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::CharConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::ShortConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::LongConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },
      [&](const Term::FloatConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getFloatTy(C), x.value); },
      [&](const Term::DoubleConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getDoubleTy(C), x.value); },
      [&](const Term::StringConst &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__); });
}

llvm::Value *LLVMAstTransformer::mkExprValue(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

  using ValPtr = llvm::Value *;
  namespace Intr = llvm::Intrinsic;

  const auto unaryExpr = [&](const Term::Any &l, const Type::Any &rtn, const std::function<ValPtr(ValPtr)> &fn) {
    if (tpe(l) != rtn) {
      throw std::logic_error("Semantic error: lhs type " + to_string(tpe(l)) + " of binary numeric operation in " +
                             to_string(expr) + " doesn't match return type " + to_string(rtn));
    }

    return fn(conditionalLoad(mkRef(l)));
  };

  const auto binaryExpr = [&](const Term::Any &l, const Term::Any &r, const Type::Any &rtn,
                              const std::function<ValPtr(ValPtr, ValPtr)> &fn) {
    if (tpe(l) != rtn) {
      throw std::logic_error("Semantic error: lhs type " + to_string(tpe(l)) + " of binary numeric operation in " +
                             to_string(expr) + " doesn't match return type " + to_string(rtn));
    }
    if (tpe(r) != rtn) {
      throw std::logic_error("Semantic error: rhs type " + to_string(tpe(r)) + " of binary numeric operation in " +
                             to_string(expr) + " doesn't match return type " + to_string(rtn));
    }

    return fn(conditionalLoad(mkRef(l)), conditionalLoad(mkRef(r)));
  };

  const auto unaryNumOp = [&](const Term::Any &arg, const Type::Any &rtn,
                              const std::function<ValPtr(ValPtr)> &integralFn,
                              const std::function<ValPtr(ValPtr)> &fractionalFn) -> ValPtr {
    return unaryExpr(arg, rtn, [&](auto lhs) -> ValPtr {
      if (holds<TypeKind::Integral>(kind(rtn))) {
        return integralFn(lhs);
      } else if (holds<TypeKind::Fractional>(kind(rtn))) {
        return fractionalFn(lhs);
      } else {
        return undefined(__FILE_NAME__, __LINE__);
      }
    });
  };

  const auto binaryNumOp = [&](const Term::Any &l, const Term::Any &r, const Type::Any &rtn,
                               const std::function<ValPtr(ValPtr, ValPtr)> &integralFn,
                               const std::function<ValPtr(ValPtr, ValPtr)> &fractionalFn) -> ValPtr {
    return binaryExpr(l, r, rtn, [&](auto lhs, auto rhs) -> ValPtr {
      if (holds<TypeKind::Integral>(kind(rtn))) {
        return integralFn(lhs, rhs);
      } else if (holds<TypeKind::Fractional>(kind(rtn))) {
        return fractionalFn(lhs, rhs);
      } else {
        return undefined(__FILE_NAME__, __LINE__);
      }
    });
  };

  const auto externUnaryCall = [&](const std::string &name, const Type::Any &tpe, const Term::Any &arg) {
    auto t = mkTpe(tpe);
    auto ft = llvm::FunctionType::get(t, {t}, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, fn->getParent());
    return B.CreateCall(f, conditionalLoad(mkRef(arg)));
  };

  const auto externBinaryCall = [&](const std::string &name, const Type::Any &tpe, //
                                    const Term::Any &lhs, const Term::Any &rhs) {
    auto t = mkTpe(tpe);
    auto ft = llvm::FunctionType::get(t, {t, t}, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, fn->getParent());
    return B.CreateCall(f, {conditionalLoad(mkRef(lhs)), conditionalLoad(mkRef(rhs))});
  };

  const auto unaryIntrinsic = [&](llvm::Intrinsic::ID id, const Type::Any &overload, const Term::Any &arg) {
    auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, conditionalLoad(mkRef(arg)));
  };

  const auto binaryIntrinsic = [&](llvm::Intrinsic::ID id, const Type::Any &overload, //
                                   const Term::Any &lhs, const Term::Any &rhs) {
    // XXX the overload type here is about the overloading of intrinsic names, not about the parameter types
    // i.e. f32 is for foo.f32(float %a, float %b, float %c)
    auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, {conditionalLoad(mkRef(lhs)), conditionalLoad(mkRef(rhs))});
  };

  return variants::total(
      *expr, //
      [&](const Expr::UnaryIntrinsic &x) {
        auto tpe = x.rtn;
        auto arg = x.lhs;
        return variants::total(
            *x.kind,                                                                                        //
            [&](const UnaryIntrinsicKind::Sin &) -> ValPtr { return unaryIntrinsic(Intr::sin, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Cos &) -> ValPtr { return unaryIntrinsic(Intr::cos, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Tan &) -> ValPtr { return externUnaryCall("tan", tpe, arg); },    //

            [&](const UnaryIntrinsicKind::Asin &) -> ValPtr { return externUnaryCall("asin", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Acos &) -> ValPtr { return externUnaryCall("acos", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Atan &) -> ValPtr { return externUnaryCall("atan", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Sinh &) -> ValPtr { return externUnaryCall("sinh", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Cosh &) -> ValPtr { return externUnaryCall("cosh", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Tanh &) -> ValPtr { return externUnaryCall("tanh", tpe, arg); }, //

            [&](const UnaryIntrinsicKind::Signum &) -> ValPtr {
              return unaryNumOp(
                  arg, tpe, //
                  [&](auto x) {
                    auto msbOffset = x->getType()->getPrimitiveSizeInBits() - 1;
                    return B.CreateOr(B.CreateAShr(x, msbOffset), B.CreateLShr(B.CreateNeg(x), msbOffset));
                  },
                  [&](auto x) {
                    auto nan = B.CreateFCmpUNO(x, x);
                    auto zero = B.CreateFCmpUNO(x, llvm::ConstantFP::get(x->getType(), 0));
                    Term::Any magnitude;
                    if (holds<Type::Float>(tpe))        //
                      magnitude = Term::FloatConst(1);  //
                    else if (holds<Type::Double>(tpe))  //
                      magnitude = Term::DoubleConst(1); //
                    else
                      error(__FILE_NAME__, __LINE__);
                    return B.CreateSelect(B.CreateLogicalOr(nan, zero), x,
                                          binaryIntrinsic(Intr::copysign, tpe, magnitude, arg));
                  });
            }, //
            [&](const UnaryIntrinsicKind::Abs &) -> ValPtr {
              return unaryNumOp(
                  arg, tpe, //
                  [&](auto x) { return unaryIntrinsic(Intr::abs, tpe, arg); },
                  [&](auto x) { return unaryIntrinsic(Intr::fabs, tpe, arg); });
            },                                                                                                  //
            [&](const UnaryIntrinsicKind::Round &) -> ValPtr { return unaryIntrinsic(Intr::round, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Ceil &) -> ValPtr { return unaryIntrinsic(Intr::ceil, tpe, arg); },   //
            [&](const UnaryIntrinsicKind::Floor &) -> ValPtr { return unaryIntrinsic(Intr::floor, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Rint &) -> ValPtr { return unaryIntrinsic(Intr::rint, tpe, arg); },   //

            [&](const UnaryIntrinsicKind::Sqrt &) -> ValPtr { return unaryIntrinsic(Intr::sqrt, tpe, arg); },   //
            [&](const UnaryIntrinsicKind::Cbrt &) -> ValPtr { return externUnaryCall("cbrt", tpe, arg); },      //
            [&](const UnaryIntrinsicKind::Exp &) -> ValPtr { return unaryIntrinsic(Intr::exp, tpe, arg); },     //
            [&](const UnaryIntrinsicKind::Expm1 &) -> ValPtr { return externUnaryCall("expm1", tpe, arg); },    //
            [&](const UnaryIntrinsicKind::Log &) -> ValPtr { return unaryIntrinsic(Intr::log, tpe, arg); },     //
            [&](const UnaryIntrinsicKind::Log1p &) -> ValPtr { return externUnaryCall("log1p", tpe, arg); },    //
            [&](const UnaryIntrinsicKind::Log10 &) -> ValPtr { return unaryIntrinsic(Intr::log10, tpe, arg); }, //

            [&](const UnaryIntrinsicKind::BNot &) -> ValPtr {
              return unaryExpr(arg, tpe, [&](auto x) { return B.CreateNot(x); });
            });
      },
      [&](const Expr::BinaryIntrinsic &x) {
        return variants::total(
            *x.kind, //
            [&](const BinaryIntrinsicKind::Add &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateAdd(l, r, key); },
                  [&](auto l, auto r) { return B.CreateFAdd(l, r, key); });
            },
            [&](const BinaryIntrinsicKind::Sub &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSub(l, r, key); },
                  [&](auto l, auto r) { return B.CreateFSub(l, r, key); });
            },
            [&](const BinaryIntrinsicKind::Div &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSDiv(l, r, key); },
                  [&](auto l, auto r) { return B.CreateFDiv(l, r, key); });
            },
            [&](const BinaryIntrinsicKind::Mul &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateMul(l, r, key); },
                  [&](auto l, auto r) { return B.CreateFMul(l, r, key); });
            },
            [&](const BinaryIntrinsicKind::Rem &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSRem(l, r, key); },
                  [&](auto l, auto r) { return B.CreateFRem(l, r, key); });
            },
            [&](const BinaryIntrinsicKind::Pow &) -> ValPtr {
              return binaryIntrinsic(Intr::pow, x.rtn, x.lhs, x.rhs); //
            },

            [&](const BinaryIntrinsicKind::Min &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSelect(B.CreateICmpSLT(l, r), l, r); },
                  [&](auto l, auto r) { return B.CreateMinimum(l, r); });
            },
            [&](const BinaryIntrinsicKind::Max &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSelect(B.CreateICmpSLT(l, r), r, l); },
                  [&](auto l, auto r) { return B.CreateMaximum(l, r); });
            },

            [&](const BinaryIntrinsicKind::Atan2 &) -> ValPtr {
              return externBinaryCall("atan2", x.rtn, x.lhs, x.rhs);
            }, //
            [&](const BinaryIntrinsicKind::Hypot &) -> ValPtr {
              return externBinaryCall("hypot", x.rtn, x.lhs, x.rhs);
            }, //

            [&](const BinaryIntrinsicKind::BAnd &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateAnd(l, r); });
            },
            [&](const BinaryIntrinsicKind::BOr &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateOr(l, r); });
            },
            [&](const BinaryIntrinsicKind::BXor &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateXor(l, r); });
            },
            [&](const BinaryIntrinsicKind::BSL &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateShl(l, r); });
            },
            [&](const BinaryIntrinsicKind::BSR &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateAShr(l, r); });
            },
            [&](const BinaryIntrinsicKind::BZSR &) -> ValPtr {
              return binaryExpr(x.lhs, x.rhs, x.rtn, [&](auto l, auto r) { return B.CreateLShr(l, r); });
            } //
        );
      },
      [&](const Expr::UnaryLogicIntrinsic &x) -> ValPtr {
        return variants::total( //
            *x.kind,            //
            [&](const UnaryLogicIntrinsicKind::Not &) -> ValPtr {
              return B.CreateNot(conditionalLoad(mkRef(x.lhs)), key);
            });
      },
      [&](const Expr::BinaryLogicIntrinsic &x) -> ValPtr {
        auto lhs = conditionalLoad(mkRef(x.lhs));
        auto rhs = conditionalLoad(mkRef(x.rhs));
        return variants::total(
            *x.kind, //
            [&](const BinaryLogicIntrinsicKind::Eq &) -> ValPtr { return B.CreateICmpEQ(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Neq &) -> ValPtr { return B.CreateICmpNE(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::And &) -> ValPtr { return B.CreateLogicalAnd(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Or &) -> ValPtr { return B.CreateLogicalOr(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Lte &) -> ValPtr { return B.CreateICmpSLE(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Gte &) -> ValPtr { return B.CreateICmpSGE(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Lt &) -> ValPtr { return B.CreateICmpSLT(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Gt &x) -> ValPtr { return B.CreateICmpSGT(lhs, rhs); });
      },
      [&](const Expr::Cast &x) -> ValPtr {
        // we only allow widening or narrowing of integral and fractional types
        // pointers are not allowed to participate on either end
        auto from = conditionalLoad(mkRef(x.from));
        auto fromTpe = mkTpe(tpe(x.from));
        auto toTpe = mkTpe(x.as);
        enum class NumKind { Fractional, Integral };

        auto fromKind = variants::total(
            *kind(tpe(x.from)), [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion from ref type (" + to_string(fromTpe) +
                                     ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { error(__FILE_NAME__, __LINE__, "none!?"); });

        auto toKind = variants::total(
            *kind(x.as), //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion to ref type (" + to_string(fromTpe) +
                                     ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { error(__FILE_NAME__, __LINE__, "none!?"); });

        if (fromKind == NumKind::Fractional && toKind == NumKind::Integral) {

          // to the equally sized integral type first if narrowing; XXX narrowing directly produces a poison value

          llvm::Value *c = nullptr;
          if (fromTpe->getPrimitiveSizeInBits() > toTpe->getPrimitiveSizeInBits() || true) {
            auto min32BitIntBits = std::max<llvm::TypeSize::ScalarTy>(32, toTpe->getPrimitiveSizeInBits());
            auto toTpeMaxInFp = llvm::ConstantFP::get(fromTpe, double(nIntMax(min32BitIntBits)));
            auto toTpeMinInFp = llvm::ConstantFP::get(fromTpe, double(nIntMin(min32BitIntBits)));
            auto min32BitIntTy = llvm::Type::getIntNTy(C, min32BitIntBits);
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
          return isUnsigned(tpe(x.from)) ? B.CreateUIToFP(from, toTpe) : B.CreateSIToFP(from, toTpe);
        } else if (fromKind == NumKind::Integral && toKind == NumKind::Integral) {
          return B.CreateIntCast(from, toTpe, !isUnsigned(tpe(x.from)), "integral_cast");
        } else if (fromKind == NumKind::Fractional && toKind == NumKind::Fractional) {
          return B.CreateFPCast(from, toTpe, "fractional_cast");
        } else
          error(__FILE_NAME__, __LINE__, "unhandled cast");
      },
      [&](const Expr::Alias &x) -> ValPtr { return mkRef(x.ref); },
      [&](const Expr::Invoke &x) -> ValPtr {
        //        auto lhs = mkRef(x.lhs );
        return undefined(__FILE_NAME__, __LINE__, "Unimplemented invoke:`" + repr(x) + "`");
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (auto arrTpe = get_opt<Type::Array>(x.lhs.tpe); arrTpe) {
          auto ty = mkTpe(arrTpe->component);
          return B.CreateInBoundsGEP(ty->isPointerTy() ? ty->getPointerElementType() : ty, //
                                     mkSelectPtr(x.lhs),                                   //
                                     conditionalLoad(mkRef(x.idx)), key + "_ptr");
        } else {
          throw std::logic_error("Semantic error: array index not called on array type (" + to_string(x.lhs.tpe) +
                                 ")(" + repr(x) + ")");
        }
      },
      [&](const Expr::Alloc &x) -> ValPtr { //
        auto size = mkRef(x.size);
        auto elemSize = sizeOf(B, C, mkTpe(x.witness));
        auto ptr = invokeMalloc(B, C, fn->getParent(), B.CreateMul(size, elemSize));
        return B.CreateBitCast(ptr, mkTpe(x.witness));
      });
}

llvm::Value *LLVMAstTransformer::conditionalLoad(llvm::Value *rhs) {
  return rhs->getType()->isPointerTy() // deref the rhs if it's a pointer
             ? B.CreateLoad(rhs->getType()->getPointerElementType(), rhs)
             : rhs;
}

void LLVMAstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn) {
  return variants::total(
      *stmt,
      [&](const Stmt::Comment &x) { /* discard comments */
                                    return;
      },
      [&](const Stmt::Var &x) {
        // [T : ref] =>> t:T  = _        ; lut += &t
        // [T : ref] =>> t:T* = &(rhs:T) ; lut += t
        // [T : val] =>> t:T  =   rhs:T  ; lut += &t

        if (x.expr && tpe(*x.expr) != x.name.tpe) {
          throw std::logic_error("Semantic error: name type and rhs expr mismatch (" + repr(x) + ")");
        }
        auto rhs = map_opt(x.expr, [&](auto &&expr) { return mkExprValue(expr, fn, x.name.symbol + "_var_rhs"); });
        if (std::holds_alternative<Type::Array>(*x.name.tpe)) {
          if (rhs) {
            lut[x.name.symbol] = {x.name.tpe, *rhs};
          } else {
            undefined(__FILE_NAME__, __LINE__, "var array with no expr?");
          }
        } else if (std::holds_alternative<Type::Struct>(*x.name.tpe)) {
          if (rhs) {
            lut[x.name.symbol] = {x.name.tpe, *rhs};
          } else {
            // otherwise, heap allocate the struct and return the pointer to that
            llvm::Type *structPtrTy = mkTpe(x.name.tpe);
            if (!structPtrTy->isPointerTy()) {
              throw std::logic_error("The LLVM struct type `" + llvm_tostring(structPtrTy) +
                                     "` was not a pointer to the struct " + repr(x.name.tpe) +
                                     " in stmt:" + repr(stmt));
            }
            auto elemSize = sizeOf(B, C, structPtrTy);
            auto ptr = invokeMalloc(B, C, fn->getParent(), elemSize);
            lut[x.name.symbol] = {x.name.tpe, B.CreateBitCast(ptr, structPtrTy)};
          }
        } else {
          // plain primitives now
          auto ptr = B.CreateAlloca(mkTpe(x.name.tpe), nullptr, x.name.symbol + "_stack_ptr");
          if (rhs) {
            B.CreateStore(conditionalLoad(*rhs), ptr); //
          }
          lut[x.name.symbol] = {x.name.tpe, ptr};
        }
      },
      [&](const Stmt::Mut &x) {
        // [T : ref]        =>> t   := &(rhs:T) ; lut += t
        // [T : ref {u: U}] =>> t.u := &(rhs:U)
        // [T : val]        =>> t   :=   rhs:T
        if (tpe(x.expr) != x.name.tpe) {
          throw std::logic_error("Semantic error: name type (" + to_string(tpe(x.expr)) + ") and rhs expr (" +
                                 to_string(x.name.tpe) + ") mismatch (" + repr(x) + ")");
        }
        auto rhs = mkExprValue(x.expr, fn, qualified(x.name) + "_mut");
        auto lhsPtr = mkSelectPtr(x.name);

        if (std::holds_alternative<TypeKind::Ref>(*kind(tpe(x.expr)))) {
          if (x.name.init.empty()) { // local var, replace entry in lut or memcpy
            if (!rhs->getType()->isPointerTy()) {
              throw std::logic_error("Semantic error: rhs isn't a pointer type (" + repr(x) + ")");
            }
            if (!x.copy) lut[x.name.last.symbol] = {x.name.tpe, rhs};
            else {

              // XXX CreateMemCpyInline has an immarg size so %size must be a pure constant, this is crazy
              B.CreateMemCpy(lhsPtr, {}, rhs, {}, sizeOf(B, C, lhsPtr->getType()));
            }
          } else {
            B.CreateStore(conditionalLoad(rhs), mkSelectPtr(x.name)); // ignore copy, modify struct member
          }
        } else {
          B.CreateStore(conditionalLoad(rhs), mkSelectPtr(x.name)); // ignore copy
        }
      },
      [&](const Stmt::Update &x) {
        if (auto arrTpe = get_opt<Type::Array>(x.lhs.tpe); arrTpe) {
          if (arrTpe->component != tpe(x.value)) {
            throw std::logic_error("Semantic error: array component type (" + to_string(arrTpe->component) +
                                   ") and rhs expr (" + to_string(tpe(x.value)) + ") mismatch (" + repr(x) + ")");
          } else {
            auto dest = mkSelectPtr(x.lhs);
            auto ptr = B.CreateInBoundsGEP(                              //
                dest->getType()->getPointerElementType(), dest,          //
                conditionalLoad(mkRef(x.idx)), qualified(x.lhs) + "_ptr" //
            );                                                           //
            B.CreateStore(conditionalLoad(mkRef(x.value)), ptr);
          }
        } else {
          throw std::logic_error("Semantic error: array update not called on array type (" + to_string(x.lhs.tpe) +
                                 ")(" + repr(x) + ")");
        }
      },
      [&](const Stmt::While &x) {
        auto loopTest = llvm::BasicBlock::Create(C, "loop_test", fn);
        auto loopBody = llvm::BasicBlock::Create(C, "loop_body", fn);
        auto loopExit = llvm::BasicBlock::Create(C, "loop_exit", fn);
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          auto continue_ = mkExprValue(x.cond, fn, "loop");
          B.CreateCondBr(conditionalLoad(continue_), loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          for (auto &body : x.body)
            mkStmt(body, fn);
          B.CreateBr(loopTest);
        }
        B.SetInsertPoint(loopExit);
      },
      [&](const Stmt::Break &x) { undefined(__FILE_NAME__, __LINE__, "break"); }, //
      [&](const Stmt::Cont &x) { undefined(__FILE_NAME__, __LINE__, "cont"); },   //
      [&](const Stmt::Cond &x) {
        auto condTrue = llvm::BasicBlock::Create(C, "cond_true", fn);
        auto condFalse = llvm::BasicBlock::Create(C, "cond_false", fn);
        auto condExit = llvm::BasicBlock::Create(C, "cond_exit", fn);
        B.CreateCondBr(conditionalLoad(mkExprValue(x.cond, fn, "cond")), condTrue, condFalse);
        {
          B.SetInsertPoint(condTrue);
          for (auto &body : x.trueBr)
            mkStmt(body, fn);
          B.CreateBr(condExit);
        }
        {
          B.SetInsertPoint(condFalse);
          for (auto &body : x.falseBr)
            mkStmt(body, fn);
          B.CreateBr(condExit);
        }
        B.SetInsertPoint(condExit);
      },
      [&](const Stmt::Return &x) {
        auto rtnTpe = tpe(x.value);
        if (std::holds_alternative<Type::Unit>(*rtnTpe)) {
          B.CreateRetVoid();
        } else {
          if (std::holds_alternative<TypeKind::Ref>(*kind(rtnTpe))) {
            B.CreateRet((mkExprValue(x.value, fn, "return")));
          } else {
            B.CreateRet(conditionalLoad(mkExprValue(x.value, fn, "return")));
          }
        }
      } //

  );
}

std::pair<std::optional<std::string>, std::string>
LLVMAstTransformer::transform(const std::unique_ptr<llvm::Module> &module, const Program &program) {

  auto fnTree = program.entry;

  // set up the struct defs first so that structs in params work
  std::transform(                                    //
      program.defs.begin(), program.defs.end(),      //
      std::inserter(structTypes, structTypes.end()), //
      [&](auto &x) -> std::pair<Sym, std::pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable>> {
        return {x.name, mkStruct(x)};
      });

  auto paramTpes = map_vec<Named, llvm::Type *>(fnTree.args, [&](auto &&named) { return mkTpe(named.tpe); });

  auto rtnTpe = variants::total(
      *kind(fnTree.rtn),                                                               //
      [&](const TypeKind::Fractional &) -> llvm::Type * { return mkTpe(fnTree.rtn); }, //
      [&](const TypeKind::Integral &) -> llvm::Type * { return mkTpe(fnTree.rtn); },   //
      [&](const TypeKind::None &) -> llvm::Type * { return mkTpe(fnTree.rtn); },       //
      [&](const TypeKind::Ref &) -> llvm::Type * { return mkTpe(fnTree.rtn); }         //
  );

  auto fnTpe = llvm::FunctionType::get(rtnTpe, {paramTpes}, false);

  auto *fn = llvm::Function::Create(fnTpe, llvm::Function::ExternalLinkage, "lambda", *module);

  auto *entry = llvm::BasicBlock::Create(C, "entry", fn);
  B.SetInsertPoint(entry);

  // add function params to the lut first as function body will need these at some point
  std::transform(                                          //
      fn->arg_begin(), fn->arg_end(), fnTree.args.begin(), //
      std::inserter(lut, lut.end()),                       //
      [&](auto &arg, const auto &named) -> std::pair<std::string, std::pair<Type::Any, llvm::Value *>> {
        arg.setName(named.symbol);

        if (std::holds_alternative<TypeKind::Ref>(*kind(named.tpe))) {
          return {named.symbol, {named.tpe, &arg}};
        } else {
          auto stack = B.CreateAlloca(mkTpe(named.tpe), nullptr, named.symbol + "_stack_ptr");
          B.CreateStore(&arg, stack);
          return {named.symbol, {named.tpe, stack}};
        }
      });

  for (auto &stmt : fnTree.body) {
    mkStmt(stmt, fn);
  }

  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  module->print(irOut, nullptr);

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (llvm::verifyModule(*module, &errOut)) {
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

std::pair<std::optional<std::string>, std::string>
LLVMAstTransformer::optimise(const std::unique_ptr<llvm::Module> &module) {
  llvm::PassManagerBuilder builder;
  builder.OptLevel = 3;
  llvm::legacy::PassManager m;
  builder.populateModulePassManager(m);
  m.add(llvm::createInstructionCombiningPass());
  m.run(*module);

  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  module->print(irOut, nullptr);

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (llvm::verifyModule(*module, &errOut)) {
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

std::optional<llvm::StructType *> LLVMAstTransformer::lookup(const Sym &s) {
  if (auto x = get_opt(structTypes, s); x) return x->first;
  else
    return {};
}

backend::LLVM::LLVM() = default; // cache(), jit(mkJit(cache)) {}

compiler::Compilation backend::LLVM::run(const Program &program) {
  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("test", *ctx);

  LLVMAstTransformer xform(*ctx);

  auto rawXform = compiler::nowMono();
  auto [rawError, rawIR] = xform.transform(mod, program);
  auto rawXformElapsed = compiler::elapsedNs(rawXform);
  auto optXform = compiler::nowMono();
  auto [optError, optIR] = xform.optimise(mod);
  auto optXformElapsed = compiler::elapsedNs(optXform);

  compiler::Event ast2IR(compiler::nowMs(), rawXformElapsed, "ast_to_llvm_ir", rawIR);
  compiler::Event astOpt(compiler::nowMs(), optXformElapsed, "llvm_ir_opt", optIR);

  if (rawError || optError) {
    return compiler::Compilation({},               //
                                 {ast2IR, astOpt}, //
                                 mk_string<std::string>(
                                     {rawError.value_or(""), optError.value_or("")}, [](auto &&x) { return x; }, "\n"));
  }

  auto c = llvmc::compileModule(true, std::move(mod), *ctx);

  // at this point we know the target machine, so we derive the struct layout here
  for (const auto &def : program.defs) {
    auto x = xform.lookup(def.name);
    if (!x) {
      throw std::logic_error("Missing struct def:" + repr(def));
    } else {
      // FIXME this needs to use the same LLVM target machine context as the compiler
      c.layouts.emplace_back(compiler::layoutOf(def));
    }
  }

  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
