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

static llvm::Value *load(llvm::IRBuilder<> &B, llvm::Value *rhs) {
  return B.CreateLoad(rhs->getType()->getPointerElementType(), rhs);
}

llvm::Value *LLVMAstTransformer::invokeMalloc(llvm::Function *parent, llvm::Value *size) {
  return B.CreateCall(mkExternalFn(parent, Type::Array(Type::Byte()), "malloc", {Type::Int()}), size);
}

static bool isUnsigned(const Type::Any &tpe) {
  return holds<Type::Char>(tpe); // the only unsigned type in PolyAst
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

Pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable> LLVMAstTransformer::mkStruct(const StructDef &def) {
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

          auto pool = mk_string2<Sym, Pair<llvm::StructType *, StructMemberTable>>(
              structTypes,
              [](auto &&p) { return "`" + to_string(p.first) + "`" + " = " + std::to_string(p.second.second.size()); },
              "\n->");

          return undefined(__FILE_NAME__, __LINE__, "Unseen struct def: " + to_string(x) + ", table=\n" + pool);
        }
      }, //
      [&](const Type::Array &x) -> llvm::Type * {
        auto comp = mkTpe(x.component);
        return comp->isPointerTy() ? comp : comp->getPointerTo();
      } //
  );
}

llvm::Value *LLVMAstTransformer::findStackVar(const Named &named) {
  //  check the LUT table for known variables defined by var or brought in scope by parameters
  if (auto x = polyregion::get_opt(stackVarPtrs, named.symbol); x) {
    auto [tpe, value] = *x;
    if (named.tpe != tpe) {
      error(__FILE_NAME__, __LINE__,
            "Named local variable (" + to_string(named) + ") has different type from LUT (" + to_string(tpe) + ")");
    }
    return value;
  } else {
    auto pool = mk_string2<std::string, Pair<Type::Any, llvm::Value *>>(
        stackVarPtrs,
        [](auto &&p) {
          return "`" + p.first + "` = " + to_string(p.second.first) + "(IR=" + llvm_tostring(p.second.second) + ")";
        },
        "\n->");
    return undefined(__FILE_NAME__, __LINE__, "Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
  }
}

llvm::Value *LLVMAstTransformer::mkSelectPtr(const Term::Select &select) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> Pair<llvm::StructType *, StructMemberTable> {
    if (auto s = polyast::get_opt<Type::Struct>(tpe); s) {
      if (auto def = polyregion::get_opt(structTypes, s->name); def) return *def;
      else
        error(__FILE_NAME__, __LINE__, "Unseen struct type " + to_string(s->name) + " in select path" + fail());
    } else
      error(__FILE_NAME__, __LINE__, "Illegal select path involving non-struct type " + to_string(tpe) + fail());
  };

  if (select.init.empty()) return findStackVar(select.last); // local var lookup
  else {
    // we're in a select chain, init elements must return struct type; the head must come from LUT
    auto [head, tail] = uncons(select);
    auto localRoot = load(B, findStackVar(head));
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

    return B.CreateInBoundsGEP(structTy, localRoot, gepIndices, qualified(select) + "_ptr");
  }
}

llvm::Value *LLVMAstTransformer::mkTermVal(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> llvm::Value * { return load(B, mkSelectPtr(x)); },
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

llvm::Function *LLVMAstTransformer::mkExternalFn(llvm::Function *parent, const Type::Any &rtn, const std::string &name,
                                                 const std::vector<Type::Any> &args) {
  const Signature s(Sym({name}), {}, args, rtn);
  if (functions.find(s) == functions.end()) {
    auto llvmArgs = map_vec<Type::Any, llvm::Type *>(args, [&](auto t) { return mkTpe(t); });
    auto ft = llvm::FunctionType::get(mkTpe(rtn), llvmArgs, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, parent->getParent());
    functions[s] = f;
  }
  return functions[s];
}

llvm::Value *LLVMAstTransformer::mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

  using ValPtr = llvm::Value *;
  namespace Intr = llvm::Intrinsic;

  const auto unaryExpr = [&](const Term::Any &l, const Type::Any &rtn, const std::function<ValPtr(ValPtr)> &fn) {
    if (tpe(l) != rtn) {
      throw std::logic_error("Semantic error: lhs type " + to_string(tpe(l)) + " of binary numeric operation in " +
                             to_string(expr) + " doesn't match return type " + to_string(rtn));
    }

    return fn(mkTermVal(l));
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

    return fn(mkTermVal(l), mkTermVal(r));
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
    return B.CreateCall(mkExternalFn(fn, tpe, name, {tpe}), mkTermVal(arg));
  };

  const auto externBinaryCall = [&](const std::string &name, const Type::Any &tpe, //
                                    const Term::Any &lhs, const Term::Any &rhs) {
    return B.CreateCall(mkExternalFn(fn, tpe, name, {tpe, tpe}), {mkTermVal(lhs), mkTermVal(rhs)});
  };

  const auto unaryIntrinsic = [&](llvm::Intrinsic::ID id, const Type::Any &overload, const Term::Any &arg) {
    auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, mkTermVal(arg));
  };

  const auto binaryIntrinsic = [&](llvm::Intrinsic::ID id, const Type::Any &overload, //
                                   const Term::Any &lhs, const Term::Any &rhs) {
    // XXX the overload type here is about the overloading of intrinsic names, not about the parameter types
    // i.e. f32 is for foo.f32(float %a, float %b, float %c)
    auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, {mkTermVal(lhs), mkTermVal(rhs)});
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
            },
            [&](const UnaryIntrinsicKind::Pos &) -> ValPtr {
              return unaryNumOp(
                  arg, tpe,                  //
                  [&](auto x) { return x; }, //
                  [&](auto x) { return x; });
            },
            [&](const UnaryIntrinsicKind::Neg &) -> ValPtr {
              return unaryNumOp(
                  arg, tpe,                               //
                  [&](auto x) { return B.CreateNeg(x); }, //
                  [&](auto x) { return B.CreateFNeg(x); });
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
            [&](const UnaryLogicIntrinsicKind::Not &) -> ValPtr { return B.CreateNot(mkTermVal(x.lhs), key); });
      },
      [&](const Expr::BinaryLogicIntrinsic &x) -> ValPtr {
        auto lhs = mkTermVal(x.lhs);
        auto rhs = mkTermVal(x.rhs);

        if (tpe(x.lhs) != tpe(x.rhs)) {
          throw std::logic_error("rhs type" + to_string(tpe(x.lhs)) + " != rhs type" + to_string(tpe(x.rhs)) +
                                 " for binary logic expr:" + to_string(x));
        }

        return variants::total(
            *x.kind, //
            [&](const BinaryLogicIntrinsicKind::Eq &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                 //
                  [&](auto l, auto r) { return B.CreateICmpEQ(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOEQ(lhs, rhs); } //
              );
            },
            [&](const BinaryLogicIntrinsicKind::Neq &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                 //
                  [&](auto l, auto r) { return B.CreateICmpNE(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpONE(lhs, rhs); } //
              );
            },
            [&](const BinaryLogicIntrinsicKind::And &) -> ValPtr { return B.CreateLogicalAnd(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Or &) -> ValPtr { return B.CreateLogicalOr(lhs, rhs); },
            [&](const BinaryLogicIntrinsicKind::Lte &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                  //
                  [&](auto l, auto r) { return B.CreateICmpSLE(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOLE(lhs, rhs); }  //
              );
            },
            [&](const BinaryLogicIntrinsicKind::Gte &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                  //
                  [&](auto l, auto r) { return B.CreateICmpSGE(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOGE(lhs, rhs); }  //
              );
            },
            [&](const BinaryLogicIntrinsicKind::Lt &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                  //
                  [&](auto l, auto r) { return B.CreateICmpSLT(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOLT(lhs, rhs); }  //
              );
            },
            [&](const BinaryLogicIntrinsicKind::Gt &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                                  //
                  [&](auto l, auto r) { return B.CreateICmpSGT(lhs, rhs); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOGT(lhs, rhs); }  //
              );
            });
      },
      [&](const Expr::Cast &x) -> ValPtr {
        // we only allow widening or narrowing of integral and fractional types
        // pointers are not allowed to participate on either end
        auto from = mkTermVal(x.from);
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
      [&](const Expr::Alias &x) -> ValPtr { return mkTermVal(x.ref); },
      [&](const Expr::Invoke &x) -> ValPtr {
        //        auto lhs = mkTermVal(x.lhs );
        return undefined(__FILE_NAME__, __LINE__, "Unimplemented invoke:`" + repr(x) + "`");
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (auto arrTpe = get_opt<Type::Array>(x.lhs.tpe); arrTpe) {
          auto ty = mkTpe(arrTpe->component);

          auto ptr = B.CreateInBoundsGEP(ty->isPointerTy() ? ty->getPointerElementType() : ty, //
                                         mkTermVal(x.lhs),                                     //
                                         mkTermVal(x.idx), key + "_ptr");
          if (holds<TypeKind::Ref>(kind(arrTpe->component))) {
            return ptr;
          } else {
            return load(B, ptr);
          }
        } else {
          throw std::logic_error("Semantic error: array index not called on array type (" + to_string(x.lhs.tpe) +
                                 ")(" + repr(x) + ")");
        }
      },
      [&](const Expr::Alloc &x) -> ValPtr { //
        auto size = mkTermVal(x.size);
        auto elemSize = sizeOf(B, C, mkTpe(x.witness));
        auto ptr = invokeMalloc(fn, B.CreateMul(size, elemSize));
        return B.CreateBitCast(ptr, mkTpe(x.witness));
      });
}

llvm::Value *LLVMAstTransformer::conditionalLoad(llvm::Value *rhs) {
  return rhs->getType()->isPointerTy() // deref the rhs if it's a pointer
             ? B.CreateLoad(rhs->getType()->getPointerElementType(), rhs)
             : rhs;
}

BlockKind LLVMAstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx = {}) {
  return variants::total(
      *stmt,
      [&](const Stmt::Comment &x) -> BlockKind { // discard comments
        return BlockKind::Normal;
      },
      [&](const Stmt::Var &x) -> BlockKind {
        // [T : ref] =>> t:T  = _        ; lut += &t
        // [T : ref] =>> t:T* = &(rhs:T) ; lut += t
        // [T : val] =>> t:T  =   rhs:T  ; lut += &t

        if (x.expr && tpe(*x.expr) != x.name.tpe) {
          throw std::logic_error("Semantic error: name type " + to_string(x.name.tpe) + " and rhs expr type " +
                                 to_string(tpe(*x.expr)) + " mismatch (" + repr(x) + ")");
        }

        auto tpe = mkTpe(x.name.tpe);
        auto stackPtr = B.CreateAlloca(tpe, nullptr, x.name.symbol + "_stack_ptr");
        auto rhs = map_opt(x.expr, [&](auto &&expr) { return mkExprVal(expr, fn, x.name.symbol + "_var_rhs"); });

        stackVarPtrs[x.name.symbol] = {x.name.tpe, stackPtr};

        if (holds<Type::Array>(x.name.tpe)) {
          if (rhs) {
            B.CreateStore(*rhs, stackPtr);
          } else
            undefined(__FILE_NAME__, __LINE__, "var array with no expr?");
        } else if (holds<Type::Struct>(x.name.tpe)) {
          if (rhs) {
            B.CreateStore(*rhs, stackPtr);
          } else { // otherwise, heap allocate the struct and return the pointer to that
            if (!tpe->isPointerTy()) {
              throw std::logic_error("The LLVM struct type `" + llvm_tostring(tpe) +
                                     "` was not a pointer to the struct " + repr(x.name.tpe) +
                                     " in stmt:" + repr(stmt));
            }
            auto elemSize = sizeOf(B, C, tpe);
            auto ptr = invokeMalloc(fn, elemSize);
            B.CreateStore(B.CreateBitCast(ptr, tpe), stackPtr); //
          }
        } else { // any other primitives
          if (rhs) {
            B.CreateStore(*rhs, stackPtr); //
          }
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::Mut &x) -> BlockKind {
        // [T : ref]        =>> t   := &(rhs:T) ; lut += t
        // [T : ref {u: U}] =>> t.u := &(rhs:U)
        // [T : val]        =>> t   :=   rhs:T
        if (tpe(x.expr) != x.name.tpe) {
          throw std::logic_error("Semantic error: name type (" + to_string(tpe(x.expr)) + ") and rhs expr (" +
                                 to_string(x.name.tpe) + ") mismatch (" + repr(x) + ")");
        }
        auto rhs = mkExprVal(x.expr, fn, qualified(x.name) + "_mut");

        if (x.name.init.empty()) { // local var
          auto stackPtr = findStackVar(x.name.last);
          B.CreateStore(rhs, stackPtr);
        } else { // struct member select
          B.CreateStore(rhs, mkSelectPtr(x.name));
        }

        //        if (holds<TypeKind::Ref>(kind(tpe(x.expr)))) {
        //          if (x.name.init.empty()) { // local var, replace entry in lut or memcpy
        //            if (!rhs->getType()->isPointerTy()) {
        //              throw std::logic_error("Semantic error: rhs isn't a pointer type (" + repr(x) + ")");
        //            }
        //            if (!x.copy) {
        //              //              lut[x.name.last.symbol] = {x.name.tpe, rhs};
        //              // RHS Is a pointer here
        //              // lhs IS a pointer here
        //              B.CreateStore((rhs), mkSelectVal(x.name)); // ignore copy
        //            } else {
        //              auto lhsPtr = mkSelectVal(x.name);
        //              // XXX CreateMemCpyInline has an immarg size so %size must be a pure constant, this is crazy
        //              B.CreateMemCpy(lhsPtr, {}, rhs, {}, sizeOf(B, C, lhsPtr->getType()));
        //            }
        //          } else {
        //            if (holds<Type::Struct>(tpe(x.expr))) {
        //              B.CreateStore((rhs), mkSelectVal(x.name)); // ignore copy, modify struct member
        //            } else {
        //              B.CreateStore((rhs), mkSelectVal(x.name)); // ignore copy, modify struct member
        //            }
        //          }
        //        } else {
        //          B.CreateStore((rhs), mkSelectVal(x.name)); // ignore copy
        //        }
        return BlockKind::Normal;
      },
      [&](const Stmt::Update &x) -> BlockKind {
        if (auto arrTpe = get_opt<Type::Array>(x.lhs.tpe); arrTpe) {
          if (arrTpe->component != tpe(x.value)) {
            throw std::logic_error("Semantic error: array component type (" + to_string(arrTpe->component) +
                                   ") and rhs expr (" + to_string(tpe(x.value)) + ") mismatch (" + repr(x) + ")");
          } else {
            auto dest = mkTermVal(x.lhs);
            auto ptr = B.CreateInBoundsGEP(                     //
                dest->getType()->getPointerElementType(), dest, //
                mkTermVal(x.idx), qualified(x.lhs) + "_ptr"     //
            );                                                  //

            if (holds<Type::Struct>(tpe(x.value))) {
              B.CreateStore(load(B, mkTermVal(x.value)), ptr);
            } else {
              B.CreateStore(mkTermVal(x.value), ptr);
            }
          }
        } else {
          throw std::logic_error("Semantic error: array update not called on array type (" + to_string(x.lhs.tpe) +
                                 ")(" + repr(x) + ")");
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::While &x) -> BlockKind {
        auto loopTest = llvm::BasicBlock::Create(C, "loop_test", fn);
        auto loopBody = llvm::BasicBlock::Create(C, "loop_body", fn);
        auto loopExit = llvm::BasicBlock::Create(C, "loop_exit", fn);
        WhileCtx ctx{.exit = loopExit, .test = loopTest};
        B.CreateBr(loopTest);

        {
          B.SetInsertPoint(loopTest);
          auto kind = BlockKind::Normal;
          for (auto &test : x.tests)
            kind = mkStmt(test, fn, {ctx});
          if (kind != BlockKind::Terminal) {
            auto continue_ = mkTermVal(x.cond);
            B.CreateCondBr((continue_), loopBody, loopExit);
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
      [&](const Stmt::Break &x) -> BlockKind {
        if (whileCtx) {
          B.CreateBr(whileCtx->exit);
        } else {
          undefined(__FILE_NAME__, __LINE__, "orphaned break!");
        }
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cont &x) -> BlockKind {
        if (whileCtx) {
          B.CreateBr(whileCtx->test);
        } else {
          undefined(__FILE_NAME__, __LINE__, "orphaned cont!");
        }
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cond &x) -> BlockKind {
        auto condTrue = llvm::BasicBlock::Create(C, "cond_true", fn);
        auto condFalse = llvm::BasicBlock::Create(C, "cond_false", fn);
        auto condExit = llvm::BasicBlock::Create(C, "cond_exit", fn);
        B.CreateCondBr((mkExprVal(x.cond, fn, "cond")), condTrue, condFalse);
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
        B.SetInsertPoint(condExit);
        return BlockKind::Terminal;
      },
      [&](const Stmt::Return &x) -> BlockKind {
        auto rtnTpe = tpe(x.value);
        if (holds<Type::Unit>(rtnTpe)) {
          B.CreateRetVoid();
        } else {
          if (holds<TypeKind::Ref>(kind(rtnTpe))) {
            B.CreateRet(((mkExprVal(x.value, fn, "return"))));
          } else {
            B.CreateRet((mkExprVal(x.value, fn, "return")));
          }
        }
        return BlockKind::Terminal;
      } //

  );
}

Pair<Opt<std::string>, std::string> LLVMAstTransformer::transform(const std::unique_ptr<llvm::Module> &module,
                                                                  const Program &program) {

  auto fnTree = program.entry;

  // set up the struct defs first so that structs in params work
  std::transform(                                    //
      program.defs.begin(), program.defs.end(),      //
      std::inserter(structTypes, structTypes.end()), //
      [&](auto &x) -> Pair<Sym, Pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable>> {
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
      std::inserter(stackVarPtrs, stackVarPtrs.end()),     //
      [&](auto &arg, const auto &named) -> Pair<std::string, Pair<Type::Any, llvm::Value *>> {
        arg.setName(named.symbol);

        if (holds<TypeKind::Ref>(kind(named.tpe)) && false) {
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
    std::cerr << "Verification failed:\n" << errOut.str() << "\nIR=\n" << irOut.str() << std::endl;
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

Pair<Opt<std::string>, std::string> LLVMAstTransformer::optimise(const std::unique_ptr<llvm::Module> &module) {
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

Opt<llvm::StructType *> LLVMAstTransformer::lookup(const Sym &s) {
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

  //  // at this point we know the target machine, so we derive the struct layout here
  //  for (const auto &def : program.defs) {
  //    auto x = xform.lookup(def.name);
  //    if (!x) {
  //      throw std::logic_error("Missing struct def:" + repr(def));
  //    } else {
  //      // FIXME this needs to use the same LLVM target machine context as the compiler
  //      c.layouts.emplace_back(compiler::layoutOf(def));
  //    }
  //  }

  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
