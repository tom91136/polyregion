#include <iostream>

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"
#include "utils.hpp"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Host.h"
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
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), 1),                      //
      "sizePtr"                                                                  //
  );
  auto sizeVal = B.CreatePtrToInt(sizePtr, llvm::Type::getInt64Ty(C));
  return sizeVal;
}

static llvm::Value *load(llvm::IRBuilder<> &B, llvm::Value *rhs) {
  return B.CreateLoad(rhs->getType()->getPointerElementType(), rhs);
}

llvm::Value *LLVM::AstTransformer::invokeMalloc(llvm::Function *parent, llvm::Value *size) {
  return B.CreateCall(mkExternalFn(parent, Type::Array(Type::Byte()), "malloc", {Type::Long()}), size);
}

static bool isUnsigned(const Type::Any &tpe) {
  return holds<Type::Char>(tpe); // the only unsigned type in PolyAst
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

Pair<llvm::StructType *, LLVM::AstTransformer::StructMemberTable> LLVM::AstTransformer::mkStruct(const StructDef &def) {
  std::vector<llvm::Type *> types(def.members.size());
  std::transform(def.members.begin(), def.members.end(), types.begin(),
                 [&](const polyast::Named &n) { return mkTpe(n.tpe); });
  LLVM::AstTransformer::StructMemberTable table;
  for (size_t i = 0; i < def.members.size(); ++i) {
    table[def.members[i].symbol] = i;
  }
  return {llvm::StructType::create(C, types, qualified(def.name)), table};
}

llvm::Type *LLVM::AstTransformer::mkTpe(const Type::Any &tpe, unsigned AS, bool functionBoundary) {            //
  return variants::total(                                                                                      //
      *tpe,                                                                                                    //
      [&](const Type::Float &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },                         //
      [&](const Type::Double &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },                       //
      [&](const Type::Bool &x) -> llvm::Type * { return llvm::Type::getIntNTy(C, functionBoundary ? 8 : 1); }, //
      [&](const Type::Byte &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },                           //
      [&](const Type::Char &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },                          //
      [&](const Type::Short &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },                         //
      [&](const Type::Int &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); },                           //
      [&](const Type::Long &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); },                          //
      [&](const Type::String &x) -> llvm::Type * { return undefined(__FILE__, __LINE__); },                    //
      [&](const Type::Unit &x) -> llvm::Type * { return llvm::Type::getIntNTy(C, functionBoundary ? 8 : 1); }, //
      [&](const Type::Nothing &x) -> llvm::Type * { return undefined(__FILE__, __LINE__); },                   //
      [&](const Type::Struct &x) -> llvm::Type * {
        if (auto def = polyregion::get_opt(structTypes, x.name); def) {
          return def->first->getPointerTo(AS);
        } else {

          auto pool = mk_string2<Sym, Pair<llvm::StructType *, StructMemberTable>>(
              structTypes,
              [](auto &&p) { return "`" + to_string(p.first) + "`" + " = " + std::to_string(p.second.second.size()); },
              "\n->");

          return undefined(__FILE__, __LINE__, "Unseen struct def: " + to_string(x) + ", table=\n" + pool);
        }
      }, //
      [&](const Type::Array &x) -> llvm::Type * {
        // These two types promote to a byte when stored in an array
        if (holds<Type::Bool>(x.component) || holds<Type::Unit>(x.component)) {
          return llvm::Type::getInt8Ty(C)->getPointerTo(AS);
        } else {
          auto comp = mkTpe(x.component);
          return comp->isPointerTy() ? comp : comp->getPointerTo(AS);
        }
      },                                                                                             //
      [&](const Type::Var &x) -> llvm::Type * { return undefined(__FILE__, __LINE__, "type var"); }, //
      [&](const Type::Exec &x) -> llvm::Type * { return undefined(__FILE__, __LINE__, "exec"); }

  );
}

llvm::Value *LLVM::AstTransformer::findStackVar(const Named &named) {
  //  check the LUT table for known variables defined by var or brought in scope by parameters
  if (auto x = polyregion::get_opt(stackVarPtrs, named.symbol); x) {
    auto [tpe, value] = *x;
    if (named.tpe != tpe) {
      error(__FILE__, __LINE__,
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
    return undefined(__FILE__, __LINE__, "Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
  }
}

llvm::Value *LLVM::AstTransformer::mkSelectPtr(const Term::Select &select) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> Pair<llvm::StructType *, StructMemberTable> {
    if (auto s = polyast::get_opt<Type::Struct>(tpe); s) {
      if (auto def = polyregion::get_opt(structTypes, s->name); def) return *def;
      else
        error(__FILE__, __LINE__, "Unseen struct type " + to_string(s->name) + " in select path" + fail());
    } else
      error(__FILE__, __LINE__, "Illegal select path involving non-struct type " + to_string(tpe) + fail());
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
      auto [structTy, table] = structTypeOf(tpe);
      if (auto idx = get_opt(table, path.symbol); idx) {
        root = B.CreateInBoundsGEP(structTy, load(B, root),
                                   {llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0),
                                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), *idx)},
                                   qualified(select) + "_ptr");
        tpe = path.tpe;
      } else {
        auto pool = mk_string2<std::string, size_t>(
            table, [](auto &&p) { return "`" + p.first + "`" + " = " + std::to_string(p.second); }, "\n->");

        return undefined(__FILE__, __LINE__,
                         "Illegal select path with unknown struct member index of name `" + to_string(path) +
                             "`, pool=" + pool + fail());
      }
    }
    return root;
  }
}

llvm::Value *LLVM::AstTransformer::mkTermVal(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> llvm::Value * { return load(B, mkSelectPtr(x)); },
      [&](const Term::Poison &x) -> llvm::Value * {
        if (auto tpe = mkTpe(x.tpe); llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        } else {
          return undefined(__FILE__, __LINE__);
        }
      },
      [&](const Term::UnitConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), 0); },
      [&](const Term::BoolConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), x.value); },
      [&](const Term::ByteConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::CharConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::ShortConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::LongConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },
      [&](const Term::FloatConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getFloatTy(C), x.value); },
      [&](const Term::DoubleConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getDoubleTy(C), x.value); },
      [&](const Term::StringConst &x) -> llvm::Value * { return undefined(__FILE__, __LINE__); });
}

llvm::Function *LLVM::AstTransformer::mkExternalFn(llvm::Function *parent, const Type::Any &rtn,
                                                   const std::string &name, const std::vector<Type::Any> &args) {
  const Signature s(Sym({name}), {}, {}, args, rtn);
  if (functions.find(s) == functions.end()) {
    auto llvmArgs = map_vec<Type::Any, llvm::Type *>(args, [&](auto t) { return mkTpe(t); });
    auto ft = llvm::FunctionType::get(mkTpe(rtn), llvmArgs, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, parent->getParent());
    functions[s] = f;
  }
  return functions[s];
}

llvm::Value *LLVM::AstTransformer::mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

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
        return undefined(__FILE__, __LINE__);
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
        return undefined(__FILE__, __LINE__);
      }
    });
  };

  const auto ext1 = [&](const std::string &name, const Type::Any &tpe, const Term::Any &arg) -> ValPtr {
    return B.CreateCall(mkExternalFn(fn, tpe, name, {tpe}), mkTermVal(arg));
  };

  const auto externBinaryCall = [&](const std::string &name, const Type::Any &tpe, //
                                    const Term::Any &lhs, const Term::Any &rhs) -> ValPtr {
    return B.CreateCall(mkExternalFn(fn, tpe, name, {tpe, tpe}), {mkTermVal(lhs), mkTermVal(rhs)});
  };

  const auto intr0 = [&](Intr::ID id) -> ValPtr {
    auto callee = Intr::getDeclaration(fn->getParent(), id, {});
    return B.CreateCall(callee);
  };

  const auto intr1 = [&](Intr::ID id, const Type::Any &overload, const Term::Any &arg) -> ValPtr {
    auto callee = Intr::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, mkTermVal(arg));
  };

  const auto intr2 = [&](Intr::ID id, const Type::Any &overload, //
                         const Term::Any &lhs, const Term::Any &rhs) -> ValPtr {
    // XXX the overload type here is about the overloading of intrinsic names, not about the parameter types
    // i.e. f32 is for foo.f32(float %a, float %b, float %c)
    auto callee = Intr::getDeclaration(fn->getParent(), id, mkTpe(overload));
    return B.CreateCall(callee, {mkTermVal(lhs), mkTermVal(rhs)});
  };

  return variants::total(
      *expr, //
      [&](const Expr::UnaryIntrinsic &x) {
        auto tpe = x.rtn;
        auto arg = x.lhs;
        return variants::total(
            *x.kind,                                                                     //
            [&](const UnaryIntrinsicKind::Sin &) { return intr1(Intr::sin, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Cos &) { return intr1(Intr::cos, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Tan &) { return ext1("tan", tpe, arg); },      //

            [&](const UnaryIntrinsicKind::Asin &) { return ext1("asin", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Acos &) { return ext1("acos", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Atan &) { return ext1("atan", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Sinh &) { return ext1("sinh", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Cosh &) { return ext1("cosh", tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Tanh &) { return ext1("tanh", tpe, arg); }, //

            [&](const UnaryIntrinsicKind::Signum &) {
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
                      error(__FILE__, __LINE__);
                    return B.CreateSelect(B.CreateLogicalOr(nan, zero), x, intr2(Intr::copysign, tpe, magnitude, arg));
                  });
            }, //
            [&](const UnaryIntrinsicKind::Abs &) {
              return unaryNumOp(
                  arg, tpe, //
                  [&](auto x) { return intr1(Intr::abs, tpe, arg); },
                  [&](auto x) { return intr1(Intr::fabs, tpe, arg); });
            },                                                                               //
            [&](const UnaryIntrinsicKind::Round &) { return intr1(Intr::round, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Ceil &) { return intr1(Intr::ceil, tpe, arg); },   //
            [&](const UnaryIntrinsicKind::Floor &) { return intr1(Intr::floor, tpe, arg); }, //
            [&](const UnaryIntrinsicKind::Rint &) { return intr1(Intr::rint, tpe, arg); },   //

            [&](const UnaryIntrinsicKind::Sqrt &) { return intr1(Intr::sqrt, tpe, arg); },   //
            [&](const UnaryIntrinsicKind::Cbrt &) { return ext1("cbrt", tpe, arg); },        //
            [&](const UnaryIntrinsicKind::Exp &) { return intr1(Intr::exp, tpe, arg); },     //
            [&](const UnaryIntrinsicKind::Expm1 &) { return ext1("expm1", tpe, arg); },      //
            [&](const UnaryIntrinsicKind::Log &) { return intr1(Intr::log, tpe, arg); },     //
            [&](const UnaryIntrinsicKind::Log1p &) { return ext1("log1p", tpe, arg); },      //
            [&](const UnaryIntrinsicKind::Log10 &) { return intr1(Intr::log10, tpe, arg); }, //

            [&](const UnaryIntrinsicKind::BNot &) {
              return unaryExpr(arg, tpe, [&](auto x) { return B.CreateNot(x); });
            },
            [&](const UnaryIntrinsicKind::Pos &) {
              return unaryNumOp(
                  arg, tpe,                  //
                  [&](auto x) { return x; }, //
                  [&](auto x) { return x; });
            },
            [&](const UnaryIntrinsicKind::Neg &) {
              return unaryNumOp(
                  arg, tpe,                               //
                  [&](auto x) { return B.CreateNeg(x); }, //
                  [&](auto x) { return B.CreateFNeg(x); });
            },
            [&](const UnaryIntrinsicKind::LogicNot &) { return B.CreateNot(mkTermVal(x.lhs), key); });
      },
      [&](const Expr::NullaryIntrinsic &x) -> ValPtr {
        switch (options.target) {
          case Target::x86_64:
          case Target::AArch64:
          case Target::ARM: return undefined(__FILE__, __LINE__);
          case Target::SPIRV64: return undefined(__FILE__, __LINE__);
          case Target::NVPTX64: {
            // threadId  @llvm.nvvm.read.ptx.sreg.tid.*
            // blockIdx  @llvm.nvvm.read.ptx.sreg.ctaid.*
            // blockDim  @llvm.nvvm.read.ptx.sreg.ntid.*
            // gridDim   @llvm.nvvm.read.ptx.sreg.nctaid.*
            auto globalSize = [&](Intr::ID nctaid, Intr::ID ntid) -> ValPtr {
              return B.CreateMul(intr0(nctaid), intr0(ntid));
            };
            auto globalId = [&](Intr::ID ctaid, Intr::ID ntid, Intr::ID tid) -> ValPtr {
              return B.CreateAdd(B.CreateMul(intr0(ctaid), intr0(ntid)), intr0(tid));
            };
            return variants::total(
                *x.kind, //
                [&](const NullaryIntrinsicKind::GpuGlobalIdxX &) {
                  return globalId(Intr::nvvm_read_ptx_sreg_ctaid_x, Intr::nvvm_read_ptx_sreg_ntid_x,
                                  Intr::nvvm_read_ptx_sreg_tid_x);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalIdxY &) {
                  return globalId(Intr::nvvm_read_ptx_sreg_ctaid_y, Intr::nvvm_read_ptx_sreg_ntid_y,
                                  Intr::nvvm_read_ptx_sreg_tid_y);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalIdxZ &) {
                  return globalId(Intr::nvvm_read_ptx_sreg_ctaid_z, Intr::nvvm_read_ptx_sreg_ntid_z,
                                  Intr::nvvm_read_ptx_sreg_tid_z);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeX &) {
                  return globalSize(Intr::nvvm_read_ptx_sreg_nctaid_x, Intr::nvvm_read_ptx_sreg_ntid_x);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeY &) {
                  return globalSize(Intr::nvvm_read_ptx_sreg_nctaid_y, Intr::nvvm_read_ptx_sreg_ntid_y);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeZ &) {
                  return globalSize(Intr::nvvm_read_ptx_sreg_nctaid_z, Intr::nvvm_read_ptx_sreg_ntid_z);
                },
                [&](const NullaryIntrinsicKind::GpuGroupIdxX &) { return intr0(Intr::nvvm_read_ptx_sreg_ctaid_x); },
                [&](const NullaryIntrinsicKind::GpuGroupIdxY &) { return intr0(Intr::nvvm_read_ptx_sreg_ctaid_y); },
                [&](const NullaryIntrinsicKind::GpuGroupIdxZ &) { return intr0(Intr::nvvm_read_ptx_sreg_ctaid_z); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeX &) { return intr0(Intr::nvvm_read_ptx_sreg_nctaid_x); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeY &) { return intr0(Intr::nvvm_read_ptx_sreg_nctaid_y); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeZ &) { return intr0(Intr::nvvm_read_ptx_sreg_nctaid_z); },
                [&](const NullaryIntrinsicKind::GpuLocalIdxX &) { return intr0(Intr::nvvm_read_ptx_sreg_tid_x); },
                [&](const NullaryIntrinsicKind::GpuLocalIdxY &) { return intr0(Intr::nvvm_read_ptx_sreg_tid_y); },
                [&](const NullaryIntrinsicKind::GpuLocalIdxZ &) { return intr0(Intr::nvvm_read_ptx_sreg_tid_z); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeX &) { return intr0(Intr::nvvm_read_ptx_sreg_ntid_x); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeY &) { return intr0(Intr::nvvm_read_ptx_sreg_ntid_y); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeZ &) { return intr0(Intr::nvvm_read_ptx_sreg_ntid_z); },

                [&](const NullaryIntrinsicKind::GpuGroupBarrier &) { return intr0(Intr::nvvm_barrier0); },
                [&](const NullaryIntrinsicKind::GpuGroupFence &) { return intr0(Intr::nvvm_membar_cta); });
          }
          case Target::AMDGCN:

            // HSA Sys Arch 1.2:  2.9.6 Kernel Dispatch Packet format:
            //  15:0    header Packet header, see 2.9.1 Packet header (on page 25).
            //  17:16   dimensions Number of dimensions specified in gridSize. Valid values are 1, 2, or 3.
            //  31:18   Reserved, must be 0.
            //  47:32   workgroup_size_x x dimension of work-group (measured in work-items).
            //  63:48   workgroup_size_y y dimension of work-group (measured in work-items).
            //  79:64   workgroup_size_z z dimension of work-group (measured in work-items).
            //  95:80   Reserved, must be 0.
            //  127:96  grid_size_x x dimension of grid (measured in work-items).
            //  159:128 grid_size_y y dimension of grid (measured in work-items).
            //  191:160 grid_size_z z dimension of grid (measured in work-items).
            //  223:192 private_segment_size_bytes Total size in bytes of private memory allocation request (per
            //          work-item).
            //  255:224 group_segment_size_bytes Total size in bytes of group memory allocation request (per
            //          work-group).
            //  319:256 kernel_object Handle for an object in memory that includes an
            //          implementation-defined executable ISA image for the kernel.
            //  383:320 kernarg_address Address of memory containing kernel arguments.
            //  447:384 Reserved, must be 0. 511:448 completion_signal HSA signaling object handle used to indicate
            //          completion of the job

            // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_global_size.cl
            auto globalSizeU32 = [&](size_t dim) -> ValPtr {
              if (dim >= 3) throw std::logic_error("Dim >= 3");
              auto i32Ty = llvm::Type::getInt32Ty(C);
              auto i32ptr = B.CreatePointerCast(intr0(Intr::amdgcn_dispatch_ptr), i32Ty->getPointerTo());
              // 127:96   grid_size_x;  (32*3+(0*32)==96)
              // 159:128  grid_size_y;  (32*3+(1*32)==128)
              // 191:160  grid_size_z;  (32*3+(2*32)==160)
              return load(B, B.CreateInBoundsGEP(i32Ty, i32ptr, llvm::ConstantInt::get(i32Ty, 3 + dim)));
            };

            // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_local_size.cl
            auto localSizeU32 = [&](size_t dim) -> ValPtr {
              if (dim >= 3) throw std::logic_error("Dim >= 3");
              auto i16Ty = llvm::Type::getInt16Ty(C);
              auto i16ptr = B.CreatePointerCast(intr0(Intr::amdgcn_dispatch_ptr), i16Ty->getPointerTo());
              // 47:32   workgroup_size_x (16*2+(0*16)==32)
              // 63:48   workgroup_size_y (16*2+(1*16)==48)
              // 79:64   workgroup_size_z (16*2+(2*16)==64)
              auto size = B.CreateInBoundsGEP(i16Ty, i16ptr, llvm::ConstantInt::get(i16Ty, 2 + dim));
              return B.CreateIntCast(load(B, size), llvm::Type::getInt32Ty(C), false);
            };

            auto globalIdU32 = [&](Intr::ID workgroupId, Intr::ID workitemId, size_t dim) -> ValPtr {
              return B.CreateAdd(B.CreateMul(intr0(workgroupId), localSizeU32(dim)), intr0(workitemId));
            };

            //            // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_num_groups.cl
            auto numGroupsU32 = [&](size_t dim) -> ValPtr {
              auto n = globalSizeU32(dim);
              auto d = localSizeU32(dim);
              auto q = B.CreateUDiv(globalSizeU32(dim), localSizeU32(dim));                 // q = n / d
              auto rem = B.CreateZExt(B.CreateICmpUGT(n, B.CreateMul(q, d)), n->getType()); // ( (uint32t) (n > q*d) )
              return B.CreateAdd(q, rem);                                                   // q + rem
            };

            return variants::total(
                *x.kind, //
                [&](const NullaryIntrinsicKind::GpuGlobalIdxX &) {
                  return globalIdU32(Intr::amdgcn_workgroup_id_x, Intr::amdgcn_workitem_id_x, 0);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalIdxY &) {
                  return globalIdU32(Intr::amdgcn_workgroup_id_y, Intr::amdgcn_workitem_id_y, 1);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalIdxZ &) {
                  return globalIdU32(Intr::amdgcn_workgroup_id_z, Intr::amdgcn_workitem_id_z, 2);
                },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeX &) { return globalSizeU32(0); },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeY &) { return globalSizeU32(1); },
                [&](const NullaryIntrinsicKind::GpuGlobalSizeZ &) { return globalSizeU32(2); },

                [&](const NullaryIntrinsicKind::GpuGroupIdxX &) { return intr0(Intr::amdgcn_workgroup_id_x); },
                [&](const NullaryIntrinsicKind::GpuGroupIdxY &) { return intr0(Intr::amdgcn_workgroup_id_y); },
                [&](const NullaryIntrinsicKind::GpuGroupIdxZ &) { return intr0(Intr::amdgcn_workgroup_id_z); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeX &) { return numGroupsU32(0); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeY &) { return numGroupsU32(1); },
                [&](const NullaryIntrinsicKind::GpuGroupSizeZ &) { return numGroupsU32(2); },

                [&](const NullaryIntrinsicKind::GpuLocalIdxX &) { return intr0(Intr::amdgcn_workitem_id_x); },
                [&](const NullaryIntrinsicKind::GpuLocalIdxY &) { return intr0(Intr::amdgcn_workitem_id_y); },
                [&](const NullaryIntrinsicKind::GpuLocalIdxZ &) { return intr0(Intr::amdgcn_workitem_id_z); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeX &) { return localSizeU32(0); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeY &) { return localSizeU32(1); },
                [&](const NullaryIntrinsicKind::GpuLocalSizeZ &) { return localSizeU32(2); },

                [&](const NullaryIntrinsicKind::GpuGroupBarrier &) -> ValPtr {
                  // work_group_barrier (__memory_scope, 1, 1)
                  return undefined(__FILE__, __LINE__);
                },
                [&](const NullaryIntrinsicKind::GpuGroupFence &) -> ValPtr {
                  // atomic_work_item_fence(0, 5, 1)
                  return undefined(__FILE__, __LINE__);
                });
        }
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
              return intr2(Intr::pow, x.rtn, x.lhs, x.rhs); //
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
            },
            [&](const BinaryIntrinsicKind::LogicEq &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                             //
                  [&](auto l, auto r) { return B.CreateICmpEQ(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOEQ(l, r); } //
              );
            },
            [&](const BinaryIntrinsicKind::LogicNeq &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                             //
                  [&](auto l, auto r) { return B.CreateICmpNE(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpONE(l, r); } //
              );
            },
            [&](const BinaryIntrinsicKind::LogicAnd &) -> ValPtr {
              return B.CreateLogicalAnd(mkTermVal(x.lhs), mkTermVal(x.rhs));
            },
            [&](const BinaryIntrinsicKind::LogicOr &) -> ValPtr {
              return B.CreateLogicalOr(mkTermVal(x.lhs), mkTermVal(x.rhs));
            },
            [&](const BinaryIntrinsicKind::LogicLte &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                              //
                  [&](auto l, auto r) { return B.CreateICmpSLE(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOLE(l, r); }  //
              );
            },
            [&](const BinaryIntrinsicKind::LogicGte &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                              //
                  [&](auto l, auto r) { return B.CreateICmpSGE(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOGE(l, r); }  //
              );
            },
            [&](const BinaryIntrinsicKind::LogicLt &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                              //
                  [&](auto l, auto r) { return B.CreateICmpSLT(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOLT(l, r); }  //
              );
            },
            [&](const BinaryIntrinsicKind::LogicGt &) -> ValPtr {
              return binaryNumOp(
                  x.lhs, x.rhs, tpe(x.lhs),                              //
                  [&](auto l, auto r) { return B.CreateICmpSGT(l, r); }, //
                  [&](auto l, auto r) { return B.CreateFCmpOGT(l, r); }  //
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
            [&](const TypeKind::None &) -> NumKind { error(__FILE__, __LINE__, "none!?"); });

        auto toKind = variants::total(
            *kind(x.as), //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion to ref type (" + to_string(fromTpe) +
                                     ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { error(__FILE__, __LINE__, "none!?"); });

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
          error(__FILE__, __LINE__, "unhandled cast");
      },
      [&](const Expr::Alias &x) -> ValPtr { return mkTermVal(x.ref); },
      [&](const Expr::Invoke &x) -> ValPtr {
        //        auto lhs = mkTermVal(x.lhs );
        return undefined(__FILE__, __LINE__, "Unimplemented invoke:`" + repr(x) + "`");
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (auto arrTpe = get_opt<Type::Array>(x.lhs.tpe); arrTpe) {
          auto ty = mkTpe(*arrTpe)->getPointerElementType();

          auto ptr = B.CreateInBoundsGEP(ty->isPointerTy() ? ty->getPointerElementType() : ty, //
                                         mkTermVal(x.lhs),                                     //
                                         mkTermVal(x.idx), key + "_ptr");
          if (holds<TypeKind::Ref>(kind(arrTpe->component))) {
            return ptr;
          } else if (holds<Type::Bool>(arrTpe->component) || holds<Type::Unit>(arrTpe->component)) {
            // Narrow from i8 to i1
            return B.CreateICmpNE(load(B, ptr), llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true));
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

        auto ptr = invokeMalloc(fn, B.CreateMul(B.CreateIntCast(size, mkTpe(Type::Long()), true), elemSize));
        return B.CreateBitCast(ptr, mkTpe(x.witness));
      },
      [&](const Expr::Suspend &x) -> ValPtr { return undefined(__FILE__, __LINE__); });
}

llvm::Value *LLVM::AstTransformer::conditionalLoad(llvm::Value *rhs) {
  return rhs->getType()->isPointerTy() // deref the rhs if it's a pointer
             ? B.CreateLoad(rhs->getType()->getPointerElementType(), rhs)
             : rhs;
}

LLVM::BlockKind LLVM::AstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx = {}) {

  //  // XXX bool is i8 where non-zero values are true,
  //  //   `br` only takes i1 as the first arg, so we do the appropriate comparison now
  //  auto boolToi8 = [&](llvm::Value *cond) {
  //    return cond->getType()->isIntegerTy(1)
  //               ? cond // as-is if we're already i1
  //               : B.CreateICmpNE(cond, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true));
  //  };

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
        auto stackPtr = B.CreateAlloca(tpe, AllocaAS, nullptr, x.name.symbol + "_stack_ptr");
        auto rhs = x.expr ? std::make_optional(mkExprVal(*x.expr, fn, x.name.symbol + "_var_rhs")) : std::nullopt;

        stackVarPtrs[x.name.symbol] = {x.name.tpe, stackPtr};

        if (holds<Type::Array>(x.name.tpe)) {
          if (rhs) {
            B.CreateStore(*rhs, stackPtr);
          } else
            undefined(__FILE__, __LINE__, "var array with no expr?");
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
            } else if (holds<Type::Bool>(tpe(x.value)) || holds<Type::Unit>(tpe(x.value))) {
              // Extend from i1 to i8
              auto b = mkTermVal(x.value);
              B.CreateStore(B.CreateIntCast(b, llvm::Type::getInt8Ty(C), true), ptr);
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
      [&](const Stmt::Break &x) -> BlockKind {
        if (whileCtx) {
          B.CreateBr(whileCtx->exit);
        } else {
          undefined(__FILE__, __LINE__, "orphaned break!");
        }
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cont &x) -> BlockKind {
        if (whileCtx) {
          B.CreateBr(whileCtx->test);
        } else {
          undefined(__FILE__, __LINE__, "orphaned cont!");
        }
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cond &x) -> BlockKind {
        auto condTrue = llvm::BasicBlock::Create(C, "cond_true", fn);
        auto condFalse = llvm::BasicBlock::Create(C, "cond_false", fn);
        auto condExit = llvm::BasicBlock::Create(C, "cond_exit", fn);
        B.CreateCondBr(mkExprVal(x.cond, fn, "cond"), condTrue, condFalse);
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
        } else if (holds<Type::Bool>(rtnTpe)) {
          // Extend from i1 to i8
          auto b = mkExprVal(x.value, fn, "return");
          B.CreateRet(B.CreateIntCast(b, llvm::Type::getInt8Ty(C), true));
        } else {
          B.CreateRet(mkExprVal(x.value, fn, "return"));
        }
        return BlockKind::Terminal;
      } //

  );
}

Pair<Opt<std::string>, std::string> LLVM::AstTransformer::transform(const std::unique_ptr<llvm::Module> &module,
                                                                    const Program &program) {

  // Work out what address space we're using for arguments
  unsigned GlobalAS = 0;
  switch (options.target) {
    case Target::x86_64:
    case Target::AArch64:
    case Target::ARM:
      break; // CPUs default to generic so nothing to do here.
      // For GPUs, any pointer passed in as args should be annotated global AS.
      //             AMDGPU   |   NVVM
      //      Generic (code)  0  Generic (code)
      //       Global (Host)  1  Global
      //        Region (GDS)  2  Internal Use
      //         Local (LDS)  3  Shared
      // Constant (Internal)  4  Constant
      //             Private  5  Local
    case Target::NVPTX64:
      GlobalAS = 1;
      AllocaAS = 0;
      break;

    case Target::AMDGCN:
      GlobalAS = 1;
      AllocaAS = 5;
      break;
    case Target::SPIRV64: undefined(__FILE__, __LINE__); break;
  }

  auto fnTree = program.entry;

  // set up the struct defs first so that structs in params work
  std::transform(                                    //
      program.defs.begin(), program.defs.end(),      //
      std::inserter(structTypes, structTypes.end()), //
      [&](auto &x) -> Pair<Sym, Pair<llvm::StructType *, LLVM::AstTransformer::StructMemberTable>> {
        return {x.name, mkStruct(x)};
      });

  std::vector<Named> allArgs;
  if (fnTree.receiver) allArgs.insert(allArgs.begin(), *fnTree.receiver);
  allArgs.insert(allArgs.begin(), fnTree.args.begin(), fnTree.args.end());
  allArgs.insert(allArgs.begin(), fnTree.captures.begin(), fnTree.captures.end());

  auto paramTpes =
      map_vec<Named, llvm::Type *>(allArgs, [&](auto &&named) { return mkTpe(named.tpe, GlobalAS, true); });

  // Unit type at function return type position is void
  // Any other location, Unit is a singleton value
  auto rtnTpe = holds<Type::Unit>(fnTree.rtn) ? llvm::Type::getVoidTy(C) : mkTpe(fnTree.rtn, 0, true);

  auto fnTpe = llvm::FunctionType::get(rtnTpe, {paramTpes}, false);

  auto *fn = llvm::Function::Create(fnTpe, llvm::Function::ExternalLinkage, qualified(fnTree.name), *module);

  // setup function conventions for targets
  switch (options.target) {
    case Target::x86_64:
    case Target::AArch64:
    case Target::ARM:
      // nothing to do for CPUs
      break;
    case Target::NVPTX64:
      module->getOrInsertNamedMetadata("nvvm.annotations")
          ->addOperand(
              llvm::MDNode::get(C, // XXX the attribute name must be "kernel" here and not the function name!
                                {llvm::ValueAsMetadata::get(fn), llvm::MDString::get(C, "kernel"),
                                 llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 1))}));
      break;
    case Target::AMDGCN: fn->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL); break;
    case Target::SPIRV64: undefined(__FILE__, __LINE__); break;
  }

  auto *entry = llvm::BasicBlock::Create(C, "entry", fn);
  B.SetInsertPoint(entry);

  // add function params to the lut first as function body will need these at some point
  std::transform(                                      //
      fn->arg_begin(), fn->arg_end(), allArgs.begin(), //
      std::inserter(stackVarPtrs, stackVarPtrs.end()), //
      [&](auto &arg, const auto &named) -> Pair<std::string, Pair<Type::Any, llvm::Value *>> {
        arg.setName(named.symbol);

        auto argValue = holds<Type::Bool>(named.tpe) || holds<Type::Unit>(named.tpe)
                            ? B.CreateICmpNE(&arg, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true))
                            : &arg;

        auto stack = B.CreateAlloca(mkTpe(named.tpe, GlobalAS), AllocaAS, nullptr, named.symbol + "_stack_ptr");
        B.CreateStore(argValue, stack);
        return {named.symbol, {named.tpe, stack}};
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

Pair<Opt<std::string>, std::string> LLVM::AstTransformer::optimise(const std::unique_ptr<llvm::Module> &module) {
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

Opt<llvm::StructType *> LLVM::AstTransformer::lookup(const Sym &s) {
  if (auto x = get_opt(structTypes, s); x) return x->first;
  else
    return {};
}

llvmc::TargetInfo LLVM::Options::toTargetInfo() const {
  using llvm::Triple;
  const auto bindGpuArch = [&](Triple::ArchType archTpe, Triple::VendorType vendor, Triple::OSType os) {
    Triple triple(Triple::getArchTypeName(archTpe), Triple::getVendorTypeName(vendor), Triple::getOSTypeName(os));
    if (!archTpe) throw std::logic_error("Arch must be specified for " + triple.str());
    return llvmc::TargetInfo{
        .triple = triple,
        .target = backend::llvmc::targetFromTriple(triple),
        .cpu = {.uArch = arch, .features = {}},
    };
  };

  const auto bindCpuArch = [&](Triple::ArchType archTpe) {
    const Triple defaultTriple = backend::llvmc::defaultHostTriple();
    if (arch.empty() &&
        defaultTriple.getArch() != archTpe) // when detecting host arch, the host triple's arch must match
      throw std::logic_error("Requested arch detection with " + Triple::getArchTypeName(archTpe).str() +
                             " but the host arch is different (" +
                             Triple::getArchTypeName(defaultTriple.getArch()).str() + ")");

    Triple triple = defaultTriple;
    triple.setArch(archTpe);
    return llvmc::TargetInfo{
        .triple = triple,
        .target = backend::llvmc::targetFromTriple(triple),
        .cpu = arch.empty() || arch == "native" ? llvmc::hostCpuInfo() : llvmc::CpuInfo{.uArch = arch, .features = {}},
    };
  };

  switch (target) {
    case LLVM::Target::x86_64: return bindCpuArch(Triple::ArchType::x86_64);
    case LLVM::Target::AArch64: return bindCpuArch(Triple::ArchType::aarch64);
    case LLVM::Target::ARM: return bindCpuArch(Triple::ArchType::arm);
    case LLVM::Target::NVPTX64:
      return bindGpuArch(Triple::ArchType::nvptx64, Triple::VendorType::NVIDIA, Triple::OSType::CUDA);
    case LLVM::Target::AMDGCN:
      return bindGpuArch(Triple::ArchType::amdgcn, Triple::VendorType::AMD, Triple::OSType::AMDHSA);
    case Target::SPIRV64:
      // TODO implement this properly
      auto os = backend::llvmc::defaultHostTriple().getOS();
      return bindGpuArch(Triple::ArchType::spirv64, Triple::VendorType::UnknownVendor, os);
  }
}

compiler::Compilation backend::LLVM::run(const Program &program, const compiler::Opt &opt) {
  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("test", *ctx);

  LLVM::AstTransformer xform(options, *ctx);

  auto rawXform = compiler::nowMono();
  auto [rawError, rawIR] = xform.transform(mod, program);
  auto rawXformElapsed = compiler::elapsedNs(rawXform);
  auto optXform = compiler::nowMono();
  auto [optError, optIR] = xform.optimise(mod);
  auto optXformElapsed = compiler::elapsedNs(optXform);

  compiler::Event ast2IR(compiler::nowMs(), rawXformElapsed, "ast_to_llvm_ir", rawIR);
  compiler::Event astOpt(compiler::nowMs(), optXformElapsed, "llvm_ir_opt", optIR);

  if (rawError || optError) {
    return {{},
            {},               //
            {ast2IR, astOpt}, //
            mk_string<std::string>(
                {rawError.value_or(""), optError.value_or("")}, [](auto &&x) { return x; }, "\n")};
  }

  auto c = llvmc::compileModule(options.toTargetInfo(), opt, true, std::move(mod), *ctx);

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
