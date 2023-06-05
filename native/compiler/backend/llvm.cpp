#include <iostream>
#include <unordered_set>

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"

#include "llvm_amdgpu.h"
#include "llvm_cpu.h"
#include "llvm_nvptx.h"
#include "llvm_opencl.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Host.h"
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

ValPtr LLVMBackend::load(llvm::IRBuilder<> &B, ValPtr rhs, llvm::Type *ty) { return B.CreateLoad(ty, rhs); }
ValPtr LLVMBackend::sizeOf(llvm::IRBuilder<> &B, llvm::LLVMContext &C, llvm::Type *ptrTpe) {
  // http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
  // we want
  // %SizePtr = getelementptr %T, %T* null, i32 1
  // %Size = ptrtoint %T* %SizePtr to i32
  auto sizePtr = B.CreateGEP(                                          //
      ptrTpe,                                                          //
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(C)), //
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), 1),            //
      "sizePtr"                                                        //
  );
  auto sizeVal = B.CreatePtrToInt(sizePtr, llvm::Type::getInt64Ty(C));
  return sizeVal;
}

LLVMBackend::AstTransformer::AstTransformer(const LLVMBackend::Options &options, llvm::LLVMContext &c)
    : options(options), C(c), targetHandler(TargetSpecificHandler::from(options.target)), stackVarPtrs(), structTypes(), functions(), B(C) {
  // Work out what address space we're using for arguments
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
      LocalAS = 3;
      AllocaAS = 0;
      break;
    case Target::AMDGCN:
      GlobalAS = 1;
      LocalAS = 3;
      AllocaAS = 5;
      break;
    case Target::SPIRV32:
    case Target::SPIRV64:
      GlobalAS = 1;
      LocalAS = 3;
      AllocaAS = 0;
      break;
  }
}

std::unique_ptr<LLVMBackend::TargetSpecificHandler> LLVMBackend::TargetSpecificHandler::from(LLVMBackend::Target target) {
  switch (target) {
    case Target::x86_64:  // fallthrough
    case Target::AArch64: // fallthrough
    case Target::ARM: return std::make_unique<CPUTargetSpecificHandler>();
    case Target::NVPTX64: return std::make_unique<NVPTXTargetSpecificHandler>();
    case Target::AMDGCN: return std::make_unique<AMDGPUTargetSpecificHandler>();
    case Target::SPIRV32: // fallthrough
    case Target::SPIRV64: return std::make_unique<OpenCLTargetSpecificHandler>();
  }
}
LLVMBackend::TargetSpecificHandler::~TargetSpecificHandler() {}

ValPtr LLVMBackend::AstTransformer::invokeMalloc(llvm::Function *parent, ValPtr size) {
  return B.CreateCall(mkExternalFn(parent, Type::Array(Type::IntS8(), TypeSpace::Global()), "malloc", {Type::IntS64()}), size);
}

ValPtr LLVMBackend::AstTransformer::invokeAbort(llvm::Function *parent) {
  return B.CreateCall(mkExternalFn(parent, Type::Nothing(), "abort", {}));
}

// the only unsigned type in PolyAst
static bool isUnsigned(const Type::Any &tpe) {
  return holds<Type::IntU8>(tpe) || holds<Type::IntU16>(tpe) || holds<Type::IntU32>(tpe) || holds<Type::IntU64>(tpe);
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

Pair<llvm::StructType *, LLVMBackend::AstTransformer::StructMemberIndexTable> LLVMBackend::AstTransformer::mkStruct(const StructDef &def) {
  std::vector<llvm::Type *> types(def.members.size());
  std::transform(def.members.begin(), def.members.end(), types.begin(), [&](const StructMember &n) {
    auto tpe = mkTpe(n.named.tpe);
    return tpe->isStructTy() ? B.getPtrTy() : tpe;
  });
  LLVMBackend::AstTransformer::StructMemberIndexTable table;
  for (size_t i = 0; i < def.members.size(); ++i)
    table[def.members[i].named.symbol] = i;
  return {llvm::StructType::create(C, types, qualified(def.name)), table};
}

llvm::Type *LLVMBackend::AstTransformer::mkTpe(const Type::Any &tpe, bool functionBoundary) {                   //
  return variants::total(                                                                                       //
      *tpe,                                                                                                     //
      [&](const Type::Float16 &x) -> llvm::Type * { return llvm::Type::getHalfTy(C); },                         //
      [&](const Type::Float32 &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },                        //
      [&](const Type::Float64 &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },                       //
      [&](const Type::Bool1 &x) -> llvm::Type * { return llvm::Type::getIntNTy(C, functionBoundary ? 8 : 1); }, //

      [&](const Type::IntU8 &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },   //
      [&](const Type::IntU16 &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); }, //
      [&](const Type::IntU32 &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); }, //
      [&](const Type::IntU64 &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); }, //

      [&](const Type::IntS8 &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },   //
      [&](const Type::IntS16 &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); }, //
      [&](const Type::IntS32 &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); }, //
      [&](const Type::IntS64 &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); }, //

      [&](const Type::Unit0 &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); },   //
      [&](const Type::Nothing &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); }, //
      [&](const Type::Struct &x) -> llvm::Type * {
        if (auto def = polyregion::get_opt(structTypes, x.name); def) return def->first;
        else {
          auto pool = mk_string2<Sym, Pair<llvm::StructType *, StructMemberIndexTable>>(
              structTypes, [](auto &&p) { return "`" + to_string(p.first) + "`" + " = " + std::to_string(p.second.second.size()); },
              "\n->");

          return undefined(__FILE__, __LINE__, "Unseen struct def: " + to_string(x) + ", table=\n" + pool);
        }
      }, //
      [&](const Type::Array &x) -> llvm::Type * {
        ;

        return B.getPtrTy(variants::total(
            *x.space,                                             //
            [&](const TypeSpace::Local &_) { return LocalAS; },   //
            [&](const TypeSpace::Global &_) { return GlobalAS; }) //
        );
        //        // These two types promote to a byte when stored in an array
        //        if (holds<Type::Bool1>(x.component) || holds<Type::Unit0>(x.component)) {
        //          return llvm::Type::getInt8Ty(C)->getPointerTo(AS);
        //        } else {
        //          auto comp = mkTpe(x.component);
        //          return comp->isPointerTy() ? comp : comp->getPointerTo(AS);
        //        }
      },                                                                                             //
      [&](const Type::Var &x) -> llvm::Type * { return undefined(__FILE__, __LINE__, "type var"); }, //
      [&](const Type::Exec &x) -> llvm::Type * { return undefined(__FILE__, __LINE__, "exec"); }

  );
}

ValPtr LLVMBackend::AstTransformer::findStackVar(const Named &named) {
  if (holds<Type::Unit0>(named.tpe)) return mkTermVal(Term::Unit0Const());
  //  check the LUT table for known variables defined by var or brought in scope by parameters
  if (auto x = polyregion::get_opt(stackVarPtrs, named.symbol); x) {
    auto [tpe, value] = *x;
    if (named.tpe != tpe) {
      error(__FILE__, __LINE__, "Named local variable (" + to_string(named) + ") has different type from LUT (" + to_string(tpe) + ")");
    }
    return value;
  } else {
    auto pool = mk_string2<std::string, Pair<Type::Any, ValPtr>>(
        stackVarPtrs,
        [](auto &&p) { return "`" + p.first + "` = " + to_string(p.second.first) + "(IR=" + llvm_tostring(p.second.second) + ")"; },
        "\n->");
    return undefined(__FILE__, __LINE__, "Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
  }
}

ValPtr LLVMBackend::AstTransformer::mkSelectPtr(const Term::Select &select) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> Pair<llvm::StructType *, StructMemberIndexTable> {
    if (auto s = get_opt<Type::Struct>(tpe); s) {
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
        root = B.CreateInBoundsGEP(
            structTy, load(B, root, B.getPtrTy(AllocaAS)),
            {llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), *idx)},
            qualified(select) + "_select_ptr");
        tpe = path.tpe;
      } else {
        auto pool = mk_string2<std::string, size_t>(
            table, [](auto &&p) { return "`" + p.first + "`" + " = " + std::to_string(p.second); }, "\n->");

        return undefined(__FILE__, __LINE__,
                         "Illegal select path with unknown struct member index of name `" + to_string(path) + "`, pool=" + pool + fail());
      }
    }
    return root;
  }
}

ValPtr LLVMBackend::AstTransformer::mkTermVal(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> ValPtr {
        if (holds<Type::Unit0>(x.tpe)) return mkTermVal(Term::Unit0Const());
        auto tpe = mkTpe(x.tpe);
        return load(B, mkSelectPtr(x), tpe->isStructTy() ? B.getPtrTy(GlobalAS) : tpe);
      },
      [&](const Term::Poison &x) -> ValPtr {
        if (auto tpe = mkTpe(x.tpe); llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        } else {
          return undefined(__FILE__, __LINE__);
        }
      },
      [&](const Term::Unit0Const &x) -> ValPtr {
        // this only exists to represent the singleton
        return ConstantInt::get(llvm::Type::getInt1Ty(C), 0);
      },
      [&](const Term::Bool1Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt1Ty(C), x.value); },

      [&](const Term::IntU8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::IntU16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntU32Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::IntU64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },

      [&](const Term::IntS8Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::IntS16Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntS32Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::IntS64Const &x) -> ValPtr { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },

      [&](const Term::Float16Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getHalfTy(C), x.value); },
      [&](const Term::Float32Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getFloatTy(C), x.value); },
      [&](const Term::Float64Const &x) -> ValPtr { return ConstantFP::get(llvm::Type::getDoubleTy(C), x.value); });
}

llvm::Function *LLVMBackend::AstTransformer::mkExternalFn(llvm::Function *parent, const Type::Any &rtn, const std::string &name,
                                                          const std::vector<Type::Any> &args) {
  InvokeSignature sig(Sym({name}), {}, {}, args, {}, rtn);
  if (auto it = functions.find(sig); it != functions.end()) {
    return it->second;
  } else {
    auto llvmArgs = map_vec<Type::Any, llvm::Type *>(args, [&](auto t) { return mkTpe(t); });
    auto ft = llvm::FunctionType::get(mkTpe(rtn, true), llvmArgs, false);
    auto fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, parent->getParent());
    functions.emplace(sig, fn);
    return fn;
  }
}

ValPtr LLVMBackend::AstTransformer::mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

  return variants::total(
      *expr, //
      [&](const Expr::SpecOp &x) -> ValPtr { return targetHandler->mkSpecVal(*this, fn, x); },
      [&](const Expr::MathOp &x) -> ValPtr { return targetHandler->mkMathVal(*this, fn, x); },
      [&](const Expr::IntrOp &x) -> ValPtr {
        auto intr = x.op;
        return variants::total(
            *intr, //
            [&](const Intr::BNot &v) -> ValPtr { return unaryExpr(expr, v.x, v.tpe, [&](auto x) { return B.CreateNot(x); }); },
            [&](const Intr::LogicNot &v) -> ValPtr { return B.CreateNot(mkTermVal(v.x)); },
            [&](const Intr::Pos &v) -> ValPtr {
              return unaryNumOp(
                  expr, v.x, v.tpe, [&](auto x) { return x; }, [&](auto x) { return x; });
            },
            [&](const Intr::Neg &v) -> ValPtr {
              return unaryNumOp(
                  expr, v.x, v.tpe, [&](auto x) { return B.CreateNeg(x); }, [&](auto x) { return B.CreateFNeg(x); });
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
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpEQ(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOEQ(l, r); });
            },
            [&](const Intr::LogicNeq &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpNE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpONE(l, r); });
            },
            [&](const Intr::LogicLte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpSLE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOLE(l, r); });
            },
            [&](const Intr::LogicGte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpSGE(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOGE(l, r); });
            },
            [&](const Intr::LogicLt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpSLT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOLT(l, r); });
            },
            [&](const Intr::LogicGt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, tpe(v.x), //
                  [&](auto l, auto r) { return B.CreateICmpSGT(l, r); }, [&](auto l, auto r) { return B.CreateFCmpOGT(l, r); });
            });
      },
      [&](const Expr::Cast &x) -> ValPtr {
        // we only allow widening or narrowing of integral and fractional types
        // pointers are not allowed to participate on either end
        auto from = mkTermVal(x.from);
        auto fromTpe = mkTpe(tpe(x.from));
        auto toTpe = mkTpe(x.as);
        enum class NumKind { Fractional, Integral };

        // Same type
        if (*x.as == *tpe(x.from)) return from;

        // x.as <: x.from
        auto lhsStruct = get_opt<Type::Struct>(x.as);
        auto rhsStruct = get_opt<Type::Struct>(tpe(x.from));
        if (lhsStruct && rhsStruct &&
            (std::any_of(lhsStruct->parents.begin(), lhsStruct->parents.end(), [&](auto &x) { return x == rhsStruct->name; }) ||
             std::any_of(rhsStruct->parents.begin(), rhsStruct->parents.end(), [&](auto &x) { return x == lhsStruct->name; }))) {
          return from;
        }

        auto fromKind = variants::total(
            *kind(tpe(x.from)), [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion from ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { error(__FILE__, __LINE__, "none!?"); });

        auto toKind = variants::total(
            *kind(x.as), //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion to ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { error(__FILE__, __LINE__, "none!?"); });

        if (fromKind == NumKind::Fractional && toKind == NumKind::Integral) {

          // to the equally sized integral type first if narrowing; XXX narrowing directly produces a poison value

          ValPtr c = nullptr;
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
        std::vector<Term::Any> allArgs;
        if (x.receiver) allArgs.push_back((*x.receiver));
        for (auto &arg : x.args)
          if (!holds<Type::Unit0>(tpe(arg))) allArgs.push_back(arg);
        for (auto &arg : x.captures)
          if (!holds<Type::Unit0>(tpe(arg))) allArgs.push_back(arg);

        auto paramTerms = map_vec2(allArgs, [&](auto &&term) {
          auto val = mkTermVal(term);
          return holds<Type::Bool1>(tpe(term)) ? B.CreateZExt(val, mkTpe(Type::Bool1(), true)) : val;
        });

        InvokeSignature sig(x.name, {}, map_opt(x.receiver, [](auto &x) { return tpe(x); }),
                            map_vec2(x.args, [](auto &x) { return tpe(x); }), map_vec2(x.captures, [](auto &x) { return tpe(x); }), x.rtn);

        if (auto fn = functions.find(sig); fn != functions.end()) {
          auto call = B.CreateCall(fn->second, paramTerms);
          // in case the fn returns a unit (which is mapped to void), we just return the constant
          if (holds<Type::Unit0>(x.rtn)) {
            return mkTermVal(Term::Unit0Const());
          } else
            return call;
        } else {

          for (auto [key, v] : functions) {
            std::cerr << key << " = " << v << " =m" << (key == sig) << std::endl;
          }
          return undefined(__FILE__, __LINE__, "Cannot find function " + to_string(sig));
        }
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (auto lhs = get_opt<Term::Select>(x.lhs); lhs) {
          if (auto arrTpe = get_opt<Type::Array>(lhs->tpe); arrTpe) {

            if (holds<Type::Unit0>(arrTpe->component)) {
              // Still call GEP so that memory access and OOB effects are still present.
              auto val = mkTermVal(Term::Unit0Const());
              B.CreateInBoundsGEP(val->getType(),                  //
                                  mkTermVal(*lhs),                 //
                                  mkTermVal(x.idx), key + "_ptr"); //
              return val;
            }

            auto ty = mkTpe(arrTpe->component);
            auto ptr = B.CreateInBoundsGEP(ty->isStructTy() ? B.getPtrTy(AllocaAS) : ty, //
                                           mkTermVal(*lhs),                              //
                                           mkTermVal(x.idx), key + "_ptr");
            if (holds<TypeKind::Ref>(kind(arrTpe->component))) {
              return ptr;
            } else if (holds<Type::Bool1>(arrTpe->component)) { // Narrow from i8 to i1
              return B.CreateICmpNE(load(B, ptr, ty), llvm::ConstantInt::get(llvm::Type::getInt1Ty(C), 0, true));
            } else {
              return load(B, ptr, ty);
            }
          } else {
            throw std::logic_error("Semantic error: array index not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else
          throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (index) is not a select");
      },

      [&](const Expr::RefTo &x) -> ValPtr {
        if (auto lhs = get_opt<Term::Select>(x.lhs); lhs) {
          if (auto arrTpe = get_opt<Type::Array>(lhs->tpe); arrTpe) { // taking reference of an index in an array
            auto offset = x.idx ? mkTermVal(*x.idx) : llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), 0, true);
            auto ty = holds<Type::Unit0>(arrTpe->component) ? llvm::Type::getInt8Ty(C) : mkTpe(arrTpe->component);
            return B.CreateInBoundsGEP(ty->isStructTy() ? B.getPtrTy(AllocaAS) : ty, //
                                       mkTermVal(*lhs),                              //
                                       offset, key + "_ptr");
          } else if (auto structTpe = get_opt<Type::Struct>(lhs->tpe); structTpe) {
            return mkTermVal(*lhs);

          } else { // taking reference of a var
            if (x.idx) throw std::logic_error("Semantic error: Cannot take reference of scalar with index in " + to_string(x));
            return mkTermVal(*lhs);
          }
        } else
          throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (index) is not a select");
      },

      [&](const Expr::Alloc &x) -> ValPtr { //
        auto componentTpe = B.getPtrTy(0);
        auto size = mkTermVal(x.size);
        auto elemSize = sizeOf(B, C, componentTpe);
        auto ptr = invokeMalloc(fn, B.CreateMul(B.CreateIntCast(size, mkTpe(Type::IntS64()), true), elemSize));
        return B.CreateBitCast(ptr, componentTpe);
      });
}

static bool canAssign(Type::Any lhs, Type::Any rhs) {
  if (*lhs == *rhs) return true;
  auto lhsStruct = get_opt<Type::Struct>(lhs);
  auto rhsStruct = get_opt<Type::Struct>(rhs);
  if (lhsStruct && rhsStruct) {
    return std::any_of(lhsStruct->parents.begin(), lhsStruct->parents.end(), [&](auto &x) { return x == rhsStruct->name; });
  }
  return false;
}

LLVMBackend::BlockKind LLVMBackend::AstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn, Opt<WhileCtx> whileCtx = {}) {

  //  // XXX bool is i8 where non-zero values are true,
  //  //   `br` only takes i1 as the first arg, so we do the appropriate comparison now
  //  auto boolToi8 = [&](ValPtr cond) {
  //    return cond->getType()->isIntegerTy(1)
  //               ? cond // as-is if we're already i1
  //               : B.CreateICmpNE(cond, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true));
  //  };

  return variants::total(
      *stmt,
      [&](const Stmt::Block &x) -> BlockKind {
        auto kind = BlockKind::Normal;
        for (auto &body : x.stmts)
          kind = mkStmt(body, fn);
        return kind;
      },
      [&](const Stmt::Comment &x) -> BlockKind { // discard comments
        return BlockKind::Normal;
      },
      [&](const Stmt::Var &x) -> BlockKind {
        // [T : ref] =>> t:T  = _        ; lut += &t
        // [T : ref] =>> t:T* = &(rhs:T) ; lut += t
        // [T : val] =>> t:T  =   rhs:T  ; lut += &t

        if (x.expr && tpe(*x.expr) != x.name.tpe) {
          throw std::logic_error("Semantic error: name type " + to_string(x.name.tpe) + " and rhs expr type " + to_string(tpe(*x.expr)) +
                                 " mismatch (" + repr(x) + ")");
        }

        if (holds<Type::Unit0>(x.name.tpe)) {
          // Unit0 declaration, discard declaration but keep RHS effect.
          if (x.expr) mkExprVal(*x.expr, fn, x.name.symbol + "_var_rhs");
        } else {
          auto tpe = mkTpe(x.name.tpe);
          auto stackPtr = B.CreateAlloca(tpe->isStructTy() ? B.getPtrTy(AllocaAS) : tpe, AllocaAS, nullptr, x.name.symbol + "_stack_ptr");
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
                     //            if (!tpe->isPointerTy()) {
                     //              throw std::logic_error("The LLVM struct type `" + llvm_tostring(tpe) +
                     //                                     "` was not a pointer to the struct " + repr(x.name.tpe) +
                     //                                     " in stmt:" + repr(stmt));
                     //            }
              auto elemSize = sizeOf(B, C, tpe);
              auto ptr = invokeMalloc(fn, elemSize);
              B.CreateStore(ptr, stackPtr); //
            }
          } else { // any other primitives
            if (rhs) {
              B.CreateStore(*rhs, stackPtr); //
            }
          }
        }
        return BlockKind::Normal;
      },
      [&](const Stmt::Mut &x) -> BlockKind {
        // [T : ref]        =>> t   := &(rhs:T) ; lut += t
        // [T : ref {u: U}] =>> t.u := &(rhs:U)
        // [T : val]        =>> t   :=   rhs:T
        if (auto lhs = get_opt<Term::Select>(x.name); lhs) {
          if (tpe(x.expr) != lhs->tpe) {
            throw std::logic_error("Semantic error: name type (" + to_string(tpe(x.expr)) + ") and rhs expr (" + to_string(lhs->tpe) +
                                   ") mismatch (" + repr(x) + ")");
          }
          if (holds<Type::Unit0>(lhs->tpe)) return BlockKind::Normal;
          auto rhs = mkExprVal(x.expr, fn, qualified(*lhs) + "_mut");
          if (lhs->init.empty()) { // local var
            auto stackPtr = findStackVar(lhs->last);
            B.CreateStore(rhs, stackPtr);
          } else { // struct member select
            B.CreateStore(rhs, mkSelectPtr(*lhs));
          }
        } else
          throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (mut) is not a select");
        return BlockKind::Normal;
      },
      [&](const Stmt::Update &x) -> BlockKind {
        if (auto lhs = get_opt<Term::Select>(x.lhs); lhs) {
          if (auto arrTpe = get_opt<Type::Array>(lhs->tpe); arrTpe) {
            auto rhs = x.value;
            if (arrTpe->component != tpe(rhs)) {
              throw std::logic_error("Semantic error: array component type (" + to_string(arrTpe->component) + ") and rhs expr (" +
                                     to_string(tpe(rhs)) + ") mismatch (" + repr(x) + ")");
            } else {
              auto dest = mkTermVal(*lhs);
              if (holds<Type::Bool1>(tpe(rhs)) || holds<Type::Unit0>(tpe(rhs))) { // Extend from i1 to i8
                auto ty = llvm::Type::getInt8Ty(C);
                auto ptr = B.CreateInBoundsGEP(ty, dest, mkTermVal(x.idx), qualified(*lhs) + "_update_ptr");
                B.CreateStore(B.CreateIntCast(mkTermVal(rhs), ty, true), ptr);
              } else {
                auto ty = mkTpe(tpe(rhs));
                auto ptr = B.CreateInBoundsGEP(                         //
                    ty->isStructTy() ? B.getPtrTy(AllocaAS) : ty, dest, //
                    mkTermVal(x.idx), qualified(*lhs) + "_update_ptr"   //
                );                                                      //
                B.CreateStore(mkTermVal(rhs), ptr);
              }
            }
          } else {
            throw std::logic_error("Semantic error: array update not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else
          throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (update) is not a select");

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
        if (whileCtx) B.CreateBr(whileCtx->exit);
        else
          undefined(__FILE__, __LINE__, "orphaned break!");
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cont &x) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->test);
        else
          undefined(__FILE__, __LINE__, "orphaned cont!");
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
        if (condExit->getNumUses() > 0) {
          B.SetInsertPoint(condExit);
          return BlockKind::Normal;
        } else {
          condExit->removeFromParent();
          return BlockKind::Terminal;
        }
      },
      [&](const Stmt::Return &x) -> BlockKind {
        auto rtnTpe = tpe(x.value);
        if (holds<Type::Unit0>(rtnTpe)) {
          B.CreateRetVoid();
        } else if (holds<Type::Nothing>(rtnTpe)) {
          B.CreateUnreachable();
        } else {
          auto expr = mkExprVal(x.value, fn, "return");
          if (holds<Type::Bool1>(rtnTpe)) {
            // Extend from i1 to i8
            B.CreateRet(B.CreateIntCast(expr, llvm::Type::getInt8Ty(C), true));
          } else {
            B.CreateRet(expr);
          }
        }
        return BlockKind::Terminal;
      });
}

void LLVMBackend::AstTransformer::addDefs(const std::vector<StructDef> &defs) {
  // TODO handle recursive defs
  std::unordered_set<StructDef> defsWithDependencies(defs.begin(), defs.end());
  while (!defsWithDependencies.empty()) {
    std::vector<StructDef> zeroDeps;
    std::copy_if( //
        defsWithDependencies.begin(), defsWithDependencies.end(), std::back_inserter(zeroDeps), [&](auto &def) {
          return std::all_of(def.members.begin(), def.members.end(), [&](auto &m) {
            if (auto s = get_opt<Type::Struct>(m.named.tpe); s) return structTypes.find(s->name) != structTypes.end();
            else
              return true;
          });
        });
    if (!zeroDeps.empty()) {
      for (auto &r : zeroDeps) {
        structTypes.emplace(r.name, mkStruct(r));
        defsWithDependencies.erase(r);
      }
    } else
      throw std::logic_error("Recursive defs cannot be resolved: " + mk_string<StructDef>(
                                                                         zeroDeps, [](auto &r) { return to_string(r); }, ","));
  }
}

std::vector<Pair<Sym, llvm::StructType *>> LLVMBackend::AstTransformer::getStructTypes() const {
  std::vector<Pair<Sym, llvm::StructType *>> results;
  for (auto &[k, v] : structTypes)
    results.emplace_back(k, v.first);
  return results;
}

static std::vector<Arg> collectFnDeclarationNames(const Function &f) {
  std::vector<Arg> allArgs;
  if (f.receiver) allArgs.push_back(*f.receiver);
  auto addAddExcludingUnit = [&](auto &xs) {
    for (auto &x : xs) {
      if (!holds<Type::Unit0>(x.named.tpe)) allArgs.push_back(x);
    }
  };
  addAddExcludingUnit(f.args);
  addAddExcludingUnit(f.moduleCaptures);
  addAddExcludingUnit(f.termCaptures);
  return allArgs;
}

void LLVMBackend::AstTransformer::addFn(llvm::Module &mod, const Function &f, bool entry) {

  auto args = collectFnDeclarationNames(f);
  auto llvmArgTpes = map_vec<Arg, llvm::Type *>(args, [&](auto &&arg) {
    auto tpe = mkTpe(arg.named.tpe, true);
    return tpe->isStructTy() ? B.getPtrTy(GlobalAS) : tpe;
  });

  // Unit type at function return type position is void, any other location, Unit is a singleton value
  auto rtnTpe = holds<Type::Unit0>(f.rtn) ? llvm::Type::getVoidTy(C) : mkTpe(f.rtn, true);

  auto fnTpe = llvm::FunctionType::get(rtnTpe, {llvmArgTpes}, false);

  auto *fn = llvm::Function::Create(fnTpe,                                                                     //
                                    entry ? llvm::Function::ExternalLinkage : llvm::Function::ExternalLinkage, // TODO
                                    qualified(f.name),                                                         //
                                    mod);

  fn->setDSOLocal(true);

  if (entry || true) { // setup external function conventions for targets

    //    targetHandler->witnessEntry(*this, mod, fn);

    switch (options.target) {
      case Target::x86_64:
      case Target::AArch64:
      case Target::ARM:
        // nothing to do for CPUs
        break;
      case Target::NVPTX64:

        mod.getOrInsertNamedMetadata("nvvm.annotations")
            ->addOperand(llvm::MDNode::get(C, // XXX the attribute name must be "kernel" here and not the function name!
                                           {llvm::ValueAsMetadata::get(fn), llvm::MDString::get(C, "kernel"),
                                            llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 1))}));
        break;
      case Target::AMDGCN: fn->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL); break;
      case Target::SPIRV32:
      case Target::SPIRV64:

        fn->addFnAttr(llvm::Attribute::Convergent);
        fn->addFnAttr(llvm::Attribute::NoRecurse);
        fn->addFnAttr(llvm::Attribute::NoUnwind);

        // See logic defined in clang/lib/CodeGen/CodeGenModule.cpp @ CodeGenModule::GenKernelArgMetadata
        // We need to insert OpenCL metadata for clspv to pick up and identify the arg types

        llvm::SmallVector<llvm::Metadata *, 8> addressQuals;     // MDNode for the kernel argument address space qualifiers.
        llvm::SmallVector<llvm::Metadata *, 8> accessQuals;      // MDNode for the kernel argument access qualifiers (images only).
        llvm::SmallVector<llvm::Metadata *, 8> argTypeNames;     // MDNode for the kernel argument type names.
        llvm::SmallVector<llvm::Metadata *, 8> argBaseTypeNames; // MDNode for the kernel argument base type names.
        llvm::SmallVector<llvm::Metadata *, 8> argTypeQuals;     // MDNode for the kernel argument type qualifiers.
        llvm::SmallVector<llvm::Metadata *, 8> argNames;         // MDNode for the kernel argument names.

        for (size_t i = 0; i < args.size(); ++i) {
          auto arg = args[i];
          auto llvmTpe = llvmArgTpes[i];
          addressQuals.push_back(llvm::ConstantAsMetadata::get( //
              B.getInt32(llvmTpe->isPointerTy() ? llvmTpe->getPointerAddressSpace() : 0)));
          accessQuals.push_back(llvm::MDString::get(C, "none")); // write_only | read_only | read_write | none

          auto typeName = [](Type::Any tpe) -> std::string {
            auto impl = [](Type::Any tpe, auto &thunk) -> std::string {
              return variants::total(
                  *tpe,                                                                                        //
                  [&](const Type::Float16 &) -> std::string { return "half"; },                                //
                  [&](const Type::Float32 &) -> std::string { return "float"; },                               //
                  [&](const Type::Float64 &) -> std::string { return "double"; },                              //
                  [&](const Type::IntU8 &) -> std::string { return "uchar"; },                                 //
                  [&](const Type::IntU16 &) -> std::string { return "ushort"; },                               //
                  [&](const Type::IntU32 &) -> std::string { return "uint"; },                                 //
                  [&](const Type::IntU64 &) -> std::string { return "ulong"; },                                //
                  [&](const Type::IntS8 &) -> std::string { return "char"; },                                  //
                  [&](const Type::IntS16 &) -> std::string { return "short"; },                                //
                  [&](const Type::IntS32 &) -> std::string { return "int"; },                                  //
                  [&](const Type::IntS64 &) -> std::string { return "long"; },                                 //
                  [&](const Type::Bool1 &) -> std::string { return "char"; },                                  //
                  [&](const Type::Unit0 &) -> std::string { return "void"; },                                  //
                  [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },                         //
                  [&](const Type::Struct &x) -> std::string { return qualified(x.name); },                     //
                  [&](const Type::Array &x) -> std::string { return thunk(x.component, thunk) + "*"; },        //
                  [&](const Type::Var &) -> std::string { return undefined(__FILE__, __LINE__, "type var"); }, //
                  [&](const Type::Exec &) -> std::string { return undefined(__FILE__, __LINE__, "exec"); }     //
              );
            };
            return impl(tpe, impl);
          };

          argTypeNames.push_back(llvm::MDString::get(C, typeName(arg.named.tpe)));
          argBaseTypeNames.push_back(llvm::MDString::get(C, typeName(arg.named.tpe)));

          argTypeQuals.push_back(llvm::MDString::get(C, "")); // const | restrict | volatile | pipe | ""
          argNames.push_back(llvm::MDString::get(C, arg.named.symbol));
        }

        fn->setMetadata("kernel_arg_addr_space", llvm::MDNode::get(C, addressQuals));
        fn->setMetadata("kernel_arg_access_qual", llvm::MDNode::get(C, accessQuals));
        fn->setMetadata("kernel_arg_type", llvm::MDNode::get(C, argTypeNames));
        fn->setMetadata("kernel_arg_base_type", llvm::MDNode::get(C, argBaseTypeNames));
        fn->setMetadata("kernel_arg_type_qual", llvm::MDNode::get(C, argTypeQuals));
        fn->setMetadata("kernel_arg_name", llvm::MDNode::get(C, argNames));
        fn->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
        break;
    }
  }

  std::vector<Type::Any> extraArgTpes;
  for (auto &n : f.moduleCaptures)
    extraArgTpes.push_back(n.named.tpe);
  for (auto &n : f.termCaptures)
    extraArgTpes.push_back(n.named.tpe);

  functions.emplace(InvokeSignature(f.name,                                                   //
                                    {},                                                       //
                                    map_opt(f.receiver, [](auto &x) { return x.named.tpe; }), //
                                    map_vec2(f.args, [](auto &x) { return x.named.tpe; }),    //
                                    extraArgTpes,                                             //
                                    f.rtn),
                    fn);
}

Pair<Opt<std::string>, std::string> LLVMBackend::AstTransformer::transform(llvm::Module &mod, const Program &program) {

  transform(mod, program.entry);
  for (auto &f : program.functions)
    transform(mod, f);

  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  mod.print(irOut, nullptr);

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (llvm::verifyModule(mod, &errOut)) {
    std::cerr << "Verification failed:\n" << errOut.str() << "\nIR=\n" << irOut.str() << std::endl;
    return {errOut.str(), irOut.str()};
  } else {
    return {{}, irOut.str()};
  }
}

void LLVMBackend::AstTransformer::transform(llvm::Module &mod, const Function &fnTree) {

  std::vector<Type::Any> argTpes;
  for (auto &n : fnTree.moduleCaptures)
    argTpes.push_back(n.named.tpe);
  for (auto &n : fnTree.termCaptures)
    argTpes.push_back(n.named.tpe);

  InvokeSignature sig(fnTree.name,                                                   //
                      {},                                                            //
                      map_opt(fnTree.receiver, [](auto &x) { return x.named.tpe; }), //
                      map_vec2(fnTree.args, [](auto &x) { return x.named.tpe; }),    //
                      argTpes,                                                       //
                      fnTree.rtn);

  auto it = functions.find(sig);
  if (it == functions.end()) {

    for (auto [key, v] : functions) {
      std::cerr << key << " = " << v << " = m " << (key == sig) << std::endl;
    }

    throw std::logic_error("Cannot find function " + to_string(sig) + ", function was not added before xform?");
  }

  auto *fn = it->second;

  auto *entry = llvm::BasicBlock::Create(C, "entry", fn);
  B.SetInsertPoint(entry);

  // add function params to the lut first as function body will need these at some point
  auto allArgs = collectFnDeclarationNames(fnTree);
  std::transform(                                      //
      fn->arg_begin(), fn->arg_end(), allArgs.begin(), //
      std::inserter(stackVarPtrs, stackVarPtrs.end()), //
      [&](auto &arg, const auto &fnArg) -> Pair<std::string, Pair<Type::Any, ValPtr>> {
        arg.setName(fnArg.named.symbol);

        auto argValue = holds<Type::Bool1>(fnArg.named.tpe) || holds<Type::Unit0>(fnArg.named.tpe)
                            ? B.CreateICmpNE(&arg, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true))
                            : &arg;

        //        auto as = holds<ArgKind::Local>(fnArg.kind) ? LocalAS : GlobalAS;
        auto tpe = mkTpe(fnArg.named.tpe);
        auto stack = B.CreateAlloca(tpe->isStructTy() ? B.getPtrTy(GlobalAS) : tpe, AllocaAS, nullptr, fnArg.named.symbol + "_stack_ptr");
        B.CreateStore(argValue, stack);
        return {fnArg.named.symbol, {fnArg.named.tpe, stack}};
      });

  for (auto &stmt : fnTree.body)
    mkStmt(stmt, fn);

  stackVarPtrs.clear();
}
ValPtr LLVMBackend::AstTransformer::unaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyType &rtn, const ValPtrFn1 &fn) { //
  if (tpe(l) != rtn) {
    throw std::logic_error("Semantic error: lhs type " + to_string(tpe(l)) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkTermVal(l));
}
ValPtr LLVMBackend::AstTransformer::binaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn,
                                               const ValPtrFn2 &fn) { //
  if (tpe(l) != rtn) {
    throw std::logic_error("Semantic error: lhs type " + to_string(tpe(l)) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  if (tpe(r) != rtn) {
    throw std::logic_error("Semantic error: rhs type " + to_string(tpe(r)) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkTermVal(l), mkTermVal(r));
}
ValPtr LLVMBackend::AstTransformer::unaryNumOp(const AnyExpr &expr, const AnyTerm &arg, const AnyType &rtn, //
                                               const ValPtrFn1 &integralFn, const ValPtrFn1 &fractionalFn) {
  return unaryExpr(expr, arg, rtn, [&](auto lhs) -> ValPtr {
    if (holds<TypeKind::Integral>(kind(rtn))) {
      return integralFn(lhs);
    } else if (holds<TypeKind::Fractional>(kind(rtn))) {
      return fractionalFn(lhs);
    } else {
      return undefined(__FILE__, __LINE__);
    }
  });
}
ValPtr LLVMBackend::AstTransformer::binaryNumOp(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, //
                                                const ValPtrFn2 &integralFn, const ValPtrFn2 &fractionalFn) {
  return binaryExpr(expr, l, r, rtn, [&](auto lhs, auto rhs) -> ValPtr {
    if (holds<TypeKind::Integral>(kind(rtn))) {
      return integralFn(lhs, rhs);
    } else if (holds<TypeKind::Fractional>(kind(rtn))) {
      return fractionalFn(lhs, rhs);
    } else {
      return undefined(__FILE__, __LINE__);
    }
  });
}
ValPtr LLVMBackend::AstTransformer::extFn1(llvm::Function *fn, const std::string &name, const AnyType &rtn, const AnyTerm &arg) { //
  auto fn_ = mkExternalFn(fn, rtn, name, {tpe(arg)});
  if (options.target == Target::SPIRV32 || options.target == Target::SPIRV64) {
    fn_->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    //    fn_->addFnAttr(llvm::Attribute::NoBuiltin);
    //    fn_->addFnAttr(llvm::Attribute::Convergent);
  }
  if (!holds<Type::Unit0>(rtn)) {
    fn_->addFnAttr(llvm::Attribute::WillReturn);
  }
  auto call = B.CreateCall(fn_, mkTermVal(arg));
  call->setCallingConv(fn_->getCallingConv());
  return call;
}
ValPtr LLVMBackend::AstTransformer::extFn2(llvm::Function *fn, const std::string &name, const AnyType &rtn, const AnyTerm &lhs,
                                           const AnyTerm &rhs) { //
  auto fn_ = mkExternalFn(fn, rtn, name, {tpe(lhs), tpe(rhs)});
  if (options.target == Target::SPIRV32 || options.target == Target::SPIRV64) {
    fn_->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    fn_->addFnAttr(llvm::Attribute::NoBuiltin);
    fn_->addFnAttr(llvm::Attribute::Convergent);
  }
  auto call = B.CreateCall(fn_, {mkTermVal(lhs), mkTermVal(rhs)});
  call->setCallingConv(fn_->getCallingConv());
  return call;
}
ValPtr LLVMBackend::AstTransformer::intr0(llvm::Function *fn, llvm::Intrinsic::ID id) { //
  auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, {});
  return B.CreateCall(callee);
}
ValPtr LLVMBackend::AstTransformer::intr1(llvm::Function *fn, llvm::Intrinsic::ID id, const AnyType &overload, const AnyTerm &arg) { //
  auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
  return B.CreateCall(callee, mkTermVal(arg));
}
ValPtr LLVMBackend::AstTransformer::intr2(llvm::Function *fn, llvm::Intrinsic::ID id, const AnyType &overload, //
                                          const AnyTerm &lhs, const AnyTerm &rhs) {                            //
  // XXX the overload type here is about the overloading of intrinsic names, not about the parameter types
  // i.e. f32 is for foo.f32(float %a, float %b, float %c)
  auto callee = llvm::Intrinsic::getDeclaration(fn->getParent(), id, mkTpe(overload));
  return B.CreateCall(callee, {mkTermVal(lhs), mkTermVal(rhs)});
}

llvmc::TargetInfo LLVMBackend::Options::targetInfo() const {
  using llvm::Triple;
  const auto bindGpuArch = [&](Triple::ArchType archTpe, Triple::VendorType vendor, Triple::OSType os) {
    Triple triple(Triple::getArchTypeName(archTpe), Triple::getVendorTypeName(vendor), Triple::getOSTypeName(os));

    switch (archTpe) {
      case Triple::ArchType::UnknownArch: throw std::logic_error("Arch must be specified for " + triple.str());
      case Triple::ArchType::spirv32:
        // XXX We don't have a SPIRV target machine in LLVM yet, but we do know the data layout from Clang:
        // See clang/lib/Basic/Targets/SPIR.h
        return llvmc::TargetInfo{
            .triple = triple,                                                                                                         //
            .layout = llvm::DataLayout("e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"), //
            .target = nullptr,                                                                                                        //
            .cpu = {.uArch = arch, .features = {}}};
      case Triple::ArchType::spirv64: // Same thing for SPIRV64
        return llvmc::TargetInfo{
            .triple = triple,                                                                                                 //
            .layout = llvm::DataLayout("e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"), //
            .target = nullptr,                                                                                                //
            .cpu = {.uArch = arch, .features = {}}};
      default:
        return llvmc::TargetInfo{
            .triple = triple,
            .layout = {},
            .target = backend::llvmc::targetFromTriple(triple),
            .cpu = {.uArch = arch, .features = {}},
        };
    }
  };

  const auto bindCpuArch = [&](Triple::ArchType archTpe) {
    const Triple defaultTriple = backend::llvmc::defaultHostTriple();
    if (arch.empty() && defaultTriple.getArch() != archTpe) // when detecting host arch, the host triple's arch must match
      throw std::logic_error("Requested arch detection with " + Triple::getArchTypeName(archTpe).str() +
                             " but the host arch is different (" + Triple::getArchTypeName(defaultTriple.getArch()).str() + ")");

    Triple triple = defaultTriple;
    triple.setArch(archTpe);
    return llvmc::TargetInfo{
        .triple = triple,
        .target = backend::llvmc::targetFromTriple(triple),
        .cpu = arch.empty() || arch == "native" ? llvmc::hostCpuInfo() : llvmc::CpuInfo{.uArch = arch, .features = {}},
    };
  };

  switch (target) {
    case LLVMBackend::Target::x86_64: return bindCpuArch(Triple::ArchType::x86_64);
    case LLVMBackend::Target::AArch64: return bindCpuArch(Triple::ArchType::aarch64);
    case LLVMBackend::Target::ARM: return bindCpuArch(Triple::ArchType::arm);
    case LLVMBackend::Target::NVPTX64: return bindGpuArch(Triple::ArchType::nvptx64, Triple::VendorType::NVIDIA, Triple::OSType::CUDA);
    case LLVMBackend::Target::AMDGCN: return bindGpuArch(Triple::ArchType::amdgcn, Triple::VendorType::AMD, Triple::OSType::AMDHSA);
    case Target::SPIRV32: return bindGpuArch(Triple::ArchType::spirv32, Triple::VendorType::UnknownVendor, Triple::OSType::UnknownOS);
    case Target::SPIRV64: return bindGpuArch(Triple::ArchType::spirv64, Triple::VendorType::UnknownVendor, Triple::OSType::UnknownOS);
  }
}

std::vector<polyast::Layout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &defs,
                                                         const backend::LLVMBackend::AstTransformer &xform) const {

  auto dataLayout = options.targetInfo().resolveDataLayout();

  std::unordered_map<polyast::Sym, polyast::StructDef> lut(defs.size());
  for (auto &d : defs)
    lut.emplace(d.name, d);

  std::vector<polyast::Layout> layouts;
  for (auto &[sym, structTy] : xform.getStructTypes()) {
    if (auto it = lut.find(sym); it != lut.end()) {
      auto layout = dataLayout.getStructLayout(structTy);
      std::vector<polyast::Member> members;
      for (size_t i = 0; i < it->second.members.size(); ++i) {
        members.emplace_back(it->second.members[i].named,                             //
                             layout->getElementOffset(i),                             //
                             dataLayout.getTypeAllocSize(structTy->getElementType(i)) //
        );
      }
      layouts.emplace_back(sym, layout->getSizeInBytes(), layout->getAlignment().value(), members);
    } else
      throw std::logic_error("Cannot find symbol " + to_string(sym) + " from domain");
  }
  return layouts;
}

std::vector<polyast::Layout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &defs, const polyast::OptLevel &opt) {
  llvm::LLVMContext ctx;
  backend::LLVMBackend::AstTransformer xform(options, ctx);
  xform.addDefs(defs);
  return resolveLayouts(defs, xform);
}

polyast::Compilation backend::LLVMBackend::compileProgram(const Program &program, const polyast::OptLevel &opt) {
  using namespace llvm;

  llvm::LLVMContext ctx;
  auto mod = std::make_unique<llvm::Module>("program", ctx);

  LLVMBackend::AstTransformer xform(options, ctx);
  xform.addDefs(program.defs);
  xform.addFn(*mod, program.entry, true);
  for (auto &f : program.functions)
    xform.addFn(*mod, f, false);

  auto transformStart = compiler::nowMono();
  auto [maybeTransformErr, transformMsg] = xform.transform(*mod, program);
  polyast::Event ast2IR(compiler::nowMs(), compiler::elapsedNs(transformStart), "ast_to_llvm_ir", transformMsg);

  auto verifyStart = compiler::nowMono();
  auto [maybeVerifyErr, verifyMsg] = llvmc::verifyModule(*mod);
  polyast::Event astOpt(compiler::nowMs(), compiler::elapsedNs(verifyStart), "llvm_ir_verify", verifyMsg);

  if (maybeTransformErr || maybeVerifyErr) {
    std::vector<std::string> errors;
    if (maybeTransformErr) errors.push_back(*maybeTransformErr);
    if (maybeVerifyErr) errors.push_back(*maybeVerifyErr);
    return {{},
            {},               //
            {ast2IR, astOpt}, //
            {},               //
            mk_string<std::string>(
                errors, [](auto &&x) { return x; }, "\n")};
  }

  auto c = llvmc::compileModule(options.targetInfo(), opt, true, std::move(mod));
  c.layouts = resolveLayouts(program.defs, xform);
  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
