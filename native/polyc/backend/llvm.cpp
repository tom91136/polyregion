#include <iostream>
#include <unordered_set>

#include "aspartame/optional.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"
#include "aspartame/string.hpp"

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
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Scalar.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;
using namespace aspartame;

template <typename T> static std::string llvm_tostring(const T *t) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  t->print(rso);
  return rso.str();
}

ValPtr LLVMBackend::load(llvm::IRBuilder<> &B, ValPtr rhs, llvm::Type *ty) {
  //  assert(!ty->isArrayTy());
  return B.CreateLoad(ty, rhs);
}
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
      GlobalAS = 0; // When inspecting Clang's output, they don't explicitly annotate addrspace(3) for globals
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
LLVMBackend::TargetSpecificHandler::~TargetSpecificHandler() = default;

ValPtr LLVMBackend::AstTransformer::invokeMalloc(llvm::Function *parent, ValPtr size) {
  return B.CreateCall(mkExternalFn(parent, Type::Ptr(Type::IntS8(), {}, TypeSpace::Global()), "malloc", {Type::IntS64()}), size);
}

ValPtr LLVMBackend::AstTransformer::invokeAbort(llvm::Function *parent) {
  return B.CreateCall(mkExternalFn(parent, Type::Nothing(), "abort", {}));
}

// the only unsigned type in PolyAst
static bool isUnsigned(const Type::Any &tpe) {
  return tpe.is<Type::IntU8>() || tpe.is<Type::IntU16>() || tpe.is<Type::IntU32>() || tpe.is<Type::IntU64>();
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

LLVMBackend::AstTransformer::StructInfo LLVMBackend::AstTransformer::mkStruct(const StructDef &def) {
  std::vector<llvm::Type *> types;
  StructMemberIndexTable table;
  for (const auto &p : def.parents) {
    if (auto it = structTypes.find(p); it != structTypes.end()) {
      auto [parentDef, parentStructTpe, parentTable] = it->second;
      types.push_back(parentStructTpe);
      table[qualified(parentDef.name)] = types.size() - 1;
      // for (const auto &[sym, idx] : parentTable) {
      //   types.push_back(parentStructTpe->getElementType(idx));
      //   table[sym] = types.size() - 1;
      // }
    } else throw std::logic_error("Unseen struct def in inheritance chain: " + repr(p));
  }
  for (const auto &m : def.members) {
    types.push_back(mkTpe(m.named.tpe));
    table[m.named.symbol] = types.size() - 1;
  }
  std::cout << "Sym: " << repr(def) << "\n";
  for (auto &[k, v] : table) {
    std::cout << " =>" << k << " " << v << "\n";
  }
  return {def, llvm::StructType::create(C, types, qualified(def.name)), table};
}

llvm::Type *LLVMBackend::AstTransformer::mkTpe(const Type::Any &tpe, bool functionBoundary) {                   //
  return tpe.match_total(                                                                                       //
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
        return structTypes ^ get(x.name) ^
               fold([](auto &info) { return info.tpe; },
                    [&]() -> llvm::StructType * {
                      auto pool = structTypes | values() | mk_string("\n", "\n", "\n", [](auto &info) {
                                    return " -> " + repr(info.def) + ", IR=" + to_string(info.tpe);
                                  });
                      throw std::logic_error("Unseen struct def: " + to_string(x) + ", currently in-scope structs:" + pool);
                    });
      }, //
      [&](const Type::Ptr &x) -> llvm::Type * {
        if (x.length) return llvm::ArrayType::get(mkTpe(x.component), *x.length);
        else {
          return B.getPtrTy(x.space.match_total(                    //
              [&](const TypeSpace::Local &_) { return LocalAS; },   //
              [&](const TypeSpace::Global &_) { return GlobalAS; }) //
          );
        }

        //        // These two types promote to a byte when stored in an array
        //        if (HOLDS(Type::Bool1, x.component) || HOLDS(Type::Unit0, x.component)) {
        //          return llvm::Type::getInt8Ty(C)->getPointerTo(AS);
        //        } else {
        //          auto comp = mkTpe(x.component);
        //          return comp->isPointerTy() ? comp : comp->getPointerTo(AS);
        //        }
      },                                                                               //
      [&](const Type::Var &x) -> llvm::Type * { throw std::logic_error("type var"); }, //
      [&](const Type::Exec &x) -> llvm::Type * { throw std::logic_error("exec"); }

  );
}

ValPtr LLVMBackend::AstTransformer::findStackVar(const Named &named) {
  if (named.tpe.is<Type::Unit0>()) return mkTermVal(Term::Unit0Const());
  //  check the LUT table for known variables defined by var or brought in scope by parameters
  return stackVarPtrs ^ get(named.symbol) ^
         fold(
             [&](auto &tpe, auto &value) {
               if (named.tpe != tpe)
                 throw std::logic_error("Named local variable (" + to_string(named) + ") has different type from LUT (" + to_string(tpe) +
                                        ")");
               return value;
             },
             [&]() -> ValPtr {
               auto pool = stackVarPtrs | mk_string("\n", "\n", "\n", [](auto &k, auto &v) {
                             auto &[tpe, ir] = v;
                             return " -> `" + k + "` = " + to_string(tpe) + "(IR=" + llvm_tostring(ir) + ")";
                           });
               throw std::logic_error("Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
             });
}

ValPtr LLVMBackend::AstTransformer::mkSelectPtr(const Term::Select &select) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> StructInfo {
    auto findTy = [&](const Type::Struct &s) -> StructInfo {
      return structTypes ^ get(s.name) ^
             fold([&]() -> StructInfo { throw std::logic_error("Unseen struct type " + to_string(s.name) + " in select path" + fail()); });
    };

    if (auto s = tpe.get<Type::Struct>(); s) {
      return findTy(*s);
    } else if (auto p = tpe.get<Type::Ptr>(); p) {
      if (auto _s = p->component.get<Type::Struct>(); _s) return findTy(*_s);
      else
        throw std::logic_error("Illegal select path involving pointer to non-struct type " + to_string(s->name) + " in select path" +
                               fail());
    } else throw std::logic_error("Illegal select path involving non-struct type " + to_string(tpe) + fail());
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
      auto selectFinal = [&](llvm::StructType *structTy, size_t idx, bool conditionalLoad = true, const std::string &suffix = "") {
        if (auto p = tpe.get<Type::Ptr>(); conditionalLoad && p && !p->length) {
          root = B.CreateInBoundsGEP(
              structTy, load(B, root, B.getPtrTy(AllocaAS)),
              {//
               llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), idx)},
              qualified(select) + "_select_ptr_" + suffix);
        } else {
          root = B.CreateInBoundsGEP(
              structTy, root,
              {//
               llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), idx)},
              qualified(select) + "_select_ptr_" + suffix);
        }
      };
      auto [s, structTy, table] = structTypeOf(tpe);
      if (auto idx = table ^ get( path.symbol); idx) {
        selectFinal(structTy, *idx);
        tpe = path.tpe;
      } else {
        if (auto inHeirachy = findSymbolInHeirachy<std::pair<size_t, llvm::StructType *>>(
                s.name,
                [&](auto, auto ty, auto xs) -> std::optional<std::pair<size_t, llvm::StructType *>> {
                  auto o = xs ^ get( path.symbol);
                  return o ? std::optional{std::pair{*o, ty}} : std::nullopt;
                  ;
                });
            inHeirachy) {
          auto &[inheritanceChain, lastIndex] = *inHeirachy;
          auto chainPrev = structTy;
          // TODO conditional deref only on the first chain; garbage code but semantically correct,redo this whole thing with views
          size_t c = 0;
          inheritanceChain.push_back(lastIndex.second);
          for (auto chain : inheritanceChain) {
            size_t N = 0;
            if (chain != chainPrev) { // skip the first chain; it's 0 offset
              if (auto relativeIdxIt =
                      std::find_if(chainPrev->element_begin(), chainPrev->element_end(), [&](auto t) { return t == chain; });
                  relativeIdxIt != chainPrev->element_end()) {
                N = std::distance(chainPrev->element_begin(), relativeIdxIt);
                selectFinal(chainPrev, N, c == 0, "in_chain_" + chain->getName().str());
                c++;
              } else {
                throw BackendException("Illegal select path with out of bounds parent `" + to_string(path) + "`" + fail());
              }
            }
            chainPrev = chain;
          }
          selectFinal(lastIndex.second, lastIndex.first, false, "in_chain_final_" + lastIndex.second->getName().str());
          tpe = path.tpe;
        } else {
          auto pool = table | mk_string("\n", "\n", "\n", [](auto &k, auto &v) { return " -> `" + k + "` = " + std::to_string(v) + ")"; });
          throw std::logic_error("Illegal select path with unknown struct member index of name `" + to_string(path) + "`, pool=" + pool +
                                 fail());
        }
      }
    }
    return root;
  }
}

// Opt<Pair<std::vector<llvm::StructType *>, size_t>> LLVMBackend::AstTransformer::findSymbolInHeirachy( //
//     const Sym &structName, const std::string &member, const std::vector<llvm::StructType *> &xs) const {
//   if (auto it = structTypes.find(structName); it != structTypes.end()) {
//     auto [sdef, structTy, table] = it->second;
//     if (auto idx = get_opt(table, member); idx) {
//       return {std::pair{xs, *idx}};
//     } else {
//       if (sdef.parents.empty()) return {};
//       auto ys = xs;
//       ys.push_back(structTy);
//       for (auto parent : sdef.parents) {
//         if (auto x = findSymbolInHeirachy(parent, member, ys); x) return x;
//       }
//       return {};
//     }
//   } else {
//     throw std::logic_error( "Unseen struct type " + to_string(structName) + " in heirachy");
//   }
// }

template <typename T>
Opt<Pair<std::vector<llvm::StructType *>, T>> LLVMBackend::AstTransformer::findSymbolInHeirachy( //
    const Sym &structName, std::function<Opt<T>(StructDef, llvm::StructType *, StructMemberIndexTable)> f,
    const std::vector<llvm::StructType *> &xs) const {
  if (auto it = structTypes.find(structName); it != structTypes.end()) {
    auto [sdef, structTy, table] = it->second;
    if (auto x = f(sdef, structTy, table); x) {
      return {std::pair{xs, *x}};
    } else {
      if (sdef.parents.empty()) return {};
      auto ys = xs;
      ys.push_back(structTy);
      for (const auto &parent : sdef.parents) {
        if (auto x = findSymbolInHeirachy(parent, f, ys); x) return x;
      }
      return {};
    }
  } else {
    throw std::logic_error("Unseen struct type " + to_string(structName) + " in heirachy");
  }
}

ValPtr LLVMBackend::AstTransformer::mkTermVal(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return ref.match_total( //
      [&](const Term::Select &x) -> ValPtr {
        if (x.tpe.is<Type::Unit0>()) return mkTermVal(Term::Unit0Const());
        if (auto ptr = x.tpe.get<Type::Ptr>(); ptr && ptr->length) return mkSelectPtr(x);
        else return load(B, mkSelectPtr(x), mkTpe(x.tpe));
      },
      [&](const Term::Poison &x) -> ValPtr {
        if (auto tpe = mkTpe(x.tpe); llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        } else {
          throw BackendException("unimplemented");
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
  return functions ^= get_or_emplace(InvokeSignature(Sym({name}), {}, {}, args, {}, rtn), [&](auto &sig) -> llvm::Function * {
           auto llvmArgs = args ^ map([&](auto t) { return mkTpe(t); });
           auto ft = llvm::FunctionType::get(mkTpe(rtn, true), llvmArgs, false);
           auto fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, parent->getParent());
           return fn;
         });
}

ValPtr LLVMBackend::AstTransformer::mkExprVal(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

  return expr.match_total( //
      [&](const Expr::SpecOp &x) -> ValPtr { return targetHandler->mkSpecVal(*this, fn, x); },
      [&](const Expr::MathOp &x) -> ValPtr { return targetHandler->mkMathVal(*this, fn, x); },
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
        auto fromTpe = mkTpe(x.from.tpe());
        auto toTpe = mkTpe(x.as);
        enum class NumKind { Fractional, Integral };

        // Same type
        if (x.as == x.from.tpe()) return from;

        // x.as <: x.from

        if (auto rhsPtr = x.from.tpe().get<Type::Ptr>(); rhsPtr) {
          if (auto lhsPtr = x.as.get<Type::Ptr>(); lhsPtr) {
            auto lhsStruct = lhsPtr->component.get<Type::Struct>();
            auto rhsStruct = rhsPtr->component.get<Type::Struct>();
            if (lhsStruct && rhsStruct &&
                (std::any_of(lhsStruct->parents.begin(), lhsStruct->parents.end(), [&](auto &x) { return x == rhsStruct->name; }) ||
                 std::any_of(rhsStruct->parents.begin(), rhsStruct->parents.end(), [&](auto &x) { return x == lhsStruct->name; }))) {

              auto lhsTpe = mkTpe(*lhsStruct);

              // B.to[A]

              std::cout << "R " << llvm_tostring(lhsTpe) << std::endl;

              if (auto inHeirachy = findSymbolInHeirachy<bool>(
                      rhsStruct->name,
                      [&](auto, auto structTy, auto) -> Opt<bool> { return structTy == lhsTpe ? std::optional{true} : std::nullopt; });
                  inHeirachy) {

                auto &[inheritanceChain, finaIdx] = *inHeirachy;

                auto chainPrev =
                    lhsTpe->isStructTy() ? static_cast<llvm::StructType *>(lhsTpe) : throw std::logic_error("Illegal lhs tpe!");
                for (auto chain : inheritanceChain) {

                  std::cout << "@@ " << llvm_tostring(chain) << std::endl;

                  size_t N = 0;
                  if (chain != chainPrev) { // skip the first chain; it's 0 offset
                    if (auto relativeIdxIt =
                            std::find_if(chain->element_begin(), chain->element_end(), [&](auto t) { return t == chainPrev; });
                        relativeIdxIt != chain->element_end()) {
                      N = std::distance(chain->element_begin(), relativeIdxIt);
                    } else {
                      throw std::logic_error("Illegal select path with out of bounds parent `" + to_string(path) + "`");
                    }
                  }

                  from = B.CreateInBoundsGEP(
                      chain, from,
                      {//
                       llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), N)},
                      "_upcast_ptr");
                }
              }

              // find the offset

              return from;
            }
          }
        }

        auto fromKind = x.from.tpe().kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion from ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw std::logic_error("none!?"); });

        auto toKind = x.as.kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw std::logic_error("Semantic error: conversion to ref type (" + to_string(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw std::logic_error("none!?"); });

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
          return isUnsigned(x.from.tpe()) ? B.CreateUIToFP(from, toTpe) : B.CreateSIToFP(from, toTpe);
        } else if (fromKind == NumKind::Integral && toKind == NumKind::Integral) {
          return B.CreateIntCast(from, toTpe, !isUnsigned(x.from.tpe()), "integral_cast");
        } else if (fromKind == NumKind::Fractional && toKind == NumKind::Fractional) {
          return B.CreateFPCast(from, toTpe, "fractional_cast");
        } else throw std::logic_error("unhandled cast");
      },
      [&](const Expr::Alias &x) -> ValPtr { return mkTermVal(x.ref); },
      [&](const Expr::Invoke &x) -> ValPtr {
        std::vector<Term::Any> allArgs;
        if (x.receiver) allArgs.push_back((*x.receiver));
        for (auto &arg : x.args)
          if (!arg.tpe().is<Type::Unit0>()) allArgs.push_back(arg);
        for (auto &arg : x.captures)
          if (!arg.tpe().is<Type::Unit0>()) allArgs.push_back(arg);

        auto paramTerms = allArgs ^ map([&](auto &&term) {
                            auto val = mkTermVal(term);
                            return term.tpe().template is<Type::Bool1>() ? B.CreateZExt(val, mkTpe(Type::Bool1(), true)) : val;
                          });

        InvokeSignature sig(x.name, {},                                        //
                            x.receiver ^ map([](auto &x) { return x.tpe(); }), //
                            x.args ^ map([](auto &x) { return x.tpe(); }),     //
                            x.captures ^ map([](auto &x) { return x.tpe(); }), //
                            x.rtn);

        if (auto fn = functions.find(sig); fn != functions.end()) {
          auto call = B.CreateCall(fn->second, paramTerms);
          // in case the fn returns a unit (which is mapped to void), we just return the constant
          if (x.rtn.is<Type::Unit0>()) {
            return mkTermVal(Term::Unit0Const());
          } else return call;
        } else {

          for (auto &[key, v] : functions) {
            std::cerr << repr(key) << " = " << v << " match = " << (key == sig) << std::endl;
          }
          throw std::logic_error("Cannot find function " + repr(sig));
        }
      },
      [&](const Expr::Index &x) -> ValPtr {
        if (auto lhs = x.lhs.get<Term::Select>(); lhs) {
          if (auto arrTpe = lhs->tpe.get<Type::Ptr>(); arrTpe) {

            if (arrTpe->component.is<Type::Unit0>()) {
              // Still call GEP so that memory access and OOB effects are still present.
              auto val = mkTermVal(Term::Unit0Const());
              B.CreateInBoundsGEP(val->getType(),                  //
                                  mkTermVal(*lhs),                 //
                                  mkTermVal(x.idx), key + "_ptr"); //
              return val;
            }
            if (arrTpe->length) {
              auto ty = mkTpe(*arrTpe);
              auto ptr = B.CreateInBoundsGEP(ty,              //
                                             mkTermVal(*lhs), //
                                             {llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), mkTermVal(x.idx)}, key + "_idx_ptr");
              return load(B, ptr, mkTpe(arrTpe->component));
            } else {
              auto ty = mkTpe(arrTpe->component);
              auto ptr = B.CreateInBoundsGEP(ty,              //
                                             mkTermVal(*lhs), //
                                             mkTermVal(x.idx), key + "_idx_ptr");
              if (arrTpe->component.is<Type::Bool1>()) { // Narrow from i8 to i1
                return B.CreateICmpNE(load(B, ptr, ty), llvm::ConstantInt::get(llvm::Type::getInt1Ty(C), 0, true));
              } else {

                //                if(auto sizedArr = get_opt<Type::Ptr>(arrTpe->component); sizedArr && sizedArr->length){
                //                  return ptr;
                //                }else{
                //                  return load(B, ptr, ty);
                //                }

                //                return arrTpe->component->length ? ptr :load(B, ptr, ty);
                return load(B, ptr, ty);
              }
            }
          } else {
            throw std::logic_error("Semantic error: array index not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (index) is not a select");
      },

      [&](const Expr::RefTo &x) -> ValPtr {
        if (auto lhs = x.lhs.get<Term::Select>(); lhs) {
          if (auto arrTpe = lhs->tpe.get<Type::Ptr>(); arrTpe) { // taking reference of an index in an array
            auto offset = x.idx ? mkTermVal(*x.idx) : llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), 0, true);
            if (auto nestedArrTpe = arrTpe->component.get<Type::Ptr>(); nestedArrTpe && nestedArrTpe->length) {
              auto ty = arrTpe->component.is<Type::Unit0>() ? llvm::Type::getInt8Ty(C) : mkTpe(arrTpe->component);
              return B.CreateInBoundsGEP(ty,              //
                                         mkTermVal(*lhs), //
                                         {llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), offset},
                                         key + "_ref_to_" + llvm_tostring(ty));

            } else {
              auto ty = arrTpe->component.is<Type::Unit0>() ? llvm::Type::getInt8Ty(C) : mkTpe(arrTpe->component);
              return B.CreateInBoundsGEP(ty,              //
                                         mkTermVal(*lhs), //
                                         offset, key + "_ref_to_ptr");
            }
          } else { // taking reference of a var
            if (x.idx) throw std::logic_error("Semantic error: Cannot take reference of scalar with index in " + to_string(x));

            if (lhs->tpe.is<Type::Unit0>())
              throw std::logic_error("Semantic error: Cannot take reference of an select with unit type in " + to_string(x));
            return mkSelectPtr(*lhs);
          }
        } else
          throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (index) is not a select, can't take reference of a constant");
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
  if (lhs == rhs) return true;
  auto lhsStruct = lhs.get<Type::Struct>();
  auto rhsStruct = rhs.get<Type::Struct>();
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

  return stmt.match_total(
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

        if (x.expr && x.expr->tpe() != x.name.tpe) {
          throw std::logic_error("Semantic error: name type " + to_string(x.name.tpe) + " and rhs expr type " + to_string(x.expr->tpe()) +
                                 " mismatch (" + repr(x) + ")");
        }

        if (x.name.tpe.is<Type::Unit0>()) {
          // Unit0 declaration, discard declaration but keep RHS effect.
          if (x.expr) mkExprVal(*x.expr, fn, x.name.symbol + "_var_rhs");
        } else {
          auto tpe = mkTpe(x.name.tpe);

          std::cout << llvm_tostring(tpe) << " = " << x.name.tpe << "\n";
          auto stackPtr = B.CreateAlloca(tpe, AllocaAS, nullptr, x.name.symbol + "_stack_ptr");
          auto rhs = x.expr ? std::make_optional(mkExprVal(*x.expr, fn, x.name.symbol + "_var_rhs")) : std::nullopt;
          stackVarPtrs.emplace(x.name.symbol, Pair<Type::Any, llvm::Value *>{x.name.tpe, stackPtr});
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
        if (auto lhs = x.name.get<Term::Select>(); lhs) {
          if (x.expr.tpe() != lhs->tpe) {
            throw std::logic_error("Semantic error: name type (" + to_string(x.expr.tpe()) + ") and rhs expr (" + to_string(lhs->tpe) +
                                   ") mismatch (" + repr(x) + ")");
          }
          if (lhs->tpe.is<Type::Unit0>()) return BlockKind::Normal;
          auto rhs = mkExprVal(x.expr, fn, qualified(*lhs) + "_mut");
          if (lhs->init.empty()) { // local var
            auto stackPtr = findStackVar(lhs->last);
            B.CreateStore(rhs, stackPtr);
          } else { // struct member select
            B.CreateStore(rhs, mkSelectPtr(*lhs));
          }
        } else throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (mut) is not a select");
        return BlockKind::Normal;
      },
      [&](const Stmt::Update &x) -> BlockKind {
        if (auto lhs = x.lhs.get<Term::Select>(); lhs) {
          if (auto arrTpe = lhs->tpe.get<Type::Ptr>(); arrTpe) {
            auto rhs = x.value;

            bool componentIsSizedArray = false;
            if (auto p = arrTpe->component.get<Type::Ptr>(); p && p->length) {
              componentIsSizedArray = true;
            }

            if (arrTpe->component != rhs.tpe()) {
              throw std::logic_error("Semantic error: array component type (" + to_string(arrTpe->component) + ") and rhs expr (" +
                                     to_string(rhs.tpe()) + ") mismatch (" + repr(x) + ")");
            } else {
              auto dest = mkTermVal(*lhs);
              if (rhs.tpe().is<Type::Bool1>() || rhs.tpe().is<Type::Unit0>()) { // Extend from i1 to i8
                auto ty = llvm::Type::getInt8Ty(C);
                auto ptr = B.CreateInBoundsGEP( //
                    ty, dest,
                    componentIsSizedArray ? llvm::ArrayRef<ValPtr>{llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), mkTermVal(x.idx)}
                                          : llvm::ArrayRef<ValPtr>{mkTermVal(x.idx)},
                    qualified(*lhs) + "_update_ptr");
                B.CreateStore(B.CreateIntCast(mkTermVal(rhs), ty, true), ptr);
              } else {

                auto ptr = B.CreateInBoundsGEP( //
                    mkTpe(rhs.tpe()), dest,     //
                    componentIsSizedArray ? llvm::ArrayRef<ValPtr>{llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0), mkTermVal(x.idx)}
                                          : llvm::ArrayRef<ValPtr>{mkTermVal(x.idx)},
                    qualified(*lhs) + "_update_ptr" //
                );                                  //
                B.CreateStore(mkTermVal(rhs), ptr);
              }
            }
          } else {
            throw std::logic_error("Semantic error: array update not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
          }
        } else throw std::logic_error("Semantic error: LHS of " + to_string(x) + " (update) is not a select");

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
        else throw std::logic_error("orphaned break!");
        return BlockKind::Normal;
      }, //
      [&](const Stmt::Cont &x) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->test);
        else throw std::logic_error("orphaned cont!");
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
        auto rtnTpe = x.value.tpe();
        if (rtnTpe.is<Type::Unit0>()) {
          B.CreateRetVoid();
        } else if (rtnTpe.is<Type::Nothing>()) {
          B.CreateUnreachable();
        } else {
          auto expr = mkExprVal(x.value, fn, "return");
          if (rtnTpe.is<Type::Bool1>()) {
            // Extend from i1 to i8
            B.CreateRet(B.CreateIntCast(expr, llvm::Type::getInt8Ty(C), true));
          } else if (auto ptr = rtnTpe.get<Type::Ptr>(); ptr && ptr->length) {
            B.CreateRet(load(B, expr, mkTpe(rtnTpe)));
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
          bool noInheritanceDependency =
              std::all_of(def.parents.begin(), def.parents.end(), [&](auto &p) { return structTypes.find(p) != structTypes.end(); });
          bool noMemberDependencies = std::all_of(def.members.begin(), def.members.end(), [&](auto &m) {
            if (auto s = m.named.tpe.template get<Type::Struct>(); s) return structTypes.find(s->name) != structTypes.end();
            else return true;
          });
          return noMemberDependencies && noInheritanceDependency;
        });
    if (!zeroDeps.empty()) {
      for (auto &r : zeroDeps) {

        auto v = structTypes.emplace(r.name, mkStruct(r));
        std::cout << "Add " << llvm_tostring(v.first->second.tpe) << " from " << r << "\n";
        defsWithDependencies.erase(r);
      }
    } else
      throw std::logic_error("Recursive defs cannot be resolved: " + (zeroDeps ^ mk_string(",", [](auto &r) { return to_string(r); })));
  }
}

std::vector<Pair<Sym, llvm::StructType *>> LLVMBackend::AstTransformer::getStructTypes() const {
  return structTypes | map([](auto &k, auto &v) { return std::pair{k, v.tpe}; }) | to_vector();
}

static std::vector<Arg> collectFnDeclarationNames(const Function &f) {
  std::vector<Arg> allArgs;
  if (f.receiver) allArgs.push_back(*f.receiver);
  auto addAddExcludingUnit = [&](auto &xs) {
    for (auto &x : xs) {
      if (!x.named.tpe.template is<Type::Unit0>()) allArgs.push_back(x);
    }
  };
  addAddExcludingUnit(f.args);
  addAddExcludingUnit(f.moduleCaptures);
  addAddExcludingUnit(f.termCaptures);
  return allArgs;
}

void LLVMBackend::AstTransformer::addFn(llvm::Module &mod, const Function &f, bool entry) {

  auto args = collectFnDeclarationNames(f);
  auto llvmArgTpes = args ^ map([&](auto &&arg) { return mkTpe(arg.named.tpe, true); });

  // Unit type at function return type position is void, any other location, Unit is a singleton value
  auto rtnTpe = f.rtn.is<Type::Unit0>() ? llvm::Type::getVoidTy(C) : mkTpe(f.rtn, true);

  auto fnTpe = llvm::FunctionType::get(rtnTpe, {llvmArgTpes}, false);

  // XXX Normalise names as NVPTX has a relatively limiting range of supported characters in symbols
  auto cleanName=  qualified(f.name) ^ map([](char c) { return !std::isalnum(c) && c != '_' ? '_' : c; });

  auto *fn = llvm::Function::Create(fnTpe,                                                                     //
                                    (entry || f.kind == FunctionKind::Exported()) //
                                        ? llvm::Function::ExternalLinkage
                                        : llvm::Function::InternalLinkage,
                                    cleanName,                                                         //
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
              return tpe.match_total(                                                                 //
                  [&](const Type::Float16 &) -> std::string { return "half"; },                       //
                  [&](const Type::Float32 &) -> std::string { return "float"; },                      //
                  [&](const Type::Float64 &) -> std::string { return "double"; },                     //
                  [&](const Type::IntU8 &) -> std::string { return "uchar"; },                        //
                  [&](const Type::IntU16 &) -> std::string { return "ushort"; },                      //
                  [&](const Type::IntU32 &) -> std::string { return "uint"; },                        //
                  [&](const Type::IntU64 &) -> std::string { return "ulong"; },                       //
                  [&](const Type::IntS8 &) -> std::string { return "char"; },                         //
                  [&](const Type::IntS16 &) -> std::string { return "short"; },                       //
                  [&](const Type::IntS32 &) -> std::string { return "int"; },                         //
                  [&](const Type::IntS64 &) -> std::string { return "long"; },                        //
                  [&](const Type::Bool1 &) -> std::string { return "char"; },                         //
                  [&](const Type::Unit0 &) -> std::string { return "void"; },                         //
                  [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },                //
                  [&](const Type::Struct &x) -> std::string { return qualified(x.name); },            //
                  [&](const Type::Ptr &x) -> std::string { return thunk(x.component, thunk) + "*"; }, //
                  [&](const Type::Var &) -> std::string { throw std::logic_error("type var"); },      //
                  [&](const Type::Exec &) -> std::string { throw std::logic_error("exec"); }          //
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

  functions.emplace(InvokeSignature(f.name,                                                //
                                    {},                                                    //
                                    f.receiver ^ map([](auto &x) { return x.named.tpe; }), //
                                    f.args ^ map([](auto &x) { return x.named.tpe; }),     //
                                    extraArgTpes,                                          //
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

  InvokeSignature sig(fnTree.name,                                                //
                      {},                                                         //
                      fnTree.receiver ^ map([](auto &x) { return x.named.tpe; }), //
                      fnTree.args ^ map([](auto &x) { return x.named.tpe; }),     //
                      argTpes,                                                    //
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

        auto argValue = fnArg.named.tpe.template is<Type::Bool1>() || fnArg.named.tpe.template is<Type::Unit0>()
                            ? B.CreateICmpNE(&arg, llvm::ConstantInt::get(llvm::Type::getInt8Ty(C), 0, true))
                            : &arg;

        //        auto as = HOLDS(ArgKind::Local, fnArg.kind) ? LocalAS : GlobalAS;
        auto stack = B.CreateAlloca(mkTpe(fnArg.named.tpe), AllocaAS, nullptr, fnArg.named.symbol + "_stack_ptr");
        B.CreateStore(argValue, stack);
        return {fnArg.named.symbol, {fnArg.named.tpe, stack}};
      });

  for (auto &stmt : fnTree.body)
    mkStmt(stmt, fn);

  stackVarPtrs.clear();
}
ValPtr LLVMBackend::AstTransformer::unaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyType &rtn, const ValPtrFn1 &fn) { //
  if (l.tpe() != rtn) {
    throw std::logic_error("Semantic error: lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkTermVal(l));
}
ValPtr LLVMBackend::AstTransformer::binaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn,
                                               const ValPtrFn2 &fn) { //
  if (l.tpe() != rtn) {
    throw std::logic_error("Semantic error: lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }
  if (r.tpe() != rtn) {
    throw std::logic_error("Semantic error: rhs type " + to_string(r.tpe()) + " of binary numeric operation in " + to_string(expr) +
                           " doesn't match return type " + to_string(rtn));
  }

  return fn(mkTermVal(l), mkTermVal(r));
}
ValPtr LLVMBackend::AstTransformer::unaryNumOp(const AnyExpr &expr, const AnyTerm &arg, const AnyType &rtn, //
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
ValPtr LLVMBackend::AstTransformer::binaryNumOp(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, //
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
ValPtr LLVMBackend::AstTransformer::extFn1(llvm::Function *fn, const std::string &name, const AnyType &rtn, const AnyTerm &arg) { //
  auto fn_ = mkExternalFn(fn, rtn, name, {arg.tpe()});
  if (options.target == Target::SPIRV32 || options.target == Target::SPIRV64) {
    fn_->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    //    fn_->addFnAttr(llvm::Attribute::NoBuiltin);
    //    fn_->addFnAttr(llvm::Attribute::Convergent);
  }
  if (!rtn.is<Type::Unit0>()) {
    fn_->addFnAttr(llvm::Attribute::WillReturn);
  }
  auto call = B.CreateCall(fn_, mkTermVal(arg));
  call->setCallingConv(fn_->getCallingConv());
  return call;
}
ValPtr LLVMBackend::AstTransformer::extFn2(llvm::Function *fn, const std::string &name, const AnyType &rtn, const AnyTerm &lhs,
                                           const AnyTerm &rhs) { //
  auto fn_ = mkExternalFn(fn, rtn, name, {lhs.tpe(), rhs.tpe()});
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

std::vector<polyast::CompileLayout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &defs,
                                                                const backend::LLVMBackend::AstTransformer &xform) const {

  auto dataLayout = options.targetInfo().resolveDataLayout();

  std::unordered_map<polyast::Sym, polyast::StructDef> lut(defs.size());
  for (auto &d : defs)
    lut.emplace(d.name, d);

  std::vector<polyast::CompileLayout> layouts;
  for (auto &[sym, structTy] : xform.getStructTypes()) {
    if (auto it = lut.find(sym); it != lut.end()) {
      auto layout = dataLayout.getStructLayout(structTy);
      std::vector<polyast::CompileLayoutMember> members;
      for (size_t i = 0; i < it->second.members.size(); ++i) {
        members.emplace_back(it->second.members[i].named,                             //
                             layout->getElementOffset(i),                             //
                             dataLayout.getTypeAllocSize(structTy->getElementType(i)) //
        );
      }
      layouts.emplace_back(sym, layout->getSizeInBytes(), layout->getAlignment().value(), members);
    } else throw std::logic_error("Cannot find symbol " + to_string(sym) + " from domain");
  }
  return layouts;
}

std::vector<polyast::CompileLayout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &defs, const compiletime::OptLevel &opt) {
  llvm::LLVMContext ctx;
  backend::LLVMBackend::AstTransformer xform(options, ctx);
  xform.addDefs(defs);
  return resolveLayouts(defs, xform);
}

polyast::CompileResult backend::LLVMBackend::compileProgram(const Program &program, const compiletime::OptLevel &opt) {
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
  polyast::CompileEvent ast2IR(compiler::nowMs(), compiler::elapsedNs(transformStart), "ast_to_llvm_ir", transformMsg);

  auto verifyStart = compiler::nowMono();
  auto [maybeVerifyErr, verifyMsg] = llvmc::verifyModule(*mod);
  polyast::CompileEvent astOpt(compiler::nowMs(), compiler::elapsedNs(verifyStart), "llvm_ir_verify", verifyMsg);

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

  auto c = llvmc::compileModule(options.targetInfo(), opt, true, std::move(mod));
  c.layouts = resolveLayouts(program.defs, xform);
  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
