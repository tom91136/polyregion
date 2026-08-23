#include "llvm.h"

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/conventions.h"
#include "polyregion/types.h"

#include "ast.h"
#include "llvm_amdgpu.h"
#include "llvm_cpu.h"
#include "llvm_nvptx.h"
#include "llvm_spirv_cl.h"
#include "llvm_vulkan.h"
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
    case LLVMBackend::Target::ARM: [[fallthrough]];
    case LLVMBackend::Target::RISCV64: [[fallthrough]];
    case LLVMBackend::Target::PPC64LE: return std::make_unique<CPUTargetSpecificHandler>();
    case LLVMBackend::Target::NVPTX64: return std::make_unique<NVPTXTargetSpecificHandler>();
    case LLVMBackend::Target::AMDGCN: return std::make_unique<AMDGPUTargetSpecificHandler>();
    case LLVMBackend::Target::SPIRV32_Kernel: [[fallthrough]];
    case LLVMBackend::Target::SPIRV64_Kernel: return std::make_unique<SPIRVOpenCLTargetSpecificHandler>();
    case LLVMBackend::Target::SPIRV_GLCompute: return std::make_unique<SPIRVVulkanTargetSpecificHandler>();
    default: throw BackendException(fmt::format("Unknown target {}", magic_enum::enum_name(target)));
  }
}

TargetSpecificHandler::~TargetSpecificHandler() = default;
ValPtr TargetSpecificHandler::isNaN(CodeGen &gen, llvm::Value *from) { return gen.B.CreateFCmpUNO(from, from); }

namespace {

// step a base pointer by `offset` elements of `compTpe`. used where a length-0 Local Arr collapsed to a
// scalar on SPIR-V kernel (no [0 x T] runtime array), so the [N x T] two-index GEP has no aggregate base.
// SPIR-V kernel routes through a byte-offset i8 chain (a scalar base takes a single index and matches the
// Ptr-branch's Arc OpenCL handling)
ValPtr elemStepPtr(CodeGen &gen, const Type::Any &compTpe, llvm::Value *base, llvm::Value *offset, const std::string &key) {
  auto &B = gen.B;
  auto &C = gen.C;
  auto *elemTy = gen.resolveType(compTpe);
  if (C.isSpirvKernel()) {
    auto *elemSize = llvm::ConstantInt::get(C.i64Ty(), gen.M.getDataLayout().getTypeAllocSize(elemTy));
    return gen.byteOffsetPtr(base, B.CreateMul(offset, elemSize), key);
  }
  return B.CreateInBoundsGEP(elemTy, base, offset, key);
}

// centralises the "scalar-collapsed length-0 Local Arr vs [N x T] aggregate" GEP decision so no pointer-forming
// site re-derives the isArrayTy guard. resolveType collapses a length-0 dynamic-local Arr to a scalar on SPIR-V
// kernel (no [0 x T]), so the two-index aggregate GEP is invalid there and `collapsedStep` forms the pointer
// instead; the normal [N x T] arr indexes the aggregate with a leading 0
template <typename CollapsedStep>
ValPtr arrElemPtr(CodeGen &gen, llvm::Type *arrTy, llvm::Value *base, llvm::Value *offset, const std::string &key,
                  const CollapsedStep &collapsedStep) {
  if (arrTy->isArrayTy()) return gen.B.CreateInBoundsGEP(arrTy, base, {llvm::ConstantInt::get(gen.C.i32Ty(), 0), offset}, key);
  return collapsedStep();
}

ValPtr arrElemPtr(CodeGen &gen, llvm::Type *arrTy, const Type::Any &comp, llvm::Value *base, llvm::Value *offset, const std::string &key) {
  return arrElemPtr(gen, arrTy, base, offset, key, [&] { return elemStepPtr(gen, comp, base, gen.i64SExt(offset), key); });
}

ValPtr physicalIndexVal(CodeGen &gen, const Expr::Index &x, const std::string &key) {
  auto &B = gen.B;
  auto &C = gen.C;
  auto &M = gen.M;
  using llvm::ConstantInt;
  // sign-extend the index to i64: SPIR-V treats access-chain Element as unsigned, so a 32-bit -1 jumps ~16 GB
  if (const auto lhs = x.lhs.template get<Term::Select>()) {
    if (const auto arrTpe = lhs->tpe.template get<Type::Ptr>()) {
      if (arrTpe->comp.is<Type::Unit0>()) {
        const auto val = gen.mkTermVal(Term::Unit0Const());
        B.CreateInBoundsGEP(val->getType(), gen.mkTermVal(*lhs), gen.i64SExt(gen.mkTermVal(x.idx)), key + "_ptr");
        return val;
      } else if (auto innerArr = arrTpe->comp.get<Type::Arr>()) {
        const auto arrLlvmTy = gen.resolveType(*innerArr);
        const auto compLlvmTy = gen.resolveType(innerArr->comp);
        const auto basePtr = gen.mkTermVal(*lhs);
        const auto ptr =
            B.CreateInBoundsGEP(arrLlvmTy, basePtr, {ConstantInt::get(C.i32Ty(), 0), gen.i64SExt(gen.mkTermVal(x.idx))}, key + "_idx_ptr");
        if (innerArr->comp.is<Type::Bool1>()) {
          return B.CreateICmpNE(C.load(B, ptr, compLlvmTy), ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0, true));
        }
        return C.load(B, ptr, compLlvmTy);
      } else {
        const auto ty = gen.resolveType(arrTpe->comp);
        auto *basePtr = gen.mkTermVal(*lhs);
        llvm::Value *ptr;
        if (gen.spirvStructByMemcpy()) {
          auto *elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(ty));
          auto *byteOff = B.CreateMul(gen.i64SExt(gen.mkTermVal(x.idx)), elemSize);
          ptr = gen.byteOffsetPtr(basePtr, byteOff, key + "_idx_ptr");
        } else {
          ptr = B.CreateInBoundsGEP(ty, basePtr, gen.i64SExt(gen.mkTermVal(x.idx)), key + "_idx_ptr");
        }
        if (arrTpe->comp.is<Type::Bool1>()) {
          return B.CreateICmpNE(C.load(B, ptr, ty), ConstantInt::get(llvm::Type::getInt1Ty(C.actual), 0, true));
        }
        if (gen.structByPtr() && arrTpe->comp.template is<Type::Struct>()) return ptr;
        return C.load(B, ptr, ty);
      }
    } else if (const auto arrTpe = lhs->tpe.template get<Type::Arr>()) {
      const auto ptr =
          arrElemPtr(gen, gen.resolveType(*arrTpe), arrTpe->comp, gen.mkTermVal(*lhs), gen.i64SExt(gen.mkTermVal(x.idx)), key + "_idx_ptr");
      if (gen.structByPtr() && arrTpe->comp.template is<Type::Struct>()) return ptr;
      return C.load(B, ptr, gen.resolveType(arrTpe->comp));
    } else {
      throw BackendException::semantic("array index not called on array type (" + to_string(lhs->tpe) + ")(" + repr(x) + ")");
    }
  } else throw BackendException::semantic("LHS of " + to_string(x) + " (index) is not a select");
}

static ValPtr physicalRefToPtr(CodeGen &gen, const Expr::RefTo &x, const std::string &key) {
  auto &B = gen.B;
  auto &C = gen.C;
  auto &M = gen.M;
  // `&base[idx]` where base is a pointer-typed value: the base already is the address, so step it by idx elements
  const auto stepPtr = [&](const Type::Ptr &ptrTpe, const Term::Any &baseTerm) -> ValPtr {
    auto offset = x.idx ? gen.i64SExt(gen.mkTermVal(*x.idx)) : llvm::ConstantInt::get(C.i64Ty(), 0, true);
    auto *base = gen.mkTermVal(baseTerm);
    if (auto innerArr = ptrTpe.comp.template get<Type::Arr>())
      return B.CreateGEP(gen.resolveType(*innerArr), base, {llvm::ConstantInt::get(C.i32Ty(), 0), offset}, key + "_ref_to_ptr_arr");
    auto ty = ptrTpe.comp.template is<Type::Unit0>() ? llvm::Type::getInt8Ty(C.actual) : gen.resolveType(ptrTpe.comp);
    // kernel SPIR-V: ptrtoint round-trip works around Arc OpenCL mis-handling negative OpPtrAccessChain elements
    if (C.isSpirvKernel()) {
      auto elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(ty));
      auto *byteOffset = B.CreateMul(offset, elemSize);
      return gen.byteOffsetPtr(base, byteOffset, key + "_ref_to_ptr");
    }
    return B.CreateInBoundsGEP(ty, base, offset, key + "_ref_to_ptr");
  };
  if (auto lhs = x.lhs.template get<Term::Select>()) {
    if (auto arrTpe = lhs->tpe.template get<Type::Ptr>(); arrTpe) {
      // `&p` on a pointer-typed variable (no index, comp is the operand's own pointer type = one more
      // indirection) is the address of its slot, not a decay `&p[0]`. lowering it as base+offset loads the
      // pointer value and GEPs off that - null for an unwritten local - so a later `*dest = x` through the
      // stored address faults
      if (!x.idx && x.comp == lhs->tpe) return gen.mkSelectPtr(*lhs);
      return stepPtr(*arrTpe, x.lhs);
    } else if (auto arrTpe = lhs->tpe.template get<Type::Arr>(); arrTpe) {
      auto offset = x.idx ? gen.i64SExt(gen.mkTermVal(*x.idx)) : llvm::ConstantInt::get(C.i64Ty(), 0, true);
      auto arrLlvmTy = gen.resolveType(*arrTpe);
      auto *base = gen.mkTermVal(*lhs);
      return arrElemPtr(gen, arrLlvmTy, base, offset, key + "_ref_to_" + llvm_tostring(arrLlvmTy),
                        [&] { return elemStepPtr(gen, arrTpe->comp, base, offset, key + "_ref_to_ptr"); });
    } else {
      if (x.idx) throw BackendException::semantic("Cannot take reference of scalar with index in " + to_string(x));
      if (lhs->tpe.is<Type::Unit0>())
        throw BackendException::semantic("Cannot take reference of an select with unit type in " + to_string(x));
      return gen.mkSelectPtr(*lhs);
    }
  } else {
    // a constant pointer base (a null/poison argument substituted into a by-value pointer param by inlining) has
    // no slot to spill, but it is itself the address, so step it like a pointer-typed select
    if (const auto ptrTpe = x.lhs.tpe().template get<Type::Ptr>(); ptrTpe && x.idx) return stepPtr(*ptrTpe, x.lhs);
    if (x.idx) throw BackendException::semantic("Cannot take reference of a constant with index in " + to_string(x));
    const auto val = gen.mkTermVal(x.lhs);
    // a struct-typed parameter is lowered to a pointer at the function boundary; the value IS
    // already the address, so return it directly instead of boxing it in an alloca.
    if (val->getType()->isPointerTy() && !x.lhs.tpe().template is<Type::Ptr>() && !x.lhs.tpe().template is<Type::Arr>()) return val;
    const auto slot = C.allocaAS(B, val->getType(), C.AllocaAS, key + "_const_ref");
    const auto _ = C.store(B, val, slot);
    return slot;
  }
}

// a RefTo's declared space is its ABI: the address must land in that space rather than inherit whatever the
// base happened to lower to, or a workgroup address reaches a flat consumer (and back) by reinterpretation
ValPtr physicalRefToVal(CodeGen &gen, const Expr::RefTo &x, const std::string &key) {
  const auto ptr = physicalRefToPtr(gen, x, key);
  if (!ptr->getType()->isPointerTy()) return ptr;
  const auto want = gen.B.getPtrTy(gen.C.addressSpace(x.space));
  return ptr->getType() == want ? ptr : gen.B.CreateAddrSpaceCast(ptr, want);
}

void physicalStoreUpdate(CodeGen &gen, const Term::Select &lhs, const Term::Any &idx, const Term::Any &value) {
  auto &B = gen.B;
  auto &C = gen.C;
  auto &M = gen.M;
  const bool componentIsSizedArray = lhs.tpe.template is<Type::Arr>();
  const auto dest = gen.mkTermVal(lhs);
  const auto valTy = value.tpe().template is<Type::Bool1>() ? llvm::Type::getInt8Ty(C.actual) : gen.resolveType(value.tpe());
  const auto gepTy = componentIsSizedArray ? gen.resolveType(lhs.tpe) : valTy;
  const auto getPtr = [&]() -> llvm::Value * {
    if (componentIsSizedArray)
      return arrElemPtr(gen, gepTy, lhs.tpe.template get<Type::Arr>()->comp, dest, gen.mkTermVal(idx), qualified(lhs) + "_update_ptr");
    if (gen.spirvStructByMemcpy()) {
      auto *elemSize = llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(valTy));
      auto *byteOff = B.CreateMul(gen.i64SExt(gen.mkTermVal(idx)), elemSize);
      return gen.byteOffsetPtr(dest, byteOff, qualified(lhs) + "_update_ptr");
    }
    return B.CreateInBoundsGEP(gepTy, dest, {gen.mkTermVal(idx)}, qualified(lhs) + "_update_ptr");
  };
  const auto ptr = getPtr();
  if (gen.structByPtr() && value.tpe().template is<Type::Struct>()) {
    gen.copyStruct(ptr, gen.mkTermVal(value), value.tpe());
  } else if (value.tpe().template is<Type::Bool1>()) {
    const auto _ = C.store(B, B.CreateZExt(gen.mkTermVal(value), valTy), ptr);
  } else {
    const auto _ = C.store(B, gen.mkTermVal(value), ptr);
  }
}

// oneGep (logical SPIR-V) folds a run of struct-field steps into one multi-index GEP; physical targets GEP per field
ValPtr selectPtrImpl(CodeGen &gen, const Term::Select &select, const bool oneGep) {
  auto &B = gen.B;
  auto &C = gen.C;

  auto fail = [&] { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> StructInfo {
    auto findTy = [&](const Type::Struct &s) -> StructInfo {
      return gen.structTypes ^ get_maybe(repr(s.name)) ^ fold([&]() -> StructInfo {
               throw BackendException("Unseen struct type " + to_string(s.name) + " in select path" + fail());
             });
    };

    if (auto s = tpe.get<Type::Struct>(); s) {
      return findTy(*s);
    } else if (auto p = tpe.get<Type::Ptr>(); p) {
      if (auto _s = p->comp.get<Type::Struct>(); _s) return findTy(*_s);
      else
        throw BackendException("Illegal select path involving pointer to non-struct type " + to_string(s->name) + " in select path"
                               + fail());
    } else throw BackendException("Illegal select path involving non-struct type " + to_string(tpe) + fail());
  };

  // a `#base_X` step may name a transitively-inherited base reached through intermediate `#base_*` members;
  // return the member-index hops from `si` down to and including the target, or empty if unreachable. only
  // descends base subobjects, so it never crosses a data field
  std::function<std::vector<size_t>(const StructInfo &, const std::string &)> baseChain =
      [&](const StructInfo &si, const std::string &name) -> std::vector<size_t> {
    return si.memberIndices | collect_first([&](const auto &field, const size_t index) -> Opt<std::vector<size_t>> {
             if (!(field ^ starts_with(conventions::BaseFieldPrefix)) || index >= si.def.members.size()) return std::nullopt;
             if (!si.def.members[index].tpe.template is<Type::Struct>()) return std::nullopt;
             const auto sub = structTypeOf(si.def.members[index].tpe);
             if (const auto direct = sub.memberIndices ^ get_maybe(name)) return std::vector<size_t>{index, *direct};
             const auto deeper = baseChain(sub, name);
             return deeper.empty() ? std::nullopt : Opt<std::vector<size_t>>{deeper ^ prepend(index)};
           })
           | get_or_else(std::vector<size_t>{});
  };

  if (select.steps.empty()) return gen.findStackVar(select.root);
  auto tpe = select.root.tpe;
  auto root = gen.findStackVar(select.root);

  // an inline Arr local sits behind a ref-ptr slot (see Stmt::Var); load it before indexing (SPIR-V doesn't)
  if (auto arr = tpe.template get<Type::Arr>(); arr && !C.isSpirv()) root = C.load(B, root, B.getPtrTy(C.addressSpace(arr->space)));

  llvm::SmallVector<llvm::Value *, 8> idxs;
  llvm::Type *gepBaseTy = nullptr;
  auto flush = [&]() {
    if (idxs.empty()) return;
    root = B.CreateInBoundsGEP(gepBaseTy, root, idxs, qualified(select) + "_select_ptr");
    idxs.clear();
    gepBaseTy = nullptr;
  };

  // GEP to member `idx` of `info`, advancing root/tpe (shared by direct fields and transitive-base hops)
  auto emitMember = [&](const StructInfo &info, const size_t idx) {
    // a union reinterprets its members over shared storage. an overlaid union puts every member at offset 0; a
    // de-aliased union gives its discontinuity members a distinct byte offset (see resolveLayouts) so the AMDGPU
    // O3 optimiser cannot confuse the tail-flag storage with an overlapping block_exchange buffer
    if (info.def.isUnion) {
      flush();
      if (idx < info.def.members.size()) {
        if (info.deAliased)
          if (const auto off = info.layout.members[idx].offsetInBytes; off != 0)
            root = gen.byteOffsetPtr(root, llvm::ConstantInt::get(C.i64Ty(), off), qualified(select) + "_select_ptr");
        tpe = info.def.members[idx].tpe;
      }
      return;
    }
    if (gen.spirvStructByMemcpy()) {
      const auto offsetBytes = static_cast<size_t>(info.layout.members[idx].offsetInBytes);
      auto *off = llvm::ConstantInt::get(C.i64Ty(), offsetBytes);
      root = gen.byteOffsetPtr(root, off, qualified(select) + "_select_ptr");
    } else if (oneGep) {
      if (idxs.empty()) {
        gepBaseTy = info.tpe;
        idxs.push_back(llvm::ConstantInt::get(C.i32Ty(), 0));
      }
      idxs.push_back(llvm::ConstantInt::get(C.i32Ty(), static_cast<unsigned>(idx)));
    } else {
      root = B.CreateInBoundsGEP(info.tpe, root, {llvm::ConstantInt::get(C.i32Ty(), 0), llvm::ConstantInt::get(C.i32Ty(), idx)},
                                 qualified(select) + "_select_ptr");
    }
    if (idx < info.def.members.size()) tpe = info.def.members[idx].tpe;
    // a pointer wrapper of a Struct (functionBoundary lowering) needs a deref
    const auto fieldLlvmType = info.tpe->getElementType(idx);
    if (fieldLlvmType->isPointerTy() && tpe.template is<Type::Struct>()) {
      flush();
      root = C.load(B, root, llvm::cast<llvm::PointerType>(fieldLlvmType));
    }
  };

  for (auto &step : select.steps) {
    if (step.template is<PathStep::Deref>()) {
      if (auto p = tpe.template get<Type::Ptr>()) {
        flush();
        root = C.load(B, root, C.loadedPtrTy(B, p->space));
        tpe = p->comp;
        continue;
      }
      throw BackendException("Deref step on non-pointer type " + to_string(tpe) + fail());
    }
    if (auto idx = step.template get<PathStep::Index>()) {
      auto arr = tpe.template get<Type::Arr>();
      if (!arr) throw BackendException("Index step on non-array type " + to_string(tpe) + fail());
      auto *idxV = llvm::ConstantInt::get(C.i64Ty(), idx->idx);
      if (oneGep) {
        if (idxs.empty()) {
          gepBaseTy = gen.resolveType(tpe);
          idxs.push_back(llvm::ConstantInt::get(C.i32Ty(), 0));
        }
        idxs.push_back(idxV);
      } else {
        // scalar-collapsed length-0 Local Arr keeps a typed single-index element chain (not the byte-offset step)
        root = arrElemPtr(gen, gen.resolveType(tpe), root, idxV, qualified(select) + "_select_ptr",
                          [&] { return B.CreateInBoundsGEP(gen.resolveType(arr->comp), root, idxV, qualified(select) + "_select_ptr"); });
      }
      tpe = arr->comp;
      continue;
    }
    // runtime index into an inline array element; folds into the one access chain
    if (auto dyn = step.template get<PathStep::IndexDyn>()) {
      auto *idxV = gen.i64SExt(gen.mkTermVal(dyn->idx));
      auto arr = tpe.template get<Type::Arr>();
      if (!arr) throw BackendException("IndexDyn step on non-array type " + to_string(tpe) + fail());
      // typed access chain even on SPIR-V kernel: a byte-offset inttoptr loses the per-lane provenance IGC needs
      if (oneGep) {
        if (idxs.empty()) {
          gepBaseTy = gen.resolveType(tpe);
          idxs.push_back(llvm::ConstantInt::get(C.i32Ty(), 0));
        }
        idxs.push_back(idxV);
      } else {
        root = B.CreateInBoundsGEP(gen.resolveType(tpe), root, {llvm::ConstantInt::get(C.i32Ty(), 0), idxV},
                                   qualified(select) + "_select_ptr");
      }
      tpe = arr->comp;
      continue;
    }
    const auto fieldStep = step.template get<PathStep::Field>();
    if (!fieldStep) throw BackendException("Unhandled PathStep variant" + fail());
    // a Field on a Ptr type means implicit deref (load) then GEP
    if (auto p = tpe.template get<Type::Ptr>()) {
      flush();
      root = C.load(B, root, C.loadedPtrTy(B, p->space));
      tpe = p->comp;
    }
    // a Field on an Arr type means implicit index [0]: a __shared__ struct is backed as Arr(Struct,1,Local)
    // (annotateLocalSpace), and by-ref decay reads &ts[0] - a direct member access reaches element 0 the same way
    if (auto a = tpe.template get<Type::Arr>()) {
      flush();
      root = arrElemPtr(gen, gen.resolveType(tpe), a->comp, root, llvm::ConstantInt::get(C.i32Ty(), 0), qualified(select) + "_select_arr0");
      tpe = a->comp;
    }
    const auto info = structTypeOf(tpe);
    auto idxOpt = info.memberIndices ^ get_maybe(fieldStep->name);
    if (!idxOpt) {
      // EBO'd empty base: resolve to the empty struct's own address (offset 0) rather than fail
      if (info.memberIndices.empty()) continue;
      auto pool = info.memberIndices ^ mk_string("\n", "\n", "\n", [](const auto &k, const auto &v) {
                    return " -> `" + k + "` = " + std::to_string(v) + ")";
                  });
      throw BackendException("Illegal select path with unknown struct member index of name `" + fieldStep->name + "`, pool=" + pool
                             + fail());
    }
    emitMember(info, *idxOpt);
  }
  flush();
  return root;
}

struct PhysicalPointerModel final : PointerModel {
  ValPtr selectPtr(CodeGen &gen, const Term::Select &select) override { return selectPtrImpl(gen, select, /*oneGep*/ false); }
  void copyAggregate(CodeGen &gen, ValPtr dst, ValPtr src, const AnyType &tpe) override {
    // memcpy copies between two pointers; an aggregate rvalue (poison/constant/SSA, e.g. an uninitialised
    // struct local) arrives as a value, not a pointer, so store it directly as the by-value path does
    if (!src->getType()->isPointerTy()) {
      const auto _ = gen.C.store(gen.B, src, dst);
      return;
    }
    if (auto s = tpe.get<Type::Struct>()) {
      const auto &info = gen.structTypes.at(repr(s->name));
      if (info.layout.sizeInBytes != 0)
        gen.B.CreateMemCpy(dst, llvm::MaybeAlign(info.layout.alignment), src, llvm::MaybeAlign(info.layout.alignment),
                           info.layout.sizeInBytes);
    } else { // by-value array (e.g. std::array's _M_elems)
      auto *ty = gen.resolveType(tpe);
      const auto &dl = gen.M.getDataLayout();
      const auto al = dl.getABITypeAlign(ty);
      const auto size = dl.getTypeAllocSize(ty);
      if (!size.isZero()) gen.B.CreateMemCpy(dst, al, src, al, size);
    }
  }
  void zeroAggregate(CodeGen &gen, ValPtr dst, const AnyType &tpe) override {
    const auto _ = gen.C.store(gen.B, llvm::Constant::getNullValue(gen.resolveType(tpe)), dst);
  }
  ValPtr indexVal(CodeGen &gen, const Expr::Index &index, const std::string &key) override { return physicalIndexVal(gen, index, key); }
  ValPtr refToVal(CodeGen &gen, const Expr::RefTo &refTo, const std::string &key) override { return physicalRefToVal(gen, refTo, key); }
  void storeUpdate(CodeGen &gen, const Term::Select &lhs, const Term::Any &idx, const Term::Any &value) override {
    physicalStoreUpdate(gen, lhs, idx, value);
  }
};

struct LogicalPointerModel final : VulkanLowering {
  using VulkanLowering::VulkanLowering;
  ValPtr selectPtr(CodeGen &gen, const Term::Select &select) override { return selectPtrImpl(gen, select, /*oneGep*/ true); }
  void copyAggregate(CodeGen &gen, ValPtr dst, ValPtr src, const AnyType &tpe) override {
    structFieldCopy(dst, src, gen.resolveType(tpe), tpe, {llvm::ConstantInt::get(gen.C.i32Ty(), 0)});
  }
  void zeroAggregate(CodeGen &gen, ValPtr dst, const AnyType &tpe) override {
    structFieldZero(dst, gen.resolveType(tpe), tpe, {llvm::ConstantInt::get(gen.C.i32Ty(), 0)});
  }
  ValPtr indexVal(CodeGen &gen, const Expr::Index &index, const std::string &key) override {
    if (const auto lhs = index.lhs.template get<Term::Select>())
      if (auto v = mkIndex(*lhs, index.idx)) return *v;
    return physicalIndexVal(gen, index, key);
  }
  ValPtr refToVal(CodeGen &gen, const Expr::RefTo &refTo, const std::string &key) override {
    if (auto lhs = refTo.lhs.template get<Term::Select>())
      if (auto v = mkRefTo(*lhs, refTo.idx)) return *v;
    return physicalRefToVal(gen, refTo, key);
  }
  void storeUpdate(CodeGen &gen, const Term::Select &lhs, const Term::Any &idx, const Term::Any &value) override {
    if (mkUpdate(lhs, idx, value)) return;
    physicalStoreUpdate(gen, lhs, idx, value);
  }
};

} // namespace

static bool isUnsigned(const Type::Any &tpe) { // unsigned types in PolyAst; a bool is 0/1, so it zero-extends
  return tpe.is<Type::IntU8>() || tpe.is<Type::IntU16>() || tpe.is<Type::IntU32>() || tpe.is<Type::IntU64>() || tpe.is<Type::Bool1>();
}

// a CondBr on a constant leaves the untaken successor unreachable; the SPIR-V structurizer cannot resolve one
static void condBr(llvm::IRBuilder<> &B, ValPtr cond, llvm::BasicBlock *whenTrue, llvm::BasicBlock *whenFalse) {
  if (const auto c = llvm::dyn_cast<llvm::ConstantInt>(cond)) B.CreateBr(c->isOne() ? whenTrue : whenFalse);
  else B.CreateCondBr(cond, whenTrue, whenFalse);
}

// a FnRef-typed value is a stubbed kernel handle (Specialisation/FnInline thread it through as a type-erased
// pointer that is never dereferenced device-side), so the Var/Mut store paths store null for it
static void storeStubbedHandle(CodeGen &gen, llvm::Type *slotTy, llvm::Value *dst) {
  const auto _ = gen.C.store(gen.B, llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(slotTy)), dst);
}

static constexpr int64_t nIntMin(uint64_t bits) { return -(int64_t(1) << (bits - 1)); }
static constexpr int64_t nIntMax(uint64_t bits) { return (int64_t(1) << (bits - 1)) - 1; }

CodeGen::CodeGen(const LLVMBackend::Options &options, const std::string &moduleName)
    : C(options), targetHandler(TargetSpecificHandler::from(options.target)), B(C.actual), M(moduleName, C.actual) {
  if (C.isVulkan()) ptrModel = std::make_unique<LogicalPointerModel>(*this);
  else ptrModel = std::make_unique<PhysicalPointerModel>();
  // bind the target datalayout up front so codegen-time getTypeAllocSize/alloca/GEP offsets match the authoritative StructInfo layout
  M.setDataLayout(C.options.targetInfo().resolveDataLayout());
}

CodeGen::~CodeGen() = default;

llvm::Type *CodeGen::resolveType(const AnyType &tpe, const bool functionBoundary, const bool kernelEntryArg) {
  return C.resolveType(tpe, structTypes, functionBoundary, kernelEntryArg);
}

llvm::Value *CodeGen::byteOffsetPtr(llvm::Value *base, llvm::Value *byteOff, const std::string &name) {
  // a byte-typed access chain, not a ptr<->int round-trip IGC can't track per-lane
  return B.CreateInBoundsGEP(B.getInt8Ty(), base, byteOff, name);
}

llvm::Value *CodeGen::i64SExt(llvm::Value *v) { return B.CreateSExtOrTrunc(v, C.i64Ty()); }

bool CodeGen::sameIntSlot(const AnyType &a, const AnyType &b) {
  return a.kind().is<TypeKind::Integral>() && b.kind().is<TypeKind::Integral>() && resolveType(a) == resolveType(b);
}

ValPtr CodeGen::toI1(const AnyTerm &p) {
  auto *v = mkTermVal(p);
  return v->getType()->isIntegerTy(1) ? v : B.CreateICmpNE(v, llvm::ConstantInt::get(v->getType(), 0));
}

llvm::Function *CodeGen::resolveExtFn(const Type::Any &rtn, const std::string &name, const std::vector<Type::Any> &args) {
  return get_or_emplace(externalFunctions, Signature(Sym({name}), {}, {}, args, {}, {}, rtn), [&](const auto &sig) -> llvm::Function * {
    auto tpe = llvm::FunctionType::get(
        /*Result*/ resolveType(rtn, true),
        /*Params*/ args ^ map([&](const auto &t) { return resolveType(t, true); }),
        /*isVarArg*/ false);
    auto fn = llvm::Function::Create(tpe, llvm::Function::ExternalLinkage, name, M);
    return fn;
  });
}

ValPtr CodeGen::invokeMalloc(ValPtr size) {
  return B.CreateCall(resolveExtFn(Type::Ptr(Type::IntS8(), TypeSpace::Global()), "malloc", {Type::IntS64()}), size);
}

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
ValPtr CodeGen::intrAbs(const AnyType &overload, const AnyTerm &arg) { //
  // llvm.abs.iN takes an is_int_min_poison flag; false because abs(MIN_VALUE) == MIN_VALUE in both langs.
  const auto callee = llvm::Intrinsic::getOrInsertDeclaration(&M, llvm::Intrinsic::abs, resolveType(overload));
  return B.CreateCall(callee, {mkTermVal(arg), B.getFalse()});
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
  return stackVarPtrs              //
         ^ get_maybe(named.symbol) //
         ^ fold(
             [&](const auto &tpe, const auto &value) {
               if (named.tpe != tpe)
                 throw BackendException("Named local variable (" + to_string(named) + ") has different type from LUT (" + to_string(tpe)
                                        + ")");
               return value;
             },
             [&]() -> ValPtr {
               auto pool = stackVarPtrs ^ mk_string("\n", "\n", "\n", [](const auto &k, const auto &v) {
                             auto &[tpe, ir] = v;
                             return " -> `" + k + "` = " + to_string(tpe) + "(IR=" + llvm_tostring(ir) + ")";
                           });
               throw BackendException("Unseen variable: " + to_string(named) + ", variable table=\n->" + pool);
             });
}

ValPtr CodeGen::mkSelectPtr(const Term::Select &select) { return ptrModel->selectPtr(*this, select); }

void CodeGen::copyStruct(llvm::Value *dst, llvm::Value *src, const AnyType &tpe) { ptrModel->copyAggregate(*this, dst, src, tpe); }

void CodeGen::zeroStruct(llvm::Value *dst, const AnyType &tpe) { ptrModel->zeroAggregate(*this, dst, tpe); }

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
        auto tpe = resolveType(x.t, x.t.template is<Type::FnRef>());
        if (llvm::isa<llvm::PointerType>(tpe)) {
          return llvm::ConstantPointerNull::get(static_cast<llvm::PointerType *>(tpe));
        }
        return llvm::PoisonValue::get(tpe);
      },
      [&](const Term::StringConst &x) -> ValPtr {
        // XXX __constant, not __global: rusticl's program loader panics on a __global initialised string
        return B.CreateGlobalString(x.value, "strlit", C.addressSpace(TypeSpace::Constant()), &M);
      },
      [&](const Term::Select &x) -> ValPtr {
        if (x.tpe.is<Type::Unit0>()) return mkTermVal(Term::Unit0Const());
        if (auto v = ptrModel->termSelectVal(*this, x)) return *v;
        // a no-steps Arr arg/local holds a `ptr` slot to load, except on SPIR-V where the array is a direct alloca
        if (x.tpe.template is<Type::Arr>()) {
          if (x.steps.empty())
            return C.isSpirv() ? mkSelectPtr(x)
                               : C.load(B, mkSelectPtr(x), B.getPtrTy(C.addressSpace(x.tpe.template get<Type::Arr>()->space)));
          return mkSelectPtr(x);
        }
        if (structByPtr() && x.tpe.template is<Type::Struct>()) return mkSelectPtr(x);
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
            [&](const Intr::BNot &v) -> ValPtr { return unaryExpr(expr, v.x, v.tpe, [&](const auto &x) { return B.CreateNot(x); }); },
            [&](const Intr::LogicNot &v) -> ValPtr { return B.CreateNot(mkTermVal(v.x)); },
            [&](const Intr::Pos &v) -> ValPtr {
              return unaryNumOp(expr, v.x, v.tpe, [&](const auto &x) { return x; }, [&](const auto &x) { return x; });
            },
            [&](const Intr::Neg &v) -> ValPtr {
              return unaryNumOp(
                  expr, v.x, v.tpe, [&](const auto &x) { return B.CreateNeg(x); }, [&](const auto &x) { return B.CreateFNeg(x); });
            },
            [&](const Intr::Add &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) { return B.CreateAdd(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFAdd(l, r); });
            },
            [&](const Intr::Sub &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) { return B.CreateSub(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFSub(l, r); });
            },
            [&](const Intr::Mul &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) { return B.CreateMul(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFMul(l, r); });
            },
            [&](const Intr::Div &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.tpe) ? B.CreateUDiv(l, r) : B.CreateSDiv(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFDiv(l, r); });
            },
            [&](const Intr::Rem &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.tpe) ? B.CreateURem(l, r) : B.CreateSRem(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFRem(l, r); });
            },
            [&](const Intr::Min &v) -> ValPtr {
              // XXX minnum: aarch32 SelectionDAG can't legalize llvm.minimum.f{32,64}
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) {
                    return B.CreateSelect(isUnsigned(v.tpe) ? B.CreateICmpULT(l, r) : B.CreateICmpSLT(l, r), l, r);
                  },
                  [&](const auto &l, const auto &r) { return B.CreateMinNum(l, r); });
            },
            [&](const Intr::Max &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.tpe, //
                  [&](const auto &l, const auto &r) {
                    return B.CreateSelect(isUnsigned(v.tpe) ? B.CreateICmpULT(l, r) : B.CreateICmpSLT(l, r), r, l);
                  },
                  [&](const auto &l, const auto &r) { return B.CreateMaxNum(l, r); });
            }, //
            [&](const Intr::BAnd &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](const auto &l, const auto &r) { return B.CreateAnd(l, r); });
            },
            [&](const Intr::BOr &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](const auto &l, const auto &r) { return B.CreateOr(l, r); });
            },
            [&](const Intr::BXor &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](const auto &l, const auto &r) { return B.CreateXor(l, r); });
            },
            [&](const Intr::BSL &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](const auto &l, const auto &r) { return B.CreateShl(l, r); });
            },
            [&](const Intr::BSR &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe,
                                [&](const auto &l, const auto &r) { return isUnsigned(v.tpe) ? B.CreateLShr(l, r) : B.CreateAShr(l, r); });
            },
            [&](const Intr::BZSR &v) -> ValPtr {
              return binaryExpr(expr, v.x, v.y, v.tpe, [&](const auto &l, const auto &r) { return B.CreateLShr(l, r); });
            },                                                                                                     //
            [&](const Intr::LogicAnd &v) -> ValPtr { return B.CreateLogicalAnd(mkTermVal(v.x), mkTermVal(v.y)); }, //
            [&](const Intr::LogicOr &v) -> ValPtr { return B.CreateLogicalOr(mkTermVal(v.x), mkTermVal(v.y)); },   //
            [&](const Intr::LogicEq &v) -> ValPtr {
              if (v.x.tpe().is<Type::Ptr>())
                return binaryExpr(expr, v.x, v.y, v.x.tpe(), [&](const auto &l, const auto &r) { return B.CreateICmpEQ(l, r); });
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return B.CreateICmpEQ(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpOEQ(l, r); });
            },
            [&](const Intr::LogicNeq &v) -> ValPtr {
              if (v.x.tpe().is<Type::Ptr>())
                return binaryExpr(expr, v.x, v.y, v.x.tpe(), [&](const auto &l, const auto &r) { return B.CreateICmpNE(l, r); });
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return B.CreateICmpNE(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpONE(l, r); });
            },
            [&](const Intr::LogicLte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.x.tpe()) ? B.CreateICmpULE(l, r) : B.CreateICmpSLE(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpOLE(l, r); });
            },
            [&](const Intr::LogicGte &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.x.tpe()) ? B.CreateICmpUGE(l, r) : B.CreateICmpSGE(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpOGE(l, r); });
            },
            [&](const Intr::LogicLt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.x.tpe()) ? B.CreateICmpULT(l, r) : B.CreateICmpSLT(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpOLT(l, r); });
            },
            [&](const Intr::LogicGt &v) -> ValPtr {
              return binaryNumOp(
                  expr, v.x, v.y, v.x.tpe(), //
                  [&](const auto &l, const auto &r) { return isUnsigned(v.x.tpe()) ? B.CreateICmpUGT(l, r) : B.CreateICmpSGT(l, r); },
                  [&](const auto &l, const auto &r) { return B.CreateFCmpOGT(l, r); });
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

        if (x.from.tpe().is<Type::Struct>() && x.as.is<Type::Struct>()) {
          const auto &dl = M.getDataLayout();
          const auto fromSize = dl.getTypeAllocSize(fromTpe).getFixedValue();
          const auto toSize = dl.getTypeAllocSize(toTpe).getFixedValue();
          if (toSize > fromSize) {
            throw BackendException::semantic("cast from " + to_string(x.from.tpe()) + " (" + std::to_string(fromSize)
                                             + " bytes) to the larger " + to_string(x.as) + " (" + std::to_string(toSize)
                                             + " bytes) would read past the source allocation");
          }
          if (const auto sel = x.from.template get<Term::Select>()) {
            const auto ptr = mkSelectPtr(*sel);
            return structByPtr() ? ptr : C.load(B, ptr, toTpe);
          }
          if (structByPtr()) return from;
          const auto slot = C.allocaAS(B, fromTpe, C.AllocaAS, key + "_struct_cast");
          B.CreateStore(from, slot);
          return C.load(B, slot, toTpe);
        }

        // Reinterpret aggregates through their storage, materialising a slot when needed.
        if (x.from.tpe().kind().is<TypeKind::Ref>() && x.as.is<Type::Ptr>()) {
          if (const auto sel = x.from.template get<Term::Select>())
            return x.from.tpe().is<Type::Arr>() ? mkTermVal(x.from) : mkSelectPtr(*sel);
          const auto slot = C.allocaAS(B, fromTpe, C.AllocaAS, key + "_aggregate_ptr");
          B.CreateStore(from, slot);
          return slot;
        }

        // Casts to/from a None-kind type (Nothing/Unit0/Exec) are no-ops: void-shaped types carry no value.
        if (x.from.tpe().kind().is<TypeKind::None>() || x.as.kind().is<TypeKind::None>()) {
          return from;
        }

        // Both disallowed on Logical SPIR-V; permitted elsewhere.
        if (x.from.tpe().is<Type::Ptr>() && x.as.kind().is<TypeKind::Integral>()) {
          return B.CreatePtrToInt(from, toTpe);
        }
        // inttoptr: reinterpret an integer as a pointer (e.g. aligning a base `(void*)(size_t(p) & MASK)`)
        if (x.from.tpe().kind().is<TypeKind::Integral>() && x.as.is<Type::Ptr>()) {
          return B.CreateIntToPtr(from, toTpe);
        }

        auto fromKind = x.from.tpe().kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException::semantic("conversion from ref type (" + llvm_tostring(fromTpe) + ") is not allowed");
            },
            [&](const TypeKind::None &) -> NumKind { throw BackendException("none!?"); });

        auto toKind = x.as.kind().match_total( //
            [&](const TypeKind::Integral &) -> NumKind { return NumKind::Integral; },
            [&](const TypeKind::Fractional &) -> NumKind { return NumKind::Fractional; },
            [&](const TypeKind::Ref &) -> NumKind {
              throw BackendException::semantic("conversion to ref type (" + llvm_tostring(fromTpe) + ") is not allowed");
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
          auto isNan = targetHandler->isNaN(*this, from);
          return B.CreateSelect(isNan, zero, c);
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
        auto allArgs = x.args;
        if (x.receiver) allArgs ^= prepend(*x.receiver);
        // Mirror the declaration filter: drop Unit0/Nothing args; both lower to LLVM void at the boundary.
        const auto argNoUnit = allArgs ^ filter([](const auto &arg) {
                                 return !arg.tpe().template is<Type::Unit0>() //
                                        && !arg.tpe().template is<Type::Nothing>();
                               });
        // a kernel reference (`&trampoline_kernel`) is a FnRef poison, but the function-pointer formals it feeds are
        // harvested as Ptr(Nothing); both lower to the same stub pointer, so try that form of the signature too when
        // the exact lookup misses (some defs keep FnRef formals for direct lambda handles)
        const auto sig =
            Signature(calleeName(x), /*tpeVars*/ {}, /*receiver*/ {}, argNoUnit ^ map([](const auto &arg) { return arg.tpe(); }),
                      /*moduleCaptures*/ {}, /*termCaptures*/ {}, x.rtn);
        const auto sigPtr = Signature(calleeName(x), /*tpeVars*/ {}, /*receiver*/ {},
                                      argNoUnit ^ map([](const auto &arg) -> Type::Any {
                                        return arg.tpe().template is<Type::FnRef>()
                                                   ? Type::Any(Type::Ptr(Type::Nothing(), TypeSpace::Global()))
                                                   : arg.tpe();
                                      }),
                                      /*moduleCaptures*/ {}, /*termCaptures*/ {}, x.rtn);
        return functions                                                                           //
                   ^ get_maybe(sig)                                                                //
               | or_else([&]() -> Opt<llvm::Function *> { return functions ^ get_maybe(sigPtr); }) //
               | fold(
                   [&](const auto &fn) -> ValPtr {
                     const bool calleeUsesSret = fn->arg_size() > 0 && fn->getArg(0)->hasStructRetAttr();
                     const auto sretOffset = calleeUsesSret ? 1 : 0;
                     std::vector<ValPtr> params;
                     params.reserve(argNoUnit.size() + sretOffset);
                     if (calleeUsesSret) {
                       auto *sretSlotTy = resolveType(x.rtn, /*functionBoundary*/ false);
                       params.push_back(C.allocaAS(B, sretSlotTy, C.AllocaAS, "sret_slot"));
                     }
                     for (size_t i = 0; i < argNoUnit.size(); ++i) {
                       const auto &term = argNoUnit[i];
                       auto *formalTy = fn->getFunctionType()->getParamType(i + sretOffset);
                       // a FnRef is a stubbed kernel handle: its boundary lowering depends on the formal (FnRef
                       // formals take a scalar i8, Ptr(Nothing) formals take a pointer), so materialise the poison
                       // in the formal type instead of guessing in mkTermVal
                       if (term.tpe().template is<Type::FnRef>()) {
                         params.push_back(llvm::PoisonValue::get(formalTy));
                         continue;
                       }
                       if (term.tpe().template is<Type::Struct>()) {
                         if (auto sel = term.template get<Term::Select>()) {
                           params.push_back(mkSelectPtr(*sel));
                           continue;
                         }
                         // non-Select struct term: the function boundary lowers struct params
                         // to pointers, so box the value in an alloca and pass its address
                         if (formalTy->isPointerTy()) {
                           auto *slotTy = resolveType(term.tpe(), /*functionBoundary*/ false);
                           auto *slot = C.allocaAS(B, slotTy, C.AllocaAS, "struct_arg");
                           auto *val = mkTermVal(term);
                           const auto _ = C.store(B, val, slot);
                           params.push_back(slot);
                           continue;
                         }
                       }
                       auto val = mkTermVal(term);
                       params.push_back(term.tpe().template is<Type::Bool1>() ? B.CreateZExt(val, resolveType(Type::Bool1(), true)) : val);
                     }
                     // an actual whose AS differs from the formal's crosses spaces explicitly: SPIR-V widens
                     // Function/CrossWorkgroup to Generic, and on AMDGCN/NVPTX a workgroup address reaching a
                     // flat formal takes the aperture. reinterpreting instead would narrow the pointer silently
                     for (size_t i = 0; i < params.size(); ++i) {
                       auto *formal = fn->getFunctionType()->getParamType(i);
                       auto *actual = params[i]->getType();
                       if (formal != actual && formal->isPointerTy() && actual->isPointerTy())
                         params[i] = B.CreateAddrSpaceCast(params[i], formal);
                     }
                     if (params.size() != fn->arg_size())
                       throw BackendException(fmt::format("Invocation {} passes {} LLVM arguments to {} parameters", repr(sig),
                                                          params.size(), fn->arg_size()));
                     for (size_t i = 0; i < params.size(); ++i) {
                       auto *formal = fn->getFunctionType()->getParamType(i);
                       if (formal != params[i]->getType())
                         throw BackendException(fmt::format("Invocation {} argument {} lowers to {}, expected {}", repr(sig), i,
                                                            llvm_tostring(params[i]->getType()), llvm_tostring(formal)));
                     }
                     const auto call = B.CreateCall(fn, params);
                     if (calleeUsesSret) return params[0];
                     return x.rtn.is<Type::Unit0>() ? mkTermVal(Term::Unit0Const()) : call;
                   },
                   [&]() -> ValPtr {
                     throw BackendException(fmt::format("Unhandled invocation {}, known functions are:\n{}", repr(sig),
                                                        functions | keys() | mk_string("\n -> ", show_repr)));
                   });
      },
      [&](const Expr::ForeignCall &x) -> ValPtr {
        auto *fn = resolveExtFn(x.rtn, x.name, x.args ^ map([](const auto &a) { return a.tpe(); }));
        const auto call = B.CreateCall(fn, x.args ^ map([&](const auto &a) { return mkTermVal(a); }));
        return x.rtn.is<Type::Unit0>() ? mkTermVal(Term::Unit0Const()) : call;
      },
      [&](const Expr::OffsetOf &x) -> ValPtr {
        const auto s = x.structTpe.template get<Type::Struct>();
        if (!s) throw BackendException::semantic("OffsetOf on non-struct type " + to_string(x.structTpe));
        const auto info = structTypes                //
                          ^ get_maybe(repr(s->name)) //
                          ^ fold([&]() -> StructInfo { throw BackendException("Unseen struct in OffsetOf: " + repr(s->name)); });
        const auto idx = info.memberIndices   //
                         ^ get_maybe(x.field) //
                         ^ fold([&]() -> size_t { throw BackendException("Unknown field `" + x.field + "` in OffsetOf"); });
        return llvm::ConstantInt::get(C.i64Ty(), static_cast<uint64_t>(info.layout.members[idx].offsetInBytes));
      },
      [&](const Expr::SizeOf &x) -> ValPtr {
        // alloc size (includes trailing padding) so it doubles as the array element stride
        return llvm::ConstantInt::get(C.i64Ty(), M.getDataLayout().getTypeAllocSize(resolveType(x.forTpe)).getFixedValue());
      },
      [&](const Expr::Index &x) -> ValPtr { return ptrModel->indexVal(*this, x, key); },
      [&](const Expr::RefTo &x) -> ValPtr { return ptrModel->refToVal(*this, x, key); },
      [&](const Expr::Alloc &x) -> ValPtr { //
        const auto componentTpe = B.getPtrTy(0);
        const auto size = mkTermVal(x.size);
        const auto elemSize = C.sizeOf(B, componentTpe);
        const auto ptr = invokeMalloc(B.CreateMul(B.CreateIntCast(size, resolveType(Type::IntS64()), true), elemSize));
        return B.CreateBitCast(ptr, componentTpe);
      });
}

CodeGen::BlockKind CodeGen::mkStmt(const Stmt::Any &stmt, llvm::Function &fn, const Opt<WhileCtx> &whileCtx) {
  return stmt.match_total(
      [&](const Stmt::Var &x) -> BlockKind {
        // [T : ref] =>> t:T  = _        ; lut += &t
        // [T : ref] =>> t:T* = &(rhs:T) ; lut += t
        // [T : val] =>> t:T  =   rhs:T  ; lut += &t
        if (x.expr && x.expr->tpe() != x.name.tpe) {
          throw BackendException::semantic("name type " + to_string(x.name.tpe) + " and rhs expr type " + to_string(x.expr->tpe())
                                           + " mismatch (" + repr(x) + ")");
        }

        if (C.isVulkan() && x.expr) {
          std::optional<Term::StringConst> sc;
          if (const auto alias = x.expr->template get<Expr::Alias>()) sc = alias->ref.template get<Term::StringConst>();
          else if (const auto cast = x.expr->template get<Expr::Cast>()) sc = cast->from.template get<Term::StringConst>();
          if (sc)
            if (const auto pc = x.name.tpe.template get<Type::Ptr>())
              if (ptrModel->defineLocalString(*this, x.name.symbol, sc->value, pc->comp)) return BlockKind::Normal;
        }

        if (x.name.tpe.is<Type::Unit0>()) {
          // Unit0 declaration, discard declaration but keep RHS effect.
          if (x.expr) auto _ = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
        } else {
          const auto tpe = resolveType(x.name.tpe);
          auto allocTy = ptrModel->localAllocType(*this, x.name.tpe, tpe);
          const auto localArr =
              x.name.tpe.template get<Type::Arr>() ^ exists([](const auto &a) { return a.space.template is<TypeSpace::Local>(); });
          const auto dynShared = x.name.tpe.template get<Type::Arr>()
                                 ^ exists([](const auto &a) { return a.space.template is<TypeSpace::Local>() && a.length == 0; });
          llvm::Value *stackPtr;
          const auto logicalLocal =
              localArr ? ptrModel->allocateLocalArray(*this, x.name.symbol, x.name.tpe, allocTy) : std::optional<ValPtr>{};
          // physical/SPIR-V-kernel back a workgroup array with an addrspace(3) global; Vulkan logical SPIR-V excluded
          if (logicalLocal) {
            stackPtr = *logicalLocal;
          } else if (dynShared && C.isNVPTX()) {
            // NVPTX dynamic shared memory is a single module-level external `[0 x i8]` addrspace(3) global; the
            // launch-configured shared bytes back it, and align 16 keeps reinterpreted block-exchange scratch
            // naturally aligned. reuse-by-name so every extern __shared__ decl (and postProcessModule) refers to
            // the same storage
            if (auto *existing = M.getNamedGlobal(details::PolycDynSharedGlobal)) stackPtr = existing;
            else {
              auto *dynTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(C.actual), 0);
              auto *g = new llvm::GlobalVariable(M, dynTy, /*isConstant*/ false, llvm::GlobalValue::ExternalLinkage,
                                                 /*Initializer*/ nullptr, details::PolycDynSharedGlobal, nullptr,
                                                 llvm::GlobalValue::NotThreadLocal, C.LocalAS);
              g->setAlignment(llvm::Align(16));
              stackPtr = g;
            }
          } else if (dynShared && C.isSpirvKernel()) {
            // SPIR-V (OpenCL) forbids a runtime-sized [0 x T] workgroup variable, and there is no
            // launch-sized dynamic shared global as on NVPTX. A sycl::local_accessor inlines a fresh
            // length-0 Local array per operator[], so minting a per-var global scatters one block's scratch
            // across several 1-byte buffers and the reduction's threads never share storage. Back them all
            // with one fixed-size named workgroup global (reused by name), sized to cover a full work-group's
            // packed accessors; the remapper's per-accessor __off offsets then address this single region.
            if (auto *existing = M.getNamedGlobal(details::PolycDynSharedGlobal)) stackPtr = existing;
            else {
              if (sharedDynamicLocalBytes == 0)
                throw BackendException(
                    fmt::format("workgroup storage exceeds configured capacity of {} bytes", C.options.workgroupMemoryBytes));
              auto *dynTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(C.actual), sharedDynamicLocalBytes);
              auto *g = new llvm::GlobalVariable(M, dynTy, /*isConstant*/ false, llvm::GlobalValue::InternalLinkage,
                                                 llvm::Constant::getNullValue(dynTy), details::PolycDynSharedGlobal, nullptr,
                                                 llvm::GlobalValue::NotThreadLocal, C.LocalAS);
              g->setAlignment(llvm::Align(16));
              stackPtr = g;
            }
          } else if (localArr && (!C.isSpirv() || C.isSpirvKernel())) {
            // zero-init not undef/poison: a __shared__ slot one thread writes and another reads after a barrier is
            // a store the optimiser cannot connect cross-thread, so an un-analysed read falls back to the initialiser.
            // undef there makes a branch on the read UB - NVPTX O2 then proves the dependent store dead and drops it;
            // poison is worse (AMDGPU folds it, DCEing the LDS). the initialiser is dropped in codegen, so shared
            // memory stays runtime-uninitialised either way and this only tightens the optimiser's model.
            // AMDGPU is the exception: its asm printer rejects any non-undef initialiser on an LDS (addrspace 3)
            // global, so match what clang emits for HIP __shared__ and use undef there
            auto *init = C.isAMDGPU() ? llvm::UndefValue::get(allocTy) : llvm::Constant::getNullValue(allocTy);
            auto *wg = new llvm::GlobalVariable(M, allocTy, /*isConstant*/ false, llvm::GlobalValue::InternalLinkage, init,
                                                x.name.symbol + "_wg", nullptr, llvm::GlobalValue::NotThreadLocal, C.LocalAS);
            // a shared struct backing a byte buffer (an `alignas(16) char[N]`) reinterprets the storage to a wider
            // type; the polyAST drops the alignas, so the natural i8 alignment is 1 and the reinterpreted load
            // faults. floor the global at 16 (over-alignment is always sound)
            std::function<bool(llvm::Type *)> hasByteBuffer = [&](llvm::Type *t) -> bool {
              if (auto *at = llvm::dyn_cast<llvm::ArrayType>(t))
                return at->getElementType()->isIntegerTy(8) ? at->getNumElements() > 8 : hasByteBuffer(at->getElementType());
              if (auto *st = llvm::dyn_cast<llvm::StructType>(t))
                return llvm::any_of(st->elements(), [&](auto *e) { return hasByteBuffer(e); });
              return false;
            };
            if (hasByteBuffer(allocTy)) wg->setAlignment(llvm::Align(16));
            stackPtr = wg;
          } else {
            stackPtr = C.allocaAS(B, allocTy, C.AllocaAS, x.name.symbol + "_stack_ptr");
          }
          // inline Type::Arr needs a flat ptr slot (AMDGCN's 32-bit alloca AS overflows the 64-bit store); not on SPIR-V
          if (x.name.tpe.template is<Type::Arr>() && !C.isSpirv()) {
            llvm::Value *refSlot;
            if (localArr && C.AllocaAS != 0) {
              // LDS pointer slot needs a plain alloca, no addrspacecast: the cast blocks AMDGPU SROA from tracing the addrspace(3) global
              const auto slotTy = B.getPtrTy(C.LocalAS);
              auto *fn = B.GetInsertBlock()->getParent();
              auto &entry = fn->getEntryBlock();
              llvm::IRBuilder<> entryB(&entry, entry.getFirstNonPHIOrDbgOrAlloca());
              refSlot = entryB.CreateAlloca(slotTy, C.AllocaAS, nullptr, x.name.symbol + "_ref_ptr");
            } else refSlot = C.allocaAS(B, B.getPtrTy(localArr ? C.LocalAS : 0u), C.AllocaAS, x.name.symbol + "_ref_ptr");
            const auto _ = C.store(B, stackPtr, refSlot);
            stackPtr = refSlot;
          }
          // Rebind on same-name redeclaration (adjacent `for (int l = 0; ...)` loops);
          // `emplace` would keep the prior slot and the second loop would see the stale value.
          if (auto it = stackVarPtrs.find(x.name.symbol); it != stackVarPtrs.end() && it->second.first != x.name.tpe) {
            throw BackendException("Re-declaration of " + x.name.symbol + " changes type from " + to_string(it->second.first) + " to "
                                   + to_string(x.name.tpe));
          }
          stackVarPtrs.insert_or_assign(x.name.symbol, Pair<Type::Any, llvm::Value *>{x.name.tpe, stackPtr});
          if (dynShared) {
            // A length-zero Local array is a view of the kernel's one work-group arena, not a value-bearing
            // declaration.  Inlined local_accessor::operator[] creates several such aliases.  Initialising each
            // alias used to store poison at byte zero of polyc_dyn_shared; every work-item then raced with the
            // reduction's legitimate local[0] accesses.  Keep any RHS effects, but never write a zero-length view.
            if (x.expr) auto _ = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
          } else if (x.expr && x.expr->tpe().is<Type::FnRef>() && tpe->isPointerTy()) {
            storeStubbedHandle(*this, tpe, stackPtr);
          } else if (x.expr) {
            auto rhs = mkExprVal(*x.expr, x.name.symbol + "_var_rhs");
            if (structByPtr() && (x.name.tpe.template is<Type::Struct>() || x.name.tpe.template is<Type::Arr>())) {
              copyStruct(stackPtr, rhs, x.name.tpe);
            } else {
              if (tpe->isPointerTy() && rhs->getType()->isPointerTy() && rhs->getType() != tpe) rhs = B.CreateAddrSpaceCast(rhs, tpe);
              const auto _ = C.store(B, rhs, stackPtr); //
            }
          } else if (x.name.tpe.template is<Type::Struct>() && !localArr) {
            // zero an uninitialised struct declaration: a ctor that sets only some members otherwise leaves the
            // rest at stale stack bytes, and a later by-value copy or destruction derefs the garbage
            zeroStruct(stackPtr, x.name.tpe);
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
          throw BackendException::semantic("name type (" + to_string(x.expr.tpe()) + ") and rhs expr (" + to_string(lhs.tpe)
                                           + ") mismatch (" + repr(x) + ")");
        }
        if (lhs.tpe.is<Type::Unit0>()) return BlockKind::Normal;
        auto rhs = mkExprVal(x.expr, qualified(lhs) + "_mut");
        const auto dst = lhs.steps.empty() ? findStackVar(lhs.root) : mkSelectPtr(lhs);
        // by-value aggregate: rhs is a pointer to the source, so copy contents rather than store the pointer.
        // an Arr behind a select path is inline [N x T] storage, so it copies on every target; only a step-less
        // Arr lhs is a local's ref-ptr slot, which rebinds by pointer as Stmt::Var does
        const bool inlineArr = lhs.tpe.template is<Type::Arr>() && (structByPtr() || !lhs.steps.empty());
        if (inlineArr || (structByPtr() && lhs.tpe.template is<Type::Struct>())) {
          copyStruct(dst, rhs, lhs.tpe);
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
          throw BackendException::semantic("array update not called on array type (" + to_string(lhs.tpe) + ")(" + repr(x) + ")");
        }
        if (*compTpe != x.value.tpe()) {
          throw BackendException::semantic("array comp type (" + to_string(*compTpe) + ") and rhs term (" + to_string(x.value.tpe())
                                           + ") mismatch (" + repr(x) + ")");
        }
        // XXX Unit0 store: no-op. Host storage may be a JVM Object[]; a byte write clobbers the first ref.
        if (x.value.tpe().template is<Type::Unit0>()) return BlockKind::Normal;
        ptrModel->storeUpdate(*this, lhs, x.idx, x.value);
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
          condBr(B, mkTermVal(x.cond), loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          auto kind = BlockKind::Normal;
          for (auto &body : x.body) {
            kind = mkStmt(body, fn, {ctx});
            if (kind == BlockKind::Terminal) break;
          }
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
        static_cast<void>(mkStmt(Stmt::Var(x.induction, std::optional<Expr::Any>{}, /*isMutable*/ true), fn, whileCtx));
        static_cast<void>(mkStmt(Stmt::Mut(inductionSelect, Expr::Alias(x.lbIncl)), fn, whileCtx));
        WhileCtx ctx{.exit = loopExit, .test = loopTest};
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          condBr(B, mkExprVal(Expr::IntrOp(Intr::LogicLt(inductionTerm, x.ubExcl))), loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          auto kind = BlockKind::Normal;
          for (auto &body : x.body) {
            kind = mkStmt(body, fn, {ctx});
            if (kind == BlockKind::Terminal) break;
          }
          if (kind != BlockKind::Terminal) {
            [[maybe_unused]] auto _0 =
                mkStmt(Stmt::Mut(inductionSelect, Expr::IntrOp(Intr::Add(inductionTerm, x.step, x.induction.tpe))), fn, {ctx});
            B.CreateBr(loopTest);
          }
        }
        B.SetInsertPoint(loopExit);
        return BlockKind::Normal;
      },
      [&](const Stmt::Break &) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->exit);
        else throw BackendException("orphaned break!");
        return BlockKind::Terminal;
      }, //
      [&](const Stmt::Cont &) -> BlockKind {
        if (whileCtx) B.CreateBr(whileCtx->test);
        else throw BackendException("orphaned cont!");
        return BlockKind::Terminal;
      }, //
      [&](const Stmt::Cond &x) -> BlockKind {
        const auto condTrue = llvm::BasicBlock::Create(C.actual, "cond_true", &fn);
        const auto condFalse = llvm::BasicBlock::Create(C.actual, "cond_false", &fn);
        const auto condExit = llvm::BasicBlock::Create(C.actual, "cond_exit", &fn);
        condBr(B, mkTermVal(x.cond, "cond"), condTrue, condFalse);
        {
          B.SetInsertPoint(condTrue);
          auto kind = BlockKind::Normal;
          for (auto &body : x.trueBr) {
            kind = mkStmt(body, fn, whileCtx);
            if (kind == BlockKind::Terminal) break;
          }
          if (kind != BlockKind::Terminal) B.CreateBr(condExit);
        }
        {
          B.SetInsertPoint(condFalse);
          auto kind = BlockKind::Normal;
          for (auto &body : x.falseBr) {
            kind = mkStmt(body, fn, whileCtx);
            if (kind == BlockKind::Terminal) break;
          }
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
          static_cast<void>(mkExprVal(x.value, "return_unit"));
          B.CreateRetVoid();
        } else if (rtnTpe.is<Type::Nothing>()) {
          B.CreateUnreachable();
        } else if (currentSretParam && rtnTpe.is<Type::Struct>()) {
          const auto val = mkExprVal(x.value, "return_sret_val");
          if (C.isVulkan()) {
            copyStruct(currentSretParam, val, rtnTpe); // val is a struct pointer (structByPtr)
          } else {
            const auto structInfo = structTypes.at(repr(rtnTpe.get<Type::Struct>()->name));
            auto spill = C.allocaAS(B, structInfo.tpe, C.AllocaAS, "return_sret_spill");
            auto _ = C.store(B, val, spill);
            const auto size = structInfo.layout.sizeInBytes;
            const auto align = structInfo.layout.alignment;
            B.CreateMemCpy(currentSretParam, llvm::MaybeAlign(align), spill, llvm::MaybeAlign(align), size);
          }
          B.CreateRetVoid();
        } else {
          const auto expr = mkExprVal(x.value, "return");
          if (rtnTpe.is<Type::Bool1>()) {
            // Extend from i1 to i8
            B.CreateRet(B.CreateZExt(expr, llvm::Type::getInt8Ty(C.actual)));
          } else {
            B.CreateRet(expr);
          }
        }
        return BlockKind::Terminal;
      },
      [&](const Stmt::Annotated &x) -> BlockKind { return mkStmt(x.inner, fn, whileCtx); },
      [&](const Stmt::Try &) -> BlockKind { throw BackendException("Stmt::Try should be erased"); },
      [&](const Stmt::Raise &) -> BlockKind { throw BackendException("Stmt::Raise should be erased"); },
      [&](const Stmt::Rethrow &) -> BlockKind { throw BackendException("Stmt::Rethrow should be erased"); });
}

// SPIR-V: struct-by-value returns get coerced to a single i32 by the pre-legaliser. Convert
// to sret form (leading out-pointer, void return) so no struct crosses a function boundary.
static bool shouldUseSret(const CodeGen &cg, const Function &fn) { return fn.decl.rtn.is<Type::Struct>() && cg.C.isSpirv(); }

static auto createPrototype(CodeGen &cg, llvm::Module &mod, const Function &fn) {
  // CPU HostThreaded kernels receive `tid` as a leading arg from the runtime; GPU launches
  // provide it via intrinsics, so adding `__tid` there would off-by-one the kernel ABI.
  const auto cpuTarget = LLVMBackend::isCpuTarget(cg.C.options.target);
  auto allArgs = fn.decl.moduleCaptures | concat(fn.decl.termCaptures) | concat(fn.decl.args) | to_vector();
  if (fn.decl.receiver) allArgs ^= prepend(*fn.decl.receiver);
  if (fn.isEntry && cpuTarget) allArgs ^= prepend(Arg(Named("__tid", Type::IntS64()), {}));

  // Drop Unit0/Nothing args: both lower to void, which FunctionType::get's isValidArgumentType asserts.
  const auto argsNoUnit = allArgs ^ filter([](const auto &arg) {
                            return !arg.named.tpe.template is<Type::Unit0>() //
                                   && !arg.named.tpe.template is<Type::Nothing>();
                          });

  const bool useSret = shouldUseSret(cg, fn);

  // Structs are returned by-value (functionBoundary=false); other args travel as opaque pointers.
  const auto rtnTpe = (fn.decl.rtn.is<Type::Unit0>() || useSret) ? llvm::Type::getVoidTy(cg.C.actual)
                      : fn.decl.rtn.is<Type::Struct>()           ? cg.resolveType(fn.decl.rtn, false)
                                                                 : cg.resolveType(fn.decl.rtn, true);

  auto argTys = argsNoUnit ^ map([&](const auto &arg) { return cg.resolveType(arg.named.tpe, true, fn.isEntry); });

  for (std::size_t i = 0; i < argTys.size(); ++i) {
    if (argTys[i]->isEmptyTy()) {
      throw BackendException(fmt::format("Function {} argument {} ({}) lowers to an empty parameter type", repr(fn.decl.name),
                                         argsNoUnit[i].named.symbol, repr(argsNoUnit[i].named.tpe)));
    }
  }

  // Vulkan compute entry takes no kernel params; args become descriptor-bound resources in the body
  if (cg.C.isVulkan() && fn.isEntry) argTys.clear();

  if (useSret) argTys.insert(argTys.begin(), llvm::PointerType::get(cg.C.actual, cg.C.AllocaAS));
  llvm::Type *sretStructTy = useSret ? cg.resolveType(fn.decl.rtn, /*functionBoundary*/ false) : nullptr;

  // Internal PolyAST identities must not occupy process ABI names: a harvested wrapper named
  // `malloc` may itself contain a ForeignCall("malloc"), which would otherwise recurse into the
  // wrapper. Exported entry points keep their declared ABI identity.
  const bool exported = fn.visibility.is<FunctionVisibility::Exported>();
  const auto normalisedName = (exported ? std::string{} : "polyregion_internal_") + normaliseSymbol(fn.decl.name);

  Signature sig(fn.decl.name, /*tpeVars*/ {}, /*receiver*/ {}, argsNoUnit ^ map([](const auto &x) { return x.named.tpe; }),
                /*moduleCaptures*/ {}, /*termCaptures*/ {}, fn.decl.rtn);
  llvm::Function *llvmFn = llvm::Function::Create(llvm::FunctionType::get(/*Result*/ rtnTpe, /*Params*/ argTys, /*isVarArg*/ false), //
                                                  exported ? llvm::Function::ExternalLinkage : llvm::Function::InternalLinkage,
                                                  normalisedName, //
                                                  mod);

  if (fn.decl.affinity.is<FunctionAffinity::Host>()) llvmFn->addFnAttr(POLYREFLECT_RT_PROTECT_ANNOTATION);

  // Attach sret attributes here, before any other prototype's body emission can look this
  // function up via Expr::Invoke. Deferring until the body loop would let the first caller see
  // the function without the sret marker and trip an LLVM signature mismatch.
  if (useSret) {
    auto *sretArg = llvmFn->getArg(0);
    sretArg->setName("sret");
    sretArg->addAttr(llvm::Attribute::get(cg.C.actual, llvm::Attribute::StructRet, sretStructTy));
    sretArg->addAttr(llvm::Attribute::NoAlias);
  }

  cg.targetHandler->witnessFn(cg, *llvmFn, fn);

  cg.functions.emplace(sig, llvmFn);
  return std::tuple{llvmFn, fn, argsNoUnit};
}

// unions used as reused (not type-punned) storage in LDS/shared memory: rocPRIM's block-primitive temp_storage
// overlays its phase buffers through one union, which the AMDGPU O3 memory optimiser miscompiles. de-aliasing lays
// such a union out struct-like. scope by address space (Local): a union reached inline from a Local Arr/Ptr holds
// barrier-separated phase storage, never a cross-member reinterpret; genuine type-punning unions (std::optional,
// std::variant, SSO string) live in private/global and are left overlaid
static Set<std::string> localReuseUnionsRaw(const Program &program) {
  const auto byName = program.defs | map([](const auto &d) { return std::pair{repr(d.name), &d}; }) | to<Map>();

  std::vector<Type::Any> roots;
  const auto addLocalRoots = [&](const auto &node) {
    roots ^= concat(node.template collect_all<Type::Arr>() ^ collect([](const auto &a) -> Opt<Type::Any> {
                      return a.space.template is<TypeSpace::Local>() ? Opt<Type::Any>{a.comp} : std::nullopt;
                    }));
    roots ^= concat(node.template collect_all<Type::Ptr>() ^ collect([](const auto &p) -> Opt<Type::Any> {
                      return p.space.template is<TypeSpace::Local>() ? Opt<Type::Any>{p.comp} : std::nullopt;
                    }));
  };
  addLocalRoots(program.entry);
  program.functions ^ for_each(addLocalRoots);
  program.defs ^ for_each([&](const auto &d) { d.members ^ for_each([&](const auto &m) { addLocalRoots(m.tpe); }); });

  Set<std::string> unions, visited;
  std::function<void(const Type::Any &)> taint = [&](const Type::Any &t) {
    if (const auto s = t.template get<Type::Struct>()) {
      const auto name = repr(s->name);
      if (!visited.insert(name).second) return;
      const auto def = byName ^ get_maybe(name);
      if (!def) return;
      if ((*def)->isUnion) unions.insert(name);
      (*def)->members ^ for_each([&](const auto &m) { taint(m.tpe); }); // inline members share the enclosing object's space
    } else if (const auto a = t.template get<Type::Arr>()) taint(a->comp);
    // a Ptr member holds an address; its pointee is not inline in this object's LDS storage, so don't follow it
  };
  roots ^ for_each(taint);
  return unions;
}

// the AMDGPU O3 temp_storage-reuse miscompile; keyed on the kernel name since scan/reduce_by_key carry the same
// block_discontinuity reuse storage but must stay O3, so a structural trigger would over-clamp them
static bool isSelectReuseUnionName(const std::string &name) { return name ^ contains_slice("partition_kernel_impl"); }

// rocPRIM block_discontinuity storage is the tail-flag phase whose overlay the de-aliased layout separates; the
// layout engine takes this as a predicate so it carries no vendor knowledge
static bool isDiscontinuityName(const std::string &name) { return name ^ contains_slice("block_discontinuity"); }

// de-aliasing grows a union from its largest member to the sum of members; drop any union whose de-aliased size
// would blow the workgroup LDS budget, re-checking until stable. scoped to AMDGPU, the only miscompiling target
static Set<std::string> localReuseUnions(const Program &program, const LLVMBackend::Options &options, const Set<std::string> &raw) {
  if (options.target != LLVMBackend::Target::AMDGCN) return {};
  auto deAlias = raw;
  TargetedContext ctx(options);
  for (;;) {
    const auto layouts = ctx.resolveLayouts(program.defs, deAlias, isDiscontinuityName);
    const auto over =
        layouts ^ collect_to<Set>([&](const auto &name, const auto &info) -> Opt<std::string> {
          return info.deAliased && info.layout.sizeInBytes > options.workgroupMemoryBytes ? Opt<std::string>{name} : std::nullopt;
        });
    if (over.empty()) break;
    over ^ for_each([&](const auto &name) { deAlias.erase(name); });
  }
  return deAlias;
}

// true if the program's LDS storage includes the select/partition reuse union; such a program is clamped to O0
static bool hasSelectReuseUnion(const Set<std::string> &rawLocalUnions) {
  return rawLocalUnions ^ exists([](auto &n) { return isSelectReuseUnionName(n); });
}

std::string polyregion::backend::normaliseSymbol(const Sym &sym) {
  return repr(sym) ^ map([](const char c) { return !std::isalnum(c) && c != '_' ? '_' : c; });
}

Pair<Opt<std::string>, std::string> CodeGen::transform(const Program &program, const Set<std::string> &rawLocalUnions) {
  deAliasedUnions = localReuseUnions(program, C.options, rawLocalUnions);
  structTypes = C.resolveLayouts(program.defs, deAliasedUnions, isDiscontinuityName);

  auto allFns = program.functions;
  allFns ^= prepend(program.entry);

  sharedDynamicLocalBytes = 0;
  if (!LLVMBackend::isCpuTarget(C.options.target)) {
    std::vector<uint64_t> ownStaticBytes(allFns.size(), 0);
    Map<Signature, size_t> functionBySignature;
    bool hasAnyDynamicLocal = false;
    for (size_t i = 0; i < allFns.size(); ++i) {
      auto allArgs = allFns[i].decl.moduleCaptures | concat(allFns[i].decl.termCaptures) | concat(allFns[i].decl.args) | to_vector();
      if (allFns[i].decl.receiver) allArgs ^= prepend(*allFns[i].decl.receiver);
      const auto argsNoUnit = allArgs ^ filter([](const auto &arg) {
                                return !arg.named.tpe.template is<Type::Unit0>() && !arg.named.tpe.template is<Type::Nothing>();
                              });
      functionBySignature.emplace(Signature(allFns[i].decl.name, /*tpeVars*/ {}, /*receiver*/ {},
                                            argsNoUnit ^ map([](const auto &arg) { return arg.named.tpe; }), /*moduleCaptures*/ {},
                                            /*termCaptures*/ {}, allFns[i].decl.rtn),
                                  i);
      for (const auto &local : allFns[i].template collect_all<Stmt::Var>()) {
        const auto arr = local.name.tpe.template get<Type::Arr>();
        if (!arr || !arr->space.template is<TypeSpace::Local>()) continue;
        if (arr->length == 0) {
          hasAnyDynamicLocal = true;
          continue;
        }
        const auto bytes = M.getDataLayout().getTypeAllocSize(resolveType(local.name.tpe)).getFixedValue();
        if (bytes > C.options.workgroupMemoryBytes || ownStaticBytes[i] > C.options.workgroupMemoryBytes - bytes)
          throw BackendException(fmt::format("workgroup storage exceeds configured capacity of {} bytes", C.options.workgroupMemoryBytes));
        ownStaticBytes[i] += bytes;
      }
    }

    uint64_t maxReachableStaticBytes = 0;
    for (size_t root = 0; root < allFns.size(); ++root) {
      if (!allFns[root].isEntry) continue;
      std::vector<bool> reachable(allFns.size(), false);
      std::vector<size_t> pending{root};
      uint64_t total = 0;
      while (!pending.empty()) {
        const auto i = pending.back();
        pending.pop_back();
        if (reachable[i]) continue;
        reachable[i] = true;
        if (ownStaticBytes[i] > C.options.workgroupMemoryBytes - total)
          throw BackendException(fmt::format("workgroup storage exceeds configured capacity of {} bytes", C.options.workgroupMemoryBytes));
        total += ownStaticBytes[i];
        for (const auto &invoke : allFns[i].template collect_all<Expr::Invoke>()) {
          auto args = invoke.args;
          if (invoke.receiver) args ^= prepend(*invoke.receiver);
          const auto argsNoUnit = args ^ filter([](const auto &arg) {
                                    return !arg.tpe().template is<Type::Unit0>() && !arg.tpe().template is<Type::Nothing>();
                                  });
          const auto sig = Signature(calleeName(invoke), /*tpeVars*/ {}, /*receiver*/ {},
                                     argsNoUnit ^ map([](const auto &arg) { return arg.tpe(); }), /*moduleCaptures*/ {},
                                     /*termCaptures*/ {}, invoke.rtn);
          const auto sigPtr = Signature(calleeName(invoke), /*tpeVars*/ {}, /*receiver*/ {},
                                        argsNoUnit ^ map([](const auto &arg) -> Type::Any {
                                          return arg.tpe().template is<Type::FnRef>()
                                                     ? Type::Any(Type::Ptr(Type::Nothing(), TypeSpace::Global()))
                                                     : arg.tpe();
                                        }),
                                        /*moduleCaptures*/ {}, /*termCaptures*/ {}, invoke.rtn);
          const auto callee = functionBySignature ^ get_maybe(sig) ^ or_else([&] { return functionBySignature ^ get_maybe(sigPtr); });
          if (callee && !reachable[*callee]) pending.push_back(*callee);
        }
      }
      maxReachableStaticBytes = std::max(maxReachableStaticBytes, total);
    }
    if (hasAnyDynamicLocal) sharedDynamicLocalBytes = C.options.workgroupMemoryBytes - maxReachableStaticBytes;
  }
  const auto prototypes = allFns ^ map([&](const auto &fn) { return createPrototype(*this, M, fn); });

  prototypes ^ for_each([&](const auto &llvmFn, const auto &fn, const auto &argsNoUnit) {
    B.SetInsertPoint(llvm::BasicBlock::Create(C.actual, "entry", llvmFn));
    const bool useSret = shouldUseSret(*this, fn);
    currentSretParam = useSret ? llvmFn->getArg(0) : nullptr;
    const size_t argOffset = useSret ? 1 : 0;
    ptrModel->reset();
    // Vulkan entry: the model binds args as descriptor resources; helpers flow through the generic path below
    if (fn.isEntry && ptrModel->bindEntryArgs(*llvmFn, argsNoUnit, fn)) {
      stackVarPtrs.clear();
      currentSretParam = nullptr;
      return;
    }
    stackVarPtrs = argsNoUnit                                                                                //
                   | zip_with_index()                                                                        //
                   | map([&](const auto &arg, const auto &i) -> Pair<std::string, Pair<Type::Any, ValPtr>> { //
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
                       auto *slotTy = resolveType(arg.named.tpe, arg.named.tpe.template is<Type::FnRef>());
                       if (llvmArgValue->getType() != slotTy && llvmArgValue->getType()->isPointerTy() && slotTy->isPointerTy()) {
                         llvmArgValue = B.CreateAddrSpaceCast(llvmArgValue, slotTy);
                       }
                       auto stackPtr = C.allocaAS(B, slotTy, C.AllocaAS, arg.named.symbol + "_stack_ptr");
                       auto _ = C.store(B, llvmArgValue, stackPtr);
                       return {arg.named.symbol, {arg.named.tpe, stackPtr}};
                     }) //
                   | to<Map>();
    for (auto &stmt : fn.body)
      if (mkStmt(stmt, *llvmFn) == BlockKind::Terminal) break;
    // Abstract method bodies (e.g. typeclass methods like `Monoid.mempty`) emit no terminator.
    // Insert an `unreachable` so LLVM module verification is happy — the symbol should never
    // actually be invoked since DynamicDispatchPass routes calls through a vtable.
    if (auto *bb = B.GetInsertBlock(); bb && bb->getTerminator() == nullptr) {
      B.CreateUnreachable();
    }
    stackVarPtrs.clear();
    ptrModel->reset();
    currentSretParam = nullptr;
  });

  targetHandler->postProcessModule(*this);

  const auto moduleIr = [&] {
    std::string ir;
    llvm::raw_string_ostream irOut(ir);
    M.print(irOut, nullptr);
    return ir;
  };

  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (verifyModule(M, &errOut)) {
    auto ir = moduleIr();
    fmt::print(stderr, "Verification failed:\n{}\nIR=\n{}\n", errOut.str(), ir);
    return {errOut.str(), ir};
  }
  return {{}, llvmc::captureModuleIr() ? moduleIr() : std::string{}};
}

ValPtr CodeGen::unaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyType &rtn, const ValPtrFn1 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException::semantic("lhs type " + to_string(l.tpe()) + " of unary numeric operation in " + to_string(expr)
                                     + " doesn't match return type " + to_string(rtn));
  }
  return fn(mkTermVal(l));
}
ValPtr CodeGen::binaryExpr(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn,
                           const ValPtrFn2 &fn) { //
  if (l.tpe() != rtn) {
    throw BackendException::semantic("lhs type " + to_string(l.tpe()) + " of binary numeric operation in " + to_string(expr)
                                     + " doesn't match return type " + to_string(rtn));
  }
  if (r.tpe() != rtn) {
    throw BackendException::semantic("rhs type " + to_string(r.tpe()) + " of binary numeric operation in " + to_string(expr)
                                     + " doesn't match return type " + to_string(rtn));
  }
  return fn(mkTermVal(l), mkTermVal(r));
}
ValPtr CodeGen::unaryNumOp(const AnyExpr &expr, const AnyTerm &arg, const AnyType &rtn, //
                           const ValPtrFn1 &integralFn, const ValPtrFn1 &fractionalFn) {
  return unaryExpr(expr, arg, rtn, [&](const auto &lhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) return integralFn(lhs);
    if (rtn.kind().is<TypeKind::Fractional>()) return fractionalFn(lhs);
    // None-kind result (Nothing/Unit0/Exec) needs a sized poison; void poison is unrepresentable, so use i8.
    if (rtn.kind().is<TypeKind::None>()) return llvm::PoisonValue::get(llvm::Type::getInt8Ty(C.actual));
    throw BackendException("unimplemented");
  });
}
ValPtr CodeGen::binaryNumOp(const AnyExpr &expr, const AnyTerm &l, const AnyTerm &r, const AnyType &rtn, //
                            const ValPtrFn2 &integralFn, const ValPtrFn2 &fractionalFn) {
  return binaryExpr(expr, l, r, rtn, [&](const auto &lhs, const auto &rhs) -> ValPtr {
    if (rtn.kind().is<TypeKind::Integral>()) return integralFn(lhs, rhs);
    if (rtn.kind().is<TypeKind::Fractional>()) return fractionalFn(lhs, rhs);
    if (rtn.kind().is<TypeKind::None>()) return llvm::PoisonValue::get(llvm::Type::getInt8Ty(C.actual));
    throw BackendException("unimplemented");
  });
}
ValPtr CodeGen::mkSignumVal(const AnyExpr &expr, const AnyTerm &x, const AnyType &tpe) {
  return unaryNumOp(
      expr, x, tpe,
      [&](const auto &v) -> ValPtr {
        auto msb = v->getType()->getPrimitiveSizeInBits() - 1;
        return B.CreateOr(B.CreateAShr(v, msb), B.CreateLShr(B.CreateNeg(v), msb));
      },
      [&](const auto &v) -> ValPtr {
        auto isNaN = B.CreateFCmpUNO(v, v);
        auto isZero = B.CreateFCmpOEQ(v, llvm::ConstantFP::get(v->getType(), 0.0));
        return B.CreateSelect(B.CreateLogicalOr(isNaN, isZero), v, intr2(llvm::Intrinsic::copysign, tpe, dsl::fractional(tpe, 1.0L), x));
      });
}

static llvm::AtomicOrdering atomicOrdering(const MemOrder::Any &o) {
  using AO = llvm::AtomicOrdering;
  return o.match_total([](const MemOrder::Relaxed &) { return AO::Monotonic; }, [](const MemOrder::Acquire &) { return AO::Acquire; },
                       [](const MemOrder::Release &) { return AO::Release; }, [](const MemOrder::AcqRel &) { return AO::AcquireRelease; },
                       [](const MemOrder::SeqCst &) { return AO::SequentiallyConsistent; });
}

// ArenaView expresses a typed Vulkan buffer element address as a PolyAST RefTo binding. LLVM therefore
// sees `resource.getpointer(as11) -> addrspacecast(as0) -> alloca -> load` before the memory operation.
// InferAddressSpaces can normally fold that bridge back to StorageBuffer, but volatile/atomic operations
// intentionally block the inference. Recover the original resource pointer for those operations so the
// SPIR-V translator receives a legal StorageBuffer access instead of an unselectable address-space cast.
static llvm::Value *vulkanResourcePointer(llvm::Value *ptr) {
  auto resource = [](llvm::Value *candidate) -> llvm::Value * {
    if (auto *cast = llvm::dyn_cast<llvm::AddrSpaceCastInst>(candidate))
      if (cast->getSrcAddressSpace() == 11) return cast->getOperand(0);
    return nullptr;
  };
  if (auto *direct = resource(ptr)) return direct;
  auto *load = llvm::dyn_cast<llvm::LoadInst>(ptr);
  if (!load) return ptr;
  auto *slot = llvm::dyn_cast<llvm::AllocaInst>(load->getPointerOperand()->stripPointerCasts());
  if (!slot) return ptr;
  llvm::Value *stored = nullptr;
  for (auto *user : slot->users()) {
    auto *store = llvm::dyn_cast<llvm::StoreInst>(user);
    if (!store || store->getPointerOperand()->stripPointerCasts() != slot) continue;
    if (stored) return ptr;
    stored = store->getValueOperand();
  }
  return stored ? (resource(stored) ? resource(stored) : ptr) : ptr;
}

NumericKind polyregion::backend::details::classifyNumeric(const AnyType &tpe) {
  return {tpe.is<Type::Float16>() || tpe.is<Type::Float32>() || tpe.is<Type::Float64>(),
          tpe.is<Type::IntS8>() || tpe.is<Type::IntS16>() || tpe.is<Type::IntS32>() || tpe.is<Type::IntS64>()};
}

ValPtr CodeGen::mkAtomicRMW(const Spec::GpuAtomicRMW &op, const std::string &scope) {
  const auto nk = classifyNumeric(op.value.tpe());
  using Op = llvm::AtomicRMWInst::BinOp;
  const auto binop = op.op.match_total(                                                                   //
      [&](const AtomicOp::Xchg &) { return Op::Xchg; },                                                   //
      [&](const AtomicOp::Add &) { return nk.isFloat ? Op::FAdd : Op::Add; },                             //
      [&](const AtomicOp::Sub &) { return nk.isFloat ? Op::FSub : Op::Sub; },                             //
      [&](const AtomicOp::And &) { return Op::And; },                                                     //
      [&](const AtomicOp::Or &) { return Op::Or; },                                                       //
      [&](const AtomicOp::Xor &) { return Op::Xor; },                                                     //
      [&](const AtomicOp::Min &) { return nk.isFloat ? Op::FMin : (nk.isSigned ? Op::Min : Op::UMin); },  //
      [&](const AtomicOp::Max &) { return nk.isFloat ? Op::FMax : (nk.isSigned ? Op::Max : Op::UMax); }); //
  auto *ptr = mkTermVal(op.ptr);
  if (C.isVulkan()) ptr = vulkanResourcePointer(ptr);
  return B.CreateAtomicRMW(binop, ptr, mkTermVal(op.value), llvm::MaybeAlign(), atomicOrdering(op.order),
                           C.actual.getOrInsertSyncScopeID(scope));
}

// volatile load/store keep the access uncached and coherent across blocks and, crucially, stop LLVM
// hoisting/eliding it (a decoupled look-back spins on a peer tile's status)
ValPtr CodeGen::mkVolatileLoad(const Spec::GpuVolatileLoad &op) {
  auto *ty = resolveType(op.rtn);
  auto *ptr = mkTermVal(op.ptr);
  if (C.isVulkan()) ptr = vulkanResourcePointer(ptr);
  if (!ptr->getType()->isPointerTy())
    throw BackendException(fmt::format("volatile load pointer {} lowered to {}", to_string(op.ptr), llvm_tostring(ptr->getType())));
  const bool aggregateByPtr = structByPtr() && (op.rtn.is<Type::Struct>() || op.rtn.is<Type::Arr>());
  const auto sz = M.getDataLayout().getTypeStoreSize(ty).getFixedValue();
  // an aggregate volatile load lowers to per-field ld.volatile, which tears an 8-byte descriptor when a peer
  // block writes it concurrently (new status, stale value). access POD aggregates through the same-width
  // integer so NVPTX emits a single atomic transaction, then reinterpret via a stack slot
  if (!C.isVulkan() && ty->isAggregateType() && (sz == 2 || sz == 4 || sz == 8)) {
    auto *ld = B.CreateLoad(llvm::Type::getIntNTy(C.actual, sz * 8), ptr);
    ld->setVolatile(true);
    ld->setAlignment(M.getDataLayout().getABITypeAlign(ty));
    auto *slot = C.allocaAS(B, ty, C.AllocaAS, "vld");
    auto *unpack = B.CreateStore(ld, slot);
    unpack->setAlignment(M.getDataLayout().getABITypeAlign(ty));
    return aggregateByPtr ? slot : C.load(B, slot, ty);
  }
  auto *ld = B.CreateLoad(ty, ptr);
  ld->setVolatile(true);
  if (aggregateByPtr) {
    auto *slot = C.allocaAS(B, ty, C.AllocaAS, "vld");
    const auto _ = C.store(B, ld, slot);
    return slot;
  }
  return ld;
}
ValPtr CodeGen::mkVolatileStore(const Spec::GpuVolatileStore &op) {
  auto *val = mkTermVal(op.value);
  auto *ptr = mkTermVal(op.ptr);
  if (C.isVulkan()) ptr = vulkanResourcePointer(ptr);
  auto *ty = resolveType(op.value.tpe());
  const bool aggregateByPtr = structByPtr() && (op.value.tpe().is<Type::Struct>() || op.value.tpe().is<Type::Arr>());
  const auto sz = M.getDataLayout().getTypeStoreSize(ty).getFixedValue();
  if (!C.isVulkan() && ty->isAggregateType() && (sz == 2 || sz == 4 || sz == 8)) {
    auto *packedTy = llvm::Type::getIntNTy(C.actual, sz * 8);
    ValPtr packedPtr = val;
    if (!aggregateByPtr) {
      auto *slot = C.allocaAS(B, ty, C.AllocaAS, "vst");
      const auto _ = C.store(B, val, slot);
      packedPtr = slot;
    }
    if (!packedPtr->getType()->isPointerTy())
      throw BackendException(
          fmt::format("volatile store value {} lowered to {}", to_string(op.value), llvm_tostring(packedPtr->getType())));
    auto *packed = B.CreateLoad(packedTy, packedPtr);
    packed->setAlignment(M.getDataLayout().getABITypeAlign(ty));
    auto *st = B.CreateStore(packed, ptr);
    st->setVolatile(true);
    st->setAlignment(M.getDataLayout().getABITypeAlign(ty));
    return st;
  }
  if (aggregateByPtr) val = C.load(B, val, ty);
  auto *st = B.CreateStore(val, ptr);
  st->setVolatile(true);
  return st;
}

ValPtr CodeGen::shuffleStage(llvm::Type *valTy, llvm::Type *bufTy, uint64_t words, ValPtr srcVal, const std::string &tag,
                             const std::function<ValPtr(ValPtr)> &perWord) {
  auto *i32Ty = C.i32Ty();
  const bool byPtr = srcVal->getType()->isPointerTy();
  auto *stageTy = llvm::ArrayType::get(i32Ty, words);
  auto *srcPtr = C.allocaAS(B, stageTy, C.AllocaAS, tag + "_src");
  const auto stageAlign = llvm::Align(std::max<uint64_t>(4, M.getDataLayout().getABITypeAlign(valTy).value()));
  llvm::cast<llvm::AllocaInst>(srcPtr->stripPointerCasts())->setAlignment(stageAlign);
  const auto _ = C.store(B, llvm::Constant::getNullValue(stageTy), srcPtr);
  const auto valueBytes = M.getDataLayout().getTypeStoreSize(valTy);
  if (byPtr) B.CreateMemCpy(srcPtr, stageAlign, srcVal, M.getDataLayout().getABITypeAlign(valTy), valueBytes);
  else {
    const auto _ = C.store(B, srcVal, srcPtr);
  }
  auto *dstPtr = C.allocaAS(B, stageTy, C.AllocaAS, tag + "_dst");
  llvm::cast<llvm::AllocaInst>(dstPtr->stripPointerCasts())->setAlignment(stageAlign);
  for (uint64_t i = 0; i < words; ++i) {
    auto *idx = llvm::ConstantInt::get(i32Ty, i);
    auto *word = C.load(B, B.CreateGEP(i32Ty, srcPtr, {idx}), i32Ty);
    const auto _ = C.store(B, perWord(word), B.CreateGEP(i32Ty, dstPtr, {idx}));
  }
  return C.load(B, dstPtr, valTy);
}

LLVMBackend::LLVMBackend(const Options &options) : options(options) {}

std::vector<StructLayout> LLVMBackend::resolveLayouts(const std::vector<StructDef> &structs) {
  return TargetedContext(options).resolveLayouts(structs) | values() | map([&](const auto &i) { return i.layout; }) | to_vector();
}

CompileResult LLVMBackend::compileProgram(const Program &program, const compiletime::OptLevel &opt) {
  using namespace llvm;

  auto compileOptions = options;
  if (options.target == Target::SPIRV32_Kernel || options.target == Target::SPIRV64_Kernel) {
    compileOptions.spirvVersion13 =
        !program.template collect_all<Spec::GpuShuffleDown>().empty() || !program.template collect_all<Spec::GpuShuffleUp>().empty()
        || !program.template collect_all<Spec::GpuShuffleIdx>().empty() || !program.template collect_all<Spec::GpuShuffleXor>().empty();
  }

  // AMDGPU O3 still miscompiles rocPRIM's unique/select temp_storage reuse (partition_kernel_impl) regardless of
  // union layout; clamp just those programs to O0. scan/sort/reduce-by-key lack the union and keep the requested opt
  const auto rawLocalUnions = compileOptions.target == Target::AMDGCN ? localReuseUnionsRaw(program) : Set<std::string>{};
  auto effectiveOpt = opt;
  if (compileOptions.target == Target::AMDGCN && opt != compiletime::OptLevel::O0 && hasSelectReuseUnion(rawLocalUnions))
    effectiveOpt = compiletime::OptLevel::O0;

  CodeGen cg(compileOptions, "program");
  auto transformStart = compiler::nowMono();
  auto [maybeTransformErr, transformMsg] = cg.transform(program, rawLocalUnions);
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
            errors ^ mk_string("\n"),
            {}};
  }

  auto c = compileModule(compileOptions.targetInfo(), effectiveOpt, /*emitDisassembly*/ true, cg.M, compileOptions.emitBitcode);
  c.layouts = cg.structTypes | values() | map([&](auto &i) { return i.layout; }) | to_vector();
  c.events.emplace_back(ast2IR);
  c.events.emplace_back(astOpt);

  return c;
}
