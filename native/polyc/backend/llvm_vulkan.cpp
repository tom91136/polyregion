#include "llvm_vulkan.h"

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#include "aspartame/all.hpp"

#include "polyregion/types.h"

using namespace aspartame;
using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;
using namespace polyregion::backend::details;

static constexpr unsigned VkStorageBufferAS = 11;
// SPIR-V StorageClass::StorageBuffer, distinct from the address space
static constexpr unsigned VkStorageClassStorageBuffer = 12;

void VulkanLowering::reset() {
  bufferHandles.clear();
  scalarSlots.clear();
  scalarBlock = nullptr;
  localBases.clear();
}

llvm::Value *VulkanLowering::i64Zero() const { return llvm::ConstantInt::get(cg.C.i64Ty(), 0, true); }

llvm::Value *VulkanLowering::bufferHandle(llvm::Type *elemTy, unsigned binding, const std::string &name) {
  auto *arrTy = llvm::ArrayType::get(elemTy, 0);
  // VulkanBuffer type params are (StorageClass, isWritable)
  auto *handleTy = llvm::TargetExtType::get(cg.C.actual, "spirv.VulkanBuffer", {arrTy}, {VkStorageClassStorageBuffer, /*writable*/ 1u});
  auto *nameStr = cg.B.CreateGlobalString(name, "." + name);
  return cg.B.CreateIntrinsic(llvm::Intrinsic::spv_resource_handlefrombinding, {handleTy},
                              {/*set*/ cg.B.getInt32(0), /*binding*/ cg.B.getInt32(binding),
                               /*arraySize*/ cg.B.getInt32(1), /*index*/ cg.B.getInt32(0), nameStr});
}

llvm::Value *VulkanLowering::uniformBlockHandle(llvm::Type *blockTy, unsigned binding, const std::string &name) {
  auto *handleTy = llvm::TargetExtType::get(cg.C.actual, "spirv.VulkanBuffer", {blockTy}, {2u /*StorageClass::Uniform*/, /*read-only*/ 0u});
  auto *nameStr = cg.B.CreateGlobalString(name, "." + name);
  return cg.B.CreateIntrinsic(llvm::Intrinsic::spv_resource_handlefrombinding, {handleTy},
                              {/*set*/ cg.B.getInt32(0), /*binding*/ cg.B.getInt32(binding),
                               /*arraySize*/ cg.B.getInt32(1), /*index*/ cg.B.getInt32(0), nameStr});
}

llvm::Value *VulkanLowering::bufferElementPtr(const AnyType &ptrTpe, llvm::Value *handle, llvm::Value *index) {
  auto *elemPtrTy = cg.B.getPtrTy(VkStorageBufferAS);
  auto *idx32 = cg.B.CreateIntCast(index, cg.B.getInt32Ty(), /*isSigned*/ true);
  auto *ptr = cg.B.CreateIntrinsic(llvm::Intrinsic::spv_resource_getpointer, {elemPtrTy, handle->getType()}, {handle, idx32});
  // bridge the StorageBuffer pointer into the body's flat AS; InferAddressSpaces hoists it back
  return cg.B.CreateAddrSpaceCast(ptr, cg.B.getPtrTy(cg.C.GlobalAS));
}

llvm::Value *VulkanLowering::handleOf(const Term::Select &select) {
  if (!select.steps.empty()) return nullptr;
  return bufferHandles ^ get_maybe(select.root.symbol) ^
         fold([](auto &, auto &h) -> llvm::Value * { return h; }, []() -> llvm::Value * { return nullptr; });
}

llvm::Value *VulkanLowering::elementPtr(const Term::Select &lhs, llvm::Value *idx) {
  if (auto *h = handleOf(lhs)) return bufferElementPtr(lhs.tpe, h, idx);
  if (auto org = localOrigin(lhs)) {
    auto &[base, arrTy, idxs] = *org;
    return localChainPtr(base, arrTy, extendIndices(idxs, lhs.tpe, idx));
  }
  return nullptr;
}

llvm::Value *VulkanLowering::localElementPtr(const Term::Select &select, llvm::Value *index) {
  if (!select.steps.empty()) return nullptr;
  const auto base = localBases ^ get_maybe(select.root.symbol);
  if (!base) return nullptr;
  const auto &[elemTpe, arrTy, gv] = *base;
  return cg.B.CreateInBoundsGEP(arrTy, gv, {llvm::ConstantInt::get(cg.C.i32Ty(), 0), cg.i64SExt(index)}, qualified(select) + "_wg_ptr");
}

std::optional<std::tuple<llvm::Value *, llvm::Type *, std::vector<llvm::Value *>>> VulkanLowering::localOrigin(const Term::Select &select) {
  llvm::Value *base;
  llvm::Type *arrTy;
  if (auto b = localBases ^ get_maybe(select.root.symbol)) {
    base = std::get<2>(*b);
    arrTy = std::get<1>(*b);
  } else if (select.root.tpe.template is<Type::Arr>()) {
    base = cg.mkTermVal(Term::Select(select.root, {}, select.root.tpe));
    arrTy = cg.resolveType(select.root.tpe);
  } else return std::nullopt;
  if (!(select.steps ^ forall([](auto &step) { return step.template is<PathStep::IndexDyn>(); }))) return std::nullopt;
  const auto indices =
      select.steps ^ map([&](auto &step) { return cg.i64SExt(cg.mkTermVal(step.template get<PathStep::IndexDyn>()->idx)); });
  return std::tuple{base, arrTy, indices};
}

std::pair<llvm::ArrayType *, std::vector<uint64_t>> VulkanLowering::flattenArray(llvm::Type *arrTy) {
  std::vector<uint64_t> dims;
  llvm::Type *t = arrTy;
  while (auto *a = llvm::dyn_cast<llvm::ArrayType>(t)) {
    dims.push_back(a->getNumElements());
    t = a->getElementType();
  }
  std::vector<uint64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * dims[i + 1];
  const auto total = dims ^ fold_left(uint64_t{1}, [](auto acc, auto d) { return acc * d; });
  return {llvm::ArrayType::get(t, total), strides};
}

llvm::Value *VulkanLowering::localChainPtr(llvm::Value *base, llvm::Type *arrTy, const std::vector<llvm::Value *> &indices) {
  // multi-dim access chains are unlowerable, so flatten to [total x scalar] and address by one index
  const auto [flatTy, strides] = flattenArray(arrTy);
  llvm::Value *flatIdx = indices | zip(strides) | fold_left(i64Zero(), [&](llvm::Value *acc, auto &is) {
                           return cg.B.CreateAdd(acc, cg.B.CreateMul(is.first, llvm::ConstantInt::get(cg.C.i64Ty(), is.second)));
                         });
  return cg.B.CreateInBoundsGEP(flatTy, base, {llvm::ConstantInt::get(cg.C.i32Ty(), 0), flatIdx}, "wg_flat_ptr");
}

std::vector<llvm::Value *> VulkanLowering::extendIndices(const std::vector<llvm::Value *> &base, const AnyType &lhsTpe, llvm::Value *idx) {
  const bool descend = lhsTpe.template is<Type::Arr>() ||
                       (lhsTpe.template get<Type::Ptr>() ^ exists([](auto &p) { return p.comp.template is<Type::Arr>(); }));
  std::vector<llvm::Value *> out = base;
  if (descend || out.empty()) out.push_back(cg.i64SExt(idx));
  else out.back() = cg.B.CreateAdd(out.back(), cg.i64SExt(idx));
  return out;
}

llvm::Value *VulkanLowering::scalarValueOf(const Term::Select &select) {
  if (!select.steps.empty() || !scalarBlock) return nullptr;
  const auto slot = scalarSlots ^ get_maybe(select.root.symbol);
  if (!slot) return nullptr;
  const auto &[memberIdx, tpe] = *slot;
  auto *ty = cg.resolveType(tpe);
  auto *ptr = cg.B.CreateIntrinsic(llvm::Intrinsic::spv_resource_getpointer, {cg.B.getPtrTy(VkStorageBufferAS), scalarBlock->getType()},
                                   {scalarBlock, cg.B.getInt32(memberIdx)});
  if (tpe.is<Type::Bool1>())
    return cg.B.CreateICmpNE(cg.C.load(cg.B, ptr, ty), llvm::ConstantInt::get(llvm::Type::getInt1Ty(cg.C.actual), 0, true));
  return cg.C.load(cg.B, ptr, ty);
}

void VulkanLowering::structFieldCopy(llvm::Value *dst, llvm::Value *src, llvm::Type *rootTy, const AnyType &tpe,
                                     std::vector<llvm::Value *> idxs) {
  if (auto s = tpe.template get<Type::Struct>()) {
    const auto &info = cg.structTypes.at(repr(s->name));
    info.def.members | zip_with_index<size_t>() | for_each([&](auto &m, auto i) {
      structFieldCopy(dst, src, rootTy, m.tpe, idxs ^ append(llvm::ConstantInt::get(cg.C.i32Ty(), i)));
    });
  } else if (auto a = tpe.template get<Type::Arr>()) {
    for (int e = 0; e < a->length; ++e)
      structFieldCopy(dst, src, rootTy, a->comp, idxs ^ append(llvm::ConstantInt::get(cg.C.i32Ty(), e)));
  } else {
    auto *scalarTy = tpe.template is<Type::Bool1>() ? llvm::Type::getInt8Ty(cg.C.actual) : cg.resolveType(tpe);
    auto *dstP = cg.B.CreateInBoundsGEP(rootTy, dst, idxs, "vkcopy_dst");
    auto *srcP = cg.B.CreateInBoundsGEP(rootTy, src, idxs, "vkcopy_src");
    const auto _ = cg.C.store(cg.B, cg.C.load(cg.B, srcP, scalarTy), dstP);
  }
}

std::optional<ValPtr> VulkanLowering::termSelectVal(CodeGen &, const Term::Select &x) {
  if (auto *v = scalarValueOf(x)) return v;
  if (auto *p = localElementPtr(x, i64Zero())) return p;
  return std::nullopt;
}

std::optional<ValPtr> VulkanLowering::mkIndex(const Term::Select &lhs, const Term::Any &idx) {
  auto &B = cg.B;
  auto *ptr = elementPtr(lhs, cg.i64SExt(cg.mkTermVal(idx)));
  if (!ptr) return std::nullopt;
  Opt<AnyType> compTpe;
  if (auto p = lhs.tpe.template get<Type::Ptr>()) compTpe = p->comp;
  else if (auto a = lhs.tpe.template get<Type::Arr>()) compTpe = a->comp;
  if (!compTpe) return std::nullopt;
  const auto ty = cg.resolveType(*compTpe);
  if (compTpe->template is<Type::Bool1>())
    return ValPtr{B.CreateICmpNE(cg.C.load(B, ptr, ty), llvm::ConstantInt::get(llvm::Type::getInt1Ty(cg.C.actual), 0, true))};
  if (compTpe->template is<Type::Struct>()) return ValPtr{ptr};
  return ValPtr{cg.C.load(B, ptr, ty)};
}

std::optional<ValPtr> VulkanLowering::mkRefTo(const Term::Select &lhs, const Opt<Term::Any> &idx) {
  auto *ptr = elementPtr(lhs, idx ? cg.i64SExt(cg.mkTermVal(*idx)) : i64Zero());
  return ptr ? std::optional{ValPtr{ptr}} : std::nullopt;
}

bool VulkanLowering::mkUpdate(const Term::Select &lhs, const Term::Any &idx, const Term::Any &value) {
  auto &B = cg.B;
  auto *ptr = elementPtr(lhs, cg.i64SExt(cg.mkTermVal(idx)));
  if (!ptr) return false;
  if (value.tpe().template is<Type::Struct>()) {
    cg.copyStruct(ptr, cg.mkTermVal(value), value.tpe());
    return true;
  }
  const auto valTy = value.tpe().template is<Type::Bool1>() ? llvm::Type::getInt8Ty(cg.C.actual) : cg.resolveType(value.tpe());
  const auto _ =
      cg.C.store(B, value.tpe().template is<Type::Bool1>() ? B.CreateIntCast(cg.mkTermVal(value), valTy, true) : cg.mkTermVal(value), ptr);
  return true;
}

llvm::Type *VulkanLowering::localAllocType(CodeGen &, const Type::Any &nameTpe, llvm::Type *tpe) {
  if (nameTpe.template is<Type::Arr>()) return flattenArray(tpe).first;
  return tpe;
}

bool VulkanLowering::bindEntryArgs(llvm::Function &llvmFn, const std::vector<Arg> &argsNoUnit, const Function &fn) {
  auto &B = cg.B;
  cg.stackVarPtrs.clear();
  const auto localPtr = [](auto &arg) {
    return arg.named.tpe.template get<Type::Ptr>() ^ exists([](auto &p) { return p.space.template is<TypeSpace::Local>(); });
  };
  // local-AS pointers back workgroup-shared globals
  localBases ^= concat(argsNoUnit | filter(localPtr) | map([&](auto &arg) {
                         const auto p = *arg.named.tpe.template get<Type::Ptr>();
                         auto *arrTy = llvm::ArrayType::get(cg.resolveType(p.comp), cg.vkWorkgroupSizeX);
                         auto *gv = new llvm::GlobalVariable(cg.M, arrTy, /*isConstant*/ false, llvm::GlobalValue::InternalLinkage,
                                                             llvm::PoisonValue::get(arrTy), "wg_" + arg.named.symbol, nullptr,
                                                             llvm::GlobalValue::NotThreadLocal, AddrSpace::Workgroup);
                         return std::pair{arg.named.symbol, std::tuple{p.comp, arrTy, static_cast<llvm::Value *>(gv)}};
                       }) |
                       to_vector());
  // global pointers bind to sequential storage-buffer descriptors; accumulate once, derive both maps
  const auto bound = argsNoUnit                                                                                    //
                     | filter([&](auto &arg) { return arg.named.tpe.template is<Type::Ptr>() && !localPtr(arg); }) //
                     | zip_with_index<unsigned>() | map([&](auto &arg, auto binding) {
                         const auto p = *arg.named.tpe.template get<Type::Ptr>();
                         auto *handle = bufferHandle(cg.resolveType(p.comp), binding, arg.named.symbol);
                         auto *base = bufferElementPtr(arg.named.tpe, handle, i64Zero());
                         auto *slot = cg.C.allocaAS(B, base->getType(), cg.C.AllocaAS, arg.named.symbol + "_base");
                         auto _ = cg.C.store(B, base, slot);
                         return std::tuple{arg.named.symbol, arg.named.tpe, handle, slot};
                       }) //
                     | to_vector();
  bufferHandles ^=
      concat(bound ^ map([](auto &symbol, auto &tpe, auto &handle, auto &slot) { return std::pair{symbol, std::pair{tpe, handle}}; }));
  cg.stackVarPtrs ^=
      concat(bound ^ map([](auto &symbol, auto &tpe, auto &handle, auto &slot) { return std::pair{symbol, std::pair{tpe, slot}}; }));
  const auto scalars = argsNoUnit ^ collect([](auto &arg) -> std::optional<std::pair<std::string, Type::Any>> {
                         if (arg.named.tpe.template is<Type::Ptr>()) return std::nullopt;
                         return std::pair{arg.named.symbol, arg.named.tpe};
                       });
  if (!scalars.empty()) {
    // std140 padding must mirror the host packing exactly, else a strict driver reads a member as garbage
    const auto &dl = cg.M.getDataLayout();
    auto *i8Ty = llvm::Type::getInt8Ty(cg.C.actual);
    const auto scalarTys = scalars ^ map([&](auto &s) { return cg.resolveType(s.second); });
    const std::vector<size_t> sizes = scalarTys ^ map([&](auto *ty) { return static_cast<size_t>(dl.getTypeAllocSize(ty)); });
    const auto offsets = runtime::std140ScalarLayout(sizes).first;
    std::vector<llvm::Type *> blockMembers;
    uint64_t off = 0;
    for (size_t i = 0; i < scalars.size(); ++i) {
      if (offsets[i] > off) blockMembers.push_back(llvm::ArrayType::get(i8Ty, offsets[i] - off));
      scalarSlots.emplace(scalars[i].first, std::pair{static_cast<unsigned>(blockMembers.size()), scalars[i].second});
      blockMembers.push_back(scalarTys[i]);
      off = offsets[i] + sizes[i];
    }
    auto *blockTy = llvm::StructType::get(cg.C.actual, blockMembers);
    scalarBlock = uniformBlockHandle(blockTy, static_cast<unsigned>(bound.size()), "_scalars");
  }
  for (auto &stmt : fn.body)
    auto _ = cg.mkStmt(stmt, llvmFn);
  if (auto *bb = B.GetInsertBlock(); bb && bb->getTerminator() == nullptr) B.CreateUnreachable();
  return true;
}
