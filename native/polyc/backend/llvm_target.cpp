#include <unordered_set>

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"

#include "fmt/core.h"
#include "llvm/TargetParser/Host.h"

using namespace aspartame;
using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;
using namespace polyregion::backend::details;

TargetedContext::TargetedContext(const LLVMBackend::Options &options) : options(options) {
  switch (options.target) {
    case LLVMBackend::Target::x86_64:
    case LLVMBackend::Target::AArch64:
    case LLVMBackend::Target::ARM: break;
    // CPUs default to generic, so nothing to do here.
    // For GPUs, any pointer passed in as args should be annotated global AS.
    //             AMDGPU   |   NVVM
    //      Generic (code)  0  Generic (code)
    //       Global (Host)  1  Global
    //        Region (GDS)  2  Internal Use
    //         Local (LDS)  3  Shared
    // Constant (Internal)  4  Constant
    //             Private  5  Local
    case LLVMBackend::Target::NVPTX64:
      GlobalAS = 0; // When inspecting Clang's output, they don't explicitly annotate addrspace(1) for globals
      LocalAS = 3;
      AllocaAS = 0;
      break;
    case LLVMBackend::Target::AMDGCN:
      GlobalAS = 0;
      LocalAS = 3;
      AllocaAS = 5;
      break;
    case LLVMBackend::Target::SPIRV32:
    case LLVMBackend::Target::SPIRV64:
      GlobalAS = 1;
      LocalAS = 3;
      AllocaAS = 0;
      break;
  }
}

llvm::Type *TargetedContext::i32Ty() { return llvm::Type::getInt32Ty(actual); }

TargetedContext::AS TargetedContext::addressSpace(const TypeSpace::Any &s) const {
  return s.match_total(                                  //
      [&](const TypeSpace::Local &) { return LocalAS; }, //
      [&](const TypeSpace::Global &) { return GlobalAS; }, [&](const TypeSpace::Private &) { return GlobalAS; });
}

ValPtr TargetedContext::allocaAS(llvm::IRBuilder<> &B, llvm::Type *ty, const unsigned int AS, const std::string &key) const {
  // LLVM requires allocas to dominate every use. If the current insertion point is inside a loop
  // body or after a branch, an alloca emitted there fails verification when reads of the slot
  // happen on a different control-flow path. Place all allocas at the start of the function entry
  // block — the canonical pattern that mem2reg expects — and leave the builder's insertion point
  // untouched so subsequent stores/loads still land where the caller intended.
  const auto allocaTy = ty->isPointerTy() ? B.getPtrTy() : ty;
  llvm::Value *stackPtr;
  if (auto fn = B.GetInsertBlock() ? B.GetInsertBlock()->getParent() : nullptr) {
    auto &entry = fn->getEntryBlock();
    llvm::IRBuilder<> entryB(&entry, entry.getFirstNonPHIOrDbgOrAlloca());
    stackPtr = entryB.CreateAlloca(allocaTy, AS, nullptr, key);
    if (AS != 0) {
      // The address-space cast must also live in entry so its result dominates any use.
      stackPtr = entryB.CreateAddrSpaceCast(stackPtr, B.getPtrTy());
    }
    return stackPtr;
  }
  stackPtr = B.CreateAlloca(allocaTy, AS, nullptr, key);
  return AS != 0 ? B.CreateAddrSpaceCast(stackPtr, B.getPtrTy()) : stackPtr;
}

ValPtr TargetedContext::load(llvm::IRBuilder<> &B, const ValPtr rhs, llvm::Type *ty) const {
  if (ty->isPointerTy()) {
    const ValPtr spaceNormalisedRhs = rhs->getType()->getPointerAddressSpace() != GlobalAS ? B.CreateAddrSpaceCast(rhs, B.getPtrTy()) : rhs;
    return B.CreateLoad(B.getPtrTy(), spaceNormalisedRhs);
  }
  return B.CreateLoad(ty, rhs);
}

ValPtr TargetedContext::store(llvm::IRBuilder<> &B, const ValPtr rhsVal, const ValPtr lhsPtr) const {
  if (const llvm::Type *rhsTy = rhsVal->getType();
      rhsTy->isPointerTy() && rhsTy->getPointerAddressSpace() != lhsPtr->getType()->getPointerAddressSpace()) {
    return B.CreateStore(B.CreateAddrSpaceCast(rhsVal, lhsPtr->getType()), lhsPtr);
  }
  return B.CreateStore(rhsVal, lhsPtr);
}

ValPtr TargetedContext::sizeOf(llvm::IRBuilder<> &B, llvm::Type *ptrTpe) {
  // See http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
  // We want:
  //   %SizePtr = getelementptr %T, %T* null, i32 1
  //   %Size = ptrtoint %T* %SizePtr to i32
  const auto sizePtr = B.CreateGEP(                                         //
      ptrTpe,                                                               //
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(actual)), //
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(actual), 1),            //
      "sizePtr"                                                             //
  );
  return B.CreatePtrToInt(sizePtr, llvm::Type::getInt64Ty(actual));
}

llvm::Type *TargetedContext::resolveType(const AnyType &tpe, const Map<std::string, StructInfo> &structs, const bool functionBoundary) {
  return tpe.match_total(                                                                     //
      [&](const Type::Float16 &) -> llvm::Type * { return llvm::Type::getHalfTy(actual); },   //
      [&](const Type::Float32 &) -> llvm::Type * { return llvm::Type::getFloatTy(actual); },  //
      [&](const Type::Float64 &) -> llvm::Type * { return llvm::Type::getDoubleTy(actual); }, //

      [&](const Type::IntU8 &) -> llvm::Type * { return llvm::Type::getInt8Ty(actual); },   //
      [&](const Type::IntU16 &) -> llvm::Type * { return llvm::Type::getInt16Ty(actual); }, //
      [&](const Type::IntU32 &) -> llvm::Type * { return llvm::Type::getInt32Ty(actual); }, //
      [&](const Type::IntU64 &) -> llvm::Type * { return llvm::Type::getInt64Ty(actual); }, //

      [&](const Type::IntS8 &) -> llvm::Type * { return llvm::Type::getInt8Ty(actual); },   //
      [&](const Type::IntS16 &) -> llvm::Type * { return llvm::Type::getInt16Ty(actual); }, //
      [&](const Type::IntS32 &) -> llvm::Type * { return llvm::Type::getInt32Ty(actual); }, //
      [&](const Type::IntS64 &) -> llvm::Type * { return llvm::Type::getInt64Ty(actual); }, //

      [&](const Type::Nothing &) -> llvm::Type * { return llvm::Type::getVoidTy(actual); },                         //
      [&](const Type::Unit0 &) -> llvm::Type * { return llvm::Type::getVoidTy(actual); },                           //
      [&](const Type::Bool1 &) -> llvm::Type * { return llvm::Type::getIntNTy(actual, functionBoundary ? 8 : 1); }, //

      [&](const Type::Struct &x) -> llvm::Type * {
        // At a function boundary, structs travel as opaque pointers (the kernel runtime passes a
        // ptr to the encoded struct). Inside the function body, they're materialised on the stack.
        if (functionBoundary) return llvm::PointerType::get(actual, addressSpace(TypeSpace::Global()));
        return structs ^ get_maybe(repr(x.name)) ^
               fold([](auto &info) { return info.tpe; },
                    [&]() -> llvm::StructType * {
                      throw BackendException(fmt::format("Unseen struct def {}, currently in-scope structs: {}", repr(x),
                                                         structs | values() | mk_string("\n", "\n", "\n", [](auto &ty) {
                                                           return fmt::format(" -> {}", to_string(ty.def));
                                                         })));
                    });
      }, //
      [&](const Type::Ptr &x) -> llvm::Type * {
        if (x.length) return llvm::ArrayType::get(resolveType(x.comp, structs, functionBoundary), *x.length);
        return llvm::PointerType::get(actual, addressSpace(x.space));
      }, //
      [&](const Type::Var &x) -> llvm::Type * { throw BackendException("Type::Var should be erased before LLVM lowering"); },
      [&](const Type::Exec &x) -> llvm::Type * { throw BackendException("Type::Exec should be erased before LLVM lowering"); },
      [&](const Type::Annotated &x) -> llvm::Type * { return resolveType(x.tpe, structs, functionBoundary); } //
  );
}

StructInfo TargetedContext::resolveStruct(const StructDef &def, const Map<std::string, StructInfo> &structs) {
  const auto types = def.members ^ map([&](auto &m) { return resolveType(m.tpe, structs); });
  const auto table = (def.members | map([](auto &m) { return m.symbol; }) | zip_with_index<size_t>() | to_vector()) ^ to<Map>();
  const auto tpe = llvm::StructType::create(actual, types, repr(def.name));
  const auto dataLayout = options.targetInfo().resolveDataLayout();
  const auto structLayout = dataLayout.getStructLayout(tpe);
  const StructLayout layout(/*name*/ repr(def.name),
                            /*sizeInBytes*/ structLayout->getSizeInBytes(),
                            /*alignment*/ static_cast<int64_t>(structLayout->getAlignment().value()),
                            /*members*/ def.members | zip_with_index<size_t>() | map([&](auto &named, auto i) {
                              return StructLayoutMember(
                                  /*name*/ named,                                                            //
                                  /*offsetInBytes*/ static_cast<int64_t>(structLayout->getElementOffset(i)), //
                                  /*sizeInBytes*/ dataLayout.getTypeAllocSize(tpe->getElementType(i))        //
                              );
                            }) | to_vector());
  return {.def = def, .layout = layout, .tpe = tpe, .memberIndices = table};
}

Map<std::string, StructInfo> TargetedContext::resolveLayouts(const std::vector<StructDef> &structs) {
  // Two-phase resolution to handle recursive defs (e.g., Node → Option[Node] → Node).
  // Phase 1: create opaque struct types for every def so subsequent type resolution can refer
  // to them by name without requiring the full body.
  Map<std::string, llvm::StructType *> opaqueTypes;
  for (auto &def : structs) {
    opaqueTypes.emplace(repr(def.name), llvm::StructType::create(actual, repr(def.name)));
  }
  // Also create opaque types for any structs referenced (transitively through members) but not
  // present in the defs list. The Scala frontend currently doesn't propagate field-type deps,
  // so e.g. `Node.next: Option[Node]` lands here without an Option struct def. The kernel can't
  // dereference these opaque structs, but type-only references (capturing Node by-value while
  // only reading `node.elem`) work fine.
  std::function<void(const Type::Any &)> walk = [&](const Type::Any &t) {
    if (auto s = t.template get<Type::Struct>()) {
      auto name = repr(s->name);
      if (!opaqueTypes.contains(name)) {
        opaqueTypes.emplace(name, llvm::StructType::create(actual, name));
      }
      for (auto &arg : s->args)
        walk(arg);
    } else if (auto p = t.template get<Type::Ptr>()) {
      walk(p->comp);
    } else if (auto a = t.template get<Type::Annotated>()) {
      walk(a->tpe);
    } else if (auto e = t.template get<Type::Exec>()) {
      for (auto &arg : e->args)
        walk(arg);
      walk(e->rtn);
    }
  };
  for (auto &def : structs) {
    for (auto &m : def.members)
      walk(m.tpe);
  }
  // Synthesise StructDefs for any types we created opaque shells for but weren't in the input.
  // We give them a single i8 placeholder member so LLVM treats them as sized (size 1) — empty
  // structs aren't well-supported by `DataLayout::getAlignment`.
  std::vector<StructDef> allDefs(structs.begin(), structs.end());
  Set<std::string> originalNames;
  for (auto &d : structs)
    originalNames.emplace(repr(d.name));
  for (auto &[name, _] : opaqueTypes) {
    if (!originalNames.contains(name)) {
      // Fabricate a Sym from the dotted name. The kernel can only treat this as opaque (no
      // member access) — adequate for type-only references that the Scala frontend doesn't
      // bring along.
      allDefs.push_back(StructDef(Sym({name}), {}, {Named("__opaque", Type::IntS8())}, {}));
    }
  }

  // Phase 2a: register stubs so resolveType can refer to any struct by name during body resolution.
  Map<std::string, StructInfo> resolved;
  for (auto &def : allDefs) {
    const auto stub = StructInfo{
        .def = def, .layout = StructLayout(repr(def.name), 0, 0, {}), .tpe = opaqueTypes.at(repr(def.name)), .memberIndices = {}};
    resolved.emplace(repr(def.name), stub);
  }
  // Phase 2b: setBody on every struct so all types are sized (recursive defs can ask each other for size).
  // Struct-typed fields are laid out inline (functionBoundary=false) so the layout matches what a
  // host C++ compiler produces for a captured-by-value struct — the kernel receives a pointer to the
  // outer struct and reads inline fields directly. The previous design used functionBoundary=true
  // (8-byte pointer slots) to defend against empty-struct fields computing to 0 bytes; we keep that
  // defence narrowly by giving empty structs a single placeholder byte.
  for (auto &def : allDefs) {
    auto tpe = opaqueTypes.at(repr(def.name));
    if (!tpe->isOpaque()) continue; // body already set; safe in case of duplicate StructDefs
    auto memberTypes = def.members ^ map([&](auto &m) { return resolveType(m.tpe, resolved, /*functionBoundary*/ false); });
    tpe->setBody(memberTypes);
  }
  // Phase 2c: now that all bodies are set, compute layouts.
  for (auto &def : allDefs) {
    auto tpe = opaqueTypes.at(repr(def.name));
    const auto table = (def.members | map([](auto &m) { return m.symbol; }) | zip_with_index<size_t>() | to_vector()) ^ to<Map>();
    const auto dataLayout = options.targetInfo().resolveDataLayout();
    const auto structLayout = dataLayout.getStructLayout(tpe);
    const StructLayout layout(/*name*/ repr(def.name),
                              /*sizeInBytes*/ structLayout->getSizeInBytes(),
                              /*alignment*/ static_cast<int64_t>(structLayout->getAlignment().value()),
                              /*members*/ def.members | zip_with_index<size_t>() | map([&](auto &named, auto i) {
                                return StructLayoutMember(
                                    /*name*/ named,                                                            //
                                    /*offsetInBytes*/ static_cast<int64_t>(structLayout->getElementOffset(i)), //
                                    /*sizeInBytes*/ dataLayout.getTypeAllocSize(tpe->getElementType(i))        //
                                );
                              }) | to_vector());
    resolved.insert_or_assign(repr(def.name), StructInfo{.def = def, .layout = layout, .tpe = tpe, .memberIndices = table});
  }
  return resolved;
}

llvmc::TargetInfo LLVMBackend::Options::targetInfo() const {
  using llvm::Triple;
  const auto bindGpuArch = [&](const Triple::ArchType &archTpe, const Triple::VendorType &vendor, const Triple::OSType &os) {
    const Triple triple(Triple::getArchTypeName(archTpe), Triple::getVendorTypeName(vendor), Triple::getOSTypeName(os));

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
            .target = llvmc::targetFromTriple(triple),
            .cpu = {.uArch = arch, .features = {}},
        };
    }
  };

  const auto bindCpuArch = [&](const Triple::ArchType &archTpe) {
    const Triple defaultTriple = llvmc::defaultHostTriple();
    if (arch.empty() && defaultTriple.getArch() != archTpe) // when detecting host arch, the host triple's arch must match
      throw BackendException("Requested arch detection with " + Triple::getArchTypeName(archTpe).str() +
                             " but the host arch is different (" + Triple::getArchTypeName(defaultTriple.getArch()).str() + ")");

    Triple triple = defaultTriple;
    triple.setArch(archTpe);
    return llvmc::TargetInfo{
        .triple = triple,
        .layout = {},
        .target = llvmc::targetFromTriple(triple),
        .cpu = arch.empty() || arch == "native" ? llvmc::hostCpuInfo() : llvmc::CpuInfo{.uArch = arch, .features = {}},
    };
  };

  switch (target) {
    case Target::x86_64: return bindCpuArch(Triple::ArchType::x86_64);
    case Target::AArch64: return bindCpuArch(Triple::ArchType::aarch64);
    case Target::ARM: return bindCpuArch(Triple::ArchType::arm);
    case Target::NVPTX64: return bindGpuArch(Triple::ArchType::nvptx64, Triple::VendorType::NVIDIA, Triple::OSType::CUDA);
    case Target::AMDGCN: return bindGpuArch(Triple::ArchType::amdgcn, Triple::VendorType::AMD, Triple::OSType::AMDHSA);
    case Target::SPIRV32: return bindGpuArch(Triple::ArchType::spirv32, Triple::VendorType::UnknownVendor, Triple::OSType::UnknownOS);
    case Target::SPIRV64: return bindGpuArch(Triple::ArchType::spirv64, Triple::VendorType::UnknownVendor, Triple::OSType::UnknownOS);
    default: throw BackendException(fmt::format("Unexpected target {}", magic_enum::enum_name(target)));
  }
}
