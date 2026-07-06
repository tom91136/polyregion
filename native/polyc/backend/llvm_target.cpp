#include "aspartame/all.hpp"
#include "fmt/core.h"
#include "magic_enum/magic_enum.hpp"

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"

using namespace aspartame;
using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;
using namespace polyregion::backend::details;

TargetedContext::TargetedContext(const LLVMBackend::Options &options) : options(options) {
  switch (options.target) {
    case LLVMBackend::Target::x86_64:
    case LLVMBackend::Target::AArch64:
    case LLVMBackend::Target::ARM:
    case LLVMBackend::Target::RISCV64:
    case LLVMBackend::Target::PPC64LE: break;
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
      GlobalAS = AddrSpace::Default; // When inspecting Clang's output, they don't explicitly annotate addrspace(1) for globals
      LocalAS = AddrSpace::Workgroup;
      AllocaAS = AddrSpace::Default;
      ConstantAS = AddrSpace::CrossWorkgroup; // module-level globals must be in a named AS; AS1=.global is used for constant data
      break;
    case LLVMBackend::Target::AMDGCN:
      GlobalAS = AddrSpace::Default;
      LocalAS = AddrSpace::Workgroup;
      AllocaAS = AddrSpace::Private;
      ConstantAS = AddrSpace::CrossWorkgroup;
      break;
    case LLVMBackend::Target::SPIRV32_Kernel:
    case LLVMBackend::Target::SPIRV64_Kernel:
      GlobalAS = AddrSpace::CrossWorkgroup;
      LocalAS = AddrSpace::Workgroup;
      AllocaAS = AddrSpace::Default; // Function/private
      GenericAS = AddrSpace::Generic;
      break;
    case LLVMBackend::Target::SPIRV_GLCompute:
      // logical SPIR-V can't cast Function <-> StorageBuffer; emit the body in the flat AS and let
      // InferAddressSpaces promote buffers back from a StorageBuffer->flat cast
      GlobalAS = AddrSpace::Default;
      LocalAS = AddrSpace::Workgroup;
      AllocaAS = AddrSpace::Default;
      GenericAS = 0;
      break;
  }
  spirvKernel = options.target == LLVMBackend::Target::SPIRV32_Kernel || //
                options.target == LLVMBackend::Target::SPIRV64_Kernel;
}

llvm::Type *TargetedContext::i32Ty() { return llvm::Type::getInt32Ty(actual); }
llvm::Type *TargetedContext::i64Ty() { return llvm::Type::getInt64Ty(actual); }

TargetedContext::AS TargetedContext::addressSpace(const TypeSpace::Any &s) const {
  // SPIR-V Kernel keeps private in the Function AS: widening to Generic makes IGC's SIMD vectoriser
  // read private arrays as shared, not per-lane
  const auto privateAS = spirvKernel ? AllocaAS : (GenericAS != 0 ? GenericAS : GlobalAS);
  return s.match_total(                                  //
      [&](const TypeSpace::Local &) { return LocalAS; }, //
      [&](const TypeSpace::Global &) { return GlobalAS; }, [&](const TypeSpace::Constant &) { return ConstantAS; },
      [&](const TypeSpace::Private &) { return privateAS; });
}

TargetedContext::AS TargetedContext::addressSpaceForKernelArg(const TypeSpace::Any &s) const {
  return s.match_total(                                  //
      [&](const TypeSpace::Local &) { return LocalAS; }, //
      [&](const TypeSpace::Global &) { return GlobalAS; }, [&](const TypeSpace::Constant &) { return GlobalAS; },
      [&](const TypeSpace::Private &) { return GlobalAS; });
}

llvm::PointerType *TargetedContext::loadedPtrTy(llvm::IRBuilder<> &B, const TypeSpace::Any &space) const {
  // a loaded pointer keeps its own AS so private (Function) pointers don't round-trip through Generic
  return B.getPtrTy(addressSpace(space));
}

ValPtr TargetedContext::allocaAS(llvm::IRBuilder<> &B, llvm::Type *ty, const unsigned int AS, const std::string &key) const {
  // SPIRV slots must use the value's AS (stores would otherwise cross disjoint spaces); other
  // targets are happy with default-AS slots. Non-default-AS pointers (NVPTX `addrspace(3)` for
  // shared memory) need their AS preserved -- coercing to default AS would have NVPTX emit
  // `st.b64` instead of `st.shared.b64` on the load.
  const auto preserveAS = ty->isPointerTy() && llvm::cast<llvm::PointerType>(ty)->getAddressSpace() != 0;
  const auto allocaTy = (ty->isPointerTy() && GenericAS == 0 && !preserveAS) ? B.getPtrTy() : ty;
  llvm::Value *stackPtr;
  if (auto fn = B.GetInsertBlock() ? B.GetInsertBlock()->getParent() : nullptr) {
    auto &entry = fn->getEntryBlock();
    llvm::IRBuilder<> entryB(&entry, entry.getFirstNonPHIOrDbgOrAlloca());
    stackPtr = entryB.CreateAlloca(allocaTy, AS, nullptr, key);
    if (AS != 0) stackPtr = entryB.CreateAddrSpaceCast(stackPtr, B.getPtrTy());
    return stackPtr;
  }
  stackPtr = B.CreateAlloca(allocaTy, AS, nullptr, key);
  return AS != 0 ? B.CreateAddrSpaceCast(stackPtr, B.getPtrTy()) : stackPtr;
}

ValPtr TargetedContext::load(llvm::IRBuilder<> &B, const ValPtr rhs, llvm::Type *ty) const { return B.CreateLoad(ty, rhs); }

ValPtr TargetedContext::store(llvm::IRBuilder<> &B, const ValPtr rhsVal, const ValPtr lhsPtr) const {
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

llvm::Type *TargetedContext::resolveType(const AnyType &tpe, const Map<std::string, StructInfo> &structs, const bool functionBoundary,
                                         const bool kernelEntryArg) {
  const auto ptrAS = [&](const TypeSpace::Any &space) {
    return (kernelEntryArg && functionBoundary) ? addressSpaceForKernelArg(space) : addressSpace(space);
  };
  // void is fine as a return or discarded arg, but DataLayout::getAlignment asserts on a void
  // struct member. Use [0 x i8] (size 0, align 1) inside a struct so layout queries succeed;
  // SPIR-V Kernel lowers that to OpTypeRuntimeArray and rejects it outside shader modules, so
  // fall back to plain i8 there.
  auto voidLike = [&]() -> llvm::Type * {
    if (functionBoundary) return llvm::Type::getVoidTy(actual);
    if (spirvKernel) return llvm::Type::getInt8Ty(actual);
    return llvm::ArrayType::get(llvm::Type::getInt8Ty(actual), 0);
  };
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

      [&](const Type::Nothing &) -> llvm::Type * { return voidLike(); }, [&](const Type::Unit0 &) -> llvm::Type * { return voidLike(); },
      [&](const Type::Bool1 &) -> llvm::Type * { return llvm::Type::getIntNTy(actual, functionBoundary ? 8 : 1); }, //

      [&](const Type::Struct &x) -> llvm::Type * {
        // At a function boundary, structs travel as opaque pointers (the kernel runtime passes a
        // ptr to the encoded struct). Inside the function body, they're materialised on the stack.
        if (functionBoundary) return llvm::PointerType::get(actual, ptrAS(TypeSpace::Global()));
        return structs ^ get_maybe(repr(x.name)) ^
               fold([](auto &info) { return info.tpe; },
                    [&]() -> llvm::StructType * {
                      throw BackendException(fmt::format("Unseen struct def {}, currently in-scope structs: {}", repr(x),
                                                         structs | values() | mk_string("\n", "\n", "\n", [](auto &ty) {
                                                           return fmt::format(" -> {}", to_string(ty.def));
                                                         })));
                    });
      },                                                                                                  //
      [&](const Type::Ptr &x) -> llvm::Type * { return llvm::PointerType::get(actual, ptrAS(x.space)); }, //
      [&](const Type::Arr &x) -> llvm::Type * {
        // Sized arrays decay to a pointer at function boundaries (matching C); inside a struct
        // body keep [N x T] for [0, idx] GEPs. SPIR-V Kernel forbids [0 x T] (see voidLike), so
        // collapse to i8.
        if (functionBoundary) return llvm::PointerType::get(actual, ptrAS(TypeSpace::Global()));
        if (spirvKernel && x.length == 0) return llvm::Type::getInt8Ty(actual);
        return llvm::ArrayType::get(resolveType(x.comp, structs, functionBoundary), x.length);
      }, //
      [&](const Type::Var &x) -> llvm::Type * { throw BackendException("Type::Var should be erased before LLVM lowering"); },
      [&](const Type::Exec &x) -> llvm::Type * { throw BackendException("Type::Exec should be erased before LLVM lowering"); });
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
  // Also create opaque types for any structs referenced (transitively through members) but not
  // present in the defs list. The Scala frontend currently doesn't propagate field-type deps,
  // so e.g. `Node.next: Option[Node]` lands here without an Option struct def. The kernel can't
  // dereference these opaque structs, but type-only references (capturing Node by-value while
  // only reading `node.elem`) work fine.
  std::function<Vector<std::string>(const Type::Any &)> referencedStructNames = [&](const Type::Any &t) -> Vector<std::string> {
    if (auto s = t.template get<Type::Struct>()) {
      return Vector<std::string>{repr(s->name)} | concat(s->args | flat_map(referencedStructNames)) | to_vector();
    } else if (auto p = t.template get<Type::Ptr>()) {
      return referencedStructNames(p->comp);
    } else if (auto a = t.template get<Type::Arr>()) {
      return referencedStructNames(a->comp);
    } else if (auto e = t.template get<Type::Exec>()) {
      return e->args | flat_map(referencedStructNames) | concat(referencedStructNames(e->rtn)) | to_vector();
    }
    return {};
  };
  const auto structNames = structs | flat_map([&](auto &def) {
                             return Vector<std::string>{repr(def.name)} |
                                    concat(def.members | flat_map([&](auto &m) { return referencedStructNames(m.tpe); })) | to_vector();
                           }) |
                           distinct() | to_vector();
  const auto opaqueTypes =
      structNames | map([&](auto &name) { return std::pair{name, llvm::StructType::create(actual, name)}; }) | to<Map>();
  // Synthesise StructDefs for any types we created opaque shells for but weren't in the input.
  // We give them a single i8 placeholder member so LLVM treats them as sized (size 1) — empty
  // structs aren't well-supported by `DataLayout::getAlignment`.
  const auto originalNames = structs | map([](auto &d) { return repr(d.name); }) | to<Set>();
  const auto syntheticDefs = structNames | collect([&](auto &name) -> Opt<StructDef> {
                               if (originalNames.contains(name)) return {};
                               return StructDef(Sym({name}), {}, {Named("__opaque", Type::IntS8())}, {}, /*isUnion*/ false);
                             });
  const auto allDefs = structs | concat(syntheticDefs) | to_vector();

  // Phase 2a: register stubs so resolveType can refer to any struct by name during body resolution.
  const auto resolved =
      allDefs | map([&](auto &def) {
        const auto name = repr(def.name);
        return std::pair{name,
                         StructInfo{.def = def, .layout = StructLayout(name, 0, 0, {}), .tpe = opaqueTypes.at(name), .memberIndices = {}}};
      }) |
      to<Map>();
  // Phase 2b: setBody on every struct so all types are sized (recursive defs can ask each other for size).
  // Struct-typed fields are laid out inline (functionBoundary=false) so the layout matches what a
  // host C++ compiler produces for a captured-by-value struct — the kernel receives a pointer to the
  // outer struct and reads inline fields directly. The previous design used functionBoundary=true
  // (8-byte pointer slots) to defend against empty-struct fields computing to 0 bytes; we keep that
  // defence narrowly by giving empty structs a single placeholder byte.
  const auto dataLayout = options.targetInfo().resolveDataLayout();
  const auto typeReadyForUnionStorage = [&](const Type::Any &tpe) {
    std::function<bool(const Type::Any &)> sized = [&](const Type::Any &t) {
      if (auto s = t.template get<Type::Struct>()) return !opaqueTypes.at(repr(s->name))->isOpaque();
      if (auto a = t.template get<Type::Arr>()) return sized(a->comp);
      return true;
    };
    return sized(tpe);
  };
  auto setBody = [&](const StructDef &def) {
    auto *tpe = opaqueTypes.at(repr(def.name));
    if (!tpe->isOpaque()) return true; // body already set; safe in case of duplicate StructDefs
    if (def.isUnion && !(def.members | forall([&](auto &m) { return typeReadyForUnionStorage(m.tpe); }))) return false;
    const auto memberTypes =
        def.members | map([&](auto &m) { return resolveType(m.tpe, resolved, /*functionBoundary*/ false); }) | to_vector();
    if (def.isUnion && !memberTypes.empty()) {
      const auto maxSize = memberTypes | fold_left(uint64_t{0}, [&](auto acc, auto *mt) { //
                             return std::max(acc, dataLayout.getTypeAllocSize(mt).getFixedValue());
                           });
      auto *const leadTy = (memberTypes | max_by([&](auto *mt) { return dataLayout.getABITypeAlign(mt).value(); })).value();
      const auto leadSize = dataLayout.getTypeAllocSize(leadTy).getFixedValue();
      const auto storage = maxSize > leadSize
                               ? std::vector<llvm::Type *>{leadTy, llvm::ArrayType::get(llvm::Type::getInt8Ty(actual), maxSize - leadSize)}
                               : std::vector<llvm::Type *>{leadTy};
      tpe->setBody(storage);
    } else tpe->setBody(memberTypes);
    return true;
  };
  allDefs | filter([](auto &def) { return !def.isUnion; }) | for_each([&](auto &def) { setBody(def); });
  bool progressed = true;
  while (progressed) {
    progressed = allDefs                                                                                        //
                 | filter([&](auto &def) { return def.isUnion && opaqueTypes.at(repr(def.name))->isOpaque(); }) //
                 | fold_left(false, [&](auto acc, auto &def) { return setBody(def) || acc; });
  }
  allDefs | for_each([&](auto &def) {
    if (opaqueTypes.at(repr(def.name))->isOpaque()) {
      throw BackendException(fmt::format("Could not size union struct {}", repr(def.name)));
    }
  });
  // Phase 2c: now that all bodies are set, compute layouts.
  return allDefs | map([&](auto &def) {
           auto tpe = opaqueTypes.at(repr(def.name));
           const auto table = (def.members | map([](auto &m) { return m.symbol; }) | zip_with_index<size_t>() | to_vector()) ^ to<Map>();
           if (def.isUnion) {
             const StructLayout layout(
                 /*name*/ repr(def.name),
                 /*sizeInBytes*/ static_cast<int64_t>(dataLayout.getTypeAllocSize(tpe).getFixedValue()),
                 /*alignment*/ static_cast<int64_t>(dataLayout.getABITypeAlign(tpe).value()),
                 /*members*/ def.members | map([&](auto &named) {
                   return StructLayoutMember(
                       /*name*/ named, /*offsetInBytes*/ 0,
                       /*sizeInBytes*/
                       static_cast<int64_t>(
                           dataLayout.getTypeAllocSize(resolveType(named.tpe, resolved, /*functionBoundary*/ false)).getFixedValue()));
                 }) | to_vector());
             return std::pair{repr(def.name), StructInfo{.def = def, .layout = layout, .tpe = tpe, .memberIndices = table}};
           }
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
           return std::pair{repr(def.name), StructInfo{.def = def, .layout = layout, .tpe = tpe, .memberIndices = table}};
         }) |
         to<Map>();
}

llvmc::TargetInfo LLVMBackend::Options::targetInfo() const {
  using llvm::Triple;
  const auto bindGpuArch = [&](const Triple::ArchType &archTpe, const Triple::VendorType &vendor, const Triple::OSType &os) {
    const Triple triple(Triple::getArchTypeName(archTpe), Triple::getVendorTypeName(vendor), Triple::getOSTypeName(os));
    if (archTpe == Triple::ArchType::UnknownArch) throw std::logic_error("Arch must be specified for " + triple.str());
    return llvmc::TargetInfo{
        .triple = triple,
        .layout = {},
        .target = llvmc::targetFromTriple(triple),
        .cpu = {.uArch = arch, .features = {}},
    };
  };

  // SPIRV execution model is picked from the triple OS: `spirv64-unknown-unknown` -> OpenCL
  // Kernel (Physical/OpenCL); `spirv-unknown-vulkan-compute` -> Vulkan compute (Logical/GLCompute).
  const auto bindSpirv = [&](const std::string &tripleStr) {
    const Triple triple(tripleStr);
    return llvmc::TargetInfo{
        .triple = triple,
        .layout = {},
        .target = llvmc::targetFromTriple(triple),
        .cpu = {.uArch = arch, .features = {}},
    };
  };

  const auto bindCpuArch = [&](const Triple::ArchType &archTpe) {
    const Triple defaultTriple = llvmc::defaultHostTriple();
    const bool wantHost = arch.empty() || arch == "native";
    if (wantHost && defaultTriple.getArch() != archTpe)
      throw BackendException("Requested host CPU detection (`" + (arch.empty() ? std::string{"<empty>"} : arch) + "`) with " +
                             Triple::getArchTypeName(archTpe).str() + " but the host arch is different (" +
                             Triple::getArchTypeName(defaultTriple.getArch()).str() + ")");

    Triple triple = defaultTriple;
    triple.setArch(archTpe);
    return llvmc::TargetInfo{
        .triple = triple,
        .layout = {},
        .target = llvmc::targetFromTriple(triple),
        .cpu = wantHost ? llvmc::hostCpuInfo() : llvmc::CpuInfo{.uArch = arch, .features = {}},
    };
  };

  // XXX Pin to SPIR-V 1.2: OpenCL 1.2 conformant ICDs (Intel NEO included) only accept 1.0 via
  // clCreateProgramWithIL; 1.2 is what OpenCL 2.1+ environments accept. Without a version
  // suffix LLVM defaults to 1.4 and the program won't load on most current OpenCL drivers.
  constexpr const char *Spirv32KernelTriple = "spirv32v1.2-unknown-unknown";
  constexpr const char *Spirv64KernelTriple = "spirv64v1.2-unknown-unknown";
  constexpr const char *SpirvVulkanComputeTriple = "spirv-unknown-vulkan1.3-compute";

  switch (target) {
    case Target::x86_64: return bindCpuArch(Triple::ArchType::x86_64);
    case Target::AArch64: return bindCpuArch(Triple::ArchType::aarch64);
    case Target::ARM: return bindCpuArch(Triple::ArchType::arm);
    case Target::RISCV64: return bindCpuArch(Triple::ArchType::riscv64);
    case Target::PPC64LE: return bindCpuArch(Triple::ArchType::ppc64le);
    case Target::NVPTX64: return bindGpuArch(Triple::ArchType::nvptx64, Triple::VendorType::NVIDIA, Triple::OSType::CUDA);
    case Target::AMDGCN: return bindGpuArch(Triple::ArchType::amdgcn, Triple::VendorType::AMD, Triple::OSType::AMDHSA);
    case Target::SPIRV32_Kernel: return bindSpirv(Spirv32KernelTriple);
    case Target::SPIRV64_Kernel: return bindSpirv(Spirv64KernelTriple);
    case Target::SPIRV_GLCompute: return bindSpirv(SpirvVulkanComputeTriple);
    default: throw BackendException(fmt::format("Unexpected target {}", magic_enum::enum_name(target)));
  }
}
