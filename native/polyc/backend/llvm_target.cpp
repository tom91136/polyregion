#include <unordered_set>

#include "aspartame/all.hpp"
#include "magic_enum.hpp"

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
  const auto stackPtr = B.CreateAlloca(ty->isPointerTy() ? B.getPtrTy() : ty, AS, nullptr, key);
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
        return structs ^ get_maybe(x.name) ^
               fold([](auto &info) { return info.tpe; },
                    [&]() -> llvm::StructType * {
                      throw BackendException(fmt::format("Unseen struct def {}, currently in-scope structs: {}", repr(x),
                                                         structs | values() | mk_string("\n", "\n", "\n", [](auto &ty) {
                                                           return fmt::format(" -> {}", to_string(ty.tpe));
                                                         })));
                    });
      }, //
      [&](const Type::Ptr &x) -> llvm::Type * {
        if (x.length) return llvm::ArrayType::get(resolveType(x.comp, structs, functionBoundary), *x.length);
        return llvm::PointerType::get(actual, addressSpace(x.space));
      },                                                                                                      //
      [&](const Type::Annotated &x) -> llvm::Type * { return resolveType(x.tpe, structs, functionBoundary); } //
  );
}

StructInfo TargetedContext::resolveStruct(const StructDef &def, const Map<std::string, StructInfo> &structs) {
  const auto types = def.members ^ map([&](auto &m) { return resolveType(m.tpe, structs); });
  const auto table = (def.members | map([](auto &m) { return m.symbol; }) | zip_with_index() | to_vector()) ^ to<Map>();
  const auto tpe = llvm::StructType::create(actual, types, def.name);
  const auto dataLayout = options.targetInfo().resolveDataLayout();
  const auto structLayout = dataLayout.getStructLayout(tpe);
  const StructLayout layout(/*name*/ def.name,
                            /*sizeInBytes*/ structLayout->getSizeInBytes(),
                            /*alignment*/ static_cast<int64_t>(structLayout->getAlignment().value()),
                            /*members*/ def.members | zip_with_index() | map([&](auto &named, auto i) {
                              return StructLayoutMember(
                                  /*name*/ named,                                                            //
                                  /*offsetInBytes*/ static_cast<int64_t>(structLayout->getElementOffset(i)), //
                                  /*sizeInBytes*/ dataLayout.getTypeAllocSize(tpe->getElementType(i))        //
                              );
                            }) | to_vector());
  return {.def = def, .layout = layout, .tpe = tpe, .memberIndices = table};
}

Map<std::string, StructInfo> TargetedContext::resolveLayouts(const std::vector<StructDef> &structs) {
  Map<std::string, StructInfo> resolved;
  Set<StructDef> withDependencies(structs.begin(), structs.end());
  while (!withDependencies.empty()) { // TODO handle recursive defs
    std::vector<StructDef> zeroDeps =
        withDependencies | filter([&](auto &def) {
          return def.members ^ forall([&](auto &named) {
                   return named.tpe.template get<Type::Struct>() ^ forall([&](auto &s) { return resolved.contains(s.name); });
                 });
        }) |
        to_vector();
    if (!zeroDeps.empty()) {
      for (auto &r : zeroDeps) {
        resolved.emplace(r.name, resolveStruct(r, resolved));
        withDependencies.erase(r);
      }
    } else
      throw BackendException(
          fmt::format("Recursive defs cannot be resolved: {}", zeroDeps ^ mk_string(",", [](auto &r) { return to_string(r); })));
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
