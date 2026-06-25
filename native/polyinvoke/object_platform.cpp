#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#if defined(__linux__) || defined(__APPLE__)
  #include <dlfcn.h>
#endif

#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "polyinvoke/object_platform.h"
#include "polyregion/compat.h"
#include "polyregion/llvm_utils.hpp"

// keep last: libffi pollutes the global namespace with macros
#include "ffi_wrapped.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::object;

static void ffiInvoke(const char *prefix, uint64_t symbolAddress, const std::vector<Type> &types, std::vector<void *> &args) {
  const auto toFFITpe = [prefix](const Type &tpe) -> ffi_type * {
    switch (tpe) {
      case Type::Bool1:
      case Type::IntS8: return &ffi_type_sint8;
      case Type::IntU16: return &ffi_type_uint16;
      case Type::IntS16: return &ffi_type_sint16;
      case Type::IntS32: return &ffi_type_sint32;
      case Type::IntS64: return &ffi_type_sint64;
      case Type::Float32: return &ffi_type_float;
      case Type::Float64: return &ffi_type_double;
      case Type::Ptr: return &ffi_type_pointer;
      case Type::Void: return &ffi_type_void;
      default: POLYINVOKE_FATAL(prefix, "Illegal ffi type: %s", magic_enum::enum_name(tpe).data());
    }
  };
  if (types.size() != args.size()) POLYINVOKE_FATAL(prefix, "types size (%zu) != args size (%zu)", types.size(), args.size());

  // skip stripped Unit0/Nothing params; ffi_type_void as a param misaligns Win x64
  std::vector<ffi_type *> ffiParamTypes;
  std::vector<void *> ffiParamArgs;
  ffiParamTypes.reserve(args.size());
  ffiParamArgs.reserve(args.size());
  for (size_t i = 0; i + 1 < types.size(); ++i) {
    if (types[i] == Type::Void) continue;
    ffiParamTypes.push_back(toFFITpe(types[i]));
    ffiParamArgs.push_back(args[i]);
  }
  ffi_type *returnFfiTpe = toFFITpe(types.back());
  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, static_cast<unsigned>(ffiParamTypes.size()), returnFfiTpe, ffiParamTypes.data());
  switch (status) {
    case FFI_OK: ffi_call(&cif, FFI_FN(symbolAddress), args.back(), ffiParamArgs.data()); break;
    case FFI_BAD_TYPEDEF: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: FFI_BAD_TYPEDEF (%d)", status);
    case FFI_BAD_ABI: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: FFI_BAD_ABI (%d)", status);
    default: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: unknown error (%d)", status);
  }
}

int64_t ObjectDevice::id() {
  POLYINVOKE_TRACE();
  return 0;
}
PhysicalDevice ObjectDevice::physicalDevice() {
  POLYINVOKE_TRACE();
  return PhysicalDevice::host();
}
bool ObjectDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return true;
}
PagingMode ObjectDevice::pagingMode() {
  POLYINVOKE_TRACE();
  return PagingMode::System; // host CPU: any pointer is directly dereferenceable
}
bool ObjectDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> ObjectDevice::properties() {
  POLYINVOKE_TRACE();
  return {};
}
std::vector<std::string> ObjectDevice::features() {
  POLYINVOKE_TRACE();

  std::vector<std::string> features;
  for (auto &F : llvm::sys::getHostCPUFeatures()) {
    if (F.second) features.push_back(F.first().str());
  }

  polyregion::llvm_shared::collectCPUFeatures(llvm::sys::getHostCPUName().str(),
                                              llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch(), features);

  features.emplace_back("fp64");
  features.emplace_back("fp16");
  features.emplace_back("int64");
  // the CPU host device is `@native`; expose it so host@native/c11@native match by feature under strict selection
  features.emplace_back("native");
  features.emplace_back(fmt::format("paging:{}", magic_enum::enum_name(pagingMode())));

  return features;
}
uintptr_t ObjectDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  return reinterpret_cast<uintptr_t>(std::malloc(size));
}
void ObjectDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  std::free(reinterpret_cast<void *>(ptr));
}

std::optional<void *> ObjectDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return std::malloc(size);
}
void ObjectDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  std::free(ptr);
}

// ---

void ObjectDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                   const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(reinterpret_cast<void *>(dst + dstOffset), reinterpret_cast<void *>(src + srcOffset), size);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(reinterpret_cast<void *>(dst + dstOffset), src, size);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(dst, reinterpret_cast<void *>(src + srcOffset), bytes);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  bool warned = false;
  while (!latch.waitAll()) {
    if (!warned) {
      fmt::print(stderr, "polyinvoke: kernel still running past the watchdog timeout, continuing to wait\n");
      std::fflush(stderr);
      warned = true;
    }
  }
}
ObjectDeviceQueue::ObjectDeviceQueue(const std::chrono::duration<int64_t> &timeout) : latch(timeout) { POLYINVOKE_TRACE(); }

ObjectDeviceQueue::~ObjectDeviceQueue() noexcept { POLYINVOKE_TRACE(); }
std::variant<std::string, std::unique_ptr<Platform>> RelocatablePlatform::create() {
  return std::unique_ptr<Platform>(new RelocatablePlatform());
}
RelocatablePlatform::RelocatablePlatform() { POLYINVOKE_TRACE(); }
std::string RelocatablePlatform::name() {
  POLYINVOKE_TRACE();
  return "CPU (RelocatableObject)";
}
std::vector<Property> RelocatablePlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind RelocatablePlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::HostThreaded;
}
ModuleFormat ObjectDevice::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::Object;
}
std::vector<std::unique_ptr<Device>> RelocatablePlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
  return xs;
}

static constexpr const char *RELOBJ_PREFIX = "RelocatableObject";

static std::string withGlobalPrefix(char prefix, std::string_view name) {
  return prefix ? std::string(1, prefix) + std::string(name) : std::string(name);
}

static void *malloc_(size_t size) { return std::malloc(size); }

// RuntimeDyldCOFFX86_64 zero-extends the 4-byte REL32 addend, so RIP-rel insns with a trailing imm
// (vpinsrw, vextractf128) encoding -1..-5 in the patch-site overflow the INT32_MAX assert; rewrite each
// to IMAGE_REL_AMD64_REL32_N (N=1..5) with a zero patch-site so RuntimeDyld computes Delta=4+N itself.
// raw constants since winnt.h's IMAGE_REL_AMD64_* macros collide with llvm::COFF::* enums under MSVC
static void rewriteCOFFRel32NegativeAddends(llvm::MutableArrayRef<char> buf) {
  constexpr uint16_t Rel32 = 0x0004;
  constexpr uint16_t FileMachineAMD64 = 0x8664;
  constexpr size_t CoffHeaderMin = 20;

  if (buf.size() < CoffHeaderMin) return;
  uint16_t machine;
  std::memcpy(&machine, buf.data(), 2);
  if (machine != FileMachineAMD64) return; // not an x86_64 COFF; ELF/Mach-O/import-lib skip the parse

  auto memBuf = llvm::MemoryBufferRef(llvm::StringRef(buf.data(), buf.size()), "<coff>");
  auto objOrErr = llvm::object::ObjectFile::createObjectFile(memBuf);
  if (!objOrErr) {
    llvm::consumeError(objOrErr.takeError());
    return;
  }
  auto *coff = llvm::dyn_cast<llvm::object::COFFObjectFile>(objOrErr->get());
  if (!coff) return;

  char *const base = buf.data();
  const size_t size = buf.size();

  for (const auto &secRef : coff->sections()) {
    const llvm::object::coff_section *cs = coff->getCOFFSection(secRef);
    if (!cs || cs->NumberOfRelocations == 0 || cs->SizeOfRawData == 0) continue;
    constexpr size_t Reloc = 10; // sizeof(coff_relocation)
    size_t i = 0;
    for (const auto &rel : coff->getRelocations(cs)) {
      const size_t relOff = static_cast<size_t>(cs->PointerToRelocations) + i * Reloc;
      ++i;
      if (rel.Type != Rel32 || rel.VirtualAddress < cs->VirtualAddress) continue;
      const uint32_t off = rel.VirtualAddress - cs->VirtualAddress;
      if (off + 4 > cs->SizeOfRawData) continue;
      const size_t patchOff = static_cast<size_t>(cs->PointerToRawData) + off;
      if (patchOff + 4 > size || relOff + Reloc > size) continue;

      int32_t addend;
      std::memcpy(&addend, base + patchOff, 4);
      if (addend < -5 || addend > -1) continue;

      const uint16_t newType = static_cast<uint16_t>(Rel32 + (-addend));
      std::memcpy(base + relOff + offsetof(llvm::object::coff_relocation, Type), &newType, 2);
      const uint32_t zero = 0;
      std::memcpy(base + patchOff, &zero, 4);
    }
  }
}

RelocatableDevice::RelocatableDevice() {
  POLYINVOKE_TRACE();
  auto epc = llvm::orc::SelfExecutorProcessControl::Create();
  if (!epc) POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot create executor process control: %s", toString(epc.takeError()).c_str());
  es = std::make_unique<llvm::orc::ExecutionSession>(std::move(*epc));

  const llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  globalPrefix = hostTriple.isOSBinFormatMachO() ? '_' : '\0';

  // RTDyld SIGBUSes on x86_64 macOS after many libm-calling kernels; JITLink covers Mach-O but lacks
  // COFF/ARM64 + ELF backends on some hosts, so keep RTDyld elsewhere
  if (hostTriple.isOSBinFormatMachO()) {
    auto mm = llvm::jitlink::InProcessMemoryManager::Create();
    if (!mm) POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot create JITLink memory manager: %s", toString(mm.takeError()).c_str());
    jitMemMgr = std::move(*mm);
    auto layer = std::make_unique<llvm::orc::ObjectLinkingLayer>(*es, *jitMemMgr);
    // promote our object flags so duplicate symbols across modules don't fail materialisation
    layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    layer->setAutoClaimResponsibilityForObjectSymbols(true);
    ol = std::move(layer);
  } else {
    auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        *es, [](const llvm::MemoryBuffer &) { return std::make_unique<llvm::SectionMemoryManager>(); });
    layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
    layer->setAutoClaimResponsibilityForObjectSymbols(true);
    ol = std::move(layer);
  }

  processJD = &es->createBareJITDylib("<process>");

  llvm::orc::SymbolMap absSyms;
  absSyms[es->intern(withGlobalPrefix(globalPrefix, "malloc"))] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(&malloc_), llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
  if (auto err = processJD->define(llvm::orc::absoluteSymbols(std::move(absSyms))))
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot define malloc override: %s", toString(std::move(err)).c_str());

  // libm::exportAll (from invoke::init) registers sincosf et al. via sys::DynamicLibrary; this generator picks them up
  if (auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(globalPrefix)) processJD->addGenerator(std::move(*gen));
  else POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot create process symbol generator: %s", toString(gen.takeError()).c_str());
}

RelocatableDevice::~RelocatableDevice() {
  POLYINVOKE_TRACE();
  if (es) {
    if (auto err = es->endSession())
      fmt::print(stderr, "[{}] ExecutionSession::endSession failed: {}\n", RELOBJ_PREFIX, toString(std::move(err)));
  }
}

std::string RelocatableDevice::name() {
  POLYINVOKE_TRACE();
  return "RelocatableObjectDevice(llvm::orc::RTDyldObjectLinkingLayer)";
}

void RelocatableDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  WriteLock rw(mutex);
  if (auto it = objects.find(name); it != objects.end()) {
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "Module named %s was already loaded", name.c_str());
  } else {
    auto jdOrErr = es->createJITDylib(name);
    if (!jdOrErr) POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot create JITDylib for %s: %s", name.c_str(), toString(jdOrErr.takeError()).c_str());
    llvm::orc::JITDylib &jd = *jdOrErr;
    jd.addToLinkOrder(*processJD);

    auto wbuf = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(image.size(), name);
    if (!wbuf) POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot allocate %zu-byte writable buffer for %s", image.size(), name.c_str());
    std::memcpy(wbuf->getBufferStart(), image.data(), image.size());
    rewriteCOFFRel32NegativeAddends(wbuf->getBuffer());
    if (auto err = ol->add(jd, std::unique_ptr<llvm::MemoryBuffer>(std::move(wbuf)))) {
      POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot load module: %s", toString(std::move(err)).c_str());
    }
    objects.emplace_hint(it, name, std::make_unique<details::LoadedCodeObject>(jd));
  }
}

bool RelocatableDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  ReadLock r(mutex);
  return objects.find(name) != objects.end();
}
std::unique_ptr<DeviceQueue> RelocatableDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<RelocatableDeviceQueue>(timeout, objects, mutex, *es, globalPrefix);
}

RelocatableDeviceQueue::RelocatableDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(objects) objects,
                                               decltype(mutex) mutex, llvm::orc::ExecutionSession &es, char globalPrefix)
    : ObjectDeviceQueue(timeout), objects(objects), mutex(mutex), es(es), globalPrefix(globalPrefix) {
  POLYINVOKE_TRACE();
}

void validatePolicyAndArgs(const char *prefix, std::vector<Type> types, const Policy &policy) {
  if (auto scratchCount = std::count(types.begin(), types.end(), Type::Scratch); scratchCount != 0)
    POLYINVOKE_FATAL(prefix, "Scratch types are not supported on the CPU, found %td arg(s)", scratchCount);
  if (policy.global.y != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Y > 1 is not supported: %zu", policy.global.y);
  if (policy.global.z != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Z > 1 is not supported: %zu", policy.global.z);
  if (policy.local) POLYINVOKE_FATAL(prefix, "Policy local dimension is not supported: size=%zu", policy.local->second);
  if (types[0] != Type::IntS64)
    POLYINVOKE_FATAL(prefix, "Expecting first argument as index (%s), but was %s", magic_enum::enum_name(Type::IntS64).data(),
                     magic_enum::enum_name(types[0]).data());
}
void RelocatableDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                                std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  validatePolicyAndArgs(RELOBJ_PREFIX, types, policy);

  details::LoadedCodeObject *obj = nullptr;
  {
    ReadLock r(mutex);
    const auto moduleIt = objects.find(moduleName);
    if (moduleIt == objects.end()) POLYINVOKE_FATAL(RELOBJ_PREFIX, "No module named %s was loaded", moduleName.c_str());
    obj = moduleIt->second.get();
  }
  uint64_t symAddr;
  {
    ReadLock r(obj->symbolCacheMutex);
    if (const auto it = obj->symbolCache.find(symbol); it != obj->symbolCache.end()) symAddr = it->second;
    else {
      r.unlock();
      WriteLock w(obj->symbolCacheMutex);
      if (const auto it = obj->symbolCache.find(symbol); it != obj->symbolCache.end()) symAddr = it->second;
      else {
        const auto fnName = withGlobalPrefix(globalPrefix, symbol);
        auto symOrErr = es.lookup({obj->jd}, fnName);
        if (!symOrErr)
          POLYINVOKE_FATAL(RELOBJ_PREFIX, "Symbol `%s` not found in the given object: %s", fnName.c_str(),
                           toString(symOrErr.takeError()).c_str());
        symAddr = symOrErr->getAddress().getValue();
        obj->symbolCache.emplace(symbol, symAddr);
      }
    }
  }
  this->threadedLaunch(
      policy.global.x,
      [cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      },
      [symAddr, types, argData](size_t tid) {
        auto argData_ = argData;
        auto argPtrs = detail::argDataAsPointers(types, argData_);
        if (types[0] != Type::IntS64) {
          POLYINVOKE_FATAL(RELOBJ_PREFIX, "Expecting first argument as index: %s, but was %s", magic_enum::enum_name(Type::IntS64).data(),
                           magic_enum::enum_name(types[0]).data());
        }
        auto _tid = int64_t(tid);
        argPtrs[0] = &_tid;
        ffiInvoke(RELOBJ_PREFIX, symAddr, types, argPtrs);
      });
}

static constexpr const char *SHOBJ_PREFIX = "SharedObject";

std::variant<std::string, std::unique_ptr<Platform>> SharedPlatform::create() { return std::unique_ptr<Platform>(new SharedPlatform()); }

SharedPlatform::SharedPlatform() { POLYINVOKE_TRACE(); }
std::string SharedPlatform::name() {
  POLYINVOKE_TRACE();
  return "CPU (SharedObject)";
}
std::vector<Property> SharedPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind SharedPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::HostThreaded;
}
std::vector<std::unique_ptr<Device>> SharedPlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<SharedDevice>();
  return xs;
}

SharedDevice::~SharedDevice() {
  POLYINVOKE_TRACE();
  for (auto &[_, m] : modules) {
    auto &[path, handle, symbols] = m;
    if (auto code = polyregion_dl_close(handle); code != 0) {
      fmt::print(stderr, "{} Cannot unload module, code {}: {}\n", SHOBJ_PREFIX, code, polyregion_dl_error());
    }
  }
}
std::string SharedDevice::name() {
  POLYINVOKE_TRACE();
  return "SharedObjectDevice(dlopen/dlsym)";
}
void SharedDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  if (auto it = modules.find(name); it != modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "Module named %s was already loaded", name.c_str());
  else {

    // TODO fileless load: Linux memfd_create, Windows MemoryModule; dlopen needs a real file for now
    auto tmpPath = std::tmpnam(nullptr);
    if (!tmpPath) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Unable to buffer image to file for %s, tmpfile creation failed: cannot synthesise temp path",
                       name.c_str());
    }
    std::FILE *objectFile = std::fopen(tmpPath, "wb");
    if (!objectFile) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Unable to buffer image to file for %s, tmpfile creation failed: %s", name.c_str(),
                       std::strerror(errno));
    }
    std::fwrite(image.data(), image.size(), 1, objectFile);
    std::fflush(objectFile);
    std::fclose(objectFile);
    static std::vector<std::string> tmpImagePaths;
    tmpImagePaths.emplace_back(tmpPath);
    static std::mutex cleanupMutex;
    static auto cleanUp = []() {
      std::unique_lock<std::mutex> lock(cleanupMutex);
      for (auto &path : tmpImagePaths) {
        if (std::remove(path.c_str()) != 0) {
          fmt::print(stderr, "[{}] Warning: cannot remove temporary image file {}\n", SHOBJ_PREFIX, path);
        }
      }
      tmpImagePaths.clear();
    };
    std::atexit(cleanUp);
    std::set_terminate(cleanUp);

    if (auto dylib = polyregion_dl_open(tmpPath); !dylib) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load module: %s", polyregion_dl_error());
    } else modules.emplace_hint(it, name, details::LoadedModule{image, dylib, {}});
  }
}
bool SharedDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return modules.find(name) != modules.end();
}
std::unique_ptr<DeviceQueue> SharedDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<SharedDeviceQueue>(timeout, modules, mutex);
}

SharedDeviceQueue::SharedDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(modules) modules, decltype(mutex) mutex)
    : ObjectDeviceQueue(timeout), modules(modules), mutex(mutex) {
  POLYINVOKE_TRACE();
}
void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                           std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  validatePolicyAndArgs(SHOBJ_PREFIX, types, policy);

  ReadLock r(mutex);
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "No module named %s was loaded", moduleName.c_str());

  auto &[image, handle, symbolTable] = moduleIt->second;

  void *address = nullptr;
  if (const auto it = symbolTable.find(symbol); it != symbolTable.end()) address = it->second;
  else {
    address = polyregion_dl_find(handle, symbol.c_str());
    if (!address) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load symbol %s from module %s (%zd bytes): %s", symbol.c_str(), moduleName.c_str(),
                       image.size(), polyregion_dl_error());
    }
    symbolTable.emplace_hint(it, symbol, address);
  }

  this->threadedLaunch(
      policy.global.x,
      [cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      },
      [address, types, argData](size_t tid) {
        auto argData_ = argData;
        auto argPtrs = detail::argDataAsPointers(types, argData_);
        auto _tid = int64_t(tid);
        argPtrs[0] = &_tid;
        ffiInvoke(SHOBJ_PREFIX, reinterpret_cast<uint64_t>(address), types, argPtrs);
      });
  POLYINVOKE_TRACE();
}
