#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#if defined(__linux__) || defined(__APPLE__)
  #include <dlfcn.h>
  #include <pthread.h>
  #include <unistd.h>

  #include <sys/mman.h>
  #include <sys/resource.h>
#endif
#ifdef __APPLE__
  #include <sys/sysctl.h>
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "aspartame/all.hpp"

#include "polyinvoke/object_platform.h"
#include "polyregion/compat.h"
#include "polyregion/env_keys.h"
#include "polyregion/llvm_utils.hpp"
#include "polyregion/types.h"

#include "c_build.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

// keep last: libffi pollutes the global namespace with macros
#include "ffi_wrapped.h"

using namespace aspartame;
using namespace polyregion::invoke;
using namespace polyregion::invoke::object;

namespace {
thread_local uint64_t hostThreadGlobalIdx = 0;
thread_local uint64_t hostThreadGlobalSize = 1;
thread_local uint64_t hostLocalSize = 1;

constexpr size_t HostGroupLocalBytes = polyregion::program_meta::HostWorkgroupMemoryBytes;

struct HostGroup {
  std::mutex mutex;
  std::condition_variable released;
  size_t expected;
  size_t arrived = 0;
  uint64_t generation = 0;
  alignas(64) std::byte local[HostGroupLocalBytes]{};

  explicit HostGroup(const size_t lanes) : expected(lanes) {}

  void release() {
    arrived = 0;
    ++generation;
    released.notify_all();
  }
  void arrive() {
    std::unique_lock lock(mutex);
    if (++arrived == expected) return release();
    const auto seen = generation;
    released.wait(lock, [&] { return generation != seen; });
  }
  void retire() {
    const std::lock_guard lock(mutex);
    --expected;
    if (expected > 0 && arrived == expected) release();
  }
};

thread_local HostGroup *hostGroup = nullptr;
thread_local std::unique_ptr<std::byte[]> hostTaskLocal;

std::byte *hostTaskLocalBlock() {
  if (!hostTaskLocal) hostTaskLocal = std::make_unique<std::byte[]>(HostGroupLocalBytes);
  return hostTaskLocal.get();
}

constexpr size_t HostStateAlignment = 64;

struct HostStateArena {
  void *raw = nullptr;
  size_t capacity = 0;
  HostStateArena() = default;
  HostStateArena(const HostStateArena &) = delete;
  HostStateArena &operator=(const HostStateArena &) = delete;
  ~HostStateArena() { std::free(raw); }
};

thread_local HostStateArena hostStateArena;

constexpr double HostLaneStackAddressShare = 0.5;

#if defined(__linux__) && UINTPTR_MAX == UINT32_MAX
  #define POLYINVOKE_EXPLICIT_LANE_STACK
constexpr size_t HostLaneStackBytes = 2 * 1024 * 1024;
#endif

uint64_t hostStackReserveBytes() {
#if defined(POLYINVOKE_EXPLICIT_LANE_STACK)
  return HostLaneStackBytes;
#elif defined(_WIN32)
  const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(GetModuleHandleW(nullptr));
  const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(reinterpret_cast<const char *>(dos) + dos->e_lfanew);
  return nt->OptionalHeader.SizeOfStackReserve;
#else
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  size_t size = 0;
  pthread_attr_getstacksize(&attr, &size);
  pthread_attr_destroy(&attr);
  return size;
#endif
}

uint64_t hostAddressSpaceBytes() {
  uint64_t bytes = sizeof(void *) < sizeof(uint64_t) ? uint64_t(1) << (8 * sizeof(void *)) : std::numeric_limits<uint64_t>::max();
#ifndef _WIN32
  rlimit as{};
  if (getrlimit(RLIMIT_AS, &as) == 0 && as.rlim_cur != RLIM_INFINITY) bytes = std::min<uint64_t>(bytes, as.rlim_cur);
#endif
  return bytes;
}
} // namespace

size_t polyregion::invoke::object::hostMaxLanes() {
  static const size_t lanes = [] {
    const auto fit =
        static_cast<double>(hostAddressSpaceBytes()) * HostLaneStackAddressShare / static_cast<double>(hostStackReserveBytes());
    return std::clamp<size_t>(static_cast<size_t>(fit), 1, polyregion::program_meta::HostWorkgroupMaxLanes);
  }();
  return lanes;
}

extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_thread_global_idx() { return hostThreadGlobalIdx; }
extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_thread_global_size() { return hostThreadGlobalSize; }
extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_local_idx() { return hostThreadGlobalIdx % hostLocalSize; }
extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_local_size() { return hostLocalSize; }
extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_group_idx() { return hostThreadGlobalIdx / hostLocalSize; }
extern "C" POLYREGION_EXPORT uint64_t __polyregion_host_group_size() { return hostThreadGlobalSize / hostLocalSize; }
extern "C" POLYREGION_EXPORT void __polyregion_host_group_barrier() {
  if (hostGroup) hostGroup->arrive();
}
extern "C" POLYREGION_EXPORT void *__polyregion_host_group_local(const uint64_t offset) {
  if (hostGroup) return hostGroup->local + offset;
  return hostTaskLocalBlock() + offset;
}

static constexpr const char *RELOBJ_PREFIX = "RelocatableObject";

extern "C" POLYREGION_EXPORT void *__polyregion_host_group_state(const uint64_t bytes) {
  if (bytes > hostStateArena.capacity) {
    std::free(hostStateArena.raw);
    hostStateArena.raw = nullptr;
    hostStateArena.capacity = 0;
    if (bytes > std::numeric_limits<size_t>::max() - HostStateAlignment)
      POLYINVOKE_FATAL(RELOBJ_PREFIX, "work-group state of %zu bytes is too large", static_cast<size_t>(bytes));
    hostStateArena.raw = std::malloc(bytes + HostStateAlignment);
    if (!hostStateArena.raw) POLYINVOKE_FATAL(RELOBJ_PREFIX, "cannot allocate %zu bytes of work-group state", static_cast<size_t>(bytes));
    hostStateArena.capacity = bytes;
  }
  const auto address = reinterpret_cast<uintptr_t>(hostStateArena.raw);
  return reinterpret_cast<void *>((address + HostStateAlignment - 1) & ~static_cast<uintptr_t>(HostStateAlignment - 1));
}

static bool groupedPolicy(const Policy &policy) { return policy.local && policy.local->first.x > 1; }

#ifdef POLYINVOKE_EXPLICIT_LANE_STACK
class LaneThread {
  pthread_t handle{};

public:
  template <typename F> explicit LaneThread(F body) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, HostLaneStackBytes);
    auto owned = std::make_unique<std::function<void()>>(std::move(body));
    const int status = pthread_create(
        &handle, &attr,
        [](void *raw) -> void * {
          const std::unique_ptr<std::function<void()>> run(static_cast<std::function<void()> *>(raw));
          (*run)();
          return nullptr;
        },
        owned.get());
    pthread_attr_destroy(&attr);
    if (status != 0) POLYINVOKE_FATAL(RELOBJ_PREFIX, "cannot start a lane thread: %s", std::strerror(status));
    owned.release();
  }
  void join() { pthread_join(handle, nullptr); }
};
#else
using LaneThread = std::thread;
#endif

static void groupedLaunch(const size_t groups, const size_t lanes, const MaybeCallback &cb, std::function<void(size_t)> task) {
  if (lanes > hostMaxLanes())
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "work-group size %zu exceeds host capacity of %zu lanes", lanes, hostMaxLanes());
  if (groups == 0) {
    if (cb) (*cb)();
    return;
  }
  const size_t slots = std::clamp<size_t>(hostConcurrency() / lanes, 1, groups);
  auto pending = std::make_shared<std::atomic_size_t>(slots);
  auto run = std::make_shared<std::function<void(size_t)>>(std::move(task));
  for (size_t slot = 0; slot < slots; ++slot) {
    std::thread([=]() {
      for (size_t group = slot; group < groups; group += slots) {
        HostGroup state(lanes);
        std::vector<LaneThread> workers;
        workers.reserve(lanes);
        for (size_t lane = 0; lane < lanes; ++lane)
          workers.emplace_back([&state, run, lanes, tid = group * lanes + lane]() {
            struct Retire {
              HostGroup &group;
              ~Retire() { group.retire(); }
            } retire{state};
            hostLocalSize = lanes;
            hostGroup = &state;
            (*run)(tid);
          });
        for (auto &worker : workers)
          worker.join();
      }
      if (--*pending == 0 && cb) (*cb)();
    }).detach();
  }
}

static void groupAbiLaunch(const size_t groups, const size_t lanes, const size_t localBytes, const MaybeCallback &cb,
                           std::function<void(size_t)> task) {
  if (lanes == 0 || lanes > polyregion::program_meta::HostWorkgroupMaxLanes)
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "work-group size %zu is outside [1, %zu] lanes", lanes,
                     polyregion::program_meta::HostWorkgroupMaxLanes);
  if (groups == 0) {
    if (cb) (*cb)();
    return;
  }
  const size_t workers = std::min(hostConcurrency(), groups);
  auto pending = std::make_shared<std::atomic_size_t>(workers);
  auto run = std::make_shared<std::function<void(size_t)>>(std::move(task));
  for (size_t worker = 0; worker < workers; ++worker) {
    std::thread([=]() {
      hostLocalSize = lanes;
      for (size_t group = worker; group < groups; group += workers) {
        if (localBytes > 0) std::memset(hostTaskLocalBlock(), 0, localBytes);
        const size_t tid = group * lanes;
        (*run)(tid);
      }
      if (--*pending == 0 && cb) (*cb)();
    }).detach();
  }
}

static void ffiInvoke(const char *prefix, uint64_t symbolAddress, const std::vector<Type> &types, std::vector<void *> &args) {
  const auto toFFITpe = [prefix](const Type &tpe) -> ffi_type * {
    switch (tpe) {
      case Type::Bool1:
      case Type::IntS8: return &ffi_type_sint8;
      case Type::IntU8: return &ffi_type_uint8;
      case Type::IntU16: return &ffi_type_uint16;
      case Type::IntS16: return &ffi_type_sint16;
      case Type::IntU32: return &ffi_type_uint32;
      case Type::IntS32: return &ffi_type_sint32;
      case Type::IntU64: return &ffi_type_uint64;
      case Type::IntS64: return &ffi_type_sint64;
      case Type::Float16:
      case Type::Float32: return &ffi_type_float;
      case Type::Float64: return &ffi_type_double;
      case Type::Ptr: return &ffi_type_pointer;
      case Type::Void: return &ffi_type_void;
      case Type::Scratch: break;
    }
    POLYINVOKE_FATAL(prefix, "Illegal ffi type: %s", magic_enum::enum_name(tpe).data());
  };
  if (types.size() != args.size()) POLYINVOKE_FATAL(prefix, "types size (%zu) != args size (%zu)", types.size(), args.size());

  // skip stripped Unit0/Nothing params; ffi_type_void as a param misaligns Win x64
  std::vector<ffi_type *> ffiParamTypes;
  std::vector<void *> ffiParamArgs;
  std::vector<uint64_t> argSlots(args.size());
  ffiParamTypes.reserve(args.size());
  ffiParamArgs.reserve(args.size());
  for (size_t i = 0; i + 1 < types.size(); ++i) {
    if (types[i] == Type::Void) continue;
    ffiParamTypes.push_back(toFFITpe(types[i]));
    std::memcpy(&argSlots[i], args[i], byteOfType(types[i]));
    ffiParamArgs.push_back(&argSlots[i]);
  }
  ffi_type *returnFfiTpe = toFFITpe(types.back());
  uint64_t retSlot{};
  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, static_cast<unsigned>(ffiParamTypes.size()), returnFfiTpe, ffiParamTypes.data());
  switch (status) {
    case FFI_OK:
      ffi_call(&cif, FFI_FN(symbolAddress), &retSlot, ffiParamArgs.data());
      if (types.back() != Type::Void) std::memcpy(args.back(), &retSlot, byteOfType(types.back()));
      break;
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

  const auto probed = llvm::sys::getHostCPUFeatures();
  auto features = probed                                                            //
                  | filter([](const auto &feature) { return feature.getValue(); })  //
                  | map([](const auto &feature) { return feature.getKey().str(); }) //
                  | to_vector();

  if (probed.empty())
    polyregion::llvm_shared::collectCPUFeatures(llvm::sys::getHostCPUName().str(), llvm::Triple(llvm::sys::getProcessTriple()).getArch(),
                                                features);

  // XXX x87 is never CPUID-probed and collectCPUFeatures may not resolve this host's arch/model
  features.emplace_back("x87");

  features.emplace_back("fp64");
  features.emplace_back("fp16");
  features.emplace_back("int64");
  // the CPU host device is `@native`; expose it so host@native/c11@native match by feature under strict selection
  features.emplace_back("native");
  features.emplace_back(fmt::format("paging:{}", magic_enum::enum_name(pagingMode())));

  return features;
}
namespace {
struct Reservation {
  size_t reserved;
  size_t committed;
};

std::mutex reservationsMutex;
std::map<uintptr_t, Reservation> reservations;
} // namespace

uintptr_t ObjectDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  return reinterpret_cast<uintptr_t>(std::malloc(size));
}
void ObjectDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  {
    const std::lock_guard lock(reservationsMutex);
    if (const auto it = reservations.find(ptr); it != reservations.end()) {
#ifdef _WIN32
      VirtualFree(reinterpret_cast<void *>(ptr), 0, MEM_RELEASE);
#else
      munmap(reinterpret_cast<void *>(ptr), it->second.reserved);
#endif
      reservations.erase(it);
      return;
    }
  }
  std::free(reinterpret_cast<void *>(ptr));
}
uintptr_t ObjectDevice::reserveDevice(size_t maxBytes, size_t committedBytes) {
  POLYINVOKE_TRACE();
  if (committedBytes == 0 || committedBytes > maxBytes)
    throw std::invalid_argument(fmt::format("committed size {} is outside (0, {}]", committedBytes, maxBytes));
#ifdef _WIN32
  void *base = VirtualAlloc(nullptr, maxBytes, MEM_RESERVE, PAGE_READWRITE);
  if (!base || !VirtualAlloc(base, committedBytes, MEM_COMMIT, PAGE_READWRITE)) {
    if (base) VirtualFree(base, 0, MEM_RELEASE);
    throw std::bad_alloc();
  }
#else
  void *base = mmap(nullptr, maxBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (base == MAP_FAILED) throw std::bad_alloc();
#endif
  const auto handle = reinterpret_cast<uintptr_t>(base);
  const std::lock_guard lock(reservationsMutex);
  reservations.emplace(handle, Reservation{maxBytes, committedBytes});
  return handle;
}
void ObjectDevice::growDevice(uintptr_t handle, size_t committedBytes) {
  POLYINVOKE_TRACE();
  const std::lock_guard lock(reservationsMutex);
  const auto it = reservations.find(handle);
  if (it == reservations.end()) throw std::invalid_argument(fmt::format("{} is not a growable allocation", handle));
  if (committedBytes > it->second.reserved)
    throw std::length_error(
        fmt::format("device heap growth to {} bytes exceeds the limit of {} bytes", committedBytes, it->second.reserved));
  if (committedBytes <= it->second.committed) return;
#ifdef _WIN32
  if (!VirtualAlloc(reinterpret_cast<void *>(handle), committedBytes, MEM_COMMIT, PAGE_READWRITE)) throw std::bad_alloc();
#endif
  it->second.committed = committedBytes;
}

std::optional<void *> ObjectDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return std::malloc(size);
}
void ObjectDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  std::free(ptr);
}
size_t ObjectDevice::maxThreadsPerBlock() {
  POLYINVOKE_TRACE();
  return hostMaxLanes();
}
size_t ObjectDevice::localMemoryBytes() {
  POLYINVOKE_TRACE();
  return HostGroupLocalBytes;
}
size_t ObjectDevice::globalMemoryBytes() {
  POLYINVOKE_TRACE();
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? static_cast<size_t>(status.ullTotalPhys) : 0;
#elif defined(__APPLE__)
  uint64_t bytes = 0;
  size_t size = sizeof(bytes);
  return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 ? static_cast<size_t>(bytes) : 0;
#else
  const auto pages = sysconf(_SC_PHYS_PAGES);
  const auto pageSize = sysconf(_SC_PAGESIZE);
  return pages > 0 && pageSize > 0 ? static_cast<size_t>(pages) * static_cast<size_t>(pageSize) : 0;
#endif
}
size_t ObjectDevice::computeUnits() {
  POLYINVOKE_TRACE();
  return hostConcurrency();
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

  const llvm::Triple hostTriple(llvm::sys::getProcessTriple());
  globalPrefix = hostTriple.isOSBinFormatMachO() ? '_' : '\0';

  // XXX RTDyld: SIGBUS on x86_64 macOS after libm kernels; missing R_RISCV_BRANCH (riscv64), TOC relocs (ppc64le)
  // and R_ARM_REL32 (arm), which PIC uses for every string literal reference
  const auto elfArch = hostTriple.isOSBinFormatELF() ? hostTriple.getArch() : llvm::Triple::UnknownArch;
  const bool useJITLink = hostTriple.isOSBinFormatMachO() || elfArch == llvm::Triple::riscv64 || elfArch == llvm::Triple::ppc64le
                          || elfArch == llvm::Triple::arm || elfArch == llvm::Triple::thumb;
  if (useJITLink) {
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

  const std::pair<const char *, llvm::orc::ExecutorAddr> runtimeSymbols[] = {
      {"malloc", llvm::orc::ExecutorAddr::fromPtr(&malloc_)},
      {"__polyregion_host_thread_global_idx", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_thread_global_idx)},
      {"__polyregion_host_thread_global_size", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_thread_global_size)},
      {"__polyregion_host_local_idx", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_local_idx)},
      {"__polyregion_host_local_size", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_local_size)},
      {"__polyregion_host_group_idx", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_group_idx)},
      {"__polyregion_host_group_size", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_group_size)},
      {"__polyregion_host_group_barrier", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_group_barrier)},
      {"__polyregion_host_group_local", llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_group_local)},
      {polyregion::conventions::HostGroupState, llvm::orc::ExecutorAddr::fromPtr(&__polyregion_host_group_state)},
  };
  llvm::orc::SymbolMap absSyms;
  for (const auto &[name, addr] : runtimeSymbols)
    absSyms[es->intern(withGlobalPrefix(globalPrefix, name))] =
        llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
  if (auto err = processJD->define(llvm::orc::absoluteSymbols(std::move(absSyms))))
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot define runtime symbols: %s", toString(std::move(err)).c_str());

  // libm::exportAll (from invoke::init) registers sincosf et al. via sys::DynamicLibrary; this generator picks them up
  if (auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(globalPrefix)) processJD->addGenerator(std::move(*gen));
  else POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot create process symbol generator: %s", toString(gen.takeError()).c_str());

  // XXX __truncsfhf2 etc. aren't linked into this process; give the JIT its own compiler-rt builtins
  // archive. The running binary may be a test-dist sibling of the main dist, not inside it, so try
  // POLYREGION_BITCODE_DIR's dist root first, falling back to the running binary's own.
  llvm::SmallVector<std::string, 2> distRoots;
  if (const char *bitcodeDir = std::getenv(polyregion::env::PolyregionBitcodeDir); bitcodeDir && *bitcodeDir)
    distRoots.emplace_back(llvm::sys::path::parent_path(bitcodeDir).str());
  if (const auto exe = llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&withGlobalPrefix)); !exe.empty())
    distRoots.emplace_back(llvm::sys::path::parent_path(llvm::sys::path::parent_path(exe)).str());
  for (const auto &root : distRoots) {
    llvm::SmallString<256> builtinsPath(root);
    llvm::sys::path::append(builtinsPath, "lib", "clang", std::to_string(LLVM_VERSION_MAJOR), "lib");
    llvm::sys::path::append(builtinsPath, hostTriple.str(), "libclang_rt.builtins.a");
    if (!llvm::sys::fs::exists(builtinsPath)) continue;
    if (auto gen = llvm::orc::StaticLibraryDefinitionGenerator::Load(*ol, builtinsPath.c_str())) {
      processJD->addGenerator(std::move(*gen));
      break;
    } else
      fmt::print(stderr, "[{}] Cannot load compiler-rt builtins at {}: {}\n", RELOBJ_PREFIX, builtinsPath.c_str(),
                 toString(gen.takeError()));
  }
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
  return jitMemMgr ? "RelocatableObjectDevice(llvm::orc::ObjectLinkingLayer)"
                   : "RelocatableObjectDevice(llvm::orc::RTDyldObjectLinkingLayer)";
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
    // default MatchExportedSymbolsOnly hides archive-materialised symbols (e.g. compiler-rt builtins)
    jd.addToLinkOrder(*processJD, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);

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

static size_t policyTasks(const Policy &policy) { return policy.global.x * (policy.local ? policy.local->first.x : 1); }

void validatePolicyAndArgs(const char *prefix, std::vector<Type> types, const Policy &policy) {
  if (auto scratchCount = types ^ count([](Type t) { return t == Type::Scratch; }); scratchCount != 0)
    POLYINVOKE_FATAL(prefix, "Scratch types are not supported on the CPU, found %td arg(s)", scratchCount);
  if (policy.global.y != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Y > 1 is not supported: %zu", policy.global.y);
  if (policy.global.z != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Z > 1 is not supported: %zu", policy.global.z);
  if (policy.local && (policy.local->first.y != 1 || policy.local->first.z != 1))
    POLYINVOKE_FATAL(prefix, "Policy local dimension Y or Z > 1 is not supported: %zu, %zu", policy.local->first.y, policy.local->first.z);
  if (types[0] != Type::IntS64)
    POLYINVOKE_FATAL(prefix, "Expecting first argument as index (%s), but was %s", magic_enum::enum_name(Type::IntS64).data(),
                     magic_enum::enum_name(types[0]).data());
}
static size_t staticLocalBytes(const uint64_t staticBytesAddr) {
  return staticBytesAddr ? *reinterpret_cast<const uint64_t *>(staticBytesAddr) : 0;
}
static void validateLocalStorage(const char *prefix, const Policy &policy, const uint64_t staticBytesAddr) {
  const size_t staticBytes = staticLocalBytes(staticBytesAddr);
  const size_t dynamicBytes = policy.local ? policy.local->second : 0;
  if (staticBytes + dynamicBytes > HostGroupLocalBytes)
    POLYINVOKE_FATAL(prefix, "work-group storage of %zu static and %zu dynamic bytes exceeds the host capacity of %zu bytes", staticBytes,
                     dynamicBytes, HostGroupLocalBytes);
}
void ObjectDeviceQueue::launchHost(const char *prefix, const Policy &policy, const uint64_t abiAddr, const uint64_t staticLocalBytesAddr,
                                   const MaybeCallback &cb, std::function<void(size_t)> task) {
  const int64_t abi = abiAddr ? *reinterpret_cast<const int64_t *>(abiAddr) : polyregion::program_meta::HostAbiLane;
  switch (abi) {
    case polyregion::program_meta::HostAbiGroup: {
      const size_t lanes = policy.local ? policy.local->first.x : 1;
      const size_t localBytes = staticLocalBytes(staticLocalBytesAddr) + (policy.local ? policy.local->second : 0);
      return groupAbiLaunch(policy.global.x, lanes, localBytes, cb, std::move(task));
    }
    case polyregion::program_meta::HostAbiFlat: return threadedLaunch(policyTasks(policy), cb, std::move(task));
    case polyregion::program_meta::HostAbiLane:
      if (groupedPolicy(policy)) return groupedLaunch(policy.global.x, policy.local->first.x, cb, std::move(task));
      return threadedLaunch(policyTasks(policy), cb, std::move(task));
    default: POLYINVOKE_FATAL(prefix, "unknown host abi %lld", static_cast<long long>(abi));
  }
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
  const auto resolve = [&](const std::string &name, const bool required) -> uint64_t {
    ReadLock r(obj->symbolCacheMutex);
    if (const auto it = obj->symbolCache.find(name); it != obj->symbolCache.end()) return it->second;
    r.unlock();
    WriteLock w(obj->symbolCacheMutex);
    if (const auto it = obj->symbolCache.find(name); it != obj->symbolCache.end()) return it->second;
    const auto fnName = withGlobalPrefix(globalPrefix, name);
    auto symOrErr = es.lookup({obj->jd}, fnName);
    uint64_t address = 0;
    if (symOrErr) address = symOrErr->getAddress().getValue();
    else if (required)
      POLYINVOKE_FATAL(RELOBJ_PREFIX, "Symbol `%s` not found in the given object: %s", fnName.c_str(),
                       toString(symOrErr.takeError()).c_str());
    else llvm::consumeError(symOrErr.takeError());
    obj->symbolCache.emplace(name, address);
    return address;
  };
  const auto symAddr = resolve(symbol, true);
  const auto staticBytesAddr = resolve(symbol + polyregion::program_meta::HostStaticLocalBytesSuffix, false);
  validateLocalStorage(RELOBJ_PREFIX, policy, staticBytesAddr);
  const auto abiAddr = resolve(symbol + polyregion::program_meta::HostAbiSuffix, false);
  const auto done = [cb, token = latch.acquire()]() {
    if (cb) (*cb)();
  };
  const auto task = [symAddr, types, argData, global = policyTasks(policy)](size_t tid) {
    hostThreadGlobalIdx = tid;
    hostThreadGlobalSize = global;
    auto argData_ = argData;
    auto argPtrs = detail::argDataAsPointers(types, argData_);
    if (types[0] != Type::IntS64) {
      POLYINVOKE_FATAL(RELOBJ_PREFIX, "Expecting first argument as index: %s, but was %s", magic_enum::enum_name(Type::IntS64).data(),
                       magic_enum::enum_name(types[0]).data());
    }
    auto _tid = int64_t(tid);
    argPtrs[0] = &_tid;
    ffiInvoke(RELOBJ_PREFIX, symAddr, types, argPtrs);
  };
  launchHost(RELOBJ_PREFIX, policy, abiAddr, staticBytesAddr, done, task);
}

static constexpr const char *SHOBJ_PREFIX = "SharedObject";

struct HostTable {
#define POLYREGION_HOST_TABLE_FIELD(name, ret, params) ret(*name) params;
  POLYREGION_HOST_TABLE(POLYREGION_HOST_TABLE_FIELD)
#undef POLYREGION_HOST_TABLE_FIELD
};

static void fillHostTable(polyregion_dl_handle handle) {
  if (auto *table = reinterpret_cast<HostTable *>(polyregion_dl_find(handle, polyregion::program_meta::HostTableSymbol))) {
#define POLYREGION_HOST_TABLE_FILL(name, ret, params) table->name = &__polyregion_host_##name;
    POLYREGION_HOST_TABLE(POLYREGION_HOST_TABLE_FILL)
#undef POLYREGION_HOST_TABLE_FILL
  }
}

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

SharedDevice::SharedDevice() {
  POLYINVOKE_TRACE();
  auto located = locateCCompiler();
  compiler = std::move(located.path);
  compilerUnavailable = std::move(located.unavailable);
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
ModuleFormat SharedDevice::moduleFormat() {
  POLYINVOKE_TRACE();
  return compiler.empty() ? ModuleFormat::Object : ModuleFormat::Source;
}
std::vector<Property> SharedDevice::properties() {
  POLYINVOKE_TRACE();
  return {{"c-compiler", compiler.empty() ? "unavailable: " + compilerUnavailable : compiler}};
}
void SharedDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  if (auto it = modules.find(name); it != modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "Module named %s was already loaded", name.c_str());
  else {
    std::string built;
    if (llvm::identify_magic(image) == llvm::file_magic::unknown) {
      if (compiler.empty())
        POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot build C source module %s: %s", name.c_str(), compilerUnavailable.c_str());
      try {
        built = buildSharedObject({compiler, {}}, image);
      } catch (const std::exception &e) {
        POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot build C source module %s: %s", name.c_str(), e.what());
      }
    }
    const std::string &shared = built.empty() ? image : built;

    // TODO fileless load: Linux memfd_create, Windows MemoryModule; dlopen needs a real file for now
    llvm::SmallString<256> model;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, model);
    llvm::sys::path::append(model, "polyinvoke-shobj-%%%%%%%%");
    llvm::SmallString<256> tmpPath;
    int objectFd = -1;
    if (const auto ec = llvm::sys::fs::createUniqueFile(model, objectFd, tmpPath)) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Unable to buffer image to file for %s, tmpfile creation failed: %s", name.c_str(),
                       ec.message().c_str());
    }
    {
      llvm::raw_fd_ostream objectFile(objectFd, /*shouldClose=*/true);
      objectFile.write(shared.data(), shared.size());
    }

    auto *dylib = polyregion_dl_open(tmpPath.c_str());
#ifdef _WIN32
    static std::vector<std::string> tmpImagePaths;
    static std::mutex cleanupMutex;
    {
      std::unique_lock<std::mutex> lock(cleanupMutex);
      if (tmpImagePaths.empty()) {
        std::atexit([]() {
          std::unique_lock<std::mutex> lock(cleanupMutex);
          for (auto &path : tmpImagePaths)
            if (std::remove(path.c_str()) != 0)
              fmt::print(stderr, "[{}] Warning: cannot remove temporary image file {}\n", SHOBJ_PREFIX, path);
        });
      }
      tmpImagePaths.emplace_back(tmpPath.str());
    }
#else
    llvm::sys::fs::remove(tmpPath);
#endif

    if (!dylib) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load module: %s", polyregion_dl_error());
    } else {
      fillHostTable(dylib);
      modules.emplace_hint(it, name, details::LoadedModule{image, dylib, {}});
    }
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
static uint64_t sharedSymbolAddress(const std::string &moduleName, polyregion::invoke::object::details::LoadedModule &loaded,
                                    const std::string &symbol, const bool required) {
  auto &[image, handle, symbolTable] = loaded;
  if (const auto it = symbolTable.find(symbol); it != symbolTable.end()) return it->second;
  const auto address = reinterpret_cast<uintptr_t>(polyregion_dl_find(handle, symbol.c_str()));
  if (!address && required) {
    POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load symbol %s from module %s (%zu bytes): %s", symbol.c_str(), moduleName.c_str(), image.size(),
                     polyregion_dl_error());
  }
  symbolTable.emplace(symbol, address);
  return address;
}

void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                           std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  validatePolicyAndArgs(SHOBJ_PREFIX, types, policy);

  ReadLock r(mutex);
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "No module named %s was loaded", moduleName.c_str());

  const auto staticBytesAddr =
      sharedSymbolAddress(moduleName, moduleIt->second, symbol + polyregion::program_meta::HostStaticLocalBytesSuffix, false);
  validateLocalStorage(SHOBJ_PREFIX, policy, staticBytesAddr);
  const auto abiAddr = sharedSymbolAddress(moduleName, moduleIt->second, symbol + polyregion::program_meta::HostAbiSuffix, false);
  const auto address = sharedSymbolAddress(moduleName, moduleIt->second, symbol, true);

  const auto done = [cb, token = latch.acquire()]() {
    if (cb) (*cb)();
  };
  const auto task = [address, types, argData, global = policyTasks(policy)](size_t tid) {
    hostThreadGlobalIdx = tid;
    hostThreadGlobalSize = global;
    auto argData_ = argData;
    auto argPtrs = detail::argDataAsPointers(types, argData_);
    auto _tid = int64_t(tid);
    argPtrs[0] = &_tid;
    ffiInvoke(SHOBJ_PREFIX, address, types, argPtrs);
  };
  launchHost(SHOBJ_PREFIX, policy, abiAddr, staticBytesAddr, done, task);
  POLYINVOKE_TRACE();
}
