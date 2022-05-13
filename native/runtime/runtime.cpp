#include "runtime.h"

#include <iostream>
#include <utility>

#include "cl_runtime.h"
#include "cuda_runtime.h"
#include "hip_runtime.h"

#include "ffi.h"
#include "libm.h"
#include "utils.hpp"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace polyregion;

void polyregion::runtime::init() { polyregion::libm::exportAll(); }

static const char *clKernelSource{R"CLC(
__kernel void _Z6squarePii(__global int* array, int x){
  int tid = get_global_id(0);
  array[tid] = array[tid] + tid + x;
}
)CLC"};

static const char *ptxKernelSource{R"PTX(
.version 7.4
.target sm_61
.address_size 64

        // .globl       _Z6squarePii

.visible .entry _Z6squarePii(
        .param .u64 _Z6squarePii_param_0,
        .param .u32 _Z6squarePii_param_1
)
{
        .reg .b32       %r<6>;
        .reg .b64       %rd<5>;


        ld.param.u64    %rd1, [_Z6squarePii_param_0];
        ld.param.u32    %r1, [_Z6squarePii_param_1];
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r2, %ctaid.x;
        mul.wide.s32    %rd3, %r2, 4;
        add.s64         %rd4, %rd2, %rd3;
        ld.global.u32   %r3, [%rd4];
        add.s32         %r4, %r2, %r1;
        add.s32         %r5, %r4, %r3;
        st.global.u32   [%rd4], %r5;
        ret;

}
)PTX"};

static const char *hsacoKernelSource{R"PTX(

)PTX"};

void polyregion::runtime::run() {

  using namespace runtime::cuda;
  using namespace runtime::hip;
  using namespace runtime::cl;

  std::vector<std::unique_ptr<Runtime>> rts;

  try {
    rts.push_back(std::make_unique<CudaRuntime>());
  } catch (const std::exception &e) {
    std::cerr << "[CUDA] " << e.what() << std::endl;
  }

  try {
    rts.push_back(std::make_unique<ClRuntime>());
  } catch (const std::exception &e) {
    std::cerr << "[OCL] " << e.what() << std::endl;
  }

  try {
    rts.push_back(std::make_unique<HipRuntime>());
  } catch (const std::exception &e) {
    std::cerr << "[HIP] " << e.what() << std::endl;
  }

  static std::vector<int> xs;

  for (auto &rt : rts) {
    std::cout << "RT=" << rt->name() << std::endl;
    auto devices = rt->enumerate();
    std::cout << "Found " << devices.size() << " devices" << std::endl;

    for (auto &d : devices) {
      std::cout << d->id() << " = " << d->name() << std::endl;

      if (d->name().find("TITAN") != std::string::npos) {
        //        continue ;
      }

      xs = {1, 2, 3, 4};

      auto size = sizeof(decltype(xs)::value_type) * xs.size();
      auto ptr = d->malloc(size, Access::RW);

      std::string src;
      if (rt->name() == "CUDA") {
        src = ptxKernelSource;
      } else if (rt->name() == "OpenCL") {
        src = clKernelSource;
      } else if (rt->name() == "HIP") {
        src = hsacoKernelSource;
      } else {
        throw std::logic_error("?");
      }

      d->loadKernel(src);

      d->enqueueHostToDeviceAsync(xs.data(), ptr, size, []() { std::cout << "  H->D ok" << std::endl; });

      int32_t x = 4;
      d->enqueueKernelAsync("_Z6squarePii", {{Type::Ptr, &ptr}, {Type::Int32, &x}}, {xs.size(), 1, 1}, {1, 1, 1},
                            []() { std::cout << "  K 1 ok" << std::endl; });

      x = 5;

      d->enqueueKernelAsync("_Z6squarePii", {{Type::Ptr, &ptr}, {Type::Int32, &x}}, {xs.size(), 1, 1}, {1, 1, 1},
                            []() { std::cout << "  K 2 ok" << std::endl; });
      d->enqueueDeviceToHostAsync(ptr, xs.data(), size, [&]() {
        std::cout << "  D->H ok, r= "
                  << polyregion::mk_string<int>(
                         xs, [](auto x) { return std::to_string(x); }, ",")
                  << std::endl;
      });

      std::cout << d->id() << " = Done" << std::endl;

      d->free(ptr);
    }
  }

  std::cout << "Done" << std::endl;
}

struct runtime::Data {
  std::unique_ptr<llvm::object::ObjectFile> file;
  explicit Data(std::unique_ptr<llvm::object::ObjectFile> file) : file(std::move(file)) {}
};

polyregion::runtime::Object::~Object() = default;

polyregion::runtime::Object::Object(const std::vector<uint8_t> &object) {
  llvm::MemoryBufferRef buffer(llvm::StringRef(reinterpret_cast<const char *>(object.data()), object.size()), "");
  auto objfile = llvm::object::ObjectFile::createObjectFile(buffer);
  if (auto e = objfile.takeError()) {
    throw std::logic_error(toString(std::move(e)));
  } else {
    this->data = std::make_unique<runtime::Data>(std::move(*objfile));
  }
}

std::vector<std::pair<std::string, uint64_t>> polyregion::runtime::Object::enumerate() const {
  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*data->file);
  auto table = ld.getSymbolTable();
  std::vector<std::pair<std::string, uint64_t>> result(table.size());
  std::transform(table.begin(), table.end(), result.begin(),
                 [](auto &p) { return std::make_pair(p.first, p.second.getAddress()); });
  return result;
}

constexpr static ffi_type *toFFITpe(const runtime::Type &tpe) {
  switch (tpe) {
  case polyregion::runtime::Type::Bool8:
  case polyregion::runtime::Type::Byte8:
    return &ffi_type_sint8;
  case polyregion::runtime::Type::CharU16:
    return &ffi_type_uint16;
  case polyregion::runtime::Type::Short16:
    return &ffi_type_sint16;
  case polyregion::runtime::Type::Int32:
    return &ffi_type_sint32;
  case polyregion::runtime::Type::Long64:
    return &ffi_type_sint64;
  case polyregion::runtime::Type::Float32:
    return &ffi_type_float;
  case polyregion::runtime::Type::Double64:
    return &ffi_type_double;
  case polyregion::runtime::Type::Ptr:
    return &ffi_type_pointer;
  case polyregion::runtime::Type::Void:
    return &ffi_type_void;
  default:
    return nullptr;
  }
}

thread_local static std::function<void *(size_t)> threadLocalMallocFn;
EXPORT static void *polyregion::runtime::_malloc(size_t size) { return threadLocalMallocFn(size); }

class EXPORT ThreadLocalMallocFnMemoryManager : public llvm::SectionMemoryManager {

public:
  explicit ThreadLocalMallocFnMemoryManager(MemoryMapper *mm = nullptr) : SectionMemoryManager(mm) {}

private:
  uint64_t getSymbolAddress(const std::string &Name) override {
    //    std::cout << "External call `" << Name << "`" << std::endl;
    return Name == "malloc" ? (uint64_t)&polyregion::runtime::_malloc : RTDyldMemoryManager::getSymbolAddress(Name);
  }
};

void polyregion::runtime::Object::invoke(const std::string &symbol,                  //
                                         const std::function<void *(size_t)> &alloc, //
                                         const std::vector<TypedPointer> &args,      //
                                         runtime::TypedPointer rtn) const {

  static_assert(sizeof(uint8_t) == sizeof(char));
  threadLocalMallocFn = alloc;

  ThreadLocalMallocFnMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*data->file);

  auto fnName = (data->file->isMachO() || data->file->isMachOUniversalBinary()) ? std::string("_") + symbol : symbol;
  auto sym = ld.getSymbol(fnName);

  if (!sym) {
    auto table = ld.getSymbolTable();
    auto symbols = polyregion::mk_string2<llvm::StringRef, llvm::JITEvaluatedSymbol>(
        table, [](auto &x) { return "[`" + x.first.str() + "`@" + polyregion::hex(x.second.getAddress()) + "]"; }, ",");
    throw std::logic_error("Symbol `" + std::string(fnName) + "` not found in the given object, available symbols (" +
                           std::to_string(table.size()) + ") = " + symbols);
  }

  ld.finalizeWithMemoryManagerLocking();

  if (ld.hasError()) {
    throw std::logic_error("Symbol `" + std::string(symbol) +
                           "` failed to finalise for execution: " + ld.getErrorString().str());
  }

  auto rtnFFIType = toFFITpe(rtn.first);
  if (!rtnFFIType) {
    throw std::logic_error("Illegal return type " + std::to_string(polyregion::to_underlying(rtn.first)));
  }

  std::vector<void *> argPointers(args.size());
  std::vector<ffi_type *> argsFFIType(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    argPointers[i] = args[i].second;
    argsFFIType[i] = toFFITpe(args[i].first);
    if (!argsFFIType[i])
      throw std::logic_error("Illegal parameter type on arg " + std::to_string(i) + ": " +
                             std::to_string(polyregion::to_underlying(args[i].first)));
  }

  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size(), rtnFFIType, argsFFIType.data());
  switch (status) {
  case FFI_OK:
    ffi_call(&cif, FFI_FN(sym.getAddress()), rtn.second, argPointers.data());
    break;
  case FFI_BAD_TYPEDEF:
    throw std::logic_error("ffi_prep_cif: FFI_BAD_TYPEDEF");
  case FFI_BAD_ABI:
    throw std::logic_error("ffi_prep_cif: FFI_BAD_ABI");
  default:
    throw std::logic_error("ffi_prep_cif: unknown error (" + std::to_string(status) + ")");
  }
}
void *polyregion::runtime::CountedCallbackHandler::createHandle(const runtime::Callback &cb) {
  auto eventId = eventCounter++;
  auto f = [=, this]() {
    cb();
    callbacks.erase(eventId);
  };
  auto pos = callbacks.emplace(eventId, f).first;
  // just to be sure
  static_assert(std::is_same<EntryPtr, decltype(&(*pos))>());
  return &(*pos);
}
void polyregion::runtime::CountedCallbackHandler::consume(void *data) {
  auto dev = static_cast<EntryPtr>(data);
  if (dev) dev->second();
}
void polyregion::runtime::CountedCallbackHandler::clear() { callbacks.clear(); }
