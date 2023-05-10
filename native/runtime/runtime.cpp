#include "runtime.h"

#include <mutex>
#include <utility>

#include "libm.h"

#include "cl_platform.h"
#include "cuda_platform.h"
#include "hip_platform.h"
#include "hsa_platform.h"
#include "metal_platform.h"
#include "object_platform.h"
#include "vulkan_platform.h"

using namespace polyregion;

template <typename F, typename It>
constexpr static void insertIntoBufferAt(runtime::ArgBuffer &buffer, It begin, It end, F f) {
  for (auto it = begin; it != end; ++it) {
    auto &[tpe, ptr] = *it;
    buffer.types.insert(f(buffer.types), tpe);
    if (auto size = byteOfType(tpe); size != 0) {
      if (tpe == runtime::Type::Scratch || !ptr) {
        buffer.data.insert(f(buffer.data), size, std::byte(0));
      } else {
        auto offset = static_cast<std::byte *>(ptr);
        buffer.data.insert(f(buffer.data), offset, offset + size);
      }
    }
  }
}

runtime::ArgBuffer::ArgBuffer(std::initializer_list<TypedPointer> args) { append(args); }
void runtime::ArgBuffer::append(Type tpe, void *ptr) { append({{tpe, ptr}}); }
void runtime::ArgBuffer::append(const ArgBuffer &that) {
  data.insert(data.end(), that.data.begin(), that.data.end());
  types.insert(types.end(), that.types.begin(), that.types.end());
}
void runtime::ArgBuffer::append(std::initializer_list<TypedPointer> args) {
  insertIntoBufferAt(*this, args.begin(), args.end(), [](auto &it) { return it.end(); });
}
void runtime::ArgBuffer::prepend(Type tpe, void *ptr) { prepend({{tpe, ptr}}); }
void runtime::ArgBuffer::prepend(const ArgBuffer &that) {
  data.insert(data.begin(), that.data.begin(), that.data.end());
  types.insert(types.begin(), that.types.begin(), that.types.end());
}
void runtime::ArgBuffer::prepend(std::initializer_list<TypedPointer> args) {
  // begin always inserts to the front, so we traverse it backwards to keep the same order
  insertIntoBufferAt(*this, rbegin(args), rend(args), [](auto &it) { return it.begin(); });
}

void runtime::init() { libm::exportAll(); }

std::unique_ptr<runtime::Platform> runtime::Platform::of(const runtime::Backend &b) {
  using namespace polyregion::runtime;
  switch (b) {
    case Backend::CUDA: return std::make_unique<cuda::CudaPlatform>();
    case Backend::HIP: return std::make_unique<hip::HipPlatform>();
    case Backend::HSA: return std::make_unique<hsa::HsaPlatform>();
    case Backend::OpenCL: return std::make_unique<cl::ClPlatform>();
    case Backend::Vulkan: return std::make_unique<vulkan::VulkanPlatform>();
    case Backend::Metal:
#ifdef RUNTIME_ENABLE_METAL
      return std::make_unique<metal::MetalPlatform>();
#else
      throw std::logic_error("Metal backend not available");
#endif
    case Backend::SHARED_OBJ: return std::make_unique<object::SharedPlatform>();
    case Backend::RELOCATABLE_OBJ: return std::make_unique<object::RelocatablePlatform>();
  }
}

runtime::detail::CountingLatch::Token::Token(runtime::detail::CountingLatch &latch) : latch(latch) { TRACE(); }
runtime::detail::CountingLatch::Token::~Token() {
  TRACE();
  latch.pending--;
  latch.cv.notify_all();
}
std::shared_ptr<runtime::detail::CountingLatch::Token> runtime::detail::CountingLatch::acquire() {
  TRACE();
  pending++;
  cv.notify_all();
  return std::make_shared<Token>(*this);
}
runtime::detail::CountingLatch::~CountingLatch() {
  TRACE();
  auto now = std::chrono::system_clock::now();
  std::unique_lock<std::mutex> lock(mutex);
  if (!cv.wait_until(lock, now + std::chrono::seconds(10), [&]() { return pending == 0; })) {
    throw std::logic_error("Timed out with " + std::to_string(pending) + " pending latches");
  }
}

void *runtime::detail::CountedCallbackHandler::createHandle(const runtime::Callback &cb) {

  // XXX We're storing the callbacks statically to extend lifetime because the callback behaviour on different runtimes
  // is not predictable, some may transfer control back even after destruction of all related context.
  static std::atomic_uint64_t eventCounter = 0;
  static Storage callbacks;
  static std::mutex lock;

  auto eventId = eventCounter++;
  auto f = [cb, eventId]() {
    cb();
    const std::lock_guard<std::mutex> guard(lock);
    callbacks.erase(eventId);
  };

  const std::lock_guard<std::mutex> guard(lock);
  auto pos = callbacks.emplace(eventId, f).first;
  // just to be sure
  static_assert(std::is_same<EntryPtr, decltype(&(*pos))>());
  return &(*pos);
}
void runtime::detail::CountedCallbackHandler::consume(void *data) {
  auto dev = static_cast<EntryPtr>(data);
  if (dev) dev->second();
}

std::string runtime::detail::allocateAndTruncate(const std::function<void(char *, size_t)> &f, size_t length) {
  std::string xs(length, '\0');
  f(xs.data(), xs.length() - 1);
  xs.erase(xs.find('\0'));
  return xs;
}

std::vector<void *> runtime::detail::argDataAsPointers(const std::vector<Type> &types,
                                                       std::vector<std::byte> &argData) {
  std::byte *argsPtr = argData.data();
  std::vector<void *> argsPtrStore(types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    argsPtrStore[i] = types[i] == Type::Void ? nullptr : argsPtr;
    argsPtr += byteOfType((types[i]));
  }
  return argsPtrStore;
}
std::ostream &runtime::operator<<(std::ostream &os, const runtime::Dim3 &dim3) {
  return os << "Dim3{x: " << dim3.x << " y: " << dim3.y << " z: " << dim3.z << "}";
}
