#include "polyregion/compat.h"

#include <iostream>
#include <mutex>
#include <utility>

#include "polyrt/cl_platform.h"
#include "polyrt/cuda_platform.h"
#include "polyrt/hip_platform.h"
#include "polyrt/hsa_platform.h"
#include "polyrt/metal_platform.h"
#include "polyrt/object_platform.h"
#include "polyrt/runtime.h"
#include "polyrt/vulkan_platform.h"

#include "libm.h"

using namespace polyregion;

template <typename F, typename It> constexpr static void insertIntoBufferAt(runtime::ArgBuffer &buffer, It begin, It end, F f) {
  for (auto it = begin; it != end; ++it) {
    auto &[tpe, ptr] = *it;
    buffer.types.insert(f(buffer.types), tpe);
    if (auto size = byteOfType(tpe); size != 0) {
      if (tpe == runtime::Type::Scratch || !ptr) {
        buffer.data.insert(f(buffer.data), size, std::byte{});
      } else {
        auto offset = static_cast<std::byte *>(ptr);
        buffer.data.insert(f(buffer.data), offset, offset + size);
      }
    }
  }
}

runtime::ArgBuffer::ArgBuffer(const std::initializer_list<TypedPointer> args) { append(args); }
void runtime::ArgBuffer::append(Type tpe, void *ptr) { append({{tpe, ptr}}); }
void runtime::ArgBuffer::append(const ArgBuffer &that) {
  data.insert(data.end(), that.data.begin(), that.data.end());
  types.insert(types.end(), that.types.begin(), that.types.end());
}
void runtime::ArgBuffer::append(const std::initializer_list<TypedPointer> args) {
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

std::variant<std::string, std::unique_ptr<runtime::Platform>> runtime::Platform::of(const Backend &b) {
  using namespace polyregion::runtime;
  switch (b) {
    case Backend::CUDA: return cuda::CudaPlatform::create();
    case Backend::HIP: return hip::HipPlatform::create();
    case Backend::HSA: return hsa::HsaPlatform::create();
    case Backend::OpenCL: return cl::ClPlatform::create();
    case Backend::Vulkan: return vulkan::VulkanPlatform::create();
    case Backend::Metal:
#ifdef RUNTIME_ENABLE_METAL
      return metal::MetalPlatform::create();
#else
      POLYRT_FATAL("Runtime", "%s backend not available", to_string(b).data());
#endif
    case Backend::SharedObject: return object::SharedPlatform::create();
    case Backend::RelocatableObject: return object::RelocatablePlatform::create();
  }
  return "Backend " + std::string(to_string(b)) + " not available";
}

runtime::detail::CountingLatch::Token::Token(CountingLatch &latch) : latch(latch) { POLYRT_TRACE(); }
runtime::detail::CountingLatch::Token::~Token() {
  POLYRT_TRACE();
  --latch.pending;
  std::lock_guard lock(latch.mutex);
  latch.cv.notify_all();
}
runtime::detail::CountingLatch::CountingLatch(const std::chrono::duration<int64_t> &timeout) : timeout(timeout) { POLYRT_TRACE(); }
std::shared_ptr<runtime::detail::CountingLatch::Token> runtime::detail::CountingLatch::acquire() {
  POLYRT_TRACE();
  ++pending;
  std::lock_guard lock(mutex);
  cv.notify_all();
  return std::make_shared<Token>(*this);
}
bool runtime::detail::CountingLatch::waitAll() {
  POLYRT_TRACE();
  const auto now = std::chrono::system_clock::now();
  std::unique_lock lock(mutex);
  return cv.wait_until(lock, now + timeout, [&]() { return pending == 0; });
}
runtime::detail::CountingLatch::~CountingLatch() {
  POLYRT_TRACE();
  if (!waitAll()) {
    std::cerr << "Timed out with " + std::to_string(pending) + " pending latches" << std::endl;
  }
}

void *runtime::detail::CountedCallbackHandler::createHandle(const Callback &cb) {
  std::lock_guard guard(lock);
  const auto eventId = eventCounter++;
  const auto pos = callbacks.emplace(eventId, cb).first;
  return reinterpret_cast<void *>(pos->first);
}
void runtime::detail::CountedCallbackHandler::consume(void *data) {
  std::lock_guard guard(lock);
  if (const auto it = callbacks.find(reinterpret_cast<uintptr_t>(data)); it != callbacks.end()) {
    it->second();
    callbacks.erase(reinterpret_cast<uintptr_t>(data));
  } else POLYRT_FATAL("Runtime", "Cannot consume %p, callback not found!?", data);
}
runtime::detail::CountedCallbackHandler::~CountedCallbackHandler() { const std::lock_guard guard(lock); }

std::string runtime::detail::allocateAndTruncate(const std::function<void(char *, size_t)> &f, const size_t length) {
  std::string xs(length, '\0');
  f(xs.data(), xs.length() - 1);
  xs.erase(xs.find('\0'));
  return xs;
}

std::vector<void *> runtime::detail::argDataAsPointers(const std::vector<Type> &types, std::vector<std::byte> &argData) {
  std::byte *argsPtr = argData.data();
  std::vector<void *> argsPtrStore(types.size());
  for (size_t i = 0; i < types.size(); ++i) {
    argsPtrStore[i] = types[i] == Type::Void ? nullptr : argsPtr;
    argsPtr += byteOfType(types[i]);
  }
  return argsPtrStore;
}

namespace polyregion::runtime {
std::ostream &operator<<(std::ostream &os, const Dim3 &dim3) {
  return os << "Dim3{x: " << dim3.x << " y: " << dim3.y << " z: " << dim3.z << "}";
}
} // namespace polyregion::runtime
