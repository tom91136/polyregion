#include "runtime.h"

#include <mutex>
#include <utility>

#include "libm.h"

using namespace polyregion;

void runtime::init() { libm::exportAll(); }
std::optional<runtime::Access> runtime::fromUnderlying(uint8_t v) {
  auto x = static_cast<Access>(v);
  switch (x) {
    case Access::RW:
    case Access::RO:
    case Access::WO: return x;
    default: return {};
  }
}

void *runtime::detail::CountedCallbackHandler::createHandle(const runtime::Callback &cb) {

  // XXX We're storing the callbacks statically to extend lifetime because the callback behaviour on different runtimes
  // is not predictable, some may transfer control back even after destruction of all related context.
  static std::atomic_uint64_t eventCounter = 0;
  static Storage callbacks;
  static std::mutex lock;

  auto eventId = eventCounter++;
  auto f = [=]() {
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

std::vector<void *> runtime::detail::pointers(const std::vector<TypedPointer> &args) {
  std::vector<void *> ptrs(args.size());
  for (size_t i = 0; i < args.size(); ++i)
    ptrs[i] = args[i].second;
  return ptrs;
}

std::string runtime::detail::allocateAndTruncate(const std::function<void(char *, size_t)> &f, size_t length) {
  std::string xs(length, '\0');
  f(xs.data(), xs.length() - 1);
  xs.erase(xs.find('\0'));
  return xs;
}
