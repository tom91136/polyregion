#include "runtime.h"

#include <mutex>
#include <utility>

#include "libm.h"

using namespace polyregion;

void polyregion::runtime::init() { polyregion::libm::exportAll(); }

void *polyregion::runtime::detail::CountedCallbackHandler::createHandle(const runtime::Callback &cb) {

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
void polyregion::runtime::detail::CountedCallbackHandler::consume(void *data) {
  auto dev = static_cast<EntryPtr>(data);
  if (dev) dev->second();
}
