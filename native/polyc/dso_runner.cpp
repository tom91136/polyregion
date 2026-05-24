#include "dso_runner.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/dl.h"
#include "polyregion/env.h"
#include "polyregion/polypass.h"

#include "generated/polypass_symbols.h"

namespace polyregion::polypass {

using namespace aspartame;

struct DsoPassRunner::Impl {
  std::string path;
  std::string tag;
  polyregion_dl_handle dso = nullptr;
  abi::AbiVersionFn abi = nullptr;
  abi::PassCountFn count = nullptr;
  abi::PassNameFn name = nullptr;
  abi::PassDescrFn descr = nullptr;
  abi::RunPassesFn run = nullptr;
  abi::LastErrorFn err = nullptr;
  abi::FreeFn free = nullptr;
  Vector<String> names;
  std::unordered_map<String, String> descrs;
  bool loaded = false;
};

DsoPassRunner::DsoPassRunner(std::string path) : impl(std::make_unique<Impl>()) {
  impl->path = std::move(path);
  impl->tag = "PolyPass[" + impl->path + "]";
}
DsoPassRunner::~DsoPassRunner() = default;

String DsoPassRunner::load() {
  if (impl->loaded) return {};
  // XXX set before dlopen: Boehm's GC_INIT runs from the DSO's .init_array.
  env::put("GC_INITIAL_HEAP_SIZE", "512M", false);
  env::put("GC_FREE_SPACE_DIVISOR", "1", false);
  impl->dso = polyregion_dl_open(impl->path.c_str());
  if (!impl->dso) {
    const char *e = polyregion_dl_error();
    return fmt::format("dlopen({}): {}", impl->path, e ? e : "<no error>");
  }
  impl->abi = reinterpret_cast<abi::AbiVersionFn>(polyregion_dl_find(impl->dso, abi::AbiVersion));
  impl->count = reinterpret_cast<abi::PassCountFn>(polyregion_dl_find(impl->dso, abi::PassCount));
  impl->name = reinterpret_cast<abi::PassNameFn>(polyregion_dl_find(impl->dso, abi::PassName));
  impl->descr = reinterpret_cast<abi::PassDescrFn>(polyregion_dl_find(impl->dso, abi::PassDescr));
  impl->run = reinterpret_cast<abi::RunPassesFn>(polyregion_dl_find(impl->dso, abi::RunPasses));
  impl->err = reinterpret_cast<abi::LastErrorFn>(polyregion_dl_find(impl->dso, abi::LastError));
  impl->free = reinterpret_cast<abi::FreeFn>(polyregion_dl_find(impl->dso, abi::Free));
  if (!impl->abi || !impl->count || !impl->name || !impl->run || !impl->err || !impl->free)
    return fmt::format("dlsym polypass_*: missing entry point in {}", impl->path);
  if (const uint32_t v = impl->abi(); v != POLYPASS_ABI_VERSION)
    return fmt::format("PolyPass ABI mismatch in {}: plugin={} polyc={}", impl->path, v, POLYPASS_ABI_VERSION);
  const size_t n = impl->count();
  impl->names.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const char *nm = impl->name(i);
    if (!nm) return fmt::format("polypass_pass_name({}) returned NULL in {}", i, impl->path);
    impl->names.emplace_back(nm);
    if (impl->descr)
      if (const char *d = impl->descr(i); d && *d) impl->descrs.emplace(impl->names.back(), d);
  }
  impl->loaded = true;
  return {};
}

const Vector<String> &DsoPassRunner::passNames() const { return impl->names; }

std::optional<String> DsoPassRunner::passDescr(std::string_view name) const {
  const auto it = impl->descrs.find(String(name));
  if (it == impl->descrs.end()) return std::nullopt;
  return it->second;
}

Vector<uint8_t> DsoPassRunner::runPasses(const Vector<String> &steps, const Vector<uint8_t> &programBytes, String &error) {
  if (!impl->loaded) {
    error = "DSO not loaded; call load() first";
    return {};
  }
  // XXX steps as a NUL-terminated C-string array, as the ABI expects.
  auto raw = steps | map([](const auto &s) { return s.c_str(); }) | to_vector();
  raw.push_back(nullptr);
  uint8_t *out = nullptr;
  size_t out_len = 0;
  const ::polypass_status_t rc = impl->run(raw.data(), programBytes.data(), programBytes.size(), &out, &out_len);
  if (rc != POLYPASS_OK) {
    const char *m = impl->err();
    error = fmt::format("polypass_run_passes rc={}: {}", static_cast<int>(rc), m ? m : "<null>");
    if (out) impl->free(out);
    return {};
  }
  Vector<uint8_t> result(out, out + out_len);
  if (out) impl->free(out);
  return result;
}

std::string_view DsoPassRunner::tag() const { return impl->tag; }

} // namespace polyregion::polypass
