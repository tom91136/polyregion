#include "dso_runner.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/dl.h"
#include "polyregion/env.h"
#include "polyregion/polypackage.h"
#include "polyregion/polypackage_symbols.h"
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
  polypackage::abi::AbiVersionFn packageAbi = nullptr;
  polypackage::abi::LinkPackageFn linkPackage = nullptr;
  polypackage::abi::LinkProgramFn linkProgram = nullptr;
  polypackage::abi::LastErrorFn packageErr = nullptr;
  polypackage::abi::FreeFn packageFree = nullptr;
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
  namespace packageAbi = polypackage::abi;
  impl->packageAbi = reinterpret_cast<packageAbi::AbiVersionFn>(polyregion_dl_find(impl->dso, packageAbi::AbiVersion));
  impl->linkPackage = reinterpret_cast<packageAbi::LinkPackageFn>(polyregion_dl_find(impl->dso, packageAbi::LinkPackage));
  impl->linkProgram = reinterpret_cast<packageAbi::LinkProgramFn>(polyregion_dl_find(impl->dso, packageAbi::LinkProgram));
  impl->packageErr = reinterpret_cast<packageAbi::LastErrorFn>(polyregion_dl_find(impl->dso, packageAbi::LastError));
  impl->packageFree = reinterpret_cast<packageAbi::FreeFn>(polyregion_dl_find(impl->dso, packageAbi::Free));

  const bool anyPass = impl->abi || impl->count || impl->name || impl->descr || impl->run || impl->err || impl->free;
  const bool completePass = impl->abi && impl->count && impl->name && impl->run && impl->err && impl->free;
  if (anyPass && !completePass) return fmt::format("dlsym polypass_*: incomplete pass capability in {}", impl->path);
  const bool anyPackage = impl->packageAbi || impl->linkPackage || impl->linkProgram || impl->packageErr || impl->packageFree;
  const bool completePackage = impl->packageAbi && impl->linkPackage && impl->linkProgram && impl->packageErr && impl->packageFree;
  if (anyPackage && !completePackage) return fmt::format("dlsym polypackage_*: incomplete package capability in {}", impl->path);
  if (!completePass && !completePackage) return fmt::format("{} exposes neither polypass_* nor polypackage_*", impl->path);

  if (completePass) {
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
  auto raw = steps ^ map([](const auto &s) { return s.c_str(); });
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

std::optional<uint32_t> DsoPassRunner::packageAbiVersion() const {
  if (!impl->loaded || !impl->packageAbi) return std::nullopt;
  return impl->packageAbi();
}

Vector<uint8_t> DsoPassRunner::runPackage(const std::string_view operation, const Vector<uint8_t> &request, String &error) {
  if (!impl->loaded) {
    error = "DSO not loaded; call load() first";
    return {};
  }
  const auto invoke = operation == polypackage::abi::LinkPackage   ? impl->linkPackage
                      : operation == polypackage::abi::LinkProgram ? impl->linkProgram
                                                                   : nullptr;
  if (!invoke) {
    error = fmt::format("unknown or unavailable package operation {}", operation);
    return {};
  }
  uint8_t *output = nullptr;
  size_t outputSize = 0;
  const auto status = invoke(request.data(), request.size(), &output, &outputSize);
  if (status != POLYPACKAGE_OK) {
    const char *message = impl->packageErr();
    error = message ? message : fmt::format("package operation returned status {}", static_cast<int>(status));
    if (output) impl->packageFree(output);
    return {};
  }
  if (outputSize > 0 && !output) {
    error = "package operation returned a null result with non-zero size";
    return {};
  }
  Vector<uint8_t> result;
  if (outputSize > 0) result.assign(output, output + outputSize);
  if (output) impl->packageFree(output);
  return result;
}

std::string_view DsoPassRunner::tag() const { return impl->tag; }

} // namespace polyregion::polypass
