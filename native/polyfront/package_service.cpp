#include "polyfront/package_service.hpp"

#include <cstdlib>
#include <mutex>
#include <string_view>

#include "fmt/format.h"

#include "polyregion/dl.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/polypackage.h"
#include "polyregion/polypackage_symbols.h"

#include "polyast_codec.h"

namespace polyregion::polyfront::package {
namespace {

namespace abi = polyregion::polypackage::abi;

struct ServiceAbi {
  polyregion_dl_handle handle = nullptr;
  abi::LinkPackageFn linkPackage = nullptr;
  abi::ResolveSymFn resolveSym = nullptr;
  abi::LastErrorFn lastError = nullptr;
  abi::FreeFn free = nullptr;
  std::string error;
  std::mutex callMutex;
};

std::vector<std::string> pluginPaths() {
  const char *raw = std::getenv(env::PolypassPlugins);
  if (!raw || !*raw) return {};
#if defined(_WIN32)
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  std::vector<std::string> paths;
  std::string_view rest(raw);
  while (!rest.empty()) {
    const auto at = rest.find(separator);
    const auto head = rest.substr(0, at);
    if (!head.empty()) paths.emplace_back(head);
    if (at == std::string_view::npos) break;
    rest.remove_prefix(at + 1);
  }
  return paths;
}

ServiceAbi &serviceAbi() {
  static ServiceAbi value;
  static std::once_flag once;
  std::call_once(once, [&] {
    const auto paths = pluginPaths();
    if (paths.empty()) {
      value.error = std::string(env::PolypassPlugins) + " does not name a package-service DSO";
      return;
    }
    env::put("GC_INITIAL_HEAP_SIZE", "512M", false);
    env::put("GC_FREE_SPACE_DIVISOR", "1", false);
    std::vector<std::pair<std::string, polyregion_dl_handle>> matches;
    for (const auto &path : paths) {
      auto handle = polyregion_dl_open(path.c_str());
      if (!handle) continue; // The pass list may also contain JavaScript bundles.
      if (polyregion_dl_find(handle, abi::AbiVersion)) matches.emplace_back(path, handle);
      else polyregion_dl_close(handle);
    }
    if (matches.size() != 1) {
      value.error = fmt::format("expected one package-service DSO in {}, found {}", env::PolypassPlugins, matches.size());
      for (const auto &[_, handle] : matches)
        polyregion_dl_close(handle);
      return;
    }
    value.handle = matches.front().second;
    const auto version = reinterpret_cast<abi::AbiVersionFn>(polyregion_dl_find(value.handle, abi::AbiVersion));
    value.linkPackage = reinterpret_cast<abi::LinkPackageFn>(polyregion_dl_find(value.handle, abi::LinkPackage));
    value.resolveSym = reinterpret_cast<abi::ResolveSymFn>(polyregion_dl_find(value.handle, abi::ResolveSym));
    value.lastError = reinterpret_cast<abi::LastErrorFn>(polyregion_dl_find(value.handle, abi::LastError));
    value.free = reinterpret_cast<abi::FreeFn>(polyregion_dl_find(value.handle, abi::Free));
    if (!version || !value.linkPackage || !value.resolveSym || !value.lastError || !value.free) {
      value.error = "package-service DSO is missing a polypackage_* entry point";
      return;
    }
    if (const uint32_t found = version(); found != POLYPACKAGE_ABI_VERSION)
      value.error = fmt::format("PolyPackage ABI mismatch: service={}, frontend={}", found, POLYPACKAGE_ABI_VERSION);
  });
  return value;
}

template <typename T, typename Decode, typename Invoke>
ServiceResult<T> invoke(const std::vector<uint8_t> &request, Decode decode, Invoke invokeService) {
  auto &service = serviceAbi();
  if (!service.error.empty()) return {{}, {service.error}};
  std::lock_guard lock(service.callMutex);
  uint8_t *output = nullptr;
  size_t outputSize = 0;
  const auto status = invokeService(service, request.data(), request.size(), &output, &outputSize);
  if (status != POLYPACKAGE_OK) {
    const char *message = service.lastError();
    if (output) service.free(output);
    std::string diagnostic = message ? message : "<no diagnostic>";
    if (diagnostic.starts_with("PolyPackage "))
      if (const auto separator = diagnostic.find(": "); separator != std::string::npos) diagnostic.erase(0, separator + 2);
    std::vector<std::string> errors;
    for (size_t offset = 0; offset <= diagnostic.size();) {
      const auto end = diagnostic.find('\n', offset);
      errors.emplace_back(diagnostic.substr(offset, end - offset));
      if (end == std::string::npos) break;
      offset = end + 1;
    }
    if (errors.empty()) errors.emplace_back(fmt::format("package service returned status {}", static_cast<int>(status)));
    return {{}, std::move(errors)};
  }
  if (!output || outputSize == 0) {
    if (output) service.free(output);
    return {{}, {"package service returned an empty successful result"}};
  }
  try {
    auto result = decode(output, output + outputSize);
    if (output) service.free(output);
    return {{std::move(result)}, {}};
  } catch (const std::exception &error) {
    if (output) service.free(output);
    return {{}, {std::string("cannot decode package-service result: ") + error.what()}};
  }
}

} // namespace

ServiceResult<polyast::Package> PackageService::linkPackage(const polyast::PackageLinkRequest &request) {
  return invoke<polyast::Package>(
      polyast::packagelinkrequest_to_msgpack(request),
      [](const uint8_t *begin, const uint8_t *end) { return polyast::package_service_result_from_msgpack(begin, end); },
      [](ServiceAbi &service, const uint8_t *data, size_t size, uint8_t **out, size_t *outSize) {
        return service.linkPackage(data, size, out, outSize);
      });
}

ServiceResult<polyast::PackageSymResolvedProgram> PackageService::resolveSym(const polyast::PackageSymRequest &request) {
  return invoke<polyast::PackageSymResolvedProgram>(
      polyast::packagesymrequest_to_msgpack(request),
      [](const uint8_t *begin, const uint8_t *end) { return polyast::resolvedsymprogram_from_msgpack(begin, end); },
      [](ServiceAbi &service, const uint8_t *data, size_t size, uint8_t **out, size_t *outSize) {
        return service.resolveSym(data, size, out, outSize);
      });
}

} // namespace polyregion::polyfront::package
