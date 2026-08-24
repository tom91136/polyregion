#include <cstdlib>
#include <string>

#include "polyfront/package_service.hpp"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"

int main(int argc, char **argv) {
  int expectedIndex = 1;
  if (argc > 1 && std::string(argv[1]) == "--duplicate") {
    const char *path = std::getenv(polyregion::env::PolypassPlugins);
    if (!path || !*path) return 2;
#if defined(_WIN32)
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif
    const auto duplicated = std::string(path) + separator + path;
    polyregion::env::put(polyregion::env::PolypassPlugins, duplicated.c_str(), true);
    expectedIndex = 2;
  }
  if (argc != expectedIndex + 1) return 2;
  using namespace polyregion::polyast;
  const auto result =
      polyregion::polyfront::package::PackageService::linkPackage(PackageLinkRequest(Interface(Sym({"probe"}), {}, {}), {}, {}));
  if (result) return 3;
  const std::string expected = argv[expectedIndex];
  for (const auto &error : result.errors)
    if (error.find(expected) != std::string::npos) return 0;
  return 4;
}
