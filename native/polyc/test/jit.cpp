#include <cstdint>
#include <fstream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "catch2/catch_all.hpp"

#include "polyregion/enums.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/polyc_jit.h"

#include "ast.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyc_jit_cache.h"

using namespace polyregion;
using namespace polyregion::compiletime;
using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;

TEST_CASE("polyc JIT reports its wire ABI", "[jit]") { CHECK(polyc_jit_abi_version() == POLYC_JIT_ABI_VERSION); }

TEST_CASE("polyc JIT C ABI compiles and owns its result", "[jit]") {
  const auto entry = function("jit_test", {}, Type::Unit0(), FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                              CallConvention::OffloadEntry())({ret(Term::Unit0Const())});
  const auto packed = hashed_program_to_msgpack(program({}, {entry}));

  uint8_t *image = nullptr;
  size_t imageLen = 0;
  REQUIRE(polyc_jit_compile(packed.data(), packed.size(), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                            static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen)
          == POLYC_JIT_OK);
  REQUIRE(image != nullptr);
  CHECK(imageLen > 0);
  CHECK(polyc_jit_last_error() == nullptr);
  polyc_jit_free(image);
}

TEST_CASE("polyc JIT cache rebuilds from a damaged entry", "[jit]") {
  struct CacheDir {
    llvm::SmallString<256> dir;
    std::optional<env::ScopedEnv> scoped;
    CacheDir() {
      REQUIRE(!llvm::sys::fs::createUniqueDirectory("polyc-jit-cache-test", dir));
      scoped.emplace(env::PolyregionCacheDir, dir.c_str());
    }
    ~CacheDir() { llvm::sys::fs::remove_directories(dir); }
  } cacheDir;
  const auto &dir = cacheDir.dir;

  const auto entry = function("jit_cached", {}, Type::Unit0(), FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                              CallConvention::OffloadEntry())({ret(Term::Unit0Const())});
  const auto packed = hashed_program_to_msgpack(program({}, {entry}));
  const auto compile = [&](uint8_t **image, size_t *len) {
    return polyc_jit_compile(packed.data(), packed.size(), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                             static_cast<uint32_t>(OptLevel::O0), nullptr, 0, image, len);
  };

  uint8_t *fresh = nullptr;
  size_t freshLen = 0;
  REQUIRE(compile(&fresh, &freshLen) == POLYC_JIT_OK);
  REQUIRE(freshLen > 0);
  const std::vector<uint8_t> expected(fresh, fresh + freshLen);
  polyc_jit_free(fresh);

  size_t entries = 0;
  std::error_code ec;
  for (llvm::sys::fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
    if (llvm::sys::fs::is_directory(it->path())) continue;
    std::ofstream(it->path(), std::ios::binary | std::ios::trunc) << "not an object file";
    entries++;
  }
  REQUIRE(entries > 0);

  uint8_t *recovered = nullptr;
  size_t recoveredLen = 0;
  REQUIRE(compile(&recovered, &recoveredLen) == POLYC_JIT_OK);
  REQUIRE(recovered != nullptr);
  const std::vector<uint8_t> actual(recovered, recovered + recoveredLen);
  polyc_jit_free(recovered);

  CHECK(actual == expected);
}

TEST_CASE("polyc JIT cache key follows the compiler identity", "[jit]") {
  const env::ScopedEnv cacheDir(env::PolyregionCacheDir, "polyc-jit-key-test");
  const uint8_t program[] = {1, 2, 3};
  const auto key = [&](const std::string &identity) {
    return jit::cachePath(program, sizeof(program), 0, "native", nullptr, 0, nullptr, 0, identity);
  };
  CHECK(key("a") == key("a"));
  CHECK(key("a") != key("b"));

  struct TempPlugins {
    llvm::SmallString<256> dir;
    TempPlugins() { REQUIRE(!llvm::sys::fs::createUniqueDirectory("polyc-jit-identity-test", dir)); }
    ~TempPlugins() { llvm::sys::fs::remove_directories(dir); }
    std::string write(const std::string &name, const std::string &content) const {
      llvm::SmallString<256> path(dir);
      llvm::sys::path::append(path, name);
      std::ofstream(path.c_str(), std::ios::binary) << content;
      return path.str().str();
    }
  } plugins;
#if defined(_WIN32)
  constexpr auto suffix = ".dll";
#elif defined(__APPLE__)
  constexpr auto suffix = ".dylib";
#else
  constexpr auto suffix = ".so";
#endif
  const auto first = plugins.write(std::string("first") + suffix, "one");
  const auto second = plugins.write(std::string("second") + suffix, "two plus");
  const auto identityWith = [](const std::string &plugin) {
    const env::ScopedEnv scoped(env::PolypassPlugins, plugin.c_str());
    return jit::compilerIdentity();
  };
  CHECK(identityWith(first) == identityWith(first));
  CHECK(identityWith(first) != identityWith(second));
}

TEST_CASE("polyc JIT C ABI reports malformed programs", "[jit]") {
  const uint8_t malformed[] = {0xc1}; // msgpack's reserved/invalid byte
  uint8_t *image = nullptr;
  size_t imageLen = 0;
  CHECK(polyc_jit_compile(malformed, sizeof(malformed), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen)
        == POLYC_JIT_FAILED);
  CHECK(image == nullptr);
  CHECK(imageLen == 0);
  REQUIRE(polyc_jit_last_error() != nullptr);
}

TEST_CASE("polyc JIT C ABI rejects invalid argument spans", "[jit]") {
  uint8_t *image = nullptr;
  size_t imageLen = 0;
  CHECK(polyc_jit_compile(nullptr, 1, static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen)
        == POLYC_JIT_FAILED);
  REQUIRE(polyc_jit_last_error() != nullptr);

  const uint8_t program[] = {0x80};
  CHECK(polyc_jit_compile(program, sizeof(program), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 1, &image, &imageLen)
        == POLYC_JIT_FAILED);
  REQUIRE(polyc_jit_last_error() != nullptr);
}
