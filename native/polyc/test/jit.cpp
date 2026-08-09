#include <cstdint>
#include <fstream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include "catch2/catch_all.hpp"

#include "polyregion/enums.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/polyc_jit.h"

#include "ast.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

using namespace polyregion;
using namespace polyregion::compiletime;
using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;

TEST_CASE("polyc JIT C ABI compiles and owns its result", "[jit]") {
  const auto entry =
      function("jit_test", {}, Type::Unit0(), FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true)({ret(Term::Unit0Const())});
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
  llvm::SmallString<256> dir;
  REQUIRE(!llvm::sys::fs::createUniqueDirectory("polyc-jit-cache-test", dir));
  env::put(env::PolyregionCacheDir, dir.c_str(), true);

  const auto entry =
      function("jit_cached", {}, Type::Unit0(), FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true)({ret(Term::Unit0Const())});
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

  env::put(env::PolyregionCacheDir, "", true);
  llvm::sys::fs::remove_directories(dir);
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
