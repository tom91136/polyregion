#include <cstdint>

#include "catch2/catch_all.hpp"

#include "polyregion/enums.h"
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
                            static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen) == POLYC_JIT_OK);
  REQUIRE(image != nullptr);
  CHECK(imageLen > 0);
  CHECK(polyc_jit_last_error() == nullptr);
  polyc_jit_free(image);
}

TEST_CASE("polyc JIT C ABI reports malformed programs", "[jit]") {
  const uint8_t malformed[] = {0xc1}; // msgpack's reserved/invalid byte
  uint8_t *image = nullptr;
  size_t imageLen = 0;
  CHECK(polyc_jit_compile(malformed, sizeof(malformed), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen) == POLYC_JIT_FAILED);
  CHECK(image == nullptr);
  CHECK(imageLen == 0);
  REQUIRE(polyc_jit_last_error() != nullptr);
}

TEST_CASE("polyc JIT C ABI rejects invalid argument spans", "[jit]") {
  uint8_t *image = nullptr;
  size_t imageLen = 0;
  CHECK(polyc_jit_compile(nullptr, 1, static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 0, &image, &imageLen) == POLYC_JIT_FAILED);
  REQUIRE(polyc_jit_last_error() != nullptr);

  const uint8_t program[] = {0x80};
  CHECK(polyc_jit_compile(program, sizeof(program), static_cast<uint32_t>(Target::Object_LLVM_HOST), "native", nullptr,
                          static_cast<uint32_t>(OptLevel::O0), nullptr, 1, &image, &imageLen) == POLYC_JIT_FAILED);
  REQUIRE(polyc_jit_last_error() != nullptr);
}
