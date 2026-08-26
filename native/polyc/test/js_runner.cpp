#include "js_runner.h"

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"

#include "polyregion/polypackage_symbols.h"

using namespace aspartame;
using namespace polyregion::polypass;

namespace {

constexpr auto EchoBundle = R"JS(
    exports.polypass_abi_version = function() { return 2; };
    exports.polypass_pass_count = function() { return 1; };
    exports.polypass_pass_name = function(i) { return i === 0 ? "Echo" : null; };
    exports.polypass_pass_descr = function(i) { return null; };
    exports.polypass_run_passes = function(steps, bytes) {
      const joined = steps.join(",");
      const out = new Uint8Array(bytes.length + joined.length);
      for (let i = 0; i < joined.length; ++i) out[i] = joined.charCodeAt(i);
      out.set(bytes, joined.length);
      return out;
    };
  )JS";

} // namespace

TEST_CASE("enumerate + runPasses round-trip through CommonJS exports") {
  JsPassRunner r;
  REQUIRE(r.loadModule(EchoBundle).empty());
  REQUIRE(r.passNames() == std::vector<std::string>{"Echo"});

  std::string err;
  std::vector<uint8_t> in{1, 2, 3, 4};
  auto out = r.runPasses({"Echo", "Echo"}, in, err);
  REQUIRE(err.empty());
  REQUIRE(out == std::vector<uint8_t>{'E', 'c', 'h', 'o', ',', 'E', 'c', 'h', 'o', 1, 2, 3, 4});
}

TEST_CASE("reject an incomplete pass capability") {
  JsPassRunner r;
  const auto error = r.loadModule(R"JS(
    exports.polypass_abi_version = function() { return 2; };
    exports.polypass_pass_count = function() { return 0; };
    exports.polypass_pass_name = function(i) { return null; };
  )JS");
  REQUIRE((error ^ contains_slice("incomplete pass capability")));
}

TEST_CASE("reject a plugin built against the previous PolyPass ABI") {
  JsPassRunner r;
  const auto error = r.loadModule(R"JS(
    exports.polypass_abi_version = function() { return 1; };
    exports.polypass_pass_count = function() { return 0; };
    exports.polypass_pass_name = function(i) { return null; };
    exports.polypass_run_passes = function(steps, bytes) { return bytes; };
  )JS");
  REQUIRE((error ^ contains_slice("PolyPass ABI mismatch")));
}

TEST_CASE("load and invoke a package-only plugin") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    exports.polypackage_abi_version = function() { return 2; };
    exports.polypackage_link_package = function(bytes) {
      const out = new Uint8Array(bytes.length + 1);
      out[0] = 42;
      out.set(bytes, 1);
      return out;
    };
    exports.polypackage_link_program = function(bytes) { return bytes; };
  )JS")
              .empty());
  CHECK(r.passNames().empty());
  REQUIRE(r.packageAbiVersion() == 2);

  std::string error;
  const auto output = r.runPackage(polyregion::polypackage::abi::LinkPackage, {1, 2, 3}, error);
  CHECK(error.empty());
  CHECK(output == std::vector<uint8_t>{42, 1, 2, 3});

  const auto program = r.runPackage(polyregion::polypackage::abi::LinkProgram, {4, 5, 6}, error);
  CHECK(error.empty());
  CHECK(program == std::vector<uint8_t>{4, 5, 6});
}

TEST_CASE("reject an incomplete package capability") {
  JsPassRunner r;
  const auto error = r.loadModule(R"JS(
    exports.polypackage_abi_version = function() { return 2; };
    exports.polypackage_link_package = function(bytes) { return bytes; };
  )JS");
  REQUIRE((error ^ contains_slice("incomplete package capability")));
}
