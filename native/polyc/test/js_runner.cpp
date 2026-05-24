#include "js_runner.h"

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"

using namespace aspartame;
using namespace polyregion::polypass;

namespace {

constexpr auto EchoBundle = R"JS(
    exports.polypass_abi_version = function() { return 1; };
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

TEST_CASE("missing runPasses surfaces a clear error") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    exports.polypass_abi_version = function() { return 1; };
    exports.polypass_pass_count = function() { return 0; };
    exports.polypass_pass_name = function(i) { return null; };
  )JS")
              .empty());
  std::string err;
  auto out = r.runPasses({"Anything"}, {}, err);
  REQUIRE(out.empty());
  REQUIRE_FALSE(err.empty());
  REQUIRE((err ^ contains_slice("polypass_run_passes")));
}
