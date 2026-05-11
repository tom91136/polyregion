#include "qjs_runner.h"

#include "aspartame/all.hpp"
#include "catch2/catch_test_macros.hpp"

using namespace aspartame;
using namespace polyregion::polypass;

TEST_CASE("eval echoes bytes through runPipeline") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    globalThis.runPipeline = function(spec, bytes) {
      const view = bytes;
      const out = new Uint8Array(view.length + spec.length);
      for (let i = 0; i < spec.length; ++i) out[i] = spec.charCodeAt(i);
      out.set(view, spec.length);
      return out;
    };
  )JS")
              .empty());

  std::string err;
  std::vector<uint8_t> in{1, 2, 3, 4};
  auto out = r.runPipeline("xy", in, err);
  REQUIRE(err.empty());
  REQUIRE(out == std::vector<uint8_t>{'x', 'y', 1, 2, 3, 4});
}

TEST_CASE("eval echoes bytes through CommonJS exports") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    exports.runPipeline = function(spec, bytes) {
      const view = bytes;
      const out = new Uint8Array(view.length + spec.length);
      for (let i = 0; i < spec.length; ++i) out[i] = spec.charCodeAt(i);
      out.set(view, spec.length);
      return out;
    };
  )JS")
              .empty());

  std::string err;
  std::vector<uint8_t> in{1, 2, 3, 4};
  auto out = r.runPipeline("xy", in, err);
  REQUIRE(err.empty());
  REQUIRE(out == std::vector<uint8_t>{'x', 'y', 1, 2, 3, 4});
}

TEST_CASE("missing runPipeline surfaces a clear error") {
  JsPassRunner r;
  REQUIRE(r.loadModule("// empty").empty());
  std::string err;
  auto out = r.runPipeline("noop", {}, err);
  REQUIRE(out.empty());
  REQUIRE_FALSE(err.empty());
  REQUIRE((err ^ contains_slice("runPipeline")));
}
