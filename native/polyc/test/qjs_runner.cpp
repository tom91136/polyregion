#include "qjs_runner.h"

#include "catch2/catch_test_macros.hpp"

using namespace polyregion::polypass;

TEST_CASE("eval echoes bytes through runPass") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    globalThis.runPass = function(name, bytes) {
      const view = bytes;
      const out = new Uint8Array(view.length + name.length);
      for (let i = 0; i < name.length; ++i) out[i] = name.charCodeAt(i);
      out.set(view, name.length);
      return out;
    };
  )JS")
              .empty());

  std::string err;
  std::vector<uint8_t> in{1, 2, 3, 4};
  auto out = r.runPass("xy", in, err);
  REQUIRE(err.empty());
  REQUIRE(out == std::vector<uint8_t>{'x', 'y', 1, 2, 3, 4});
}

TEST_CASE("eval echoes bytes through CommonJS exports") {
  JsPassRunner r;
  REQUIRE(r.loadModule(R"JS(
    exports.runPass = function(name, bytes) {
      const view = bytes;
      const out = new Uint8Array(view.length + name.length);
      for (let i = 0; i < name.length; ++i) out[i] = name.charCodeAt(i);
      out.set(view, name.length);
      return out;
    };
  )JS")
              .empty());

  std::string err;
  std::vector<uint8_t> in{1, 2, 3, 4};
  auto out = r.runPass("xy", in, err);
  REQUIRE(err.empty());
  REQUIRE(out == std::vector<uint8_t>{'x', 'y', 1, 2, 3, 4});
}

TEST_CASE("missing runPass surfaces a clear error") {
  JsPassRunner r;
  REQUIRE(r.loadModule("// empty").empty());
  std::string err;
  auto out = r.runPass("noop", {}, err);
  REQUIRE(out.empty());
  REQUIRE_FALSE(err.empty());
}
