#pragma once

#include <cmath>
#include <functional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fmt/format.h"

namespace polyregion::polytest::cases {

struct Task {
  std::string id;
  std::string labels;
  std::function<int()> run;
};

using Discoverer = std::function<std::vector<Task>()>;

inline std::vector<Discoverer> &discoverers() {
  static std::vector<Discoverer> xs;
  return xs;
}
inline void registerDiscoverer(Discoverer d) { discoverers().push_back(std::move(d)); }

#define POLYTEST_CONCAT_(a, b) a##b
#define POLYTEST_CONCAT(a, b) POLYTEST_CONCAT_(a, b)
#define POLYTEST_DISCOVER(...)                                                                                                             \
  namespace {                                                                                                                              \
  const auto POLYTEST_CONCAT(_polytest_reg_, __LINE__) = (::polyregion::polytest::cases::registerDiscoverer(__VA_ARGS__), 0);              \
  }

struct RequireFailed : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct SkipRequested : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Context {
public:
  bool failed = false;

  void check(bool cond, std::string_view expr, std::string detail = {}, std::source_location loc = std::source_location::current()) {
    if (cond) return;
    failed = true;
    fmt::print(stderr, "[FAIL] {}:{}: CHECK({}){}{}\n", loc.file_name(), loc.line(), expr, detail.empty() ? "" : " - ", detail);
  }
  void require(bool cond, std::string_view expr, std::string detail = {}, std::source_location loc = std::source_location::current()) {
    if (cond) return;
    failed = true;
    fmt::print(stderr, "[FATAL] {}:{}: REQUIRE({}){}{}\n", loc.file_name(), loc.line(), expr, detail.empty() ? "" : " - ", detail);
    throw RequireFailed{std::string{expr}};
  }
  [[noreturn]] void skip(std::string_view reason, std::source_location loc = std::source_location::current()) {
    fmt::print(stderr, "[SKIP] {}:{}: {}\n", loc.file_name(), loc.line(), reason);
    throw SkipRequested{std::string{reason}};
  }
  [[noreturn]] void fail(std::string_view reason, std::source_location loc = std::source_location::current()) {
    failed = true;
    fmt::print(stderr, "[FAIL] {}:{}: {}\n", loc.file_name(), loc.line(), reason);
    throw RequireFailed{std::string{reason}};
  }
  void warn(std::string_view msg, std::source_location loc = std::source_location::current()) {
    fmt::print(stderr, "[WARN] {}:{}: {}\n", loc.file_name(), loc.line(), msg);
  }
  void info(std::string_view msg) { fmt::print(stderr, "[INFO] {}\n", msg); }
};

#define POLYTEST_CHECK(ctx, cond) (ctx).check((cond), #cond)
#define POLYTEST_CHECK_S(ctx, cond, ...) (ctx).check((cond), #cond, ::fmt::format(__VA_ARGS__))
#define POLYTEST_REQUIRE(ctx, cond) (ctx).require((cond), #cond)
#define POLYTEST_REQUIRE_S(ctx, cond, ...) (ctx).require((cond), #cond, ::fmt::format(__VA_ARGS__))
#define POLYTEST_SKIP(ctx, ...) (ctx).skip(::fmt::format(__VA_ARGS__))
#define POLYTEST_FAIL(ctx, ...) (ctx).fail(::fmt::format(__VA_ARGS__))
#define POLYTEST_WARN(ctx, ...) (ctx).warn(::fmt::format(__VA_ARGS__))
#define POLYTEST_INFO(ctx, ...) (ctx).info(::fmt::format(__VA_ARGS__))

template <typename T> bool approxEqual(T a, T b, T relTol = T{1e-5}, T absTol = T{1e-7}) {
  if (std::isnan(a) || std::isnan(b)) return false;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  const T diff = a > b ? a - b : b - a;
  const T mag = a > 0 ? a : -a;
  const T magB = b > 0 ? b : -b;
  const T scale = mag > magB ? mag : magB;
  const T tol = absTol > relTol * scale ? absTol : relTol * scale;
  return diff <= tol;
}

inline std::string sanitiseId(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-') out.push_back(c);
    else if (c == ' ' || c == '\t' || c == '/' || c == '\\') out.push_back('_');
  }
  return out;
}

} // namespace polyregion::polytest::cases
