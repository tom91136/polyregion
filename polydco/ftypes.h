#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "polyregion/types.h"

namespace polyregion::polydco {

struct FManagedPrelude {
  int64_t lowerBound, upperBound, step, tripCount;
};
static_assert(sizeof(FManagedPrelude) == (sizeof(int64_t) * 4));

struct FHostThreadedPrelude {
  int64_t lowerBound, upperBound, step, tripCount;
  int64_t *begins, *ends;
};
static_assert(sizeof(FHostThreadedPrelude) == ((sizeof(int64_t) * 4) + (sizeof(void *) * 2)));

struct FReduction {
  enum class Kind : uint8_t {
    Add = 1, // real|integer
    Mul,     // real|integer
    Max,     // real|integer
    Min,     // real|integer
    IAnd,    // integer
    IOr,     // integer
    IEor,    // integer, XOR
    And,     // logical
    Or,      // logical
    Eqv,     // logical
    Neqv,    // logical
  };

  static constexpr std::string_view to_string(const Kind &kind) {
    switch (kind) {
      case Kind::Add: return "Add";
      case Kind::Mul: return "Mul";
      case Kind::Max: return "Max";
      case Kind::Min: return "Min";
      case Kind::IAnd: return "IAnd";
      case Kind::IOr: return "IOr";
      case Kind::IEor: return "IEor";
      case Kind::And: return "And";
      case Kind::Or: return "Or";
      case Kind::Eqv: return "Eqv";
      case Kind::Neqv: return "Neqv";
      default: std::fprintf(stderr, "Unimplemented FReduction::Kind\n"); std::abort();
    }
  }

  Kind kind;
  runtime::Type type;
  char *dest;

private:
  static void fail(const char *reason) {
    std::fprintf(stderr, "[PolyDCO] %s\n", reason);
    std::fflush(stderr);
    std::abort();
  }

  template <typename T, typename F> static void reduceTyped(const int64_t range, char *dest, const char *data, T init, F f) {
    const auto xs = reinterpret_cast<const T *>(data);
    T acc = init;
    for (int64_t i = 0; i < range; ++i)
      acc = f(acc, xs[i]);
    auto out = reinterpret_cast<T *>(dest);
    *out = f(acc, *out);
  }

  template <typename T> void reduceInt(const int64_t range, const char *data) const {
    using Lim = std::numeric_limits<T>;
    switch (kind) {
      case Kind::Add: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x + y; });
      case Kind::Mul: return reduceTyped<T>(range, dest, data, 1, [](const auto x, const auto y) { return x * y; });
      case Kind::Max: return reduceTyped<T>(range, dest, data, Lim::lowest(), [](const auto x, const auto y) { return std::max(x, y); });
      case Kind::Min: return reduceTyped<T>(range, dest, data, Lim::max(), [](const auto x, const auto y) { return std::min(x, y); });
      case Kind::IAnd: return reduceTyped<T>(range, dest, data, ~static_cast<T>(0), [](const auto x, const auto y) { return x & y; });
      case Kind::IOr: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x | y; });
      case Kind::IEor: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x ^ y; });
      default: {
        fail("Unsupported reduction kind on integer type");
        return;
      }
    }
  }

  template <typename T> void reduceReal(const int64_t range, const char *data) const {
    using Lim = std::numeric_limits<T>;
    switch (kind) {
      case Kind::Add: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x + y; });
      case Kind::Mul: return reduceTyped<T>(range, dest, data, 1, [](const auto x, const auto y) { return x * y; });
      case Kind::Max: return reduceTyped<T>(range, dest, data, Lim::lowest(), [](const auto x, const auto y) { return std::fmax(x, y); });
      case Kind::Min: return reduceTyped<T>(range, dest, data, Lim::max(), [](const auto x, const auto y) { return std::fmin(x, y); });
      default: {
        fail("Unsupported reduction kind on real type");
        return;
      }
    }
  }

  void reduceBool(const int64_t range, const char *data) const {
    switch (kind) {
      case Kind::And: {
        reduceTyped<bool>(range, dest, data, true, [](const bool x, const bool y) { return x && y; });
        return;
      }
      case Kind::Or: {
        reduceTyped<bool>(range, dest, data, false, [](const bool x, const bool y) { return x || y; });
        return;
      }
      case Kind::Eqv: {
        reduceTyped<bool>(range, dest, data, true, [](const bool x, const bool y) { return x == y; });
        return;
      }
      case Kind::Neqv: {
        reduceTyped<bool>(range, dest, data, false, [](const bool x, const bool y) { return x != y; });
        return;
      }
      default: {
        fail("Unsupported reduction kind on bool type");
        return;
      }
    }
  }

public:
  void reduce(const int64_t range, const char *data) const {
    switch (type) {
      case runtime::Type::Bool1: {
        reduceBool(range, data);
        return;
      }
      case runtime::Type::IntU8: {
        reduceInt<uint8_t>(range, data);
        return;
      }
      case runtime::Type::IntS8: {
        reduceInt<int8_t>(range, data);
        return;
      }
      case runtime::Type::IntU16: {
        reduceInt<uint16_t>(range, data);
        return;
      }
      case runtime::Type::IntS16: {
        reduceInt<int16_t>(range, data);
        return;
      }
      case runtime::Type::IntU32: {
        reduceInt<uint32_t>(range, data);
        return;
      }
      case runtime::Type::IntS32: {
        reduceInt<int32_t>(range, data);
        return;
      }
      case runtime::Type::IntU64: {
        reduceInt<uint64_t>(range, data);
        return;
      }
      case runtime::Type::IntS64: {
        reduceInt<int64_t>(range, data);
        return;
      }
      // XXX case runtime::Type::Float16: return reduceReal<float16_t>(range, data);
      case runtime::Type::Float32: {
        reduceReal<float>(range, data);
        return;
      }
      case runtime::Type::Float64: {
        reduceReal<double>(range, data);
        return;
      }
      default: {
        fail("Unsupported type for reduction");
        return;
      }
    }
  }

  [[nodiscard]] size_t typeSizeInBytes() const { return runtime::byteOfType(type); }
};
} // namespace polyregion::polydco
