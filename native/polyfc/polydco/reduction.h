#include <span>

#include "polydco.h"

using namespace polyregion;

struct FReduction {
  enum class Kind {
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

  Kind kind;
  size_t offset;
  runtime::Type type;

private:
  static void fail(const char *reason) {
    POLYDCO_LOG("%s", reason);
    std::fflush(stderr);
    std::abort();
  }

  template <typename T> static void reduceTyped(const int64_t range, char *dest, const char *data, T init, auto f) {
    auto xs = reinterpret_cast<const T *>(data);
    T acc = init;
    for (int64_t i = 0; i < range; ++i)
      acc = f(acc, xs[i]);
    std::memcpy(dest, &acc, sizeof(T));
  }

  template <typename T> void reduceInt(const int64_t range, char *dest, const char *data) const {
    using Lim = std::numeric_limits<T>;
    switch (kind) {
      case Kind::Add: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x + y; });
      case Kind::Mul: return reduceTyped<T>(range, dest, data, 1, [](const auto x, const auto y) { return x * y; });
      case Kind::Max: return reduceTyped<T>(range, dest, data, Lim::lowest(), [](const auto x, const auto y) { return std::max(x, y); });
      case Kind::Min: return reduceTyped<T>(range, dest, data, Lim::max(), [](const auto x, const auto y) { return std::min(x, y); });
      case Kind::IAnd: return reduceTyped<T>(range, dest, data, ~static_cast<T>(0), [](const auto x, const auto y) { return x & y; });
      case Kind::IOr: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x | y; });
      case Kind::IEor: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x ^ y; });
      default: return fail("Unsupported reduction kind on integer type");
    }
  }

  template <typename T> void reduceReal(const int64_t range, char *dest, const char *data) const {
    using Lim = std::numeric_limits<T>;
    switch (kind) {
      case Kind::Add: return reduceTyped<T>(range, dest, data, 0, [](const auto x, const auto y) { return x + y; });
      case Kind::Mul: return reduceTyped<T>(range, dest, data, 1, [](const auto x, const auto y) { return x * y; });
      case Kind::Max: return reduceTyped<T>(range, dest, data, Lim::lowest(), [](const auto x, const auto y) { return std::fmax(x, y); });
      case Kind::Min: return reduceTyped<T>(range, dest, data, Lim::max(), [](const auto x, const auto y) { return std::fmin(x, y); });
      default: return fail("Unsupported reduction kind on real type");
    }
  }

  void reduceBool(const int64_t range, char *dest, const char *data) const {
    switch (kind) {
      case Kind::And: return reduceTyped<bool>(range, dest, data, true, [](const bool x, const bool y) { return x && y; });
      case Kind::Or: return reduceTyped<bool>(range, dest, data, false, [](const bool x, const bool y) { return x || y; });
      case Kind::Eqv: return reduceTyped<bool>(range, dest, data, true, [](const bool x, const bool y) { return x == y; });
      case Kind::Neqv: return reduceTyped<bool>(range, dest, data, false, [](const bool x, const bool y) { return x != y; });
      default: return fail("Unsupported reduction kind on bool type");
    }
  }

public:
  void reduce(const int64_t range, char *dest, const char *data) const {
    switch (type) {
      case runtime::Type::Bool1: return reduceBool(range, dest + offset, data);
      case runtime::Type::IntU8: return reduceInt<uint8_t>(range, dest + offset, data);
      case runtime::Type::IntS8: return reduceInt<int8_t>(range, dest + offset, data);
      case runtime::Type::IntU16: return reduceInt<uint16_t>(range, dest + offset, data);
      case runtime::Type::IntS16: return reduceInt<int16_t>(range, dest + offset, data);
      case runtime::Type::IntU32: return reduceInt<uint32_t>(range, dest + offset, data);
      case runtime::Type::IntS32: return reduceInt<int32_t>(range, dest + offset, data);
      case runtime::Type::IntU64: return reduceInt<uint64_t>(range, dest + offset, data);
      case runtime::Type::IntS64: return reduceInt<int64_t>(range, dest + offset, data);
      // XXX case runtime::Type::Float16: return reduceReal<float16_t>(dest range,  + offset, data);
      case runtime::Type::Float32: return reduceReal<float>(range, dest + offset, data);
      case runtime::Type::Float64: return reduceReal<double>(range, dest + offset, data);
      default: return fail("Unsupported type for reduction");
    }
  }

  size_t size() const { return runtime::byteOfType(type); }
};
