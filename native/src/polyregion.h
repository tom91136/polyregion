#pragma once
#include "variants.hpp"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t size;
  uint8_t *data;
} Buffer;

typedef struct {
  Buffer program;
  char *disassembly;
} Program;

Program compile(Buffer *polyast_proto);

void release(Program *buffer);

#ifdef __cplusplus
}

namespace aaa {




struct RefBase {
  int all{};

protected:
  explicit RefBase(int all) : all(all) {}

public:
  bool operator==(const RefBase &rhs) const { return all == rhs.all; }
  bool operator!=(const RefBase &rhs) const { return !(rhs == *this); }
};


struct ConstBool : RefBase {
  explicit ConstBool(bool value, int a) : RefBase{a}, value(value) {}
  bool value;
};

struct ConstInt32 : public RefBase {
  int value;
  ConstInt32(int all, int value) : RefBase(all), value(value) {}
  //  bool operator==(const ConstInt32 &rhs) const {
  //    return std::tie(static_cast<const aaa::RefBase &>(*this), value) ==
  //           std::tie(static_cast<const aaa::RefBase &>(rhs), rhs.value);
  //  }
  //  bool operator!=(const ConstInt32 &rhs) const { return !(rhs == *this); }

  bool operator==(const ConstInt32 &rhs) const {
    return static_cast<const aaa::RefBase &>(*this) == static_cast<const aaa::RefBase &>(rhs) && value == rhs.value;
  }
  bool operator!=(const ConstInt32 &rhs) const { return !(rhs == *this); }
};

struct ConstVoid : public RefBase {
protected:
  explicit ConstVoid(int all) : RefBase(all) {}

  bool operator==(const ConstVoid &rhs) const {
    return static_cast<const aaa::RefBase &>(*this) == static_cast<const aaa::RefBase &>(rhs);
  }
  bool operator!=(const ConstVoid &rhs) const { return !(rhs == *this); }

  //
public:
  static ConstVoid Value;
};

ConstVoid ConstVoid::Value = ConstVoid(1);

using Ref = std::variant<RefBase, ConstBool, ConstInt32>;

static std::string deref(const Ref &a) {

  //  return std::visit( //
  //      overloaded{
  //          [](ConstBool arg) { return "B"; },  //
  //          [](ConstInt32 arg) { return "I"; }, //
  //          [](RefBase b) { return "a"; }       //
  //      },
  //      a);
  return "";
}

static void x() {

  ConstInt32 aa(1, 1);

  std::function<int(ConstBool)> fn = [](ConstBool arg) { return 1; };

  //  function_traits<decltype(fn)>::result_type;

  //  function_traits<int (  ConstBool)> fb = [](ConstBool arg) -> int { return 1; };

  int cap = 2;
  std::string pat = polyregion::variants::total(
      Ref{aa},                            //
      [&](ConstBool arg) { return ""; },  //
      [&](ConstInt32 arg) { return ""; }, //
      [&](RefBase b) { return ""; }       //
  );

  std::string normal = std::visit( //
      polyregion::variants::overloaded{
          //
          [&](ConstBool arg) { return std::string(""); },                         //
          [&](ConstInt32 arg) -> std::string { return ""; },                      //
          [&](int u) { return ""; }, [&](RefBase b) -> std::string { return ""; } //
      },
      Ref{aa});

  //    static_assert(isVariantMember<int, int>::value);

  static_assert(polyregion::variants::is_variant_member<RefBase, Ref>::value);
  static_assert(polyregion::variants::is_variant_member<ConstBool, Ref>::value);
  static_assert(polyregion::variants::is_variant_member<ConstInt32, Ref>::value);

  ConstVoid a = ConstVoid::Value;

  std::cout << "[] = " << pat + " " + normal;
}

} // namespace aaa

#endif