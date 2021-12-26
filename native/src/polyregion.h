#pragma once
#include <cstddef>
#include <cstdint>
#include "variants.hpp"

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
};

struct ConstBool : public RefBase {
  bool value;
};

struct ConstInt32 : public RefBase {
  int value;
};

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

  RefBase a{1};
  ConstInt32 aa{1, 2};

  std::function<int(ConstBool)> fn = [](ConstBool arg) { return 1; };

  //  function_traits<decltype(fn)>::result_type;

  //  function_traits<int (  ConstBool)> fb = [](ConstBool arg) -> int { return 1; };

  int cap = 2;
  std::string pat = polyregion::variants::total (
      Ref{aa},                            //
      [&](ConstBool arg) { return "" + 2; },  //
      [&](ConstInt32 arg) { return ""; }, //
      [&](RefBase b) { return ""; }       //
  );

  std::string normal = std::visit( //
      polyregion::variants::overloaded{
          //
          [&](ConstBool arg) { return std::string(""); },    //
          [&](ConstInt32 arg) -> std::string { return ""; }, //
          [&] (int u) { return ""; },
          [&](RefBase b) -> std::string { return ""; }       //
      },
      Ref{aa});

  //    static_assert(isVariantMember<int, int>::value);

  static_assert(polyregion::variants::is_variant_member<RefBase, Ref>::value);
  static_assert(polyregion::variants::is_variant_member<ConstBool, Ref>::value);
  static_assert(polyregion::variants::is_variant_member<ConstInt32, Ref>::value);

  std::cout << "[] = " << pat + " " + normal;
}

} // namespace aaa

#endif