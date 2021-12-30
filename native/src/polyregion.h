#pragma once
#include "foo.h"
#include "variants.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

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

static polyregion::polyast::Tree::Stmt::Return mkTree() {
  using namespace polyregion::polyast;
  Tree::Expr::Alias a(Term::BoolConst(true));
  Tree::Expr::Index i(Term::Select({}, Named("a", Type::Float())), Term::IntConst(4), Type::String());

  Tree::Stmt::Return r(i);
  return r;
}

static void x() {

  using namespace polyregion::polyast;

  Tree::Expr::Alias a(Term::BoolConst(true));
  Tree::Expr::Index i(Term::Select({}, Named("a", Type::Float())), Term::IntConst(4), Type::String());

  Tree::Stmt::Return r(i);


  auto p = mkTree();


  std::cout << a << "\n" << i << "\n" << r << "\n" << p << "\n";

  //  Alt l = A();
  //  T1ALeaf xx ({}, {"a", "b"}, 23, T1BLeaf());

  Tree::Expr::Any u = a;

  auto kk = polyregion::variants::total(
      a.tpe,                                //
      [](std::shared_ptr<Type::Float> x) {return x->kind;},  //
      [](std::shared_ptr<Type::Double> x) {return x->kind;}, //
      [](std::shared_ptr<Type::Bool> x) {return x->kind;},   //
      [](std::shared_ptr<Type::Byte> x) {return x->kind;},   //
      [](std::shared_ptr<Type::Char> x) {return x->kind;},   //
      [](std::shared_ptr<Type::Short> x) {return x->kind;},  //
      [](std::shared_ptr<Type::Int> x) {return x->kind;},    //
      [](std::shared_ptr<Type::Long> x) {return x->kind;},   //
      [](std::shared_ptr<Type::String> x) {return x->kind;}, //
      [](std::shared_ptr<Type::Unit> x) {return x->kind;},   //
      [](std::shared_ptr<Type::Struct> x) {return x->kind;}, //
      [](std::shared_ptr<Type::Array> x) {return x->kind;}   //
  );
  std::cout << "k=" << p.value << p.tpe << "\n";

  int out = polyregion::variants::total<Tree::Expr::Any>(
      a,                                                                     //
      [](const std::shared_ptr<Tree::Expr::Alias> &a) -> int { return 1; },  //
      [](const std::shared_ptr<Tree::Expr::Invoke> &b) -> int { return 1; }, //
      [](const std::shared_ptr<Tree::Expr::Index> &b) -> int { return 1; }   //
  );

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
  };

  using Ref = std::variant<ConstBool, ConstInt32>;

  struct Base {};
  struct A : Base {
    int a{};
  };
  struct B : Base {
    int b{};
  };
  using R = std::variant<std::shared_ptr<A>, std::shared_ptr<B>>;
  R vv{std::make_shared<A>(A())};
  std::visit([](auto &&x) { return 1; }, vv);

  int out3 = polyregion::variants::total(
      vv, [](const std::shared_ptr<A> &a) -> int { return 1; }, [](const std::shared_ptr<B> &b) -> int { return 1; });

  ConstInt32 aa(1, 1);

  auto xxx = Ref{aa};
  std::string pat = polyregion::variants::total(
      xxx,                                      //
      [&](const ConstBool &arg) { return ""; }, //
      [&](const ConstInt32 &arg) { return ""; } //
  );

  std::cout << "[] = " << out3;
}

// ======================================================================================

} // namespace aaa

#endif