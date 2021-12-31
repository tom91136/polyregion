#pragma once
#include "generated/polyast.h"
#include "variants.hpp"
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

static void x() {

  using namespace polyregion::polyast;

  Tree::Expr::Alias a(Term::BoolConst(true));
  Tree::Expr::Index i(Term::Select({}, Named("a", Type::Float())), Term::IntConst(4), Type::String());

  Tree::Stmt::Return r(i);

  std::cout << a << "\n"
            << i << "\n"
            << r << "\n"
            << "\n";

  Tree::Expr::Any u = a;
  //
  //
  auto kk = polyregion::variants::total(
      a.tpe,                                                   //
      [](std::shared_ptr<Type::Float> x) { return x->kind; },  //
      [](std::shared_ptr<Type::Double> x) { return x->kind; }, //
      [](std::shared_ptr<Type::Bool> x) { return x->kind; },   //
      [](std::shared_ptr<Type::Byte> x) { return x->kind; },   //
      [](std::shared_ptr<Type::Char> x) { return x->kind; },   //
      [](std::shared_ptr<Type::Short> x) { return x->kind; },  //
      [](std::shared_ptr<Type::Int> x) { return x->kind; },    //
      [](std::shared_ptr<Type::Long> x) { return x->kind; },   //
      [](std::shared_ptr<Type::String> x) { return x->kind; }, //
      [](std::shared_ptr<Type::Unit> x) { return x->kind; },   //
      [](std::shared_ptr<Type::Struct> x) { return x->kind; }, //
      [](std::shared_ptr<Type::Array> x) { return x->kind; }   //
  );
  auto aaa = polyregion::variants::total(
      *a.tpe,                                       //
      [](const Type::Float &x) { return x.kind; },  //
      [](const Type::Double &x) { return x.kind; }, //
      [](const Type::Bool &x) { return x.kind; },   //
      [](const Type::Byte &x) { return x.kind; },   //
      [](const Type::Char &x) { return x.kind; },   //
      [](const Type::Short &x) { return x.kind; },  //
      [](const Type::Int &x) { return x.kind; },    //
      [](const Type::Long &x) { return x.kind; },   //
      [](const Type::String &x) { return x.kind; }, //
      [](const Type::Unit &x) { return x.kind; },   //
      [](const Type::Struct &x) { return x.kind; }, //
      [](const Type::Array &x) { return x.kind; }   //
  );
  std::cout << "k=       " << u << kk << aaa << "\n";
}


} // namespace aaa

#endif