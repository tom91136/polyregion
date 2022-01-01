#pragma once
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

#include "json.hpp"
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

// using json = nlohmann::json;
//
// void to_json(json& j, const Sym& p) {
//   j = json{ {"name", p.fqn} };
// }
//
// void  to_json(  json& j , const Term::FloatConst& p) {
//   j[0] = p.value;
// }
//
// Term::FloatConst from_json(const json& j) {
//
//
//
//   auto ord = j.at(0).get<size_t>();
//   auto t = j.at(1).get<float>();
//   return Term::FloatConst(t);
// }

static void x() {

  using namespace polyregion::polyast;

  Expr::Alias a(Term::BoolConst(true));
  Expr::Index i(Term::Select({}, Named("a", Type::Float())), Term::IntConst(4), Type::String());

  Stmt::Return r(i);

  using json = nlohmann::json;

  std::ifstream file("./ast.msgpack", std::ios::binary);
  std::vector<uint8_t> contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto j = json::from_msgpack(contents.begin(), contents.end());
  std::cout << std::setw(1) << j << std::endl;
  auto x = Stmt::any_json(j);

  std::cout << x;
  //  std::cout << a << "\n" << i << "\n" << r << "\n" << j << "\n";

  Expr::Any u = a;
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