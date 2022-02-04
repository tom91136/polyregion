#include "ast.h"
#include "catch.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

#include <iostream>

using namespace polyregion::polyast;

using namespace Stmt;
using namespace Term;
using namespace Expr;

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                        //
      Sym({"foo"}),                                         //
      {Named("a", Type::Int()), Named("b", Type::Float())}, //
      Type::Unit(),                                         //
      {
          Comment("a"),                         //
          Comment("b"),                         //
          Break(),                              //
          Return(Alias(IntConst(1))),           //
          Cond(                                 //
              Alias(BoolConst(true)),           //
              {Return(Alias(IntConst(1)))},     //
              {Return(Alias(BoolConst(false)))} //
              )                                 //
      });
  auto actual = function_from_json(function_to_json(expected));
  CHECK(expected == actual);
}

TEST_CASE("initialise more than once should work", "[compiler]") {
  polyregion_initialise();
  polyregion_initialise();
  polyregion_initialise();
  polyregion_initialise();
}

TEST_CASE("empty function should compile", "[compiler]") {
  polyregion_initialise();
  Function fn(Sym({"foo"}), {}, Type::Unit(), {});
  auto data = nlohmann::json::to_msgpack(function_to_json(fn));

  polyregion_buffer buffer{data.data(), data.size()};

  auto compilation = polyregion_compile(&buffer, true, POLYREGION_BACKEND_LLVM);

  polyregion_release_compile(compilation);
}

TEST_CASE("struct member access", "[compiler]") {
  polyregion_initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, {Named("x", Type::Int()), Named("y", Type::Int())});
  Named arg("in", Type::Struct(sdef));
  Function fn(Sym({"foo"}), {arg}, Type::Unit(),
              {
                  Var(                          //
                      Named("y2", Type::Int()), //
                      {
                          Alias(Select({arg}, Named("y", Type::Int()))) //
                      }                                                 //
                      ),
                  Return(Alias(UnitConst())),
              });

  Program p(fn, {}, {def});

  polyregion::compiler::compile(p);
}

TEST_CASE("mut prim", "[compiler]") {
  polyregion_initialise();

  Function fn(Sym({"foo"}), {}, Type::Int(),
              {
                  Var(Named("s", Type::Int()), {}),
                  Mut(Select({}, Named("s", Type::Int())), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))
  Program p(fn, {}, {});
  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("struct buffer assign", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {Named("s", Type::Array(myStruct, {}))}, Type::Int(),
              {

                  Var(Named("a", myStruct), { Index(Select({}, Named("s", myStruct)), Term::IntConst(0), myStruct) }),


                  Var(Named("b", myStruct), {  Alias(Select({Named("a", myStruct)}, defX ))  }),

//                  Mut(Select({Named("s", Type::Array(myStruct, {}))}, defX), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))
  Program p(fn, {}, {def});
  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}


TEST_CASE("mut struct buffer", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {Named("s", Type::Array(myStruct, {}))}, Type::Int(),
              {
                  Mut(Select({Named("s", Type::Array(myStruct, {}))}, defX), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))
  Program p(fn, {}, {def});
  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("mut prim buffer", "[compiler]") {
  polyregion_initialise();
  Function fn(Sym({"foo"}), {Named("s", Type::Array(Type::Int(), {}))}, Type::Int(),
              {
                  Mut(Select({}, Named("s", Type::Array(Type::Int(), {}))), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))
  Program p(fn, {}, {});
  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("struct alloc", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {Named("out", myStruct)}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntConst(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntConst(43)), false),

                  Mut(Select({}, Named("out", myStruct)), Alias(Select({}, Named("s", myStruct))), true),
                  //                  Return(Alias(UnitConst())),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))

  Program p(fn, {}, {def});

  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}