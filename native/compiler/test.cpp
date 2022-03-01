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

template <typename P> static void assertCompilationSucceeded(const P &p) {
  INFO(repr(p))
  auto c = polyregion::compiler::compile(p);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                            //
      Sym({"foo"}),                                             //
      {}, {Named("a", Type::Int()), Named("b", Type::Float())}, //
      Type::Unit(),                                             //
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

// TEST_CASE("empty function should compile", "[compiler]") {
//   polyregion_initialise();
//   Function fn(Sym({"foo"}), {}, Type::Unit(), {});
//   Program p(fn, {}, {});
//   assertCompilationSucceeded(p);
// }

TEST_CASE("struct member access", "[compiler]") {
  polyregion_initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, {Named("x", Type::Int()), Named("y", Type::Int())});
  Named arg("in", Type::Struct(sdef));
  Function fn(Sym({"foo"}), {}, {arg}, Type::Unit(),
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
  assertCompilationSucceeded(p);
}

TEST_CASE("prim mut", "[compiler]") {
  polyregion_initialise();

  Function fn(Sym({"foo"}), {}, {}, Type::Int(),
              {
                  Var(Named("s", Type::Int()), {}),
                  Mut(Select({}, Named("s", Type::Int())), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  INFO(repr(fn))
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("index struct buffer member", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(
      Sym({"foo"}), {}, {Named("s", Type::Array(myStruct))}, Type::Int(),
      {

          Var(Named("a", myStruct),
              {Index(Select({}, Named("s", Type::Array(myStruct))), Term::IntConst(0), myStruct)}),

          Var(Named("b", Type::Int()), {Alias(Select({Named("a", myStruct)}, defX))}),

          //                  Mut(Select({Named("s", Type::Array(myStruct ))}, defX), Alias(IntConst(42)), false),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("update struct array elem member", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(
      Sym({"foo"}), {}, {Named("xs", Type::Array(myStruct))}, Type::Int(),
      {
          Var(Named("x", myStruct), Index(Select({}, Named("xs", Type::Array(myStruct))), IntConst(0), myStruct)),
          Mut(Select({Named("x", myStruct)}, defX), Alias(IntConst(42)), false),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("update struct array elem", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(
      Sym({"foo"}), {}, {Named("xs", Type::Array(myStruct))}, Type::Int(),
      {
          Var(Named("data", myStruct), {}),
          Update(Select({}, Named("xs", Type::Array(myStruct))), IntConst(7), Select({}, Named("data", myStruct))),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("update prim array", "[compiler]") {
  polyregion_initialise();
  Function fn(Sym({"foo"}), {}, {Named("s", Type::Array(Type::Int()))}, Type::Int(),
              {
                  Update(Select({}, Named("s", Type::Array(Type::Int()))), IntConst(7), (IntConst(42))),
                  Return(Alias(IntConst(69))),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("struct alloc", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);
  StructDef def2(myStruct2Sym, {defX});
  Type::Struct myStruct2(myStruct2Sym);

  Function fn(Sym({"foo"}), {}, {Named("out", myStruct)}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntConst(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntConst(43)), false),
                  Var(Named("t", myStruct2), {}),

                  //                  Mut(Select({}, Named("out", myStruct)), Alias(Select({}, Named("s", myStruct))),
                  //                  true), Return(Alias(UnitConst())),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def, def2});
  assertCompilationSucceeded(p);
}

TEST_CASE("struct alias", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {}, {Named("out", myStruct)}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Var(Named("t", myStruct), {Alias(Select({}, Named("s", myStruct)))}),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("struct mut", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {defX, defY});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {}, {Named("out", myStruct)}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Var(Named("t", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntConst(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntConst(43)), false),
                  Mut(Select({}, Named("s", myStruct)), Alias(Select({}, Named("t", myStruct))), false),

                  //                  Return(Alias(UnitConst())),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("nested struct alloc", "[compiler]") {
  polyregion_initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());

  StructDef def2(myStruct2Sym, {defX});
  Type::Struct myStruct2(myStruct2Sym);
  Named defZ = Named("z", myStruct2);

  StructDef def(myStructSym, {defX, defY, defZ});
  Type::Struct myStruct(myStructSym);

  Function fn(Sym({"foo"}), {}, {}, Type::Int(),
              {
                  Var(Named("t", myStruct2), {}),
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntConst(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntConst(43)), false),
                  Mut(Select({Named("s", myStruct)}, defZ), Alias(Select({}, Named("t", myStruct2))), false),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def2, def});
  assertCompilationSucceeded(p);
}

TEST_CASE("array alloc", "[compiler]") {
  polyregion_initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("t", arr), {Alloc(arr, IntConst(20))}),
                  Var(Named("u", arr), {Alloc(arr, IntConst(30))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array mut", "[compiler]") {
  polyregion_initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("t", arr), {Alloc(arr, IntConst(20))}),
                  Var(Named("u", arr), {Alloc(arr, IntConst(30))}),
                  Mut(Select({}, Named("s", arr)), Alias(Select({}, Named("t", arr))), false),
                  Mut(Select({}, Named("t", arr)), Alias(Select({}, Named("u", arr))), false),
                  Mut(Select({}, Named("t", arr)), Alias(Select({}, Named("s", arr))), false),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array alias", "[compiler]") {
  polyregion_initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("t", arr), {Alias(Select({}, Named("s", arr)))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("cast expr", "[compiler]") {
  polyregion_initialise();

  Function fn(Sym({"foo"}), {}, {}, Type::Int(),
              {
                  Var(Named("d", Type::Double()), {Cast(IntConst(10), Type::Double())}),
                  Var(Named("i", Type::Int()), {Cast(Select({}, Named("d", Type::Double())), Type::Int())}),

                  Return(Alias(Select({}, Named("i", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  polyregion_initialise();

  //  auto from  = DoubleConst(0x1.fffffffffffffP+1023);
  auto from = FloatConst(0x1.fffffeP+127f);
  //    auto from  = IntConst( (1<<31)-1);
  auto to = Type::Int();

  Function fn(Sym({"foo"}), {}, {}, to,
              {
                  Var(Named("i", from.tpe), {Alias(from)}),

                  Var(Named("d", to), {Cast(Select({}, Named("i", from.tpe)), to)}),

                  Return(Alias(Select({}, Named("d", to)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}