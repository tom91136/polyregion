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
  auto c = polyregion::compiler::compile(
      p, polyregion::compiler::Options{polyregion::compiler::Target::Object_LLVM_x86_64, "native"},
      polyregion::compiler::Opt::O3);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                                //
      Sym({"foo"}), {},                                             //
      {}, {Named("a", Type::Int()), Named("b", Type::Float())}, {}, //
      Type::Unit(),                                                 //
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

TEST_CASE("initialise more than once", "[compiler]") {
  polyregion::compiler::initialise();
  polyregion::compiler::initialise();
  polyregion::compiler::initialise();
  polyregion::compiler::initialise();
}

// TEST_CASE("empty function should compile", "[compiler]") {
//   polyregion::compiler::initialise();
//   Function fn(Sym({"foo"}), {}, Type::Unit(), {});
//   Program p(fn, {}, {});
//   assertCompilationSucceeded(p);
// }

TEST_CASE("index prim array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("xs", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("x", Type::Int()), {Index(Select({}, Named("xs", arr)), IntConst(0), Type::Int())}),
                  Return(Alias(Select({}, Named("x", Type::Int())))),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("index bool array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Bool());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Bool(),
              {
                  Var(Named("xs", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("x", Type::Bool()), {Index(Select({}, Named("xs", arr)), IntConst(0), Type::Bool())}),
                  Return(Alias(Select({}, Named("x", Type::Bool())))),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("index unit array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Unit());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Unit(),
              {
                  Var(Named("xs", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("x", Type::Unit()), {Index(Select({}, Named("xs", arr)), IntConst(0), Type::Unit())}),
                  Return(Alias(Select({}, Named("x", Type::Unit())))),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("index struct array member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("s", Type::Array(myStruct))}, {}, Type::Int(),
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

TEST_CASE("array update struct elem member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("xs", Type::Array(myStruct))}, {}, Type::Int(),
      {
          Var(Named("x", myStruct), Index(Select({}, Named("xs", Type::Array(myStruct))), IntConst(0), myStruct)),
          Mut(Select({Named("x", myStruct)}, defX), Alias(IntConst(42)), false),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update struct elem", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("xs", Type::Array(myStruct))}, {}, Type::Int(),
      {
          Var(Named("data", myStruct), {}),
          Update(Select({}, Named("xs", Type::Array(myStruct))), IntConst(7), Select({}, Named("data", myStruct))),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update unit ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Unit()))}, {}, Type::Unit(),
              {
                  Update(Select({}, Named("s", Type::Array(Type::Unit()))), IntConst(7), (UnitConst())),
                  Return(Alias(UnitConst())),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update bool ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Bool()))}, {}, Type::Unit(),
              {
                  Update(Select({}, Named("s", Type::Array(Type::Bool()))), IntConst(7), (BoolConst(true))),
                  Return(Alias(UnitConst())),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update prim ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Int()))}, {}, Type::Unit(),
              {
                  Update(Select({}, Named("s", Type::Array(Type::Int()))), IntConst(7), (IntConst(42))),
                  Return(Alias(UnitConst())),
              });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update unit from arg ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(
      Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Unit())), Named("x", Type::Unit())}, {}, Type::Unit(),
      {
          Update(Select({}, Named("s", Type::Array(Type::Unit()))), IntConst(7), Select({}, Named("x", Type::Unit()))),
          Return(Alias(UnitConst())),
      });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update bool from arg ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(
      Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Bool())), Named("x", Type::Bool())}, {}, Type::Unit(),
      {
          Update(Select({}, Named("s", Type::Array(Type::Bool()))), IntConst(7), Select({}, Named("x", Type::Bool()))),
          Return(Alias(UnitConst())),
      });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("array update prim from arg ", "[compiler]") {
  polyregion::compiler::initialise();
  Function fn(
      Sym({"foo"}), {}, {}, {Named("s", Type::Array(Type::Int())), Named("x", Type::Int())}, {}, Type::Unit(),
      {
          Update(Select({}, Named("s", Type::Array(Type::Int()))), IntConst(7), Select({}, Named("x", Type::Int()))),
          Return(Alias(UnitConst())),
      });
  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("alias struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Var(Named("t", myStruct), {Alias(Select({}, Named("s", myStruct)))}),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def});
  assertCompilationSucceeded(p);
}

TEST_CASE("alias struct member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, {}, {Named("x", Type::Int()), Named("y", Type::Int())});
  Named arg("in", Type::Struct(sdef, {}, {}));
  Function fn(Sym({"foo"}), {}, {}, {arg}, {}, Type::Unit(),
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

TEST_CASE("alias array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("t", arr), {Alias(Select({}, Named("s", arr)))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("alias prim", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("s", Type::Int()), {Alias(IntConst(10))}),
                  Var(Named("t", Type::Int()), {Alias(Select({}, Named("s", Type::Int())))}),
                  Return(Alias(Select({}, Named("t", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("mut struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, Type::Int(),
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

TEST_CASE("mut array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, arr,
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

TEST_CASE("mut prim", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("s", Type::Int()), {Alias(IntConst(10))}),
                  Mut(Select({}, Named("s", Type::Int())), Alias(IntConst(20)), false),
                  Return(Alias(Select({}, Named("s", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("alloc struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, {}, {defX, defY});
  Type::Struct myStruct(myStructSym, {}, {});
  StructDef def2(myStruct2Sym, {}, {defX});
  Type::Struct myStruct2(myStruct2Sym, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, Type::Int(),
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

TEST_CASE("alloc struct nested", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());

  StructDef def2(myStruct2Sym, {}, {defX});
  Type::Struct myStruct2(myStruct2Sym, {}, {});
  Named defZ = Named("z", myStruct2);

  StructDef def(myStructSym, {}, {defX, defY, defZ});
  Type::Struct myStruct(myStructSym, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("t", myStruct2), {}),
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntConst(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntConst(43)), false),
                  Mut(Select({Named("s", myStruct)}, defZ), Alias(Select({}, Named("t", myStruct2))), false),
                  Mut(Select({Named("s", myStruct), defZ}, defX), Alias(IntConst(42)), false),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def2, def});
  assertCompilationSucceeded(p);
}

TEST_CASE("alloc array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr, IntConst(10))}),
                  Var(Named("t", arr), {Alloc(arr, IntConst(20))}),
                  Var(Named("u", arr), {Alloc(arr, IntConst(30))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("cast expr", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("d", Type::Double()), {Cast(IntConst(10), Type::Double())}),
                  Var(Named("i", Type::Int()), {Cast(Select({}, Named("d", Type::Double())), Type::Int())}),

                  Return(Alias(Select({}, Named("i", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  polyregion::compiler::initialise();

  //  auto from  = DoubleConst(0x1.fffffffffffffP+1023);
  auto from = FloatConst(0x1.fffffeP+127f);
  //    auto from  = IntConst( (1<<31)-1);
  auto to = Type::Int();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, to,
              {
                  Var(Named("i", from.tpe), {Alias(from)}),

                  Var(Named("d", to), {Cast(Select({}, Named("i", from.tpe)), to)}),

                  Return(Alias(Select({}, Named("d", to)))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("cond", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("out", Type::Int()), {}),
                  Cond(Alias(BoolConst(true)),                                                   //
                       {Mut(Select({}, Named("out", Type::Int())), Alias(IntConst(42)), false)}, //
                       {Mut(Select({}, Named("out", Type::Int())), Alias(IntConst(43)), false)}  //
                       ),

                  Return(Alias(Select({}, Named("out", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}

TEST_CASE("while false", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Unit(),
              {
                  While({}, BoolConst(false), {}),

                  Return(Alias(UnitConst())),
              });

  Program p(fn, {}, {});
  assertCompilationSucceeded(p);
}
