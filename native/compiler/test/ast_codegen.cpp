#include "ast.h"
#include "catch2/catch_all.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

#include <iostream>

using namespace polyregion::polyast;

using namespace Stmt;
using namespace Term;
using namespace Expr;

using namespace polyregion;
using namespace polyast::dsl;

static std::vector<Tpe::Any> PrimitiveTypes = {Float, Double, Bool, Byte, Char, Short, Int, Long, Unit};

Term::Any generateConstValue(Tpe::Any t) {
  auto unsupported = [&]() -> Term::Any { throw std::logic_error("No constant for type " + to_string(t)); };
  return variants::total(
      *t, [&](const Type::Float &) -> Term::Any { return 42.0_(t); },     //
      [&](const Type::Double &) -> Term::Any { return 42.0_(t); },        //
      [&](const Type::Bool &) -> Term::Any { return BoolConst(true); },   //
      [&](const Type::Byte &) -> Term::Any { return 0xFF_(t); },          //
      [&](const Type::Char &) -> Term::Any { return 42_(t); },            //
      [&](const Type::Short &) -> Term::Any { return 42_(t); },           //
      [&](const Type::Int &) -> Term::Any { return 42_(t); },             //
      [&](const Type::Long &) -> Term::Any { return 0xDEADBEEF_(t); },    //
      [&](const Type::Unit &t) -> Term::Any { return UnitConst(); },      //
      [&](const Type::Nothing &t) -> Term::Any { return unsupported(); }, //
      [&](const Type::Struct &t) -> Term::Any { return unsupported(); },  //
      [&](const Type::Array &t) -> Term::Any { return unsupported(); },   //
      [&](const Type::Var &t) -> Term::Any { return unsupported(); },     //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(); });
}

template <typename P> static void assertCompile(const P &p) {
  CAPTURE(repr(p));
  auto c = polyregion::compiler::compile(
      p, polyregion::compiler::Options{polyregion::compiler::Target::Object_LLVM_x86_64, "native"},
      polyregion::compiler::Opt::O3);
  CAPTURE(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
  std::cout << (c) << std::endl;
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                                    //
      Sym({"foo"}), {},                                                 //
      {}, {Named("a", Type::Int()), Named("b", Type::Float())}, {}, {}, //
      Type::Unit(),                                                     //
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

TEST_CASE("nested if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(Int), "in1"_(Int)}, Int)({
      Cond(BinaryIntrinsic("in0"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
           {
               Cond(BinaryIntrinsic("in1"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
                    {
                        ret(1_(Int)) //
                    },
                    {
                        ret(2_(Int)) //
                    }),
           },
           {
               ret(3_(Int)) //
           }),
  });
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("nested nested if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(Int), "in1"_(Int), "in2"_(Int)}, Int)({
      Cond(BinaryIntrinsic("in0"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
           {
               Cond(BinaryIntrinsic("in1"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
                    {
                        Cond(BinaryIntrinsic("in2"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
                             {
                                 ret(1_(Int)) //
                             },
                             {
                                 ret(2_(Int)) //
                             }),
                    },
                    {
                        ret(3_(Int)) //
                    }),
           },
           {
               ret(4_(Int)) //
           }),
  });
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(Int)},
                        Int)({Cond(BinaryIntrinsic("in0"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
                                   {
                                       ret(1_(Int)) //
                                   },
                                   {
                                       ret(3_(Int)) //
                                   })});
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("code after if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(Int)},
                        Int)({let("x") = 0_(Int), //
                              Cond(BinaryIntrinsic("in0"_(Int), 42_(Int), BinaryIntrinsicKind::LogicEq(), Bool),
                                   {
                                       Mut("x"_(Int), Alias(1_(Int)), true) //
                                   },
                                   {
                                       Mut("x"_(Int), Alias(3_(Int)), true) //
                                   }),
                              ret("x"_(Int))});
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("fn call", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    auto callee = function("bar", {"a"_(tpe)}, tpe)({
        ret("a"_(tpe)) //
    });
    auto entry = function("foo", {}, tpe)({
        ret(Invoke(Sym({"bar"}), {}, {}, {generateConstValue(tpe)}, {}, tpe)) //
    });
    assertCompile(program(entry, {}, {callee}));
  }
}

TEST_CASE("constant return", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    assertCompile(program(function("foo", {}, tpe)({
        ret(generateConstValue(tpe)) //
    })));
  }
}

TEST_CASE("alias prim", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe << "(local)") {
    CAPTURE(tpe);
    assertCompile(program(function("foo", {}, tpe)({
        let("v1") = generateConstValue(tpe), //
        let("v2") = "v1"_(tpe),              //
        let("v3") = "v2"_(tpe),              //
        let("v4") = "v3"_(tpe),              //
        ret("v4"_(tpe))                      //
    })));
  }
  DYNAMIC_SECTION(tpe << "(arg)") {
    CAPTURE(tpe);
    assertCompile(program(function("foo", {"in"_(tpe)}, tpe)({
        let("v1") = "in"_(tpe), //
        let("v2") = "v1"_(tpe), //
        let("v3") = "v2"_(tpe), //
        let("v4") = "v3"_(tpe), //
        ret("v4"_(tpe))         //
    })));
  }
}

TEST_CASE("index prim array", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  auto idx = GENERATE(0, 1, 3, 7, 10);
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (local)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(Int)),                 //
        let("x") = "xs"_(Array(tpe))[integral(Int, idx)], //
        ret("x"_(tpe))                                    //
    })));
  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Array(tpe))}, tpe)({
        let("x") = "xs"_(Array(tpe))[integral(Int, idx)], //
        ret("x"_(tpe))                                    //
    })));
  }
}

TEST_CASE("update prim array", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  auto idx = GENERATE(0, 1, 3, 7, 10);
  auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(xs[" << idx << "]:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(Int)),            //
        "xs"_(Array(tpe))[integral(Int, idx)] = val, //
        ret("xs"_(Array(tpe))[integral(Int, idx)])   //
    })));
  }
  invoke(Fn0::GpuGlobalIdxX(), Int);
  DYNAMIC_SECTION("(xs[" << idx << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Array(tpe))}, tpe)({
        "xs"_(Array(tpe))[integral(Int, idx)] = val, //
        ret("xs"_(Array(tpe))[integral(Int, idx)])   //
    })));
  }
}

TEST_CASE("index struct array member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("s", Type::Array(myStruct))}, {}, {}, Type::Int(),
      {

          Var(Named("a", myStruct),
              {Index(Select({}, Named("s", Type::Array(myStruct))), Term::IntConst(0), myStruct)}),

          Var(Named("b", Type::Int()), {Alias(Select({Named("a", myStruct)}, defX))}),

          //                  Mut(Select({Named("s", Type::Array(myStruct ))}, defX), Alias(IntConst(42)), false),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("array update struct elem member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("xs", Type::Array(myStruct))}, {}, {}, Type::Int(),
      {
          Var(Named("x", myStruct), Index(Select({}, Named("xs", Type::Array(myStruct))), IntConst(0), myStruct)),
          Mut(Select({Named("x", myStruct)}, defX), Alias(IntConst(42)), false),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("array update struct elem", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("xs", Type::Array(myStruct))}, {}, {}, Type::Int(),
      {
          Var(Named("data", myStruct), {}),
          Update(Select({}, Named("xs", Type::Array(myStruct))), IntConst(7), Select({}, Named("data", myStruct))),
          Return(Alias(IntConst(69))),
      });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, {}, Type::Int(),
              {
                  Var(Named("s", myStruct), {}),
                  Var(Named("t", myStruct), {Alias(Select({}, Named("s", myStruct)))}),
                  Return(Alias(IntConst(69))),
              });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias struct member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, true, {},
                {StructMember(Named("x", Type::Int()), false), StructMember(Named("y", Type::Int()), false)}, {});
  Named arg("in", Type::Struct(sdef, {}, {}, {}));
  Function fn(Sym({"foo"}), {}, {}, {arg}, {}, {}, Type::Unit(),
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
  assertCompile(p);
}

TEST_CASE("alias array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntConst(10))}),
                  Var(Named("t", arr), {Alias(Select({}, Named("s", arr)))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("mut struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(
      Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, {}, Type::Unit(),
      {
          //                  Var(Named("s", myStruct), {}),
          //                  Var(Named("t", myStruct), {}),

          //                  Mut(Select({}, Named("t", myStruct)), Alias(Select({}, Named("out", myStruct))), false),

          Mut(Select({Named("out", myStruct)}, defX), Alias(IntConst(42)), false),
          Mut(Select({Named("out", myStruct)}, defY), Alias(IntConst(43)), false),
          //                  Mut(Select({}, Named("s", myStruct)), Alias(Select({}, Named("t", myStruct))), false),

          //                  Return(Alias(UnitConst())),
          Return(Alias(UnitConst())),
      });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("mut array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntConst(10))}),
                  Var(Named("t", arr), {Alloc(arr.component, IntConst(20))}),
                  Var(Named("u", arr), {Alloc(arr.component, IntConst(30))}),
                  Mut(Select({}, Named("s", arr)), Alias(Select({}, Named("t", arr))), false),
                  Mut(Select({}, Named("t", arr)), Alias(Select({}, Named("u", arr))), false),
                  Mut(Select({}, Named("t", arr)), Alias(Select({}, Named("s", arr))), false),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("mut prim", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("s", Type::Int()), {Alias(IntConst(10))}),
                  Mut(Select({}, Named("s", Type::Int())), Alias(IntConst(20)), false),
                  Return(Alias(Select({}, Named("s", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("alloc struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());
  StructDef def(myStructSym, true, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});
  StructDef def2(myStruct2Sym, true, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Named("out", myStruct)}, {}, {}, Type::Int(),
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
  assertCompile(p);
}

TEST_CASE("alloc struct nested", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::Int());
  Named defY = Named("y", Type::Int());

  StructDef def2(myStruct2Sym, true, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});
  Named defZ = Named("z", myStruct2);

  StructDef def(myStructSym, true, {},
                {StructMember(defX, false), StructMember(defY, false), StructMember(defZ, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Int(),
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
  assertCompile(p);
}

TEST_CASE("alloc array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Type::Array(Type::Int());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntConst(10))}),
                  Var(Named("t", arr), {Alloc(arr.component, IntConst(20))}),
                  Var(Named("u", arr), {Alloc(arr.component, IntConst(30))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cast expr", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("d", Type::Double()), {Cast(IntConst(10), Type::Double())}),
                  Var(Named("i", Type::Int()), {Cast(Select({}, Named("d", Type::Double())), Type::Int())}),

                  Return(Alias(Select({}, Named("i", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  polyregion::compiler::initialise();

  //  auto from  = DoubleConst(0x1.fffffffffffffP+1023);
  auto from = FloatConst(0x1.fffffeP+127f);
  //    auto from  = IntConst( (1<<31)-1);
  auto to = Type::Int();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, to,
              {
                  Var(Named("i", from.tpe), {Alias(from)}),

                  Var(Named("d", to), {Cast(Select({}, Named("i", from.tpe)), to)}),

                  Return(Alias(Select({}, Named("d", to)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cond", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Int(),
              {
                  Var(Named("out", Type::Int()), {}),
                  Cond(Alias(BoolConst(true)),                                                   //
                       {Mut(Select({}, Named("out", Type::Int())), Alias(IntConst(42)), false)}, //
                       {Mut(Select({}, Named("out", Type::Int())), Alias(IntConst(43)), false)}  //
                       ),

                  Return(Alias(Select({}, Named("out", Type::Int())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("while false", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Unit(),
              {
                  While({}, BoolConst(false), {}),

                  Return(Alias(UnitConst())),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}
