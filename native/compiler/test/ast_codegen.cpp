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
using namespace Intr;
using namespace Spec;
using namespace Math;

using namespace polyregion;
using namespace polyast::dsl;

static std::vector<Tpe::Any> PrimitiveTypes = {
    Type::Float16(), //
    Type::Float32(), //
    Type::Float64(), //
    Type::Bool1(),   //
    Type::IntS8(),   //
    Type::IntS16(),  //
    Type::IntS32(),  //
    Type::IntS64(),  //
    Type::IntU8(),   //
    Type::IntU16(),  //
    Type::IntU32(),  //
    Type::IntU64(),  //
    Type::Unit0(),
};

Term::Any generateConstValue(Tpe::Any t) {
  auto unsupported = [&]() -> Term::Any { throw std::logic_error("No constant for type " + to_string(t)); };
  return variants::total(
      *t,                                                                 //
      [&](const Type::Float16 &) -> Term::Any { return 42.0_(t); },       //
      [&](const Type::Float32 &) -> Term::Any { return 42.0_(t); },       //
      [&](const Type::Float64 &) -> Term::Any { return 42.0_(t); },       //
      [&](const Type::Bool1 &) -> Term::Any { return Bool1Const(true); }, //

      [&](const Type::IntS8 &) -> Term::Any { return 0xFF_(t); },        //
      [&](const Type::IntS16 &) -> Term::Any { return 42_(t); },         //
      [&](const Type::IntS32 &) -> Term::Any { return 42_(t); },         //
      [&](const Type::IntS64 &) -> Term::Any { return 0xDEADBEEF_(t); }, //

      [&](const Type::IntU8 &) -> Term::Any { return 0xFF_(t); },        //
      [&](const Type::IntU16 &) -> Term::Any { return 42_(t); },         //
      [&](const Type::IntU32 &) -> Term::Any { return 42_(t); },         //
      [&](const Type::IntU64 &) -> Term::Any { return 0xDEADBEEF_(t); }, //

      [&](const Type::Unit0 &t) -> Term::Any { return Unit0Const(); },    //
      [&](const Type::Nothing &t) -> Term::Any { return unsupported(); }, //
      [&](const Type::Struct &t) -> Term::Any { return unsupported(); },  //
      [&](const Type::Array &t) -> Term::Any { return unsupported(); },   //
      [&](const Type::Var &t) -> Term::Any { return unsupported(); },     //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(); });
}

template <typename P> static void assertCompile(const P &p) {
  CAPTURE(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_x86_64, "native"},
                                         OptLevel::O3);
  CAPTURE(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
  std::cout << (c) << std::endl;
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                                                           //
      Sym({"foo"}), {},                                                                        //
      {}, {Arg(Named("a", Type::IntS32()), {}), Arg(Named("b", Type::Float32()), {})}, {}, {}, //
      Type::Unit0(),                                                                           //
      {
          Comment("a"), //
          Comment("b"), //
          Var(Named("a", SInt), {SpecOp(GpuGlobalSize(IntS32Const(1)))}),
          Break(),                               //
          Return(Alias(IntS32Const(1))),         //
          Cond(                                  //
              Alias(Bool1Const(true)),           //
              {Return(Alias(IntS32Const(1)))},   //
              {Return(Alias(Bool1Const(false)))} //
              )                                  //
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

  auto entry = function("foo", {"in0"_(SInt)(), "in1"_(SInt)()}, SInt)({
      Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
           {
               Cond(IntrOp(LogicEq("in1"_(SInt), 42_(SInt))),
                    {
                        ret(1_(SInt)) //
                    },
                    {
                        ret(2_(SInt)) //
                    }),
           },
           {
               ret(3_(SInt)) //
           }),
  });
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("nested nested if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)(), "in1"_(SInt)(), "in2"_(SInt)()}, SInt)({
      Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
           {
               Cond(IntrOp(LogicEq("in1"_(SInt), 42_(SInt))),
                    {
                        Cond(IntrOp(LogicEq("in2"_(SInt), 42_(SInt))),
                             {
                                 ret(1_(SInt)) //
                             },
                             {
                                 ret(2_(SInt)) //
                             }),
                    },
                    {
                        ret(3_(SInt)) //
                    }),
           },
           {
               ret(4_(SInt)) //
           }),
  });
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)()}, SInt)({Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
                                                             {
                                                                 ret(1_(SInt)) //
                                                             },
                                                             {
                                                                 ret(3_(SInt)) //
                                                             })});
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("code after if", "[compiler]") {
  polyregion::compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)()},
                        SInt)({let("x") = 0_(SInt), //
                               Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
                                    {
                                        Mut("x"_(SInt), Alias(1_(SInt)), true) //
                                    },
                                    {
                                        Mut("x"_(SInt), Alias(3_(SInt)), true) //
                                    }),
                               ret("x"_(SInt))});
  assertCompile(program(entry, {}, {}));
}

TEST_CASE("fn call arg0", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    auto callee = function("bar", {}, tpe)({
        ret(generateConstValue(tpe)) //
    });
    auto entry = function("foo", {}, tpe)({
        ret(Invoke(Sym({"bar"}), {}, {}, {}, {}, tpe)) //
    });
    assertCompile(program(entry, {}, {callee}));
  }
}

TEST_CASE("fn call arg1", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    auto callee = function("bar", {"a"_(tpe)()}, tpe)({
        ret("a"_(tpe)) //
    });
    auto entry = function("foo", {}, tpe)({
        ret(Invoke(Sym({"bar"}), {}, {}, {generateConstValue(tpe)}, {}, tpe)) //
    });
    assertCompile(program(entry, {}, {callee}));
  }
}

TEST_CASE("fn call arg2", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe0 = GENERATE(from_range(PrimitiveTypes));
  auto tpe1 = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe0 << "," << tpe1) {
    CAPTURE(tpe0);
    CAPTURE(tpe1);
    auto callee = function("bar", {"a"_(tpe0)(), "b"_(tpe1)()}, tpe0)({
        ret("a"_(tpe0)) //
    });
    auto entry = function("foo", {}, tpe0)({
        ret(Invoke(Sym({"bar"}), {}, {}, {generateConstValue(tpe0), generateConstValue(tpe1)}, {}, tpe0)) //
    });
    assertCompile(program(entry, {}, {callee}));
  }
}

TEST_CASE("fn call arg3", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe0 = GENERATE(from_range(PrimitiveTypes));
  auto tpe1 = GENERATE(from_range(PrimitiveTypes));
  auto tpe2 = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe0 << "," << tpe1 << "," << tpe2) {
    CAPTURE(tpe0);
    CAPTURE(tpe1);
    CAPTURE(tpe2);
    auto callee = function("bar", {"a"_(tpe0)(), "b"_(tpe1)(), "c"_(tpe2)()}, tpe0)({
        ret("a"_(tpe0)) //
    });
    auto entry = function("foo", {}, tpe0)({
        ret(Invoke(Sym({"bar"}), {}, {}, {generateConstValue(tpe0), generateConstValue(tpe1), generateConstValue(tpe2)}, {}, tpe0)) //
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
    assertCompile(program(function("foo", {"in"_(tpe)()}, tpe)({
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
        let("xs") = Alloc(tpe, 42_(SInt)),                 //
        let("x") = "xs"_(Array(tpe))[integral(SInt, idx)], //
        ret("x"_(tpe))                                     //
    })));
  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Array(tpe))()}, tpe)({
        let("x") = "xs"_(Array(tpe))[integral(SInt, idx)], //
        ret("x"_(tpe))                                     //
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
        let("xs") = Alloc(tpe, 42_(SInt)),            //
        "xs"_(Array(tpe))[integral(SInt, idx)] = val, //
        ret("xs"_(Array(tpe))[integral(SInt, idx)])   //
    })));
  }
  DYNAMIC_SECTION("(xs[" << idx << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Array(tpe))()}, tpe)({
        "xs"_(Array(tpe))[integral(SInt, idx)] = val, //
        ret("xs"_(Array(tpe))[integral(SInt, idx)])   //
    })));
  }
}

TEST_CASE("update prim array by ref", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  auto idx = GENERATE(std::optional<int>{}, 0, 1, 3, 7, 10);
  auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(SInt)), //
        let("ref") = RefTo("xs"_(Array(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe),
        ret("ref"_(Array(tpe))[integral(SInt, 0)]) //
    })));
  }
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Array(tpe))()}, tpe)({
        let("ref") = RefTo("xs"_(Array(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe),
        ret("ref"_(Array(tpe))[integral(SInt, 0)]) //
    })));
  }
}

TEST_CASE("update value by ref", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypes));
  auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("x") = val,                          //
        let("y") = RefTo("x"_(tpe), {}, tpe),    //
        ret("y"_(Array(tpe))[integral(SInt, 0)]) //
    })));
  }
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {"x"_(tpe)()}, tpe)({
        let("y") = RefTo("x"_(tpe), {}, tpe),    //
        ret("y"_(Array(tpe))[integral(SInt, 0)]) //
    })));
  }
}

TEST_CASE("index struct array member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("s", Array(myStruct)), {})}, {}, {}, Type::IntS32(),
              {

                  Var(Named("a", myStruct), {Index(Select({}, Named("s", Array(myStruct))), Term::IntS32Const(0), myStruct)}),

                  Var(Named("b", Type::IntS32()), {Alias(Select({Named("a", myStruct)}, defX))}),

                  //                  Mut(Select({Named("s",  Array(myStruct ))}, defX), Alias(IntS32Const(42)), false),
                  Return(Alias(IntS32Const(69))),
              });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("array update struct elem member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("xs", Array(myStruct)), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("x", myStruct), Index(Select({}, Named("xs", Array(myStruct))), IntS32Const(0), myStruct)),
                  Mut(Select({Named("x", myStruct)}, defX), Alias(IntS32Const(42)), false),
                  Return(Alias(IntS32Const(69))),
              });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("array update struct elem", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("xs", Array(myStruct)), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("data", myStruct), {}),
                  Update(Select({}, Named("xs", Array(myStruct))), IntS32Const(7), Select({}, Named("data", myStruct))),
                  Return(Alias(IntS32Const(69))),
              });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("out", myStruct), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("s", myStruct), {}),
                  Var(Named("t", myStruct), {Alias(Select({}, Named("s", myStruct)))}),
                  Return(Alias(IntS32Const(69))),
              });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias struct member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, {}, {StructMember(Named("x", Type::IntS32()), false), StructMember(Named("y", Type::IntS32()), false)}, {});
  Named arg("in", Type::Struct(sdef, {}, {}, {}));
  Function fn(Sym({"foo"}), {}, {}, {Arg(arg, {})}, {}, {}, Type::Unit0(),
              {
                  Var(                             //
                      Named("y2", Type::IntS32()), //
                      {
                          Alias(Select({arg}, Named("y", Type::IntS32()))) //
                      }                                                    //
                      ),
                  Return(Alias(Unit0Const())),
              });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Array(Type::IntS32());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntS32Const(10))}),
                  Var(Named("t", arr), {Alias(Select({}, Named("s", arr)))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("mut struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});

  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("out", myStruct), {})}, {}, {}, Type::Unit0(),
              {
                  //                  Var(Named("s", myStruct), {}),
                  //                  Var(Named("t", myStruct), {}),

                  //                  Mut(Select({}, Named("t", myStruct)), Alias(Select({}, Named("out", myStruct))), false),

                  Mut(Select({Named("out", myStruct)}, defX), Alias(IntS32Const(42)), false),
                  Mut(Select({Named("out", myStruct)}, defY), Alias(IntS32Const(43)), false),
                  //                  Mut(Select({}, Named("s", myStruct)), Alias(Select({}, Named("t", myStruct))), false),

                  //                  Return(Alias(Unit0Const())),
                  Return(Alias(Unit0Const())),
              });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("mut array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Array(Type::IntS32());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntS32Const(10))}),
                  Var(Named("t", arr), {Alloc(arr.component, IntS32Const(20))}),
                  Var(Named("u", arr), {Alloc(arr.component, IntS32Const(30))}),
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

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::IntS32(),
              {
                  Var(Named("s", Type::IntS32()), {Alias(IntS32Const(10))}),
                  Mut(Select({}, Named("s", Type::IntS32())), Alias(IntS32Const(20)), false),
                  Return(Alias(Select({}, Named("s", Type::IntS32())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("alloc struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});
  StructDef def2(myStruct2Sym, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("out", myStruct), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntS32Const(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntS32Const(43)), false),
                  Var(Named("t", myStruct2), {}),

                  //                  Mut(Select({}, Named("out", myStruct)), Alias(Select({}, Named("s", myStruct))),
                  //                  true), Return(Alias(Unit0Const())),
                  Return(Alias(IntS32Const(69))),
              });

  Program p(fn, {}, {def, def2});
  assertCompile(p);
}

TEST_CASE("alloc struct nested", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());

  StructDef def2(myStruct2Sym, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});
  Named defZ = Named("z", myStruct2);

  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false), StructMember(defZ, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::IntS32(),
              {
                  Var(Named("t", myStruct2), {}),
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntS32Const(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntS32Const(43)), false),
                  Mut(Select({Named("s", myStruct)}, defZ), Alias(Select({}, Named("t", myStruct2))), false),
                  Mut(Select({Named("s", myStruct), defZ}, defX), Alias(IntS32Const(42)), false),
                  Return(Alias(IntS32Const(69))),
              });

  Program p(fn, {}, {def2, def});
  assertCompile(p);
}

TEST_CASE("alloc array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Array(Type::IntS32());

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.component, IntS32Const(10))}),
                  Var(Named("t", arr), {Alloc(arr.component, IntS32Const(20))}),
                  Var(Named("u", arr), {Alloc(arr.component, IntS32Const(30))}),
                  Return(Alias(Select({}, Named("s", arr)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cast expr", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::IntS32(),
              {
                  Var(Named("d", Type::Float64()), {Cast(IntS32Const(10), Type::Float64())}),
                  Var(Named("i", Type::IntS32()), {Cast(Select({}, Named("d", Type::Float64())), Type::IntS32())}),

                  Return(Alias(Select({}, Named("i", Type::IntS32())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  polyregion::compiler::initialise();

  //  auto from  = DoubleConst(0x1.fffffffffffffP+1023);
  auto from = Float32Const(0x1.fffffeP+127f);
  //    auto from  = IntS32Const( (1<<31)-1);
  auto to = Type::IntS32();

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

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::IntS32(),
              {
                  Var(Named("out", Type::IntS32()), {}),
                  Cond(Alias(Bool1Const(true)),                                                        //
                       {Mut(Select({}, Named("out", Type::IntS32())), Alias(IntS32Const(42)), false)}, //
                       {Mut(Select({}, Named("out", Type::IntS32())), Alias(IntS32Const(43)), false)}  //
                       ),

                  Return(Alias(Select({}, Named("out", Type::IntS32())))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("while false", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Type::Unit0(),
              {
                  While({}, Bool1Const(false), {}),

                  Return(Alias(Unit0Const())),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}
