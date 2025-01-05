#include "ast.h"
#include "catch2/catch_all.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

using namespace polyregion::polyast;
using namespace polyregion::compiletime;
using Catch::Matchers::ContainsSubstring;

using namespace Stmt;
using namespace Expr;
using namespace Intr;
using namespace Spec;
using namespace Math;

using namespace polyregion;
using namespace polyast::dsl;

static std::vector<Tpe::Any> PrimitiveTypesNoUnit = {
    Float,          //
    Double,         //
    Bool,           //
    Byte,           //
    Short,          //
    Char,           //
    SInt,           //
    UInt,           //
    Long,           //
    Type::IntU8(),  //
    Type::IntU64(), //
};

static std::vector<Tpe::Any> PrimitiveTypes = {
    Float,          //
    Double,         //
    Bool,           //
    Byte,           //
    Short,          //
    Char,           //
    SInt,           //
    UInt,           //
    Long,           //
    Unit,           //
    Type::IntU8(),  //
    Type::IntU64(), //
};

Expr::Any generateConstValue(const Tpe::Any &t) {
  const auto unsupported = [&]() -> Expr::Any { throw std::logic_error("No constant for type " + to_string(t)); };
  return t.match_total(                                                   //
      [&](const Type::Float16 &) -> Expr::Any { return 42.0_(t); },       //
      [&](const Type::Float32 &) -> Expr::Any { return 42.0_(t); },       //
      [&](const Type::Float64 &) -> Expr::Any { return 42.0_(t); },       //
      [&](const Type::Bool1 &) -> Expr::Any { return Bool1Const(true); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return 0xFF_(t); },        //
      [&](const Type::IntS16 &) -> Expr::Any { return 42_(t); },         //
      [&](const Type::IntS32 &) -> Expr::Any { return 42_(t); },         //
      [&](const Type::IntS64 &) -> Expr::Any { return 0xDEADBEEF_(t); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return 0xFF_(t); },        //
      [&](const Type::IntU16 &) -> Expr::Any { return 42_(t); },         //
      [&](const Type::IntU32 &) -> Expr::Any { return 42_(t); },         //
      [&](const Type::IntU64 &) -> Expr::Any { return 0xDEADBEEF_(t); }, //

      [&](const Type::Unit0 &) -> Expr::Any { return Unit0Const(); },       //
      [&](const Type::Nothing &) -> Expr::Any { return unsupported(); },    //
      [&](const Type::Struct &) -> Expr::Any { return unsupported(); },     //
      [&](const Type::Ptr &) -> Expr::Any { return unsupported(); },        //
      [&](const Type::Annotated &) -> Expr::Any { return unsupported(); }); //
}

template <typename P> static void assertCompile(const P &p) {
  CAPTURE(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_x86_64, "native"}, OptLevel::O3);
  CAPTURE(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
  for (auto e : c.events) {
    REQUIRE_THAT(e.data, !ContainsSubstring(" undef"));
    REQUIRE_THAT(e.data, !ContainsSubstring("unreachable"));
  }
  //  FAIL("debug");
  //  INFO(c);
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                                                   //
      "foo", {Arg(Named("a", Type::IntS32()), {}), Arg(Named("b", Double), {})}, Unit, //
      {
          Comment("a"), //
          Comment("b"), //
          Var(Named("a", SInt), {SpecOp(GpuGlobalSize(IntS32Const(1)))}),
          Break(),                        //
          Return(IntS32Const(1)),         //
          Cond(                           //
              Bool1Const(true),           //
              {Return(IntS32Const(1))},   //
              {Return(Bool1Const(false))} //
              )                           //
      },
      {FunctionAttr::Exported()});
  auto actual = function_from_json(function_to_json(expected));
  CHECK(expected == actual);
}

TEST_CASE("initialise more than once", "[compiler]") {
  compiler::initialise();
  compiler::initialise();
  compiler::initialise();
  compiler::initialise();
}

TEST_CASE("inheritance", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {
    Named x("x", tpe);
    StructDef def("foo", {x});
    Type::Struct fooTpe("foo");
    auto entry = function("foo", {}, fooTpe)({
        let("foo") = fooTpe, //
        Mut(Select({"foo"_(fooTpe)}, x), generateConstValue(tpe)),
        ret("foo"_(fooTpe)) //
    });
    assertCompile(program({def}, {entry}));
  }
}

TEST_CASE("nested if", "[compiler]") {
  compiler::initialise();

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
  assertCompile(program({}, {entry}));
}

TEST_CASE("nested nested if", "[compiler]") {
  compiler::initialise();

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
  assertCompile(program({}, {entry}));
}

TEST_CASE("if", "[compiler]") {
  compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)()}, SInt)({Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
                                                             {
                                                                 ret(1_(SInt)) //
                                                             },
                                                             {
                                                                 ret(3_(SInt)) //
                                                             })});
  assertCompile(program({}, {entry}));
}

TEST_CASE("code after if", "[compiler]") {
  compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)()},
                        SInt)({let("x") = 0_(SInt), //
                               Cond(IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
                                    {
                                        Mut("x"_(SInt), 1_(SInt)) //
                                    },
                                    {
                                        Mut("x"_(SInt), 3_(SInt)) //
                                    }),
                               ret("x"_(SInt))});
  assertCompile(program({}, {entry}));
}

TEST_CASE("return ptr to struct", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {

    Named x("x", tpe);
    StructDef def("foo", {x});
    Type::Struct fooTpe("foo");
    auto entry = function("foo", {"foo"_(Ptr(fooTpe))()}, Ptr(fooTpe))({
        Mut(Select({"foo"_(Ptr(fooTpe))}, x), generateConstValue(tpe)),
        ret("foo"_(Ptr(fooTpe))) //
    });
    assertCompile(program({def}, {entry}));
  }
}

TEST_CASE("return struct", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {

    Named x("x", tpe);
    StructDef def("foo", {x});
    Type::Struct fooTpe("foo");
    auto entry = function("foo", {}, fooTpe)({
        let("foo") = fooTpe, //
        Mut(Select({"foo"_(fooTpe)}, x), generateConstValue(tpe)),
        ret("foo"_(fooTpe)) //
    });
    assertCompile(program({def}, {entry}));
  }
}

TEST_CASE("nested struct select", "[compiler]") {
  compiler::initialise();

  Named z("z", Ptr(SInt));
  StructDef barDef("bar", {z});
  Type::Struct barTpe("bar");

  Named x("x", Ptr(SInt));
  Named y("y", Ptr(barTpe));
  StructDef fooDef("foo", {x, y});
  Type::Struct fooTpe("foo");

  auto aux = function("aux", {"in"_(Ptr(barTpe))()}, SInt)({
      ret(Index(Select({"in"_(Ptr(barTpe))}, z), 0_(SInt), SInt)) //
  });

  auto entry = function("bar", {"in"_(Ptr(fooTpe))()}, Unit)({
      let("r") = Invoke("aux", {Select({"in"_(Ptr(fooTpe))}, y)}, SInt), //
      Update(Select({"in"_(Ptr(fooTpe))}, x), 0_(SInt), "r"_(SInt)),
      ret(Unit0Const()) //
  });

  assertCompile(program({fooDef, barDef}, {entry, aux}));
}

TEST_CASE("return struct and take ref", "[compiler]") {
  compiler::initialise();

  Named z("z", SInt);
  StructDef barDef("bar", {z});
  Type::Struct barTpe("bar");

  Named x("x", Ptr(SInt));
  Named y("y", Ptr(barTpe));
  StructDef fooDef("foo", {x, y});
  Type::Struct fooTpe("foo");

  auto aux = function("aux", {"out"_(Ptr(barTpe))(), "in"_(Ptr(barTpe))()}, Ptr(barTpe))({
      Mut(Select({"out"_(Ptr(barTpe))}, z), //
          Select({"in"_(Ptr(barTpe))}, z)), //
      ret("out"_(Ptr(barTpe)))              //
  });

  auto gen = function("gen", {}, barTpe)({let("a") = barTpe,                        //
                                          Mut(Select({"a"_(barTpe)}, z), 0_(SInt)), //
                                          ret("a"_(barTpe))});

  auto entry = function("bar", {"in"_(Ptr(fooTpe))()}, Unit)({
      let("a") = Invoke("gen", {}, barTpe),                                                         //
      let("ar") = RefTo("a"_(barTpe), {}, barTpe, Global),                                          //
      let("r") = Invoke("aux", {Select({"in"_(Ptr(fooTpe))}, y), "ar"_(Ptr(barTpe))}, Ptr(barTpe)), //
      ret(Unit0Const())                                                                             //
  });

  assertCompile(program({fooDef, barDef}, {entry, gen, aux}));
}

TEST_CASE("fn call arg0", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    auto callee = function("bar", {}, tpe)({
        ret(generateConstValue(tpe)) //
    });
    auto entry = function("foo", {}, tpe)({
        ret(Invoke("bar", {}, tpe)) //
    });
    assertCompile(program({}, {entry, callee}));
  }
}

TEST_CASE("fn call arg1", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    auto callee = function("bar", {"a"_(tpe)()}, tpe)({
        ret("a"_(tpe)) //
    });
    auto entry = function("foo", {}, tpe)({
        ret(Invoke("bar", {generateConstValue(tpe)}, tpe)) //
    });
    assertCompile(program({}, {entry, callee}));
  }
}

TEST_CASE("fn call arg2", "[compiler]") {
  compiler::initialise();
  auto tpe0 = GENERATE(from_range(PrimitiveTypes));
  auto tpe1 = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe0 << "," << tpe1) {
    CAPTURE(tpe0);
    CAPTURE(tpe1);
    auto callee = function("bar", {"a"_(tpe0)(), "b"_(tpe1)()}, tpe0)({
        ret("a"_(tpe0)) //
    });
    auto entry = function("foo", {}, tpe0)({
        ret(Invoke("bar", {generateConstValue(tpe0), generateConstValue(tpe1)}, tpe0)) //
    });
    assertCompile(program({}, {entry, callee}));
  }
}

TEST_CASE("fn call arg3", "[compiler]") {
  compiler::initialise();
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
        ret(Invoke("bar", {generateConstValue(tpe0), generateConstValue(tpe1), generateConstValue(tpe2)}, tpe0)) //
    });
    assertCompile(program({}, {entry, callee}));
  }
}

TEST_CASE("constant return", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
  DYNAMIC_SECTION(tpe) {
    CAPTURE(tpe);
    assertCompile(program(function("foo", {}, tpe)({
        ret(generateConstValue(tpe)) //
    })));
  }
}

TEST_CASE("alias prim", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
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

TEST_CASE("index sized prim array", "[compiler]") {
  compiler::initialise();
  const auto tpe = SInt; // GENERATE(from_range(PrimitiveTypes));

  auto idx = 3; // GENERATE(0, 1, 3, 7, 10);
  //  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (local)") {
  //    CAPTURE(tpe, idx);
  //    assertCompile(program(function("foo", {}, tpe)({
  //        let("xs") = Alloc(tpe, 42_(SInt)),
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstValue(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe, 10))()}, tpe)({
        let("x") = "xs"_(Ptr(tpe, 10))[integral(SInt, idx)], //
        ret("x"_(tpe))                                       //
    })));
  }
}

TEST_CASE("mut sized prim array", "[compiler]") {
  compiler::initialise();
  const auto tpe = SInt; // GENERATE(from_range(PrimitiveTypes));

  auto idx = 3; // GENERATE(0, 1, 3, 7, 10);
  //  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (local)") {
  //    CAPTURE(tpe, idx);
  //    assertCompile(program(function("foo", {}, tpe)({
  //        let("xs") = Alloc(tpe, 42_(SInt)),
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstValue(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe, 10))()}, Ptr(tpe, 10))({
        "xs"_(Ptr(tpe, 10))[integral(SInt, idx)] = integral(SInt, 42), //
        ret("xs"_(Ptr(tpe, 10)))                                       //
    })));
  }
}

TEST_CASE("mut ptr to sized prim array", "[compiler]") {
  compiler::initialise();
  const auto tpe = SInt; // GENERATE(from_range(PrimitiveTypes));

  auto idx = 3; // GENERATE(0, 1, 3, 7, 10);
  //  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (local)") {
  //    CAPTURE(tpe, idx);
  //    assertCompile(program(function("foo", {}, tpe)({
  //        let("xs") = Alloc(tpe, 42_(SInt)),
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstValue(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(Ptr(tpe, 10)))()}, tpe)({
        let("deref") = "xs"_(Ptr(Ptr(tpe, 10)))[integral(SInt, 2)], "deref"_(Ptr(tpe, 10))[integral(SInt, idx)] = integral(SInt, 42), //
        ret("deref"_(Ptr(tpe, 10))[integral(SInt, idx)])                                                                              //
    })));
  }
}

TEST_CASE("index prim array", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
  const auto idx = GENERATE(0, 1, 3, 7, 10);
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (local)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(SInt), Global), "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstValue(tpe), //
        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)],                                                           //
        ret("x"_(tpe))                                                                                             //
    })));
  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe))()}, tpe)({
        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
        ret("x"_(tpe))                                   //
    })));
  }
}

TEST_CASE("update prim array", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypes));
  const auto idx = GENERATE(0, 1, 3, 7, 10);
  const auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(xs[" << idx << "]:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(SInt), Global),  //
        "xs"_(Ptr(tpe))[integral(SInt, idx)] = val, //
        ret("xs"_(Ptr(tpe))[integral(SInt, idx)])   //
    })));
  }
  DYNAMIC_SECTION("(xs[" << idx << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe))()}, tpe)({
        "xs"_(Ptr(tpe))[integral(SInt, idx)] = val, //
        ret("xs"_(Ptr(tpe))[integral(SInt, idx)])   //
    })));
  }
}

TEST_CASE("update prim array by ref", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  const auto idx = GENERATE(std::optional<int>{}, 0, 1, 3, 7, 10);
  const auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(SInt), Global),                                  //
        "xs"_(Ptr(tpe))[integral(SInt, idx.value_or(0))] = generateConstValue(tpe), //
        let("ref") = RefTo("xs"_(Ptr(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe, Global),
        ret("ref"_(Ptr(tpe))[integral(SInt, 0)]) //
    })));
  }
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe))()}, tpe)({
        let("ref") = RefTo("xs"_(Ptr(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe, Global),
        ret("ref"_(Ptr(tpe))[integral(SInt, 0)]) //
    })));
  }
}

TEST_CASE("update prim value by ref", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  const auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("x") = val,                               //
        let("y") = RefTo("x"_(tpe), {}, tpe, Global), //
        ret("y"_(Ptr(tpe))[integral(SInt, 0)])        //
    })));
  }
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {"x"_(tpe)()}, tpe)({
        let("y") = RefTo("x"_(tpe), {}, tpe, Global), //
        ret("y"_(Ptr(tpe))[integral(SInt, 0)])        //
    })));
  }
}

TEST_CASE("index struct array member", "[compiler]") {
  compiler::initialise();

  Named defX("x", Type::IntS32());
  Named defY("y", Type::IntS32());
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");

  Function fn("foo", {Arg(Named("s", Ptr(myStruct)), {})}, Type::IntS32(),
              {

                  Var(Named("a", myStruct), {Index(Select({}, Named("s", Ptr(myStruct))), Expr::IntS32Const(0), myStruct)}),

                  Var(Named("b", Type::IntS32()), {(Select({Named("a", myStruct)}, defX))}),

                  //                  Mut(Select({Named("s",  Ptr(myStruct ))}, defX),  (IntS32Const(42 ),
                  Return(IntS32Const(69)),
              },
              {FunctionAttr::Exported()});
  assertCompile(Program({def}, {fn}));
}

TEST_CASE("array update struct elem member", "[compiler]") {
  compiler::initialise();
  Named defX("x", Type::IntS32());
  Named defY("y", Type::IntS32());
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");

  Function fn("foo", {Arg(Named("xs", Ptr(myStruct)), {})}, Type::IntS32(),
              {
                  Var(Named("x", myStruct), Index(Select({}, Named("xs", Ptr(myStruct))), IntS32Const(0), myStruct)),
                  Mut(Select({Named("x", myStruct)}, defX), IntS32Const(42)),
                  Return(IntS32Const(69)),
              },
              {FunctionAttr::Exported()});
  assertCompile(Program({def}, {fn}));
}

TEST_CASE("array update struct elem", "[compiler]") {
  compiler::initialise();
  Named defX("x", Type::IntS32());
  Named defY("y", Type::IntS32());
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");

  Function fn("foo", {Arg(Named("xs", Ptr(myStruct)), {})}, Type::IntS32(),
              {
                  Var(Named("data", myStruct), {}),
                  Update(Select({}, Named("xs", Ptr(myStruct))), IntS32Const(7), Select({}, Named("data", myStruct))),
                  Return(IntS32Const(69)),
              },
              {FunctionAttr::Exported()});
  assertCompile(Program({def}, {fn}));
}

TEST_CASE("alias struct", "[compiler]") {
  compiler::initialise();
  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");
  assertCompile(program({def}, {function("foo", {"in"_(myStruct)()}, myStruct)({
                                   let("s") = "in"_(myStruct),
                                   let("t") = "s"_(myStruct),
                                   ret("t"_(myStruct)),
                               })}));
}

TEST_CASE("alias struct member", "[compiler]") {
  compiler::initialise();

  StructDef def("a.b", {(Named("x", SInt)), (Named("y", SInt))});
  Named arg("in", Type::Struct("a.b"));
  Function fn("foo", {Arg(arg, {})}, Unit,
              {
                  Var(                   //
                      Named("y2", SInt), //
                      {
                          (Select({arg}, Named("y", SInt))) //
                      } //
                      ),
                  Return(Unit0Const()),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({def}, {fn}));
}

TEST_CASE("alias array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn("foo", {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.comp, IntS32Const(10), Global)}),
                  Var(Named("t", arr), {(Select({}, Named("s", arr)))}),
                  Return(Select({}, Named("s", arr))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("mut struct", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");

  Function fn("foo", {Arg(Named("out", myStruct), {})}, Unit,
              {
                  //                  Var(Named("s", myStruct), {}),
                  //                  Var(Named("t", myStruct), {}),

                  //                  Mut(Select({}, Named("t", myStruct)),  (Select({}, Named("out", myStruct ),

                  Mut(Select({Named("out", myStruct)}, defX), IntS32Const(42)),
                  Mut(Select({Named("out", myStruct)}, defY), IntS32Const(43)),
                  //                  Mut(Select({}, Named("s", myStruct)),  (Select({}, Named("t", myStruct ),

                  //                  Return( (Unit0Const())),
                  Return(Unit0Const()),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({def}, {fn}));
}

TEST_CASE("mut array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn("foo", {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.comp, IntS32Const(10), Global)}),
                  Var(Named("t", arr), {Alloc(arr.comp, IntS32Const(20), Global)}),
                  Var(Named("u", arr), {Alloc(arr.comp, IntS32Const(30), Global)}),
                  Mut(Select({}, Named("s", arr)), Select({}, Named("t", arr))),
                  Mut(Select({}, Named("t", arr)), Select({}, Named("u", arr))),
                  Mut(Select({}, Named("t", arr)), Select({}, Named("s", arr))),
                  Return(Select({}, Named("s", arr))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("mut prim", "[compiler]") {
  compiler::initialise();

  Function fn("foo", {}, SInt,
              {
                  Var(Named("s", SInt), {(IntS32Const(10))}),
                  Mut(Select({}, Named("s", SInt)), IntS32Const(20)),
                  Return(Select({}, Named("s", SInt))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("alloc struct", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def("MyStruct", {defX, defY});
  Type::Struct myStruct("MyStruct");
  StructDef def2("MyStruct2", {defX});
  Type::Struct myStruct2("MyStruct2");

  Function fn("foo", {Arg(Named("out", myStruct), {})}, SInt,
              {
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), IntS32Const(42)),
                  Mut(Select({Named("s", myStruct)}, defY), IntS32Const(43)),
                  Var(Named("t", myStruct2), {}),

                  //                  Mut(Select({}, Named("out", myStruct)),  (Select({}, Named("s", myStruct))),
                  //                  true), Return( (Unit0Const())),
                  Return(IntrOp(Add(Select({Named("s", myStruct)}, defX), Select({Named("s", myStruct)}, defY), SInt))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({def, def2}, {fn}));
}

TEST_CASE("alloc struct nested", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);

  StructDef def2("MyStruct2", {defX});
  Type::Struct myStruct2("MyStruct2");
  Named defZ("z", myStruct2);

  StructDef def("MyStruct", {defX, defY, defZ});
  Type::Struct myStruct("MyStruct");

  Function fn("foo", {}, SInt,
              {
                  Var(Named("t", myStruct2), {}),
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), IntS32Const(42)),
                  Mut(Select({Named("s", myStruct)}, defY), IntS32Const(43)),
                  Mut(Select({Named("s", myStruct)}, defZ), Select({}, Named("t", myStruct2))),
                  Mut(Select({Named("s", myStruct), defZ}, defX), IntS32Const(42)),
                  Return(IntS32Const(69)),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({def2, def}, {fn}));
}

TEST_CASE("alloc array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn("foo", {}, arr,
              {
                  Var(Named("s", arr), {Alloc(arr.comp, IntS32Const(10), Global)}),
                  Var(Named("t", arr), {Alloc(arr.comp, IntS32Const(20), Global)}),
                  Var(Named("u", arr), {Alloc(arr.comp, IntS32Const(30), Global)}),
                  Return(Select({}, Named("s", arr))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("cast expr", "[compiler]") {
  compiler::initialise();

  Function fn("foo", {}, SInt,
              {
                  Var(Named("d", Double), {Cast(IntS32Const(10), Double)}),
                  Var(Named("i", SInt), {Cast(Select({}, Named("d", Double)), SInt)}),

                  Return(Select({}, Named("i", SInt))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  compiler::initialise();

  //  auto from = DoubleConst(0x1.fffffffffffffP+1023);
  const auto from = Float32Const(0x1.fffffeP+127f);
  //    auto from = IntS32Const( (1<<31)-1);
  const auto to = SInt;

  Function fn("foo", {}, to,
              {
                  Var(Named("i", from.tpe), {from}),

                  Var(Named("d", to), {Cast(Select({}, Named("i", from.tpe)), to)}),

                  Return(Select({}, Named("d", to))),
              },
              {FunctionAttr::Exported()});

  assertCompile(Program({}, {fn}));
}

TEST_CASE("cond", "[compiler]") {
  compiler::initialise();

  Function fn("foo", {}, SInt,
              {
                  Var(Named("out", SInt), {}),
                  Cond(Bool1Const(true),                                       //
                       {Mut(Select({}, Named("out", SInt)), IntS32Const(42))}, //
                       {Mut(Select({}, Named("out", SInt)), IntS32Const(43))}  //
                       ),

                  Return(Select({}, Named("out", SInt))),
              },
              {FunctionAttr::Exported()});
  assertCompile(Program({}, {fn}));
}

TEST_CASE("while false", "[compiler]") {
  compiler::initialise();

  Function fn("foo", {}, Unit,
              {
                  While({}, Bool1Const(false), {}),
                  Return(Unit0Const()),
              },
              {FunctionAttr::Exported()});
  assertCompile(Program({}, {fn}));
}
