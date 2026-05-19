#include "catch2/catch_all.hpp"

#include "ast.h"
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

static Function mkFn(const std::string &name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body,
                     FunctionVisibility::Any visibility = FunctionVisibility::Exported(),
                     FunctionFpMode::Any fpMode = FunctionFpMode::Relaxed(), bool isEntry = false) {
  return Function(Sym({name}), {}, {}, std::move(args), {}, {}, std::move(rtn), std::move(body), std::move(visibility), std::move(fpMode),
                  isEntry);
}

static Expr::Invoke mkInvoke(const std::string &name, std::vector<Term::Any> args, Type::Any rtn) {
  return Expr::Invoke(Sym({name}), {}, {}, std::move(args), std::move(rtn));
}

// Stmt::Mut takes Expr::Any. The DSL helper auto-wraps a Term in Expr::Alias.
static Stmt::Mut MutT(const Term::Select &name, const Term::Any &term) { return Stmt::Mut(name, Expr::Any(Expr::Alias(term))); }

static Term::Any generateConstTerm(const Tpe::Any &t) {
  const auto unsupported = [&]() -> Term::Any { throw std::logic_error("No constant for type " + to_string(t)); };
  return t.match_total(                                                         //
      [&](const Type::Float16 &) -> Term::Any { return 42.0_(t); },             //
      [&](const Type::Float32 &) -> Term::Any { return 42.0_(t); },             //
      [&](const Type::Float64 &) -> Term::Any { return 42.0_(t); },             //
      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(true); }, //
      [&](const Type::IntS8 &) -> Term::Any { return 0xFF_(t); },               //
      [&](const Type::IntS16 &) -> Term::Any { return 42_(t); },                //
      [&](const Type::IntS32 &) -> Term::Any { return 42_(t); },                //
      [&](const Type::IntS64 &) -> Term::Any { return 0xDEADBEEF_(t); },        //
      [&](const Type::IntU8 &) -> Term::Any { return 0xFF_(t); },               //
      [&](const Type::IntU16 &) -> Term::Any { return 42_(t); },                //
      [&](const Type::IntU32 &) -> Term::Any { return 42_(t); },                //
      [&](const Type::IntU64 &) -> Term::Any { return 0xDEADBEEF_(t); },        //
      [&](const Type::Unit0 &) -> Term::Any { return Term::Unit0Const(); },     //
      [&](const Type::Nothing &) -> Term::Any { return unsupported(); },        //
      [&](const Type::Struct &) -> Term::Any { return unsupported(); },         //
      [&](const Type::Ptr &) -> Term::Any { return unsupported(); },            //
      [&](const Type::Arr &) -> Term::Any { return unsupported(); },            //
      [&](const Type::Var &) -> Term::Any { return unsupported(); },            //
      [&](const Type::Exec &) -> Term::Any { return unsupported(); });
}

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
  auto liftTerm = [](const Term::Any &term) -> Expr::Any { return Expr::Alias(term); };
  return t.match_total(                                                       //
      [&](const Type::Float16 &) -> Expr::Any { return liftTerm(42.0_(t)); }, //
      [&](const Type::Float32 &) -> Expr::Any { return liftTerm(42.0_(t)); }, //
      [&](const Type::Float64 &) -> Expr::Any { return liftTerm(42.0_(t)); }, //
      [&](const Type::Bool1 &) -> Expr::Any { return Expr::Alias(Term::Bool1Const(true)); },

      [&](const Type::IntS8 &) -> Expr::Any { return liftTerm(0xFF_(t)); },        //
      [&](const Type::IntS16 &) -> Expr::Any { return liftTerm(42_(t)); },         //
      [&](const Type::IntS32 &) -> Expr::Any { return liftTerm(42_(t)); },         //
      [&](const Type::IntS64 &) -> Expr::Any { return liftTerm(0xDEADBEEF_(t)); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return liftTerm(0xFF_(t)); },        //
      [&](const Type::IntU16 &) -> Expr::Any { return liftTerm(42_(t)); },         //
      [&](const Type::IntU32 &) -> Expr::Any { return liftTerm(42_(t)); },         //
      [&](const Type::IntU64 &) -> Expr::Any { return liftTerm(0xDEADBEEF_(t)); }, //

      [&](const Type::Unit0 &) -> Expr::Any { return Expr::Alias(Term::Unit0Const()); },
      [&](const Type::Nothing &) -> Expr::Any { return unsupported(); }, [&](const Type::Struct &) -> Expr::Any { return unsupported(); },
      [&](const Type::Ptr &) -> Expr::Any { return unsupported(); }, [&](const Type::Arr &) -> Expr::Any { return unsupported(); },
      [&](const Type::Var &) -> Expr::Any { return unsupported(); }, [&](const Type::Exec &) -> Expr::Any { return unsupported(); });
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
  Function expected = mkFn(                                                            //
      "foo", {Arg(Named("a", Type::IntS32()), {}), Arg(Named("b", Double), {})}, Unit, //
      {
          Var(Named("a", SInt), Expr::Any(SpecOp(GpuGlobalSize(Term::IntS32Const(1)))), false),
          Break(),                                            //
          Return(Expr::Alias(Term::IntS32Const(1))),          //
          Cond(                                               //
              Term::Bool1Const(true),                         //
              {Return(Expr::Alias(Term::IntS32Const(1)))},    //
              {Return(Expr::Alias(Term::Bool1Const(false)))}) //
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false);
  auto actual = function_from_json(function_to_json(expected));
  CHECK(expected == actual);
}

TEST_CASE("initialise more than once", "[compiler]") {
  compiler::initialise();
  compiler::initialise();
  compiler::initialise();
  compiler::initialise();
}

TEST_CASE("spirv64 minimal", "[compiler]") {
  compiler::initialise();
  auto fn = function("k", {"p"_(Ptr(SInt))()}, Unit, FunctionVisibility::Exported(), FunctionFpMode::Strict(), true)({
      "p"_(Ptr(SInt))[0_(SInt)] = 42_(SInt), //
      ret()                                  //
  });
  auto p = program({}, {fn});
  CAPTURE(repr(p));
  auto c = compiler::compile(p, compiler::Options{Target::Object_LLVM_SPIRV64_Kernel, "intel"}, OptLevel::O3);
  CAPTURE(c.messages);
  CHECK(c.binary.has_value());
}

TEST_CASE("recursive struct", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {
    Type::Struct fooTpe(Sym({"foo"}), {});
    Named x("x", tpe);
    Named next("next", Ptr(fooTpe));
    StructDef def(Sym({"foo"}), {}, {x, next}, {});
    auto entry = function("foo", {}, fooTpe)({
        let("foo") = fooTpe, //
        Mut(Select({"foo"_(fooTpe)}, x), generateConstValue(tpe)), MutT(Select({"foo"_(fooTpe)}, next), Term::NullPtrConst(fooTpe, Global)),
        ret("foo"_(fooTpe)) //
    });
    assertCompile(program({def}, {entry}));
  }
}

TEST_CASE("nested if", "[compiler]") {
  compiler::initialise();

  auto entry = function("foo", {"in0"_(SInt)(), "in1"_(SInt)()}, SInt)({
      let("c0") = IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
      Cond("c0"_(Bool),
           {
               let("c1") = IntrOp(LogicEq("in1"_(SInt), 42_(SInt))),
               Cond("c1"_(Bool),
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
      let("c0") = IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
      Cond("c0"_(Bool),
           {
               let("c1") = IntrOp(LogicEq("in1"_(SInt), 42_(SInt))),
               Cond("c1"_(Bool),
                    {
                        let("c2") = IntrOp(LogicEq("in2"_(SInt), 42_(SInt))),
                        Cond("c2"_(Bool),
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

  auto entry = function("foo", {"in0"_(SInt)()}, SInt)({let("c") = IntrOp(LogicEq("in0"_(SInt), 42_(SInt))), //
                                                        Cond("c"_(Bool),
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
                               let("c") = IntrOp(LogicEq("in0"_(SInt), 42_(SInt))),
                               Cond("c"_(Bool),
                                    {
                                        MutT("x"_(SInt), 1_(SInt)) //
                                    },
                                    {
                                        MutT("x"_(SInt), 3_(SInt)) //
                                    }),
                               ret("x"_(SInt))});
  assertCompile(program({}, {entry}));
}

TEST_CASE("return ptr to struct", "[compiler]") {
  compiler::initialise();
  const auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {

    Named x("x", tpe);
    StructDef def(Sym({"foo"}), {}, {x}, {});
    Type::Struct fooTpe(Sym({"foo"}), {});
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
    StructDef def(Sym({"foo"}), {}, {x}, {});
    Type::Struct fooTpe(Sym({"foo"}), {});
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
  StructDef barDef(Sym({"bar"}), {}, {z}, {});
  Type::Struct barTpe(Sym({"bar"}), {});

  Named x("x", Ptr(SInt));
  Named y("y", Ptr(barTpe));
  StructDef fooDef(Sym({"foo"}), {}, {x, y}, {});
  Type::Struct fooTpe(Sym({"foo"}), {});

  auto aux = function("aux", {"in"_(Ptr(barTpe))()}, SInt)({
      ret(Index(Select({"in"_(Ptr(barTpe))}, z), 0_(SInt), SInt)) //
  });

  auto entry = function("bar", {"in"_(Ptr(fooTpe))()}, Unit)({
      let("r") = mkInvoke("aux", {Select({"in"_(Ptr(fooTpe))}, y)}, SInt), //
      Update(Select({"in"_(Ptr(fooTpe))}, x), 0_(SInt), "r"_(SInt)),
      ret(Term::Unit0Const()) //
  });

  assertCompile(program({fooDef, barDef}, {entry, aux}));
}

TEST_CASE("return struct and take ref", "[compiler]") {
  compiler::initialise();

  Named z("z", SInt);
  StructDef barDef(Sym({"bar"}), {}, {z}, {});
  Type::Struct barTpe(Sym({"bar"}), {});

  Named x("x", Ptr(SInt));
  Named y("y", Ptr(barTpe));
  StructDef fooDef(Sym({"foo"}), {}, {x, y}, {});
  Type::Struct fooTpe(Sym({"foo"}), {});

  auto aux = function("aux", {"out"_(Ptr(barTpe))(), "in"_(Ptr(barTpe))()}, Ptr(barTpe))({
      MutT(Select({"out"_(Ptr(barTpe))}, z), //
           Select({"in"_(Ptr(barTpe))}, z)), //
      ret("out"_(Ptr(barTpe)))               //
  });

  auto gen = function("gen", {}, barTpe)({let("a") = barTpe,                         //
                                          MutT(Select({"a"_(barTpe)}, z), 0_(SInt)), //
                                          ret("a"_(barTpe))});

  auto entry = function("bar", {"in"_(Ptr(fooTpe))()}, Unit)({
      let("a") = mkInvoke("gen", {}, barTpe),                                                         //
      let("ar") = RefTo("a"_(barTpe), {}, barTpe, Global),                                            //
      let("r") = mkInvoke("aux", {Select({"in"_(Ptr(fooTpe))}, y), "ar"_(Ptr(barTpe))}, Ptr(barTpe)), //
      ret(Term::Unit0Const())                                                                         //
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
        ret(mkInvoke("bar", {}, tpe)) //
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
        ret(mkInvoke("bar", {generateConstTerm(tpe)}, tpe)) //
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
        ret(mkInvoke("bar", {generateConstTerm(tpe0), generateConstTerm(tpe1)}, tpe0)) //
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
        ret(mkInvoke("bar", {generateConstTerm(tpe0), generateConstTerm(tpe1), generateConstTerm(tpe2)}, tpe0)) //
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
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstTerm(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Arr(tpe, 10))()}, tpe)({
        let("x") = "xs"_(Arr(tpe, 10))[integral(SInt, idx)], //
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
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstTerm(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Arr(tpe, 10))()}, Arr(tpe, 10))({
        "xs"_(Arr(tpe, 10))[integral(SInt, idx)] = integral(SInt, 42), //
        ret("xs"_(Arr(tpe, 10)))                                       //
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
  //        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstTerm(tpe), //
  //        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
  //        ret("x"_(tpe))                                     //
  //    })));
  //  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(Arr(tpe, 10)))()}, tpe)({
        let("deref") = "xs"_(Ptr(Arr(tpe, 10)))[integral(SInt, 2)], "deref"_(Arr(tpe, 10))[integral(SInt, idx)] = integral(SInt, 42), //
        ret("deref"_(Arr(tpe, 10))[integral(SInt, idx)])                                                                              //
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
        let("xs") = Alloc(tpe, 42_(SInt), Global), "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstTerm(tpe), //
        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)],                                                          //
        ret("x"_(tpe))                                                                                            //
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
  const auto val = generateConstTerm(tpe);
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
        let("xs") = Alloc(tpe, 42_(SInt), Global),                                 //
        "xs"_(Ptr(tpe))[integral(SInt, idx.value_or(0))] = generateConstTerm(tpe), //
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
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});

  Function fn = mkFn("foo", {Arg(Named("s", Ptr(myStruct)), {})}, Type::IntS32(),
                     {

                         Var(Named("a", myStruct), {Index(Select({}, Named("s", Ptr(myStruct))), Term::IntS32Const(0), myStruct)}, false),

                         Var(Named("b", Type::IntS32()), {Expr::Alias(Select({Named("a", myStruct)}, defX))}, false),

                         //                  Mut(Select({Named("s",  Ptr(myStruct ))}, defX),  (Term::IntS32Const(42 ),
                         Return(Expr::Alias(Term::IntS32Const(69))),
                     },
                     {FunctionVisibility::Exported()});
  assertCompile(program({def}, {fn}));
}

TEST_CASE("array update struct elem member", "[compiler]") {
  compiler::initialise();
  Named defX("x", Type::IntS32());
  Named defY("y", Type::IntS32());
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});

  Function fn = mkFn("foo", {Arg(Named("xs", Ptr(myStruct)), {})}, Type::IntS32(),
                     {
                         Var(Named("x", myStruct), Index(Select({}, Named("xs", Ptr(myStruct))), Term::IntS32Const(0), myStruct), false),
                         MutT(Select({Named("x", myStruct)}, defX), Term::IntS32Const(42)),
                         Return(Expr::Alias(Term::IntS32Const(69))),
                     },
                     {FunctionVisibility::Exported()});
  assertCompile(program({def}, {fn}));
}

TEST_CASE("array update struct elem", "[compiler]") {
  compiler::initialise();
  Named defX("x", Type::IntS32());
  Named defY("y", Type::IntS32());
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});

  Function fn = mkFn("foo", {Arg(Named("xs", Ptr(myStruct)), {})}, Type::IntS32(),
                     {
                         Var(Named("data", myStruct), {}, false),
                         Update(Select({}, Named("xs", Ptr(myStruct))), Term::IntS32Const(7), Select({}, Named("data", myStruct))),
                         Return(Expr::Alias(Term::IntS32Const(69))),
                     },
                     {FunctionVisibility::Exported()});
  assertCompile(program({def}, {fn}));
}

TEST_CASE("alias struct", "[compiler]") {
  compiler::initialise();
  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});
  assertCompile(program({def}, {function("foo", {"in"_(myStruct)()}, myStruct)({
                                   let("s") = "in"_(myStruct),
                                   let("t") = "s"_(myStruct),
                                   ret("t"_(myStruct)),
                               })}));
}

TEST_CASE("alias struct member", "[compiler]") {
  compiler::initialise();

  StructDef def(Sym({"a.b"}), {}, {(Named("x", SInt)), (Named("y", SInt))}, {});
  Named arg("in", Type::Struct(Sym({"a.b"}), {}));
  Function fn = mkFn("foo", {Arg(arg, {})}, Unit,
                     {
                         Var(                   //
                             Named("y2", SInt), //
                             {
                                 Expr::Alias(Select({arg}, Named("y", SInt))) //
                             } //
                             ,
                             false),
                         Return(Expr::Alias(Term::Unit0Const())),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({def}, {fn}));
}

TEST_CASE("alias array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn = mkFn("foo", {}, arr,
                     {
                         Var(Named("s", arr), {Alloc(arr.comp, Term::IntS32Const(10), Global)}, false),
                         Var(Named("t", arr), {Expr::Alias(Select({}, Named("s", arr)))}, false),
                         Return(Expr::Alias(Select({}, Named("s", arr)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("mut struct", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});

  Function fn = mkFn("foo", {Arg(Named("out", myStruct), {})}, Unit,
                     {
                         //                  Var(Named("s", myStruct), {}, false),
                         //                  Var(Named("t", myStruct), {}, false),

                         //                  Mut(Select({}, Named("t", myStruct)),  (Select({}, Named("out", myStruct ),

                         MutT(Select({Named("out", myStruct)}, defX), Term::IntS32Const(42)),
                         MutT(Select({Named("out", myStruct)}, defY), Term::IntS32Const(43)),
                         //                  Mut(Select({}, Named("s", myStruct)),  (Select({}, Named("t", myStruct ),

                         //                  Return( (Term::Unit0Const())),
                         Return(Expr::Alias(Term::Unit0Const())),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({def}, {fn}));
}

TEST_CASE("mut array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn = mkFn("foo", {}, arr,
                     {
                         Var(Named("s", arr), {Alloc(arr.comp, Term::IntS32Const(10), Global)}, false),
                         Var(Named("t", arr), {Alloc(arr.comp, Term::IntS32Const(20), Global)}, false),
                         Var(Named("u", arr), {Alloc(arr.comp, Term::IntS32Const(30), Global)}, false),
                         MutT(Select({}, Named("s", arr)), Select({}, Named("t", arr))),
                         MutT(Select({}, Named("t", arr)), Select({}, Named("u", arr))),
                         MutT(Select({}, Named("t", arr)), Select({}, Named("s", arr))),
                         Return(Expr::Alias(Select({}, Named("s", arr)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("mut prim", "[compiler]") {
  compiler::initialise();

  Function fn = mkFn("foo", {}, SInt,
                     {
                         Var(Named("s", SInt), {Expr::Alias(Term::IntS32Const(10))}, false),
                         MutT(Select({}, Named("s", SInt)), Term::IntS32Const(20)),
                         Return(Expr::Alias(Select({}, Named("s", SInt)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("alloc struct", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);
  StructDef def(Sym({"MyStruct"}), {}, {defX, defY}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});
  StructDef def2(Sym({"MyStruct2"}), {}, {defX}, {});
  Type::Struct myStruct2(Sym({"MyStruct2"}), {});

  Function fn = mkFn("foo", {Arg(Named("out", myStruct), {})}, SInt,
                     {
                         Var(Named("s", myStruct), {}, false),
                         MutT(Select({Named("s", myStruct)}, defX), Term::IntS32Const(42)),
                         MutT(Select({Named("s", myStruct)}, defY), Term::IntS32Const(43)),
                         Var(Named("t", myStruct2), {}, false),

                         //                  Mut(Select({}, Named("out", myStruct)),  (Select({}, Named("s", myStruct))),
                         //                  true), Return( (Term::Unit0Const())),
                         Return(IntrOp(Add(Select({Named("s", myStruct)}, defX), Select({Named("s", myStruct)}, defY), SInt))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({def, def2}, {fn}));
}

TEST_CASE("alloc struct nested", "[compiler]") {
  compiler::initialise();

  Named defX("x", SInt);
  Named defY("y", SInt);

  StructDef def2(Sym({"MyStruct2"}), {}, {defX}, {});
  Type::Struct myStruct2(Sym({"MyStruct2"}), {});
  Named defZ("z", myStruct2);

  StructDef def(Sym({"MyStruct"}), {}, {defX, defY, defZ}, {});
  Type::Struct myStruct(Sym({"MyStruct"}), {});

  Function fn = mkFn("foo", {}, SInt,
                     {
                         Var(Named("t", myStruct2), {}, false),
                         Var(Named("s", myStruct), {}, false),
                         MutT(Select({Named("s", myStruct)}, defX), Term::IntS32Const(42)),
                         MutT(Select({Named("s", myStruct)}, defY), Term::IntS32Const(43)),
                         MutT(Select({Named("s", myStruct)}, defZ), Select({}, Named("t", myStruct2))),
                         MutT(Select({Named("s", myStruct), defZ}, defX), Term::IntS32Const(42)),
                         Return(Expr::Alias(Term::IntS32Const(69))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({def2, def}, {fn}));
}

TEST_CASE("alloc array", "[compiler]") {
  compiler::initialise();

  const auto arr = Ptr(SInt);

  Function fn = mkFn("foo", {}, arr,
                     {
                         Var(Named("s", arr), {Alloc(arr.comp, Term::IntS32Const(10), Global)}, false),
                         Var(Named("t", arr), {Alloc(arr.comp, Term::IntS32Const(20), Global)}, false),
                         Var(Named("u", arr), {Alloc(arr.comp, Term::IntS32Const(30), Global)}, false),
                         Return(Expr::Alias(Select({}, Named("s", arr)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("cast expr", "[compiler]") {
  compiler::initialise();

  Function fn = mkFn("foo", {}, SInt,
                     {
                         Var(Named("d", Double), {Cast(Term::IntS32Const(10), Double)}, false),
                         Var(Named("i", SInt), {Cast(Select({}, Named("d", Double)), SInt)}, false),

                         Return(Expr::Alias(Select({}, Named("i", SInt)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  compiler::initialise();

  //  auto from = DoubleConst(0x1.fffffffffffffP+1023);
  const auto from = Term::Float32Const(0x1.fffffeP+127f);
  //    auto from = Term::IntS32Const( (1<<31)-1);
  const auto to = SInt;

  Function fn = mkFn("foo", {}, to,
                     {
                         Var(Named("i", from.tpe), {Expr::Alias(from)}, false),

                         Var(Named("d", to), {Cast(Select({}, Named("i", from.tpe)), to)}, false),

                         Return(Expr::Alias(Select({}, Named("d", to)))),
                     },
                     {FunctionVisibility::Exported()});

  assertCompile(program({}, {fn}));
}

TEST_CASE("cond", "[compiler]") {
  compiler::initialise();

  Function fn = mkFn("foo", {}, SInt,
                     {
                         Var(Named("out", SInt), {}, false),
                         Cond(Term::Bool1Const(true),                                        //
                              {MutT(Select({}, Named("out", SInt)), Term::IntS32Const(42))}, //
                              {MutT(Select({}, Named("out", SInt)), Term::IntS32Const(43))}  //
                              ),

                         Return(Expr::Alias(Select({}, Named("out", SInt)))),
                     },
                     {FunctionVisibility::Exported()});
  assertCompile(program({}, {fn}));
}

TEST_CASE("while false", "[compiler]") {
  compiler::initialise();

  Function fn = mkFn("foo", {}, Unit,
                     {
                         While(Term::Bool1Const(false), {}),
                         Return(Expr::Alias(Term::Unit0Const())),
                     },
                     {FunctionVisibility::Exported()});
  assertCompile(program({}, {fn}));
}

static Function mkMathKernel(const Type::Any &tpe) {
  auto outPtr = "out"_(Ptr(tpe));
  const Term::Select inSel = "in"_(tpe);
  return function("k", {"out"_(Ptr(tpe))(), "in"_(tpe)()}, Unit, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                  /*isEntry*/ true)({
      let("rsin") = call(Sin(inSel, tpe)),            //
      let("rcos") = call(Cos(inSel, tpe)),            //
      let("rtan") = call(Tan(inSel, tpe)),            //
      let("rasin") = call(Asin(inSel, tpe)),          //
      let("racos") = call(Acos(inSel, tpe)),          //
      let("ratan") = call(Atan(inSel, tpe)),          //
      let("rsinh") = call(Sinh(inSel, tpe)),          //
      let("rcosh") = call(Cosh(inSel, tpe)),          //
      let("rtanh") = call(Tanh(inSel, tpe)),          //
      let("rsqrt") = call(Sqrt(inSel, tpe)),          //
      let("rcbrt") = call(Cbrt(inSel, tpe)),          //
      let("rexp") = call(Exp(inSel, tpe)),            //
      let("rexpm1") = call(Expm1(inSel, tpe)),        //
      let("rlog") = call(Log(inSel, tpe)),            //
      let("rlog1p") = call(Log1p(inSel, tpe)),        //
      let("rlog10") = call(Log10(inSel, tpe)),        //
      let("rceil") = call(Ceil(inSel, tpe)),          //
      let("rfloor") = call(Floor(inSel, tpe)),        //
      let("rrint") = call(Rint(inSel, tpe)),          //
      let("rpow") = call(Pow(inSel, inSel, tpe)),     //
      let("ratan2") = call(Atan2(inSel, inSel, tpe)), //
      let("rhypot") = call(Hypot(inSel, inSel, tpe)), //
      let("rsignum") = call(Signum(inSel, tpe)),      //
      outPtr[0_(SInt)] = "rsin"_(tpe),                //
      outPtr[1_(SInt)] = "rcos"_(tpe),                //
      outPtr[2_(SInt)] = "rtan"_(tpe),                //
      outPtr[3_(SInt)] = "rasin"_(tpe),               //
      outPtr[4_(SInt)] = "racos"_(tpe),               //
      outPtr[5_(SInt)] = "ratan"_(tpe),               //
      outPtr[6_(SInt)] = "rsinh"_(tpe),               //
      outPtr[7_(SInt)] = "rcosh"_(tpe),               //
      outPtr[8_(SInt)] = "rtanh"_(tpe),               //
      outPtr[9_(SInt)] = "rsqrt"_(tpe),               //
      outPtr[10_(SInt)] = "rcbrt"_(tpe),              //
      outPtr[11_(SInt)] = "rexp"_(tpe),               //
      outPtr[12_(SInt)] = "rexpm1"_(tpe),             //
      outPtr[13_(SInt)] = "rlog"_(tpe),               //
      outPtr[14_(SInt)] = "rlog1p"_(tpe),             //
      outPtr[15_(SInt)] = "rlog10"_(tpe),             //
      outPtr[16_(SInt)] = "rceil"_(tpe),              //
      outPtr[17_(SInt)] = "rfloor"_(tpe),             //
      outPtr[18_(SInt)] = "rrint"_(tpe),              //
      outPtr[19_(SInt)] = "rpow"_(tpe),               //
      outPtr[20_(SInt)] = "ratan2"_(tpe),             //
      outPtr[21_(SInt)] = "rhypot"_(tpe),             //
      outPtr[22_(SInt)] = "rsignum"_(tpe),            //
      ret()                                           //
  });
}

static void assertCompileTarget(const Program &p, compiler::Options opts) {
  CAPTURE(repr(p));
  CAPTURE(opts.target);
  CAPTURE(opts.arch);
  auto c = compiler::compile(p, opts, OptLevel::O3);
  CAPTURE(c.messages);
  CHECK(c.messages == "");
  CHECK(c.binary.has_value());
}

TEST_CASE("math ops compile across targets", "[compiler][math]") {
  compiler::initialise();
  const auto tpe = GENERATE(values<Type::Any>({Float, Double}));
  auto p = program({}, {mkMathKernel(tpe)});
  DYNAMIC_SECTION(tpe) {
    SECTION("CPU x86_64") { assertCompileTarget(p, {Target::Object_LLVM_x86_64, "native"}); }
    SECTION("NVPTX64") {
      const auto cpu = GENERATE(
          values<std::string>({"sm_35", "sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"}));
      DYNAMIC_SECTION(cpu) { assertCompileTarget(p, {Target::Object_LLVM_NVPTX64, cpu}); }
    }
    SECTION("AMDGCN") {
      // CDNA1/2/3 (gfx9xx = wave64)
      // RDNA2/3/4 (gfx10xx/11xx/12xx = wave32)
      const auto cpu = GENERATE(values<std::string>({"gfx906", "gfx908", "gfx90a", "gfx942", "gfx1030", "gfx1100", "gfx1200"}));
      DYNAMIC_SECTION(cpu) { assertCompileTarget(p, {Target::Object_LLVM_AMDGCN, cpu}); }
    }
    SECTION("SPIRV64 Kernel") { assertCompileTarget(p, {Target::Object_LLVM_SPIRV64_Kernel, "intel"}); }
  }
}
