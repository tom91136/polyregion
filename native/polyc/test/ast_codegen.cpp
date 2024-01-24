#include "ast.h"
#include "catch2/catch_all.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

#include <iostream>

using namespace polyregion::polyast;
using Catch::Matchers::ContainsSubstring;

using namespace Stmt;
using namespace Term;
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
    Type::IntU64(),  //
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
    Type::IntU64(),  //
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
      [&](const Type::Ptr &t) -> Term::Any { return unsupported(); },     //
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
  for (auto e : c.events) {
    REQUIRE_THAT(e.data, !ContainsSubstring(" undef"));
    REQUIRE_THAT(e.data, !ContainsSubstring("unreachable"));
  }
  FAIL("debug");
  INFO(c);
}

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                                                           //
      Sym({"foo"}), {},                                                                        //
      {}, {Arg(Named("a", Type::IntS32()), {}), Arg(Named("b", Double), {})}, {}, {}, //
      Unit,                                                                           //
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

TEST_CASE("inheritance", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));


  StructDef(Sym({"C"}),{},{StructMember(Named("C::a",IntS32()),1),StructMember(Named("C::b",IntS32()),1),StructMember(Named("C::c",Float),1)},{Sym({"B"}),Sym({"A"})})
  StructDef(Sym({"A"}),{},{StructMember(Named("A::x",Float32()),1),StructMember(Named("A::y",Float ),1)},{Sym({"Base"})})
  StructDef(Sym({"Base"}),{},{StructMember(Named("Base::x",IntS32()),1)},{})
  StructDef(Sym({"B"}),{},{StructMember(Named("B::x",IntS32()),1),StructMember(Named("B::y",Nothing()),1)},{})



  DYNAMIC_SECTION(tpe) {
    Sym foo({"foo"});
    Named x = Named("x", tpe);

    StructDef def(foo, {}, {StructMember(x, false)}, {});

    Type::Struct fooTpe(foo, {}, {}, {});
    auto entry = function("foo", {}, fooTpe)({
        let("foo") = fooTpe, //
        Mut(Select({"foo"_(fooTpe)}, x), Alias(generateConstValue(tpe)), false),
        ret("foo"_(fooTpe)) //
    });
    assertCompile(program(entry, {def}, {}));
  }
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

TEST_CASE("return ptr to struct", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {
    Sym foo({"foo"});
    Named x = Named("x", tpe);
    StructDef def(foo, {}, {StructMember(x, false)}, {});
    Type::Struct fooTpe(foo, {}, {}, {});
    auto entry = function("foo", {"foo"_(Ptr(fooTpe))()}, Ptr(fooTpe))({
        Mut(Select({"foo"_(Ptr(fooTpe))}, x), Alias(generateConstValue(tpe)), false),
        ret("foo"_(Ptr(fooTpe))) //
    });
    assertCompile(program(entry, {def}, {}));
  }
}


TEST_CASE("return struct", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  DYNAMIC_SECTION(tpe) {
    Sym foo({"foo"});
    Named x = Named("x", tpe);
    StructDef def(foo, {}, {StructMember(x, false)}, {});
    Type::Struct fooTpe(foo, {}, {}, {});
    auto entry = function("foo", {}, fooTpe)({
        let("foo") = fooTpe, //
        Mut(Select({"foo"_(fooTpe)}, x), Alias(generateConstValue(tpe)), false),
        ret("foo"_(fooTpe)) //
    });
    assertCompile(program(entry, {def}, {}));
  }
}


TEST_CASE("nested struct select", "[compiler]") {
  polyregion::compiler::initialise();

  Sym bar({"bar"});
  Named z = Named("z",  Ptr(SInt));
  StructDef barDef(bar, {}, {StructMember(z, false) }, {});
  Type::Struct barTpe(bar, {}, {}, {});

  Sym foo({"foo"});
  Named x = Named("x",  Ptr(SInt));
  Named y = Named("y",  Ptr(barTpe));
  StructDef fooDef(foo, {}, {StructMember(x, false), StructMember(y, false)}, {});
  Type::Struct fooTpe(foo, {}, {}, {});

  auto aux = function("aux", { "in"_(Ptr(barTpe))()  },  SInt)({
      ret(Index(Select({"in"_(Ptr(barTpe))}, z) , 0_(SInt),  SInt ))
  });

  auto entry = function("bar", { "in"_(Ptr(fooTpe))()  },  Unit)({
      let("r") =    (Invoke(Sym({"aux"}), {}, {}, { Select({"in"_(Ptr(fooTpe))}, y) }, {}, SInt)), //
      Update(Select({"in"_(Ptr(fooTpe))}, x) , 0_(SInt), "r"_(SInt)),
      ret(Unit0Const()) //
  });

  assertCompile(program(entry, {fooDef, barDef}, { aux} ));
}

TEST_CASE("return struct and take ref", "[compiler]") {
  polyregion::compiler::initialise();

  Sym bar({"bar"});
  Named z = Named("z",  SInt);
  StructDef barDef(bar, {}, {StructMember(z, false) }, {});
  Type::Struct barTpe(bar, {}, {}, {});

  Sym foo({"foo"});
  Named x = Named("x",  Ptr(SInt));
  Named y = Named("y",  Ptr(barTpe));
  StructDef fooDef(foo, {}, {StructMember(x, false), StructMember(y, false)}, {});
  Type::Struct fooTpe(foo, {}, {}, {});

  auto aux = function("aux", { "out"_(Ptr(barTpe))(), "in"_(Ptr(barTpe))()  },  Ptr(barTpe))({
      Mut(Select({"out"_(Ptr(barTpe))}, z), Alias(Select({"in"_(Ptr(barTpe))}, z)), true),
      ret("out"_(Ptr(barTpe)))
  });

  auto gen = function("gen", {   },   barTpe)({
      let("a") = barTpe, //
      Mut(Select({"a"_(barTpe)}, z), Alias(0_(SInt)), true), //
      ret("a"_( barTpe))
  });

  auto entry = function("bar", {"in"_(Ptr(fooTpe))()}, Unit)({
      let("a") = (Invoke(Sym({"gen"}), {}, {}, {}, {},  barTpe)), //
      let("ar") = RefTo("a"_(barTpe), {}, barTpe), //
      let("r") = (Invoke(Sym({"aux"}), {}, {}, {Select({"in"_(Ptr(fooTpe))}, y), "ar"_(Ptr(barTpe))}, {}, Ptr(barTpe))), //
      ret(Unit0Const()) //
  });

  assertCompile(program(entry, {fooDef, barDef}, { gen, aux} ));
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

TEST_CASE("index sized prim array", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = SInt;//GENERATE(from_range(PrimitiveTypes));

  auto idx = 3;//GENERATE(0, 1, 3, 7, 10);
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
        ret("x"_(tpe))                                     //
    })));
  }
}

TEST_CASE("mut sized prim array", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = SInt;//GENERATE(from_range(PrimitiveTypes));

  auto idx = 3;//GENERATE(0, 1, 3, 7, 10);
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
        ret("xs"_(Ptr(tpe, 10)))                                     //
    })));
  }
}


TEST_CASE("mut ptr to sized prim array", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = SInt;//GENERATE(from_range(PrimitiveTypes));

  auto idx = 3;//GENERATE(0, 1, 3, 7, 10);
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
        let("deref") = "xs"_(Ptr(Ptr(tpe, 10)))[integral(SInt, 2)],
        "deref"_(Ptr(tpe, 10))[integral(SInt, idx)] = integral(SInt, 42), //
        ret("deref"_(Ptr(tpe, 10))[integral(SInt, idx)] )                                     //
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
        let("xs") = Alloc(tpe, 42_(SInt)),
        "xs"_(Ptr(tpe))[integral(SInt, idx)] = generateConstValue(tpe), //
        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
        ret("x"_(tpe))                                     //
    })));
  }
  DYNAMIC_SECTION("xs[" << idx << "]:" << tpe << " (args)") {
    CAPTURE(tpe, idx);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe))()}, tpe)({
        let("x") = "xs"_(Ptr(tpe))[integral(SInt, idx)], //
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
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  auto idx = GENERATE(std::optional<int>{}, 0, 1, 3, 7, 10);
  auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("xs") = Alloc(tpe, 42_(SInt)), //
        "xs"_(Ptr(tpe))[integral(SInt, idx.value_or(0))] = generateConstValue(tpe), //
        let("ref") = RefTo("xs"_(Ptr(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe),
        ret("ref"_(Ptr(tpe))[integral(SInt, 0)]) //
    })));
  }
  DYNAMIC_SECTION("(xs[" << (idx ? std::to_string(*idx) : "(none)") << "]:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, idx, val);
    assertCompile(program(function("foo", {"xs"_(Ptr(tpe))()}, tpe)({
        let("ref") = RefTo("xs"_(Ptr(tpe)), idx ? std::optional{integral(SInt, *idx)} : std::nullopt, tpe),
        ret("ref"_(Ptr(tpe))[integral(SInt, 0)]) //
    })));
  }
}

TEST_CASE("update prim value by ref", "[compiler]") {
  polyregion::compiler::initialise();
  auto tpe = GENERATE(from_range(PrimitiveTypesNoUnit));
  auto val = generateConstValue(tpe);
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (local)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {}, tpe)({
        let("x") = val,                          //
        let("y") = RefTo("x"_(tpe), {}, tpe),    //
        ret("y"_(Ptr(tpe))[integral(SInt, 0)]) //
    })));
  }
  DYNAMIC_SECTION("(&x:" << tpe << ") = " << val << " (args)") {
    CAPTURE(tpe, val);
    assertCompile(program(function("foo", {"x"_(tpe)()}, tpe)({
        let("y") = RefTo("x"_(tpe), {}, tpe),    //
        ret("y"_(Ptr(tpe))[integral(SInt, 0)]) //
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

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("s", Ptr(myStruct)), {})}, {}, {}, Type::IntS32(),
              {

                  Var(Named("a", myStruct), {Index(Select({}, Named("s", Ptr(myStruct))), Term::IntS32Const(0), myStruct)}),

                  Var(Named("b", Type::IntS32()), {Alias(Select({Named("a", myStruct)}, defX))}),

                  //                  Mut(Select({Named("s",  Ptr(myStruct ))}, defX), Alias(IntS32Const(42)), false),
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

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("xs", Ptr(myStruct)), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("x", myStruct), Index(Select({}, Named("xs", Ptr(myStruct))), IntS32Const(0), myStruct)),
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

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("xs", Ptr(myStruct)), {})}, {}, {}, Type::IntS32(),
              {
                  Var(Named("data", myStruct), {}),
                  Update(Select({}, Named("xs", Ptr(myStruct))), IntS32Const(7), Select({}, Named("data", myStruct))),
                  Return(Alias(IntS32Const(69))),
              });
  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias struct", "[compiler]") {
  polyregion::compiler::initialise();
  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", SInt);
  Named defY = Named("y", SInt);
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});
  assertCompile(program(function("foo", {"in"_(myStruct)()}, myStruct)({
                            let("s") = "in"_(myStruct),
                            let("t") = "s"_(myStruct),
                            ret("t"_(myStruct)),
                        }),
                        {def}));
}

TEST_CASE("alias struct member", "[compiler]") {
  polyregion::compiler::initialise();

  Sym sdef({"a", "b"});
  StructDef def(sdef, {}, {StructMember(Named("x", SInt), false), StructMember(Named("y", SInt), false)}, {});
  Named arg("in", Type::Struct(sdef, {}, {}, {}));
  Function fn(Sym({"foo"}), {}, {}, {Arg(arg, {})}, {}, {}, Unit,
              {
                  Var(                             //
                      Named("y2", SInt), //
                      {
                          Alias(Select({arg}, Named("y", SInt))) //
                      }                                                    //
                      ),
                  Return(Alias(Unit0Const())),
              });

  Program p(fn, {}, {def});
  assertCompile(p);
}

TEST_CASE("alias array", "[compiler]") {
  polyregion::compiler::initialise();

  auto arr = Ptr(SInt);

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

  Named defX = Named("x", SInt);
  Named defY = Named("y", SInt);
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("out", myStruct), {})}, {}, {}, Unit,
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

  auto arr = Ptr(SInt);

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

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, SInt,
              {
                  Var(Named("s", SInt), {Alias(IntS32Const(10))}),
                  Mut(Select({}, Named("s", SInt)), Alias(IntS32Const(20)), false),
                  Return(Alias(Select({}, Named("s", SInt)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("alloc struct", "[compiler]") {
  polyregion::compiler::initialise();

  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", SInt);
  Named defY = Named("y", SInt);
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});
  StructDef def2(myStruct2Sym, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("out", myStruct), {})}, {}, {}, SInt,
              {
                  Var(Named("s", myStruct), {}),
                  Mut(Select({Named("s", myStruct)}, defX), Alias(IntS32Const(42)), false),
                  Mut(Select({Named("s", myStruct)}, defY), Alias(IntS32Const(43)), false),
                  Var(Named("t", myStruct2), {}),

                  //                  Mut(Select({}, Named("out", myStruct)), Alias(Select({}, Named("s", myStruct))),
                  //                  true), Return(Alias(Unit0Const())),
                  Return(IntrOp(Add(Select({Named("s", myStruct)}, defX), Select({Named("s", myStruct)}, defY), SInt))),
              });

  Program p(fn, {}, {def, def2});
  assertCompile(p);
}

TEST_CASE("alloc struct nested", "[compiler]") {
  polyregion::compiler::initialise();
  Sym myStructSym({"MyStruct"});
  Sym myStruct2Sym({"MyStruct2"});

  Named defX = Named("x", SInt);
  Named defY = Named("y", SInt);

  StructDef def2(myStruct2Sym, {}, {StructMember(defX, false)}, {});
  Type::Struct myStruct2(myStruct2Sym, {}, {}, {});
  Named defZ = Named("z", myStruct2);

  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false), StructMember(defZ, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, SInt,
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

  auto arr = Ptr(SInt);

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

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, SInt,
              {
                  Var(Named("d", Double), {Cast(IntS32Const(10), Double)}),
                  Var(Named("i", SInt), {Cast(Select({}, Named("d", Double)), SInt)}),

                  Return(Alias(Select({}, Named("i", SInt)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("cast fp to int expr", "[compiler]") {
  polyregion::compiler::initialise();

  //  auto from  = DoubleConst(0x1.fffffffffffffP+1023);
  auto from = Float32Const(0x1.fffffeP+127f);
  //    auto from  = IntS32Const( (1<<31)-1);
  auto to = SInt;

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

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, SInt,
              {
                  Var(Named("out", SInt), {}),
                  Cond(Alias(Bool1Const(true)),                                                        //
                       {Mut(Select({}, Named("out", SInt)), Alias(IntS32Const(42)), false)}, //
                       {Mut(Select({}, Named("out", SInt)), Alias(IntS32Const(43)), false)}  //
                       ),

                  Return(Alias(Select({}, Named("out", SInt)))),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}

TEST_CASE("while false", "[compiler]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, {}, Unit,
              {
                  While({}, Bool1Const(false), {}),

                  Return(Alias(Unit0Const())),
              });

  Program p(fn, {}, {});
  assertCompile(p);
}
