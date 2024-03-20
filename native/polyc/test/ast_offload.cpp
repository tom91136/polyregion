#include "ast.h"
#include "catch2/catch_all.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

#include <iostream>

using namespace polyregion::polyast;
using namespace polyregion::compiletime;
using Catch::Matchers::ContainsSubstring;

using namespace Stmt;
using namespace Term;
using namespace Expr;
using namespace Intr;
using namespace Spec;
using namespace Math;

using namespace polyregion;
using namespace polyast::dsl;

template <typename P> static void assertCompile(const P &p) {
  CAPTURE(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_AMDGCN, "gfx1036"},
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



TEST_CASE("array update struct elem member", "[compiler]") {
  polyregion::compiler::initialise();
  Sym myStructSym({"MyStruct"});
  Named defX = Named("x", Type::IntS32());
  Named defY = Named("y", Type::IntS32());
  StructDef def(myStructSym, {}, {StructMember(defX, false), StructMember(defY, false)}, {});
  Type::Struct myStruct(myStructSym, {}, {}, {});

  Function fn(Sym({"foo"}), {}, {}, {Arg(Named("xs", Ptr(myStruct)), {})}, {}, {}, Unit,
              {
//                  Var(Named("x", myStruct), Index(Select({}, Named("xs", Ptr(myStruct))), IntS32Const(0), myStruct)),
                  Mut(Select({Named("xs", Ptr(myStruct)) }, defX), Alias(IntS32Const(42)), false),
                  Return(Alias(Unit0Const())),
              },
              FunctionKind::Exported());
  Program p(fn, {}, {def});
  assertCompile(p);
}