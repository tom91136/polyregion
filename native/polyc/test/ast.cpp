#include "ast.h"

#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

#include <iostream>

using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;

using namespace Stmt;
using namespace Expr;

static Type::Struct mkStruct(const std::string &name) { return Type::Struct(Sym({name}), {}, {}, {}); }
static Signature mkSignature(const std::string &name, std::vector<Type::Any> args, Type::Any rtn) {
  return Signature(Sym({name}), {}, {}, std::move(args), {}, {}, std::move(rtn));
}

TEST_CASE("Fn identity") {
  const Type::Any tpe = mkStruct("polyregion_CollectionLengthSuite_ClassB__");
  Function a = function("foo", {"in"_(tpe)()}, tpe)({ret("in"_(tpe))});
  Function b = function("foo", {"in"_(tpe)()}, tpe)({ret("in"_(tpe))});


  CHECK(a == b);
}

TEST_CASE("Invoke identity") {
  const Signature a = mkSignature("polyregion.CollectionLengthSuite.ClassB.foo",
                                  {mkStruct("polyregion_CollectionLengthSuite_ClassB__"), Type::IntS32()}, Type::IntS32());

  const Signature b = mkSignature("polyregion.CollectionLengthSuite.ClassB.foo",
                                  {mkStruct("polyregion_CollectionLengthSuite_ClassB__"), Type::IntS32()}, Type::IntS32());

  CHECK(a == b);
}

TEST_CASE("Struct identity") {
  const Type::Any a = mkStruct("polyregion_CollectionLengthSuite_ClassB__");
  const Type::Any b = mkStruct("polyregion_CollectionLengthSuite_ClassB__");
  CHECK(a == b);
}
