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

TEST_CASE("Fn identity") {
  const Type::Any tpe = Type::Struct("polyregion_CollectionLengthSuite_ClassB__");
  Function a = function("foo", {"in"_(tpe)()}, tpe)({ret("in"_(tpe))});
  Function b = function("foo", {"in"_(tpe)()}, tpe)({ret("in"_(tpe))});


  CHECK(a == b);
}

TEST_CASE("Invoke identity") {
  const Signature a("polyregion.CollectionLengthSuite.ClassB.foo",
                    {Type::Struct("polyregion_CollectionLengthSuite_ClassB__"), Type::IntS32()}, Type::IntS32());

  const Signature b("polyregion.CollectionLengthSuite.ClassB.foo",
                    {Type::Struct("polyregion_CollectionLengthSuite_ClassB__"), Type::IntS32()}, Type::IntS32());

  CHECK(a == b);
}

TEST_CASE("Struct identity") {
  const Type::Any a = Type::Struct("polyregion_CollectionLengthSuite_ClassB__");
  const Type::Any b = Type::Struct("polyregion_CollectionLengthSuite_ClassB__");
  CHECK(a == b);
}
