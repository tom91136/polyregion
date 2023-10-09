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

TEST_CASE("Invoke identity") {

  InvokeSignature a(Sym({"polyregion", "CollectionLengthSuite", "ClassB", "foo"}), {},
      {Type::Struct(Sym({"polyregion_CollectionLengthSuite_ClassB__"}), {}, {}, {Sym({"polyregion_CollectionLengthSuite_Base__"})})},
                    {Type::IntS32()}, {}, Type::IntS32());

  InvokeSignature b(
      Sym({"polyregion", "CollectionLengthSuite", "ClassB", "foo"}),
      {},
      {Type::Struct(Sym({"polyregion_CollectionLengthSuite_ClassB__"}), {}, {}, {Sym({"polyregion_CollectionLengthSuite_Base__"})})},
                    {Type::IntS32()}, {}, Type::IntS32());

  CHECK(a == b);
}

TEST_CASE("Struct identity") {

  Type::Any a =
      Type::Struct(Sym({"polyregion_CollectionLengthSuite_ClassB__"}), {}, {}, {Sym({"polyregion_CollectionLengthSuite_Base__"})});

  Type::Any b =
      Type::Struct(Sym({"polyregion_CollectionLengthSuite_ClassB__"}), {}, {}, {Sym({"polyregion_CollectionLengthSuite_Base__"})});



  CHECK(*(a) == *(b));
}
