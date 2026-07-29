#include "ast.h"

#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"

#include "generated/polyast.h"
#include "generated/polyast_codec.h"

using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;

using namespace Stmt;
using namespace Expr;

static Type::Struct mkStruct(const std::string &name) { return Type::Struct(Sym({name}), {}); }
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

TEST_CASE("Trailing optional ctor args default") {
  const SourcePosition p("f.cpp", 1);
  CHECK(p.file == "f.cpp");
  CHECK(p.line == 1);
  CHECK(!p.col);
  CHECK(p == SourcePosition("f.cpp", 1, {}));
}

TEST_CASE("A NoIdentity type compares equal regardless of content") {
  const Origin empty;
  const Origin full(SourcePosition("a.cpp", 1, 2), std::string("let x = 1"), Sym({"f"}));
  CHECK(empty == full);
  CHECK(empty.hash_code() == full.hash_code());
  CHECK(full.pos); // the content is still carried, only identity is waived
}

TEST_CASE("Named identity ignores source provenance") {
  const Named bare("x", Type::IntS32());
  const Named decl("x", Type::IntS32(), Origin(SourcePosition("a.cpp", 1, 2)));
  const Named use("x", Type::IntS32(), Origin(SourcePosition("b.cpp", 9, {}), std::string("inlined"), Sym({"f"})));

  CHECK(bare == decl);
  CHECK(decl == use);
  CHECK(decl.hash_code() == use.hash_code());
  CHECK(std::unordered_set<Named>{bare, decl, use}.size() == 1);

  CHECK(decl != Named("y", Type::IntS32(), decl.origin));
  CHECK(decl != Named("x", Type::IntS64(), decl.origin));
}

TEST_CASE("Origin survives a msgpack round trip") {
  const Origin origin(SourcePosition("a.cpp", 7, 3), std::string("let x = 1"), Sym({"ns", "caller"}));
  const auto entry = function("origin_rt", {Arg(Named("x", Type::IntS32(), origin))}, Type::Unit0())({ret(Term::Unit0Const())});
  const auto decoded = program_from_msgpack(program_to_msgpack(program({}, {entry})));

  // any Origin compares equal to any other, so assert on the fields rather than the value
  const auto &decodedOrigin = decoded.entry.args[0].named.origin;
  REQUIRE(decodedOrigin.pos);
  CHECK(*decodedOrigin.pos == *origin.pos);
  CHECK(decodedOrigin.source == origin.source);
  CHECK(decodedOrigin.inlinedFrom == origin.inlinedFrom);

  const auto viaJson = origin_from_json(origin_to_json(origin));
  CHECK(viaJson.pos == origin.pos);
  CHECK(viaJson.source == origin.source);
  CHECK(viaJson.inlinedFrom == origin.inlinedFrom);
}

TEST_CASE("Type::FnRef survives a msgpack round trip") {
  const Type::Any callee = Type::FnRef(Sym({"ns", "callee"}));
  const auto entry = function("fnref_rt", {}, Type::IntS32())({ret(Expr::Invoke(callee, {}, {}, {}, Type::IntS32()))});
  const auto decoded = program_from_msgpack(program_to_msgpack(program({}, {entry})));

  const auto ivks = decoded.entry.collect_all<Expr::Invoke>();
  REQUIRE(ivks.size() == 1);
  CHECK(ivks[0].callee == callee);
  CHECK(calleeName(ivks[0]) == Sym({"ns", "callee"}));
  CHECK(!calleeSym(Expr::Invoke(Type::Var("F"), {}, {}, {}, Type::IntS32())));
}
