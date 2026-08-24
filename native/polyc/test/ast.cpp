#include "ast.h"

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

TEST_CASE("Semantic names do not depend on repr") {
  const Sym symbol({"vendor", "algorithm"});
  const Type::Any tpe = Type::Ptr(Type::Struct(symbol, {Type::IntS32()}), TypeSpace::Local());
  const Signature signature(symbol, {}, {}, {tpe}, {}, {}, Type::Unit0());

  CHECK(fqcn(symbol) == "vendor.algorithm");
  CHECK(canonicalName(tpe) == "vendor.algorithm<I32>*^Local");
  CHECK(signatureKey(signature) == "vendor.algorithm<>(vendor.algorithm<I32>*^Local)[;]:Unit0");

  const Signature genericSignature(symbol, {Type::Var("T", 4)}, tpe, {Type::Var("T", 4)}, {Type::IntU32()}, {Type::IntU64()},
                                   Type::Unit0());
  CHECK(signatureKey(genericSignature) == "vendor.algorithm<I32>*^Local.vendor.algorithm<#T:size=4>(#T:size=4)[U32;U64]:Unit0");
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
  REQUIRE(decoded.entry);
  const auto &decodedOrigin = decoded.entry->decl.args[0].named.origin;
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

  REQUIRE(decoded.entry);
  const auto ivks = decoded.entry->collect_all<Expr::Invoke>();
  REQUIRE(ivks.size() == 1);
  CHECK(ivks[0].callee == callee);
  CHECK(calleeName(ivks[0]) == Sym({"ns", "callee"}));
  CHECK(!calleeSym(Expr::Invoke(Type::Var("F"), {}, {}, {}, Type::IntS32())));
}

TEST_CASE("Interface packages and GPU operations survive a msgpack round trip") {
  const auto element = Type::Var("Element", 4);
  const auto pointer = Type::Ptr(element, TypeSpace::Global()).widen();
  const auto extent = ArgExtent::Elements(ArgSizeExpr::Min(ArgSizeExpr::Param(1), ArgSizeExpr::Const(64)));
  const auto publicName = Sym({"vendor", "transform"});
  const auto implementationName = Sym({"vendor", "implementation", "transform"});
  const auto declaration =
      FunctionDecl(publicName, {Type::Var("T")}, {},
                   {Arg(Named("values", Type::Ptr(Type::Var("T"), TypeSpace::Global())), {}, ArgBoundary(ArgAccess::ReadWrite(), extent)),
                    Arg(Named("size", Type::IntS32()), {})},
                   {}, {}, Type::Unit0(), FunctionAffinity::Offload());
  const auto implementationDecl =
      FunctionDecl(implementationName, {element}, {},
                   {Arg(Named("values", pointer), {}, ArgBoundary(ArgAccess::ReadWrite(), extent)), Arg(Named("size", Type::IntS32()), {})},
                   {}, {}, Type::Unit0(), FunctionAffinity::Offload());
  const auto ptr = Term::Poison(pointer).widen();
  const auto value = Term::IntU32Const(7).widen();
  const std::vector<Stmt::Any> body{
      Stmt::Var(Named("cas", Type::IntU32()),
                Expr::SpecOp(Spec::GpuAtomicCAS(ptr, value, value, MemScope::Device(), MemOrder::Relaxed(), Type::IntU32())), false),
      Stmt::Var(Named("reduce", Type::IntU32()), Expr::SpecOp(Spec::GpuGroupReduce(AtomicOp::Add(), value, Type::IntU32())), false),
      Stmt::Var(Named("inclusive", Type::IntU32()), Expr::SpecOp(Spec::GpuGroupInclusiveScan(AtomicOp::Max(), value, Type::IntU32())),
                false),
      Stmt::Var(Named("exclusive", Type::IntU32()), Expr::SpecOp(Spec::GpuGroupExclusiveScan(AtomicOp::Min(), value, Type::IntU32())),
                false),
      ret()};
  const auto implementation = Function(implementationDecl, body, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                                       CallConvention::OffloadEntry(), publicName, {"subgroup.collective"});
  const auto package = Package(Interface(Sym({"vendor"}), {declaration}, {}), Program({}, {implementation}, {}, PassPhase::Initial(), {}));

  const auto decoded = package_from_msgpack(package_to_msgpack(package));
  CHECK(decoded == package);
  CHECK_FALSE(decoded.program.entry);
  REQUIRE(decoded.program.functions.size() == 1);
  CHECK(decoded.program.functions.front().convention.is<CallConvention::OffloadEntry>());
  CHECK(decoded.program.functions.front().implements == std::optional{publicName});
  CHECK(decoded.program.functions.front().requiredCapabilities == std::vector<std::string>{"subgroup.collective"});
  CHECK(decoded.program.functions.front().collect_all<Spec::GpuAtomicCAS>().size() == 1);
  CHECK(decoded.program.functions.front().collect_all<Spec::GpuGroupReduce>().size() == 1);
  CHECK(decoded.program.functions.front().collect_all<Spec::GpuGroupInclusiveScan>().size() == 1);
  CHECK(decoded.program.functions.front().collect_all<Spec::GpuGroupExclusiveScan>().size() == 1);
}
