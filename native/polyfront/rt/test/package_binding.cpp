#include <algorithm>
#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/package.hpp"
#include "polyfront/package_driver.hpp"
#include "polyfront/package_program.hpp"

#include "polyast_codec.h"

using namespace polyregion;

namespace {

polyast::FunctionDecl transform(const polyast::Sym &name, const std::string &variable) {
  using namespace polyast;
  const auto tpe = Type::Var(variable).widen();
  const auto ptr = Type::Ptr(tpe, TypeSpace::Global()).widen();
  const auto extent = ArgExtent::Elements(ArgSizeExpr::Param(2));
  return FunctionDecl(name, {variable}, {},
                      {Arg(Named("in", ptr), {}, Boundary(ArgAccess::Read(), extent)),
                       Arg(Named("out", ptr), {}, Boundary(ArgAccess::Write(), extent)), Arg(Named("n", Type::IntS32()), {}, {})},
                      {}, {}, Type::Unit0(), FunctionAffinity::Host());
}

} // namespace

TEST_CASE("native package resolution selects an exact typed implementation") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = transform(Sym({"implementation", "transform_w4"}), "Element");
  const auto candidate = ImplementationCandidate(publicName, implementation, {}, {TypeSizeConstraint("Element", 4)});
  const auto index = PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {candidate});
  const auto package = Package(index, polyfront::packageProgram({}, {}));
  const auto bytes = package_to_msgpack(package);
  REQUIRE(package_from_msgpack(bytes) == package);
  auto incompatible = bytes;
  auto hashAt = incompatible.end();
  for (auto it = incompatible.begin(); std::distance(it, incompatible.end()) >= 32; ++it)
    if (std::all_of(it, it + 32, [](const auto ch) { return std::isxdigit(ch); })) {
      hashAt = it;
      break;
    }
  REQUIRE(hashAt != incompatible.end());
  *hashAt = *hashAt == '0' ? '1' : '0';
  CHECK(std::holds_alternative<std::string>(polyast::decodePackage(incompatible.data(), incompatible.data() + incompatible.size())));

  const auto f32 = Type::Float32().widen();
  const auto f32p = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto call = InvokeSignature(publicName, {}, {}, {f32p, f32p, Type::IntS32()}, Type::Unit0());
  const auto resolved = polyfront::package::resolve(index, call, {}, {}, {{repr(f32), 4}});
  REQUIRE(resolved);
  CHECK(resolved.value->candidate == candidate);
}

TEST_CASE("native package drivers derive staging from declaration boundaries") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = transform(Sym({"implementation", "transform_w4"}), "Element");
  const auto candidate = ImplementationCandidate(publicName, implementation, {}, {TypeSizeConstraint("Element", 4)});
  const auto index = PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {candidate});
  const auto f32 = Type::Float32().widen();
  const auto f32p = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto call = InvokeSignature(publicName, {}, {}, {f32p, f32p, Type::IntS32()}, Type::Unit0());
  const auto resolved = polyfront::package::resolve(index, call, {}, {}, {{repr(f32), 4}});
  REQUIRE(resolved);
  const auto plan = polyfront::package::buildDriver("__package_driver", *resolved.value, {{repr(f32), 4}});
  REQUIRE(plan);
  REQUIRE_FALSE(plan.value->driver.decl.args.empty());
  CHECK(plan.value->driver.decl.args.front().named.symbol == "#context");
  CHECK(plan.value->runtimeArguments == std::vector<size_t>{0, 1, 2});
  const auto usesContext = [](const auto &operation) {
    const auto selected = operation.context.template get<Term::Select>();
    return selected && selected->root.symbol == "#context";
  };
  const auto allocations = plan.value->driver.collect_all<Spec::RemoteAlloc>();
  REQUIRE(allocations.size() == 2);
  CHECK(allocations | aspartame::forall(usesContext));
  const auto copies = plan.value->driver.collect_all<Spec::RemoteMemcpy>();
  REQUIRE(copies.size() == 2);
  CHECK(copies | aspartame::forall(usesContext));
  CHECK(copies[0].direction.is<Direction::LocalToRemote>());
  CHECK(copies[1].direction.is<Direction::RemoteToLocal>());
  const auto frees = plan.value->driver.collect_all<Spec::RemoteFree>();
  REQUIRE(frees.size() == 2);
  CHECK(frees | aspartame::forall(usesContext));
  const auto casts = plan.value->driver.collect_all<Expr::Cast>();
  CHECK(casts | aspartame::exists([](const auto &cast) {
          const auto selected = cast.from.template get<Term::Select>();
          return selected && selected->root.symbol == "v2" && cast.as.template is<Type::IntU64>();
        }));
}

TEST_CASE("native package decoding reports malformed metadata") {
  const std::vector<uint8_t> malformed{0xc1};
  CHECK(std::holds_alternative<std::string>(polyast::decodePackage(malformed.data(), malformed.data() + malformed.size())));
  CHECK(std::holds_alternative<std::string>(polyast::decodeHashedProgram(malformed.data(), malformed.data() + malformed.size())));
}

TEST_CASE("Program and Package payloads have distinct wire schemas") {
  using namespace polyregion::polyast;
  const auto program = polyregion::polyfront::packageProgram({}, {});
  const auto programBytes = hashed_program_to_msgpack(program);
  const auto packageBytes = package_to_msgpack(Package(PackageIndex(InterfaceDef(Sym({"foo"}), {}, {}), {}), program));
  CHECK(std::holds_alternative<std::string>(decodePackage(programBytes.data(), programBytes.data() + programBytes.size())));
  CHECK(std::holds_alternative<std::string>(decodeHashedProgram(packageBytes.data(), packageBytes.data() + packageBytes.size())));
}

TEST_CASE("native package resolution rejects malformed extent metadata") {
  using namespace polyast;
  const auto name = Sym({"bar", "malformed"});
  const auto ptr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const auto boundary = Boundary(ArgAccess::Read(), ArgExtent::Elements(ArgSizeExpr::Const(-1)));
  const auto publicDecl =
      FunctionDecl(name, {}, {}, {Arg(Named("in", ptr), {}, boundary)}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(Sym({"implementation", "malformed"}));
  const auto index =
      PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {ImplementationCandidate(name, implementationDecl, {}, {})});
  const auto resolved = polyfront::package::resolve(index, InvokeSignature(name, {}, {}, {ptr}, Type::Unit0()), {}, {}, {});
  CHECK_FALSE(resolved);
  CHECK(resolved.errors | aspartame::exists([](const auto &error) { return error.find("negative") != std::string::npos; }));
}

TEST_CASE("native package validation rejects malformed type binders") {
  using namespace polyast;
  const auto name = Sym({"bar", "malformed"});
  const auto callable = Type::Exec({" "}, {Type::Var("Missing")}, Type::Unit0()).widen();
  const auto decl =
      FunctionDecl(name, {"T"}, {}, {Arg(Named("op", callable), {})}, {}, {}, Type::Var("AlsoMissing"), FunctionAffinity::Host());
  const auto errors = polyfront::package::validate(decl);
  CHECK(errors | aspartame::exists([](const auto &error) { return error.find("callable type variable 0 is empty") != std::string::npos; }));
  CHECK(errors
        | aspartame::exists([](const auto &error) { return error.find("undeclared type variable `Missing`") != std::string::npos; }));
  CHECK(errors
        | aspartame::exists([](const auto &error) { return error.find("undeclared type variable `AlsoMissing`") != std::string::npos; }));
}

TEST_CASE("native package binding turns callable placeholders into function references") {
  using namespace polyast;
  using namespace polyast::dsl;
  const auto t = Type::Var("T").widen();
  const auto op = Type::Exec({}, {t}, t).widen();
  const auto publicName = Sym({"bar", "apply"});
  const auto publicDecl =
      FunctionDecl(publicName, {"T"}, {}, {Arg(Named("x", t), {}), Arg(Named("op", op), {})}, {}, {}, t, FunctionAffinity::Host());
  const auto element = Type::Var("Element").widen();
  const auto implementationDecl =
      FunctionDecl(Sym({"bar", "implementation", "apply"}), {"Element", "Op"}, {},
                   {Arg(Named("x", element), {}), Arg(Named("op", Type::Var("Op")), {})}, {}, {}, element, FunctionAffinity::Host());
  const auto x = NamedBuilder(Named("x", element));
  const auto implementationFn = Function(implementationDecl, {ret(Expr::Invoke(Type::Var("Op"), {}, {}, {x}, element).widen())},
                                         FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false);
  const auto unrelated = function("unrelated", {}, Type::Unit0(), FunctionVisibility::Exported())({ret()});
  const auto candidate = ImplementationCandidate(publicName, implementationDecl, {}, {});
  const auto package = Package(PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {candidate}),
                               polyfront::packageProgram({implementationFn, unrelated}, {}));
  const auto callbackName = Sym({"user", "plusTwo"});
  const auto callbackDecl =
      FunctionDecl(callbackName, {}, {}, {Arg(Named("x", Type::IntS32()), {})}, {}, {}, Type::IntS32(), FunctionAffinity::Host());
  const auto call = InvokeSignature(publicName, {}, {}, {Type::IntS32(), Type::FnRef(callbackName)}, Type::IntS32());
  const auto resolved = polyfront::package::resolve(package.index, call, {callbackDecl}, {}, {{repr(Type::IntS32()), 4}});
  REQUIRE(resolved);
  const auto bound = polyfront::package::bindImplementationClosure(package, *resolved.value);
  REQUIRE(bound);
  CHECK(bound.value->size() == 1);
  const auto selected = *bound.value | aspartame::find([&](const auto &fn) { return fn.decl.name == implementationDecl.name; });
  REQUIRE(selected);
  CHECK(selected->decl.args.size() == 1);
  const auto invokes = selected->collect_all<Expr::Invoke>();
  REQUIRE(invokes.size() == 1);
  CHECK(invokes.front().callee == Type::FnRef(callbackName));
}

TEST_CASE("native package struct closure rejects caller conflicts") {
  using namespace polyast;
  using namespace polyast::dsl;
  const auto name = Sym({"bar", "Record"});
  const auto record = Type::Struct(name, {}).widen();
  const auto decl =
      FunctionDecl(Sym({"bar", "record"}), {}, {}, {Arg(Named("x", record), {})}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto function = Function(decl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false);
  const auto packageDef = StructDef(name, {}, {Named("x", Type::IntS32())}, {}, false);
  const auto callerDef = StructDef(name, {}, {Named("x", Type::Float32())}, {}, false);
  const auto index = PackageIndex(InterfaceDef(Sym({"foo"}), {decl}, {}), {});
  const auto package = Package(index, polyfront::packageProgram({function}, {packageDef}));
  const auto defs = polyfront::package::bindStructClosure(package, {function}, {callerDef});
  CHECK_FALSE(defs);
  CHECK(defs.errors | aspartame::exists([](const auto &error) { return error.find("conflicts") != std::string::npos; }));
}
