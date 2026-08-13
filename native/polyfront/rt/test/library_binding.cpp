#include <algorithm>
#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/library_driver.hpp"
#include "polyfront/library_emit.hpp"
#include "polyfront/library_package.hpp"

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

TEST_CASE("native library resolution selects an exact typed implementation") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = transform(Sym({"implementation", "transform_w4"}), "Element");
  const auto candidate = ImplementationCandidate(publicName, implementation, {}, {TypeSizeConstraint("Element", 4)});
  const auto index = PackageIndex(LibraryDef(Sym({"foo"}), {publicDecl}, {}), {candidate});
  const auto bytes = packageindex_to_msgpack(index);
  REQUIRE(packageindex_from_msgpack(bytes) == index);
  auto incompatible = bytes;
  auto hashAt = incompatible.end();
  for (auto it = incompatible.begin(); std::distance(it, incompatible.end()) >= 32; ++it)
    if (std::all_of(it, it + 32, [](const auto ch) { return std::isxdigit(ch); })) {
      hashAt = it;
      break;
    }
  REQUIRE(hashAt != incompatible.end());
  *hashAt = *hashAt == '0' ? '1' : '0';
  CHECK(std::holds_alternative<std::string>(polyast::decodePackageIndex(incompatible.data(), incompatible.data() + incompatible.size())));

  const auto f32 = Type::Float32().widen();
  const auto f32p = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto call = InvokeSignature(publicName, {}, {}, {f32p, f32p, Type::IntS32()}, Type::Unit0());
  const auto resolved = polyfront::library::resolve(index, call, {}, {}, {{repr(f32), 4}});
  REQUIRE(resolved);
  CHECK(resolved.value->candidate == candidate);
}

TEST_CASE("native library drivers derive staging from declaration boundaries") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = transform(Sym({"implementation", "transform_w4"}), "Element");
  const auto candidate = ImplementationCandidate(publicName, implementation, {}, {TypeSizeConstraint("Element", 4)});
  const auto index = PackageIndex(LibraryDef(Sym({"foo"}), {publicDecl}, {}), {candidate});
  const auto f32 = Type::Float32().widen();
  const auto f32p = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto call = InvokeSignature(publicName, {}, {}, {f32p, f32p, Type::IntS32()}, Type::Unit0());
  const auto resolved = polyfront::library::resolve(index, call, {}, {}, {{repr(f32), 4}});
  REQUIRE(resolved);
  const auto plan = polyfront::library::buildDriver("__library_driver", *resolved.value, {{repr(f32), 4}});
  REQUIRE(plan);
  CHECK(plan.value->runtimeArguments == std::vector<size_t>{0, 1, 2});
  CHECK(plan.value->driver.collect_all<Spec::RemoteAlloc>().size() == 2);
  const auto copies = plan.value->driver.collect_all<Spec::RemoteMemcpy>();
  REQUIRE(copies.size() == 2);
  CHECK(copies[0].direction.is<Direction::LocalToRemote>());
  CHECK(copies[1].direction.is<Direction::RemoteToLocal>());
  CHECK(plan.value->driver.collect_all<Spec::RemoteFree>().size() == 2);
  const auto casts = plan.value->driver.collect_all<Expr::Cast>();
  CHECK(casts | aspartame::exists([](const auto &cast) {
          const auto selected = cast.from.template get<Term::Select>();
          return selected && selected->root.symbol == "v2" && cast.as.template is<Type::IntU64>();
        }));
}

TEST_CASE("native package decoding reports malformed metadata") {
  const std::vector<uint8_t> malformed{0xc1};
  CHECK(std::holds_alternative<std::string>(polyast::decodePackageIndex(malformed.data(), malformed.data() + malformed.size())));
  CHECK(std::holds_alternative<std::string>(polyast::decodeHashedProgram(malformed.data(), malformed.data() + malformed.size())));
}

TEST_CASE("package metadata does not change the Program schema hash") {
  const auto program = polyregion::polyfront::libraryProgram({}, {});
  const auto bytes = polyregion::polyast::hashed_program_to_msgpack(program);
  const std::string priorProgramHash = "547fdf1f8e09544533ed52bc647322a1";
  CHECK(std::search(bytes.begin(), bytes.end(), priorProgramHash.begin(), priorProgramHash.end()) != bytes.end());
}

TEST_CASE("native library resolution rejects malformed extent metadata") {
  using namespace polyast;
  const auto name = Sym({"bar", "malformed"});
  const auto ptr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const auto boundary = Boundary(ArgAccess::Read(), ArgExtent::Elements(ArgSizeExpr::Const(-1)));
  const auto publicDecl =
      FunctionDecl(name, {}, {}, {Arg(Named("in", ptr), {}, boundary)}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(Sym({"implementation", "malformed"}));
  const auto index = PackageIndex(LibraryDef(Sym({"foo"}), {publicDecl}, {}), {ImplementationCandidate(name, implementationDecl, {}, {})});
  const auto resolved = polyfront::library::resolve(index, InvokeSignature(name, {}, {}, {ptr}, Type::Unit0()), {}, {}, {});
  CHECK_FALSE(resolved);
  CHECK(resolved.errors | aspartame::exists([](const auto &error) { return error.find("negative") != std::string::npos; }));
}

TEST_CASE("native library validation rejects malformed type binders") {
  using namespace polyast;
  const auto name = Sym({"bar", "malformed"});
  const auto callable = Type::Exec({" "}, {Type::Var("Missing")}, Type::Unit0()).widen();
  const auto decl =
      FunctionDecl(name, {"T"}, {}, {Arg(Named("op", callable), {})}, {}, {}, Type::Var("AlsoMissing"), FunctionAffinity::Host());
  const auto errors = polyfront::library::validate(decl);
  CHECK(errors | aspartame::exists([](const auto &error) { return error.find("callable type variable 0 is empty") != std::string::npos; }));
  CHECK(errors
        | aspartame::exists([](const auto &error) { return error.find("undeclared type variable `Missing`") != std::string::npos; }));
  CHECK(errors
        | aspartame::exists([](const auto &error) { return error.find("undeclared type variable `AlsoMissing`") != std::string::npos; }));
}

TEST_CASE("native library binding turns callable placeholders into function references") {
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
  const auto package = polyfront::library::Package{PackageIndex(LibraryDef(Sym({"foo"}), {publicDecl}, {}), {candidate}),
                                                   polyfront::libraryProgram({implementationFn, unrelated}, {}),
                                                   {}};
  const auto callbackName = Sym({"user", "plusTwo"});
  const auto callbackDecl =
      FunctionDecl(callbackName, {}, {}, {Arg(Named("x", Type::IntS32()), {})}, {}, {}, Type::IntS32(), FunctionAffinity::Host());
  const auto call = InvokeSignature(publicName, {}, {}, {Type::IntS32(), Type::FnRef(callbackName)}, Type::IntS32());
  const auto resolved = polyfront::library::resolve(package.index, call, {callbackDecl}, {}, {{repr(Type::IntS32()), 4}});
  REQUIRE(resolved);
  const auto bound = polyfront::library::bindImplementationClosure(package, *resolved.value);
  REQUIRE(bound);
  CHECK(bound.value->size() == 1);
  const auto selected = *bound.value | aspartame::find([&](const auto &fn) { return fn.decl.name == implementationDecl.name; });
  REQUIRE(selected);
  CHECK(selected->decl.args.size() == 1);
  const auto invokes = selected->collect_all<Expr::Invoke>();
  REQUIRE(invokes.size() == 1);
  CHECK(invokes.front().callee == Type::FnRef(callbackName));
}

TEST_CASE("native library struct closure rejects caller conflicts") {
  using namespace polyast;
  using namespace polyast::dsl;
  const auto name = Sym({"bar", "Record"});
  const auto record = Type::Struct(name, {}).widen();
  const auto decl =
      FunctionDecl(Sym({"bar", "record"}), {}, {}, {Arg(Named("x", record), {})}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto function = Function(decl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false);
  const auto packageDef = StructDef(name, {}, {Named("x", Type::IntS32())}, {}, false);
  const auto callerDef = StructDef(name, {}, {Named("x", Type::Float32())}, {}, false);
  const auto index = PackageIndex(LibraryDef(Sym({"foo"}), {decl}, {}), {});
  const auto package = polyfront::library::Package{index, polyfront::libraryProgram({function}, {packageDef}), {}};
  const auto defs = polyfront::library::bindStructClosure(package, {function}, {callerDef});
  CHECK_FALSE(defs);
  CHECK(defs.errors | aspartame::exists([](const auto &error) { return error.find("conflicts") != std::string::npos; }));
}
