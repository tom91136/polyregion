#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/options_backend.hpp"
#include "polyfront/package.hpp"
#include "polyfront/package_driver.hpp"
#include "polyfront/package_program.hpp"

#include "polyast_codec.h"

using namespace polyregion;
using namespace aspartame;

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

TEST_CASE("object driver targets follow the consumer architecture") {
  using polyregion::compiletime::Target;
  const llvm::Triple x86("x86_64-unknown-linux-gnu"), riscv("riscv64-unknown-linux-gnu"), ppc("powerpc64le-unknown-linux-gnu");
  CHECK(polyfront::objectTargetFor(llvm::Triple("x86_64-apple-darwin")) == Target::Object_LLVM_x86_64);
  CHECK(polyfront::objectTargetFor(llvm::Triple("aarch64-apple-darwin")) == Target::Object_LLVM_AArch64);
  CHECK(polyfront::objectTargetFor(llvm::Triple("armv7-unknown-linux-gnueabihf")) == Target::Object_LLVM_ARM);
  CHECK(polyfront::objectTargetFor(riscv, riscv) == Target::Object_LLVM_HOST);
  CHECK(polyfront::objectTargetFor(ppc, ppc) == Target::Object_LLVM_HOST);
  CHECK_FALSE(polyfront::objectTargetFor(riscv, x86));

  CHECK(polyfront::objectCPUFor(llvm::Triple("aarch64-unknown-linux-gnu"), "", x86) == "generic");
  CHECK(polyfront::objectCPUFor(x86, "", x86) == "native");
  CHECK(polyfront::objectCPUFor(x86, "haswell", x86) == "haswell");

  CHECK(polyfront::objectTargetsCompatible(llvm::Triple("aarch64-apple-darwin"), llvm::Triple("arm64-apple-macosx26.0.0")));
  CHECK_FALSE(polyfront::objectTargetsCompatible(llvm::Triple("aarch64-unknown-linux-gnu"), llvm::Triple("arm64-apple-macosx26.0.0")));
  CHECK_FALSE(polyfront::objectTargetsCompatible(llvm::Triple("x86_64-unknown-linux"), llvm::Triple("x86_64-unknown-freebsd")));
  CHECK_FALSE(
      polyfront::objectTargetsCompatible(llvm::Triple("armv7-unknown-linux-gnueabi"), llvm::Triple("armv7-unknown-linux-gnueabihf")));

  llvm::LLVMContext context;
  llvm::Module driver("driver", context);
  driver.setDataLayout("e-p:64:64");
  llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::PointerType::get(context, 0)}, false),
                         llvm::GlobalValue::ExternalLinkage, "entry", driver);
  CHECK(polyfront::objectLayoutsCompatible(driver, llvm::DataLayout("e-p:64:64-p270:32:32")));
  CHECK_FALSE(polyfront::objectLayoutsCompatible(driver, llvm::DataLayout("e-p:32:32")));
}

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
  const auto hashAt = incompatible | sliding(32, 1) | index_where([](auto window) {
                        return window | forall([](const auto ch) { return std::isxdigit(static_cast<unsigned char>(ch)) != 0; });
                      });
  REQUIRE(hashAt >= 0);
  incompatible[static_cast<size_t>(hashAt)] = incompatible[static_cast<size_t>(hashAt)] == '0' ? '1' : '0';
  CHECK(std::holds_alternative<std::string>(polyast::decodePackage(incompatible.data(), incompatible.data() + incompatible.size())));

  const auto f32 = Type::Float32().widen();
  const auto f32p = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto call = InvokeSignature(publicName, {}, {}, {f32p, f32p, Type::IntS32()}, Type::Unit0());
  const auto resolved = polyfront::package::resolve(index, call, {}, {}, {{repr(f32), 4}});
  REQUIRE(resolved);
  CHECK(resolved.value->candidate == candidate);

  const auto fallback = candidate.withImplementation(implementation.withName(Sym({"implementation", "fallback"}))).withTypeSizes({});
  const auto exactResolution = polyfront::package::resolve(index.withCandidates({fallback, candidate}), call, {}, {}, {{repr(f32), 4}});
  REQUIRE(exactResolution);
  CHECK(exactResolution.value->candidate == candidate);
  const auto fallbackResolution = polyfront::package::resolve(index.withCandidates({candidate, fallback}), call, {}, {}, {{repr(f32), 2}});
  REQUIRE(fallbackResolution);
  CHECK(fallbackResolution.value->candidate == fallback);
  CHECK_FALSE(polyfront::package::resolve(index, call, {}, {}, {{repr(f32), 2}}));
  CHECK_FALSE(polyfront::package::resolve(
      index.withCandidates({fallback, fallback.withImplementation(implementation.withName(Sym({"implementation", "fallback_alt"})))}), call,
      {}, {}, {{repr(f32), 2}}));
  const auto nonPositive = candidate.withTypeSizes({TypeSizeConstraint("Element", 0)});
  const auto nonPositiveResolution =
      polyfront::package::resolve(index.withCandidates({nonPositive, fallback}), call, {}, {}, {{repr(f32), 0}});
  REQUIRE(nonPositiveResolution);
  CHECK(nonPositiveResolution.value->candidate == fallback);
}

TEST_CASE("native package resolution rejects cyclic call substitutions") {
  using namespace polyast;
  const auto name = Sym({"bar", "cycle"});
  const auto publicDecl = FunctionDecl(name, {"T", "U"}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementation = publicDecl.withName(Sym({"implementation", "cycle"}));
  const auto candidate = ImplementationCandidate(name, implementation, {}, {TypeSizeConstraint("T", 4), TypeSizeConstraint("U", 4)});
  const auto index = PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {candidate});
  const auto call = InvokeSignature(name, {Type::Var("U"), Type::Var("T")}, {}, {}, Type::Unit0());

  const auto resolved = polyfront::package::resolve(index, call, {}, {}, {});
  CHECK_FALSE(resolved);
  CHECK(resolved.errors | exists([](const auto &error) { return error ^ contains_slice("cyclic substitution"); }));
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
  CHECK(allocations | forall(usesContext));
  const auto copies = plan.value->driver.collect_all<Spec::RemoteMemcpy>();
  REQUIRE(copies.size() == 2);
  CHECK(copies | forall(usesContext));
  CHECK(copies[0].direction.is<Direction::LocalToRemote>());
  CHECK(copies[1].direction.is<Direction::RemoteToLocal>());
  const auto frees = plan.value->driver.collect_all<Spec::RemoteFree>();
  REQUIRE(frees.size() == 2);
  CHECK(frees | forall(usesContext));
  const auto casts = plan.value->driver.collect_all<Expr::Cast>();
  CHECK(casts | exists([](const auto &cast) {
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
  CHECK(resolved.errors | exists([](const auto &error) { return error ^ contains_slice("negative"); }));
}

TEST_CASE("native package validation rejects malformed type binders") {
  using namespace polyast;
  const auto name = Sym({"bar", "malformed"});
  const auto callable = Type::Exec({" "}, {Type::Var("Missing")}, Type::Unit0()).widen();
  const auto decl =
      FunctionDecl(name, {"T"}, {}, {Arg(Named("op", callable), {})}, {}, {}, Type::Var("AlsoMissing"), FunctionAffinity::Host());
  const auto errors = polyfront::package::validate(decl);
  CHECK(errors | exists([](const auto &error) { return error ^ contains_slice("callable type variable 0 is empty"); }));
  CHECK(errors | exists([](const auto &error) { return error ^ contains_slice("undeclared type variable `Missing`"); }));
  CHECK(errors | exists([](const auto &error) { return error ^ contains_slice("undeclared type variable `AlsoMissing`"); }));
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
  const auto callbackOverload =
      FunctionDecl(callbackName, {}, {}, {Arg(Named("x", Type::Float32()), {})}, {}, {}, Type::Float32(), FunctionAffinity::Host());
  const auto call = InvokeSignature(publicName, {}, {}, {Type::IntS32(), Type::FnRef(callbackName)}, Type::IntS32());
  const auto resolved = polyfront::package::resolve(package.index, call, {callbackOverload, callbackDecl}, {}, {{repr(Type::IntS32()), 4}});
  REQUIRE(resolved);
  const auto bound = polyfront::package::bindImplementationClosure(package, *resolved.value);
  REQUIRE(bound);
  CHECK(bound.value->size() == 1);
  const auto selected = *bound.value | find([&](const auto &fn) { return fn.decl.name == implementationDecl.name; });
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
  CHECK(defs.errors | exists([](const auto &error) { return error ^ contains_slice("conflicts"); }));
}
