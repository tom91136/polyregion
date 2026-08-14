#include "polyfront/package_emit.hpp"

#include <atomic>
#include <string>
#include <thread>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include <catch2/catch_test_macros.hpp>

#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"

namespace {

using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;
using namespace polyregion::polyfront;
using namespace polyregion::polyfront::package;

Package fixture(int32_t increment) {
  const auto i32 = Type::IntS32().widen();
  const auto publicName = Sym({"foo", "bar", "apply"});
  const auto implementationName = Sym({"foo", "implementation", "apply"});
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(implementationName);
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation = Function(implementationDecl, {ret(call(Intr::Add(x, Term::IntS32Const(increment).widen(), i32)))},
                                       FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), false);
  return {PackageIndex(InterfaceDef(Sym({"foo"}), {publicDecl}, {}), {ImplementationCandidate(publicName, implementationDecl, {}, {})}),
          packageProgram({implementation}, {})};
}

class TemporaryDirectory {
public:
  TemporaryDirectory() { REQUIRE_FALSE(llvm::sys::fs::createUniqueDirectory("polyregion-package-test", path)); }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path); }

  llvm::SmallString<128> path;
};

} // namespace

TEST_CASE("packages emit as one file") {
  TemporaryDirectory root;
  const auto first = fixture(1);
  const auto emittedFirst = emitPackage(first, root.path.str().str());
  REQUIRE(emittedFirst);
  CHECK(*emittedFirst.value == first);
  llvm::SmallString<128> path(root.path);
  llvm::sys::path::append(path, "foo", "lib.polyast");
  CHECK(llvm::sys::fs::is_regular_file(path));

  const auto second = fixture(2);
  const auto emittedSecond = emitPackage(second, root.path.str().str());
  REQUIRE(emittedSecond);
  const auto loaded = loadPackage("foo", {root.path.str().str()});
  REQUIRE(loaded);
  CHECK(*loaded.value == second);
}

TEST_CASE("concurrent package emitters return their own staged package") {
  TemporaryDirectory root;
  std::optional<Checked<Package>> first;
  std::optional<Checked<Package>> second;
  const auto firstFixture = fixture(1);
  const auto secondFixture = fixture(2);
  std::thread firstThread([&] { first = emitPackage(firstFixture, root.path.str().str()); });
  std::thread secondThread([&] { second = emitPackage(secondFixture, root.path.str().str()); });
  firstThread.join();
  secondThread.join();
  REQUIRE(first);
  REQUIRE(*first);
  CHECK(*first->value == firstFixture);
  REQUIRE(second);
  REQUIRE(*second);
  CHECK(*second->value == secondFixture);
}

TEST_CASE("package readers tolerate concurrent replacement") {
  TemporaryDirectory root;
  REQUIRE(emitPackage(fixture(0), root.path.str().str()));
  std::atomic<bool> done = false;
  std::vector<std::string> emitterErrors;
  std::vector<std::string> readerErrors;
  std::thread emitter([&] {
    for (int32_t increment = 1; increment <= 50; ++increment)
      if (const auto result = emitPackage(fixture(increment), root.path.str().str()); !result)
        emitterErrors ^= aspartame::concat(result.errors);
    done = true;
  });
  std::thread reader([&] {
    do {
      if (const auto result = loadPackage("foo", {root.path.str().str()}); !result) readerErrors ^= aspartame::concat(result.errors);
    } while (!done);
  });
  emitter.join();
  reader.join();
  CHECK(emitterErrors.empty());
  CHECK(readerErrors.empty());
}

TEST_CASE("package emission rejects incomplete implementations without replacing the emitted package") {
  TemporaryDirectory root;
  const auto current = fixture(1);
  REQUIRE(emitPackage(current, root.path.str().str()));

  auto incomplete = fixture(2);
  incomplete.program = packageProgram({}, {});
  const auto rejected = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(rejected);
  CHECK(rejected.errors == std::vector<std::string>{"implementation `foo.implementation.apply` is absent from the package program"});

  const auto loaded = loadPackage("foo", {root.path.str().str()});
  REQUIRE(loaded);
  CHECK(*loaded.value == current);
}

TEST_CASE("package emission rejects unsafe identities and duplicate candidates") {
  TemporaryDirectory root;
  CHECK(safePathComponent("foo$bar"));
  auto unsafe = fixture(1);
  unsafe.index = unsafe.index.withInterface(unsafe.index.interface.withName(Sym({".."})));
  const auto unsafeResult = emitPackage(unsafe, root.path.str().str());
  REQUIRE_FALSE(unsafeResult);
  CHECK(unsafeResult.errors == std::vector<std::string>{"invalid package identity `..`"});

  auto reserved = fixture(1);
  reserved.index = reserved.index.withInterface(reserved.index.interface.withName(Sym({"CON.txt"})));
  const auto reservedResult = emitPackage(reserved, root.path.str().str());
  REQUIRE_FALSE(reservedResult);
  CHECK(reservedResult.errors == std::vector<std::string>{"invalid package identity `CON.txt`"});

  auto trailing = fixture(1);
  trailing.index = trailing.index.withInterface(trailing.index.interface.withName(Sym({"foo."})));
  const auto trailingResult = emitPackage(trailing, root.path.str().str());
  REQUIRE_FALSE(trailingResult);
  CHECK(trailingResult.errors == std::vector<std::string>{"invalid package identity `foo.`"});

  auto duplicate = fixture(1);
  duplicate.index = duplicate.index.withCandidates({duplicate.index.candidates.front(), duplicate.index.candidates.front()});
  const auto duplicateResult = emitPackage(duplicate, root.path.str().str());
  REQUIRE_FALSE(duplicateResult);
  CHECK(duplicateResult.errors == std::vector<std::string>{"package index contains duplicate implementation candidates"});
}

TEST_CASE("package emission rejects invalid type-size constraints") {
  TemporaryDirectory root;
  auto invalid = fixture(1);
  const auto candidate = invalid.index.candidates.front().withTypeSizes({TypeSizeConstraint("Missing", 4)});
  invalid.index = invalid.index.withCandidates({candidate});
  const auto result = emitPackage(invalid, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(
      result.errors
      == std::vector<std::string>{"implementation `foo.implementation.apply` type-size constraint references unbound variable `Missing`"});
}

TEST_CASE("package emission rejects an incomplete implementation closure") {
  TemporaryDirectory root;
  auto incomplete = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation = incomplete.program.functions.front().withBody(
      {ret(Expr::Invoke(Type::FnRef(Sym({"foo", "implementation", "helper"})), {}, {}, {x}, i32).widen())});
  incomplete.program = packageProgram({implementation}, {});
  const auto result = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"implementation closure references absent function `foo.implementation.helper`"});
}

TEST_CASE("package emission rejects an ambiguous implementation closure") {
  TemporaryDirectory root;
  auto ambiguous = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto x = NamedBuilder(Named("x", i32));
  const auto helperName = Sym({"foo", "implementation", "helper"});
  const auto helperDecl = FunctionDecl(helperName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper = Function(helperDecl, {ret(x)}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), false);
  const auto implementation =
      ambiguous.program.functions.front().withBody({ret(Expr::Invoke(Type::FnRef(helperName), {}, {}, {x}, i32).widen())});
  ambiguous.program = packageProgram({implementation, helper, helper}, {});
  const auto result = emitPackage(ambiguous, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"implementation closure references ambiguous function `foo.implementation.helper`"});
}

TEST_CASE("package emission validates helper declarations") {
  TemporaryDirectory root;
  auto invalid = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto helperName = Sym({"foo", "implementation", "helper"});
  const auto helperDecl =
      FunctionDecl(helperName, {}, {}, {Arg(Named("x", Type::Var("Missing")), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper = Function(helperDecl, {ret(Term::IntS32Const(0))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), false);
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation =
      invalid.program.functions.front().withBody({ret(Expr::Invoke(Type::FnRef(helperName), {}, {}, {x}, i32).widen())});
  invalid.program = packageProgram({implementation, helper}, {});
  const auto result = emitPackage(invalid, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors
        | aspartame::exists([](const auto &error) { return error.find("undeclared type variable `Missing`") != std::string::npos; }));
}

TEST_CASE("package emission distinguishes structural symbols in implementation closure") {
  TemporaryDirectory root;
  auto incomplete = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto presentName = Sym({"a.b"});
  const auto absentName = Sym({"a", "b"});
  const auto helperDecl = FunctionDecl(presentName, {}, {}, {}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper = Function(helperDecl, {ret(Term::IntS32Const(0))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), false);
  const auto implementation = incomplete.program.functions.front().withBody(
      {Stmt::Var(Named("present", i32), Expr::Invoke(Type::FnRef(presentName), {}, {}, {}, i32), false),
       Stmt::Var(Named("absent", i32), Expr::Invoke(Type::FnRef(absentName), {}, {}, {}, i32), false), ret(Term::IntS32Const(0))});
  incomplete.program = packageProgram({implementation, helper}, {});
  const auto result = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"implementation closure references absent function `a.b`"});
}

TEST_CASE("package emission rejects an incomplete struct closure") {
  TemporaryDirectory root;
  auto incomplete = fixture(1);
  const auto record = Type::Struct(Sym({"foo", "Record"}), {}).widen();
  const auto implementation = incomplete.program.functions.front();
  incomplete.program = packageProgram({implementation.withDecl(implementation.decl.withArgs({Arg(Named("record", record), {})}))}, {});
  const auto publicDecl = incomplete.program.functions.front().decl.withName(incomplete.index.interface.decls.front().name);
  incomplete.index = incomplete.index.withInterface(incomplete.index.interface.withDecls({publicDecl}));
  incomplete.index =
      incomplete.index.withCandidates({incomplete.index.candidates.front().withImplementation(incomplete.program.functions.front().decl)});
  const auto result = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"struct definition `foo.Record` is absent"});
}

TEST_CASE("package emission rejects an ambiguous struct closure") {
  TemporaryDirectory root;
  auto ambiguous = fixture(1);
  const auto name = Sym({"foo", "Record"});
  const auto record = Type::Struct(name, {}).widen();
  const auto implementation = ambiguous.program.functions.front();
  const auto definition = StructDef(name, {}, {Named("value", Type::IntS32())}, {}, false);
  ambiguous.program =
      packageProgram({implementation.withDecl(implementation.decl.withArgs({Arg(Named("record", record), {})}))}, {definition, definition});
  const auto publicDecl = ambiguous.program.functions.front().decl.withName(ambiguous.index.interface.decls.front().name);
  ambiguous.index = ambiguous.index.withInterface(ambiguous.index.interface.withDecls({publicDecl}));
  ambiguous.index =
      ambiguous.index.withCandidates({ambiguous.index.candidates.front().withImplementation(ambiguous.program.functions.front().decl)});
  const auto result = emitPackage(ambiguous, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"struct definition `foo.Record` is ambiguous"});
}

#ifndef _WIN32
TEST_CASE("package emission rejects a symbolic-link package directory") {
  TemporaryDirectory root;
  TemporaryDirectory external;
  llvm::SmallString<128> directory(root.path);
  llvm::sys::path::append(directory, "foo");
  REQUIRE_FALSE(llvm::sys::fs::create_link(external.path, directory));
  const auto value = fixture(1);
  const auto result = emitPackage(value, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"package emission directory cannot be a symbolic link"});
}
#endif
