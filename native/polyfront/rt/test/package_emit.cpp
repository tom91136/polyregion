#include "polyfront/package_emit.hpp"

#include <atomic>
#include <string>
#include <thread>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <catch2/catch_test_macros.hpp>

#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"

namespace {

using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;
using namespace polyregion::polyfront;
using namespace polyregion::polyfront::package;
using namespace aspartame;

Package fixture(int32_t increment) {
  const auto i32 = Type::IntS32().widen();
  const auto publicName = Sym({"foo", "bar", "apply"});
  const auto implementationName = Sym({"foo", "implementation", "apply"});
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(implementationName);
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation =
      Function(implementationDecl, {ret(call(Intr::Add(x, Term::IntS32Const(increment).widen(), i32)))}, FunctionVisibility::Exported(),
               FunctionFpMode::Relaxed(), CallConvention::RegularCall(), publicName);
  return {Interface(Sym({"foo"}), {publicDecl}, {}), packageProgram({implementation}, {})};
}

class TemporaryDirectory {
public:
  TemporaryDirectory() { REQUIRE_FALSE(llvm::sys::fs::createUniqueDirectory("polyregion-package-test", path)); }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path); }

  llvm::SmallString<128> path;
};

bool writeBytes(const llvm::StringRef path, const std::vector<uint8_t> &bytes) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  if (error) return false;
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  stream.close();
  return !stream.has_error();
}

Checked<Package> linkPackage(const PackageLinkRequest &request, const std::string &root) {
  TemporaryDirectory inputs;
  llvm::SmallString<128> interfacePath(inputs.path), errorPath(inputs.path);
  llvm::sys::path::append(interfacePath, "interface.polyast");
  llvm::sys::path::append(errorPath, "stderr.txt");
  if (!writeBytes(interfacePath, interface_to_msgpack(request.interface))) return {{}, {"cannot write package interface input"}};

  std::vector<std::string> ownedArgs{POLYC_TEST_EXECUTABLE, "package", "link"};
  for (const auto &capability : request.capabilities)
    ownedArgs.emplace_back("--capability=" + capability);
  ownedArgs.emplace_back(interfacePath.str());
  ownedArgs.emplace_back(root);
  for (size_t index = 0; index < request.programFragments.size(); ++index) {
    llvm::SmallString<128> programPath(inputs.path);
    llvm::sys::path::append(programPath, "program-" + std::to_string(index) + ".polyast");
    if (!writeBytes(programPath, hashed_program_to_msgpack(request.programFragments[index])))
      return {{}, {"cannot write package program input"}};
    ownedArgs.emplace_back(programPath.str());
  }
  std::vector<llvm::StringRef> args;
  args.reserve(ownedArgs.size());
  for (const auto &arg : ownedArgs)
    args.emplace_back(arg);
  std::string executionError;
  const int code = llvm::sys::ExecuteAndWait(POLYC_TEST_EXECUTABLE, args, std::nullopt, {std::nullopt, std::nullopt, errorPath.str()}, 0, 0,
                                             &executionError);
  if (code != 0) {
    std::vector<std::string> errors;
    if (const auto buffer = llvm::MemoryBuffer::getFile(errorPath)) {
      std::string diagnostic = (*buffer)->getBuffer().str();
      for (size_t offset = 0; offset <= diagnostic.size();) {
        const auto end = diagnostic.find('\n', offset);
        const auto line = diagnostic.substr(offset, end - offset);
        if (!line.empty()) errors.emplace_back(line);
        if (end == std::string::npos) break;
        offset = end + 1;
      }
    }
    if (!executionError.empty()) errors.emplace_back(std::move(executionError));
    if (errors.empty()) errors.emplace_back("polyc package link exited with code " + std::to_string(code));
    return {{}, std::move(errors)};
  }
  return loadPackage(symbol(request.interface.name), {root});
}

Checked<Package> emitPackage(const Package &package, const std::string &root) {
  return linkPackage(PackageLinkRequest(package.interface, {package.program}, {}), root);
}

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
  std::thread firstThread([&] { first = publishPackage(firstFixture, root.path.str().str()); });
  std::thread secondThread([&] { second = publishPackage(secondFixture, root.path.str().str()); });
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
  REQUIRE(publishPackage(fixture(0), root.path.str().str()));
  std::atomic<bool> done = false;
  std::vector<std::string> emitterErrors;
  std::vector<std::string> readerErrors;
  std::thread emitter([&] {
    for (int32_t increment = 1; increment <= 50; ++increment)
      if (const auto result = publishPackage(fixture(increment), root.path.str().str()); !result) emitterErrors ^= concat(result.errors);
    done = true;
  });
  std::thread reader([&] {
    do {
      if (const auto result = loadPackage("foo", {root.path.str().str()}); !result) readerErrors ^= concat(result.errors);
    } while (!done);
  });
  emitter.join();
  reader.join();
  CHECK(emitterErrors.empty());
  CHECK(readerErrors.empty());
}

TEST_CASE("polyc transports a large linked package without truncation") {
  TemporaryDirectory root;
  const auto publicName = Sym({"bar", "large"});
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(Sym({"implementation", "large"}));
  std::vector<Function> functions;
  functions.reserve(4097);
  functions.emplace_back(implementationDecl, std::vector<Stmt::Any>{}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                         CallConvention::RegularCall(), publicName);
  for (int i = 0; i < 4096; ++i) {
    const auto decl = publicDecl.withName(Sym({"helper", std::to_string(i)}));
    functions.emplace_back(decl, std::vector<Stmt::Any>{}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                           CallConvention::RegularCall());
  }
  const auto request = PackageLinkRequest(Interface(Sym({"large"}), {publicDecl}, {}), {packageProgram(std::move(functions), {})}, {});
  const auto result = linkPackage(request, root.path.str().str());
  REQUIRE(result);
  CHECK(result.value->program.functions.size() == 4097);
  CHECK(std::any_of(result.value->program.functions.begin(), result.value->program.functions.end(),
                    [](const auto &function) { return function.decl.name == Sym({"helper", "4095"}); }));
}

TEST_CASE("package emission rejects incomplete implementations without replacing the emitted package") {
  TemporaryDirectory root;
  const auto current = fixture(1);
  REQUIRE(emitPackage(current, root.path.str().str()));

  auto incomplete = fixture(2);
  incomplete.program = packageProgram({}, {});
  const auto rejected = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(rejected);
  CHECK(rejected.errors == std::vector<std::string>{"public declaration `foo.bar.apply` has no compatible implementation"});

  const auto loaded = loadPackage("foo", {root.path.str().str()});
  REQUIRE(loaded);
  CHECK(*loaded.value == current);
}

TEST_CASE("package emission rejects unsafe identities") {
  TemporaryDirectory root;
  CHECK(safePathComponent("foo$bar"));
  auto unsafe = fixture(1);
  unsafe.interface = unsafe.interface.withName(Sym({".."}));
  const auto unsafeResult = emitPackage(unsafe, root.path.str().str());
  REQUIRE_FALSE(unsafeResult);
  CHECK(unsafeResult.errors == std::vector<std::string>{"invalid package identity `..`"});

  auto reserved = fixture(1);
  reserved.interface = reserved.interface.withName(Sym({"CON.txt"}));
  const auto reservedResult = emitPackage(reserved, root.path.str().str());
  REQUIRE_FALSE(reservedResult);
  CHECK(reservedResult.errors == std::vector<std::string>{"invalid package identity `CON.txt`"});

  auto trailing = fixture(1);
  trailing.interface = trailing.interface.withName(Sym({"foo."}));
  const auto trailingResult = emitPackage(trailing, root.path.str().str());
  REQUIRE_FALSE(trailingResult);
  CHECK(trailingResult.errors == std::vector<std::string>{"invalid package identity `foo.`"});
}

TEST_CASE("package emission rejects invalid type-size constraints") {
  TemporaryDirectory root;
  auto invalid = fixture(1);
  const auto candidate = invalid.program.functions.front();
  invalid.program.functions.front() = candidate.withDecl(candidate.decl.withTpeVars({Type::Var("Missing", 4)}));
  const auto result = emitPackage(invalid, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors ^ exists([](const auto &error) { return error ^ contains_slice("type variable `Missing` is not bound"); }));
  CHECK(result.errors
        ^ exists([](const auto &error) { return error ^ contains_slice("public declaration `foo.bar.apply` has no compatible"); }));
}

TEST_CASE("package emission rejects malformed complete struct binders and applications") {
  TemporaryDirectory root;
  const auto structName = Sym({"foo", "Box"});
  const auto applied = Type::Struct(structName, {Type::IntS32(), Type::Float32()}).widen();
  const auto malformed = StructDef(structName, {Type::Var("T", 4)}, {Named("value", Type::Var("T", 8))}, {}, false);
  const auto publicName = Sym({"foo", "bar", "struct"});
  const auto publicDecl =
      FunctionDecl(publicName, {}, {}, {Arg(Named("box", applied), {})}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(Sym({"foo", "implementation", "struct"}));
  const auto implementation = Function(implementationDecl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                                       CallConvention::RegularCall(), publicName);
  const auto package = Package(Interface(Sym({"foo"}), {publicDecl}, {}), packageProgram({implementation}, {malformed}));

  const auto result = emitPackage(package, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors ^ exists([](const auto &error) { return error ^ contains_slice("differs from its binder"); }));
  CHECK(result.errors ^ exists([](const auto &error) { return error ^ contains_slice("type-argument count differs"); }));
}

TEST_CASE("package emission rejects partial type-size constraints") {
  TemporaryDirectory root;
  const auto name = Sym({"foo", "bar", "transform"});
  const auto publicDecl = FunctionDecl(name, {Type::Var("T"), Type::Var("U")}, {},
                                       {Arg(Named("in", Type::Var("T")), {}), Arg(Named("out", Type::Var("U")), {})}, {}, {}, Type::Unit0(),
                                       FunctionAffinity::Host());
  const auto implementationDecl =
      FunctionDecl(Sym({"foo", "implementation", "transform"}), {Type::Var("Input", 4), Type::Var("Output")}, {},
                   {Arg(Named("in", Type::Var("Input", 4)), {}), Arg(Named("out", Type::Var("Output")), {})}, {}, {}, Type::Unit0(),
                   FunctionAffinity::Host());
  const auto implementation =
      Function(implementationDecl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall(), name);
  const auto package = Package(Interface(Sym({"foo"}), {publicDecl}, {}), packageProgram({implementation}, {}));

  const auto result = emitPackage(package, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors
        == std::vector<std::string>{"implementation `foo.implementation.transform` type-size constraints must cover all type variables"});
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

TEST_CASE("package emission deduplicates identical implementation helpers") {
  TemporaryDirectory root;
  auto ambiguous = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto x = NamedBuilder(Named("x", i32));
  const auto helperName = Sym({"foo", "implementation", "helper"});
  const auto helperDecl = FunctionDecl(helperName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper =
      Function(helperDecl, {ret(x)}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto implementation =
      ambiguous.program.functions.front().withBody({ret(Expr::Invoke(Type::FnRef(helperName), {}, {}, {x}, i32).widen())});
  ambiguous.program = packageProgram({implementation, helper, helper}, {});
  const auto result = emitPackage(ambiguous, root.path.str().str());
  REQUIRE(result);
  CHECK(result.value->program.functions.size() == 2);
}

TEST_CASE("package emission validates helper declarations") {
  TemporaryDirectory root;
  auto invalid = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto helperName = Sym({"foo", "implementation", "helper"});
  const auto helperDecl =
      FunctionDecl(helperName, {}, {}, {Arg(Named("x", Type::Var("Missing")), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper = Function(helperDecl, {ret(Term::IntS32Const(0))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                               CallConvention::RegularCall());
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation =
      invalid.program.functions.front().withBody({ret(Expr::Invoke(Type::FnRef(helperName), {}, {}, {x}, i32).widen())});
  invalid.program = packageProgram({implementation, helper}, {});
  const auto result = emitPackage(invalid, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors ^ exists([](const auto &error) { return error ^ contains_slice("undeclared type variable `Missing`"); }));
}

TEST_CASE("package emission distinguishes structural symbols in implementation closure") {
  TemporaryDirectory root;
  auto incomplete = fixture(1);
  const auto i32 = Type::IntS32().widen();
  const auto presentName = Sym({"a.b"});
  const auto absentName = Sym({"a", "b"});
  const auto helperDecl = FunctionDecl(presentName, {}, {}, {}, {}, {}, i32, FunctionAffinity::Host());
  const auto helper = Function(helperDecl, {ret(Term::IntS32Const(0))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                               CallConvention::RegularCall());
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
  const auto publicDecl = incomplete.program.functions.front().decl.withName(incomplete.interface.declarations.front().name);
  incomplete.interface = incomplete.interface.withDeclarations({publicDecl});
  const auto result = emitPackage(incomplete, root.path.str().str());
  REQUIRE_FALSE(result);
  CHECK(result.errors == std::vector<std::string>{"struct definition `foo.Record` is absent"});
}

TEST_CASE("package emission deduplicates identical struct definitions") {
  TemporaryDirectory root;
  auto ambiguous = fixture(1);
  const auto name = Sym({"foo", "Record"});
  const auto record = Type::Struct(name, {}).widen();
  const auto implementation = ambiguous.program.functions.front();
  const auto definition = StructDef(name, {}, {Named("value", Type::IntS32())}, {}, false);
  ambiguous.program =
      packageProgram({implementation.withDecl(implementation.decl.withArgs({Arg(Named("record", record), {})}))}, {definition, definition});
  const auto publicDecl = ambiguous.program.functions.front().decl.withName(ambiguous.interface.declarations.front().name);
  ambiguous.interface = ambiguous.interface.withDeclarations({publicDecl});
  const auto result = emitPackage(ambiguous, root.path.str().str());
  REQUIRE(result);
  CHECK(result.value->program.defs.size() == 1);
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
