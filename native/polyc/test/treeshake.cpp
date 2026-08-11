#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"

#include "ast.h"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;
using namespace aspartame;

namespace {

constexpr auto Treeshake = "DeadFunctionElimination;DeadStructElimination";

Stmt::Any callTo(const std::string &name) {
  return Stmt::Var(Named("_" + name, Type::IntS32()), Expr::Invoke(Type::FnRef(Sym({name})), {}, {}, {}, Type::IntS32()), false);
}

Function exported(const std::string &name, const Vector<Stmt::Any> &body = {}, const Vector<Arg> &args = {}) {
  return function(name, args, Type::IntS32(), FunctionVisibility::Exported())(body ^ append(ret(Term::IntS32Const(0))));
}

Function internal(const std::string &name, const Vector<Stmt::Any> &body = {}) {
  return function(name, {}, Type::IntS32(), FunctionVisibility::Internal())(body ^ append(ret(Term::IntS32Const(0))));
}

Program library(const Vector<Function> &fns, const Vector<StructDef> &defs = {}) {
  return Program(function("__library_root", {}, Type::Unit0(), FunctionVisibility::Internal())({ret()}), fns, defs, PassPhase::Initial(),
                 {});
}

Vector<std::string> namesOf(const Vector<Function> &fns) {
  return fns ^ map([](const auto &f) { return repr(f.decl.name); }) ^ distinct() ^ sort();
}

Vector<std::string> shake(const Program &p) { return namesOf(compiler::runPipeline(p, Treeshake).functions); }

} // namespace

TEST_CASE("treeshake resolves both pass names and prunes to the export closure") {
  const auto kept = shake(library({exported("a", {callTo("shared")}), internal("shared"), internal("orphan")}));
  CHECK(kept == Vector<std::string>{"a", "shared"});
}

TEST_CASE("treeshake keeps what the entry reaches even with no exports") {
  const Program p(function("_main", {}, Type::Unit0(), FunctionVisibility::Internal())({callTo("used"), ret()}),
                  {internal("used"), internal("orphan")}, {}, PassPhase::Initial(), {});
  CHECK(shake(p) == Vector<std::string>{"used"});
}

TEST_CASE("treeshake dead-strips structs left unreferenced") {
  const auto used = Type::Struct(Sym({"Used"}), {});
  const StructDef usedDef(Sym({"Used"}), {}, {Named("x", Type::IntS32())}, {}, false);
  const StructDef unusedDef(Sym({"Unused"}), {}, {Named("y", Type::IntS32())}, {}, false);

  const auto lib = library({exported("a", {}, {Arg(Named("s", used), {})})}, {usedDef, unusedDef});
  const auto defs = compiler::runPipeline(lib, Treeshake).defs ^ map([](const auto &d) { return repr(d.name); });
  CHECK((defs ^ contains("Used")));
  CHECK(!(defs ^ contains("Unused")));
}

TEST_CASE("treeshaking twice yields identical bytes") {
  const auto lib = library({exported("a", {callTo("shared")}), exported("b", {callTo("shared")}), internal("shared"), internal("orphan")});
  const auto shakeBytes = [&] { return hashed_program_to_msgpack(compiler::runPipeline(lib, Treeshake)); };
  CHECK(shakeBytes() == shakeBytes());
}

TEST_CASE("a treeshaken library round-trips through msgpack") {
  const auto lib = library({exported("a", {callTo("shared")}), internal("shared"), internal("orphan")});
  const auto decoded = hashed_program_from_msgpack(hashed_program_to_msgpack(compiler::runPipeline(lib, Treeshake)));
  CHECK(namesOf(decoded.functions) == Vector<std::string>{"a", "shared"});
  CHECK((decoded.functions //
         ^ exists([](const auto &f) { return repr(f.decl.name) == "a" && f.visibility.template is<FunctionVisibility::Exported>(); })));
}
