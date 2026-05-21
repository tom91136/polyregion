#include <iostream>

#include "catch2/catch_all.hpp"

#include "ast.h"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

using namespace polyregion::polyast;
using namespace polyregion::compiletime;
using namespace Stmt;
using namespace Expr;
using namespace Intr;

static Function mkFn(const std::string &name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body,
                     FunctionVisibility::Any visibility = FunctionVisibility::Exported(),
                     FunctionFpMode::Any fpMode = FunctionFpMode::Relaxed(), bool isEntry = false) {
  return Function(Sym({name}), {}, std::optional<Arg>{}, std::move(args), {}, {}, std::move(rtn), std::move(body), std::move(visibility),
                  std::move(fpMode), isEntry);
}

template <typename P> static void assertCompilationSucceeded(const P &p) {
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_HOST, "native"}, OptLevel::O3);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("run", "[backend]") {
  polyregion::compiler::initialise();

  using namespace polyregion::polyast::dsl;
  const Named aN("a", Type::IntS32());
  const Named bN("b", Type::IntS32());
  Function fn =
      mkFn("foo", {}, Type::Unit0(),
           {
               Var(aN, Expr::Alias(Term::IntS32Const(42)).widen(), /*isMutable*/ false).widen(),
               Var(bN, Expr::Alias(Term::IntS32Const(42)).widen(), /*isMutable*/ false).widen(),
               Var(Named("c", Type::IntS32()), Expr::IntrOp(Add(selectNamed(aN), selectNamed(bN), Type::IntS32()).widen()).widen(),
                   /*isMutable*/ false)
                   .widen(),
               Return(Expr::Alias(Term::Unit0Const()).widen()).widen(),
           });

  Program p(fn, {}, {}, PassPhase::Initial());
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, {Target::Object_LLVM_AMDGCN, "gfx906"}, OptLevel::O3);
  INFO(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);

  c = polyregion::compiler::compile(p, {Target::Object_LLVM_AMDGCN, "gfx803"}, OptLevel::O3);
  INFO(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);

  c = polyregion::compiler::compile(p, {Target::Object_LLVM_NVPTX64, "sm_35"}, OptLevel::O3);
  INFO(c);
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}
