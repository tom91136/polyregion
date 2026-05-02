#include "ast.h"
#include "catch2/catch_all.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"

#include <iostream>

using namespace polyregion::polyast;
using namespace polyregion::compiletime;
using namespace Stmt;
using namespace Expr;
using namespace Intr;

static Function mkFn(const std::string &name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body,
                     std::set<FunctionAttr::Any> attrs = {FunctionAttr::Exported()}) {
  return Function(Sym({name}), {}, {}, std::move(args), {}, {}, std::move(rtn), std::move(body), std::move(attrs));
}

template <typename P> static void assertCompilationSucceeded(const P &p) {
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_x86_64, "native"}, OptLevel::O3);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("run", "[backend]") {
  polyregion::compiler::initialise();

  Function fn = mkFn("foo", {}, Type::Unit0(),
                     {

                         Var(Named("a", Type::IntS32()), {IntS32Const(42)}),
                         Var(Named("b", Type::IntS32()), {IntS32Const(42)}),
                         Var(Named("c", Type::IntS32()),                                        //
                             IntrOp(Add(Select({}, Named("a", Type::IntS32())),                 //
                                        Select({}, Named("b", Type::IntS32())), Type::IntS32()) //
                                    )                                                           //
                             ),
                         Return(Unit0Const()),
                     });

  Program p(fn, {}, {});
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
