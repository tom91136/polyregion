#include "ast.h"
#include "catch.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

#include <iostream>

using namespace polyregion::polyast;

using namespace Stmt;
using namespace Term;
using namespace Expr;

template <typename P> static void assertCompilationSucceeded(const P &p) {
  INFO(repr(p))
  auto c = polyregion::compiler::compile(
      p, polyregion::compiler::Options{polyregion::compiler::Target::Object_LLVM_x86_64, "native"},
      polyregion::compiler::Opt::O3);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("run", "[backend]") {
  polyregion::compiler::initialise();

  Function fn(Sym({"foo"}), {}, {}, {}, {}, Type::Unit(),
              {

                  //          Var(Named("gid", Type::Int()), {NullaryIntrinsic(NullaryIntrinsicKind::GpuGlobalIdxX(),
                  //          Type::Int())}),
                  Var(Named("a", Type::Int()), {Alias(IntConst(42))}),
                  Var(Named("b", Type::Int()), {Alias(IntConst(42))}),
                  Var(Named("c", Type::Int()),                                //
                      BinaryIntrinsic(Select({}, Named("a", Type::Int())),    //
                                      Select({}, Named("b", Type::Int())),    //
                                      BinaryIntrinsicKind::Add(), Type::Int() //
                                      )                                       //
                      ),
                  Return(Alias(UnitConst())),
              });

  Program p(fn, {}, {});
  INFO(repr(p))
  auto c = polyregion::compiler::compile(p, {polyregion::compiler::Target::Object_LLVM_AMDGCN, "sm_61"},
                                         polyregion::compiler::Opt::O3);
  std::cout << c << std::endl;
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}
