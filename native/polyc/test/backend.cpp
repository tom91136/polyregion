#include <algorithm>

#include "catch2/catch_all.hpp"
#include "fmt/format.h"

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
                  std::move(fpMode), isEntry, FunctionAffinity::Offload());
}

template <typename P> static void assertCompilationSucceeded(const P &p) {
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, polyregion::compiler::Options{Target::Object_LLVM_HOST, "native"}, OptLevel::O3);
  fmt::print("{}\n", repr(c));
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

  Program p(fn, {}, {}, PassPhase::Initial(), {});
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, {Target::Object_LLVM_AMDGCN, "gfx906"}, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);

  c = polyregion::compiler::compile(p, {Target::Object_LLVM_AMDGCN, "gfx803"}, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);

  c = polyregion::compiler::compile(p, {Target::Object_LLVM_NVPTX64, "sm_35"}, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("host prelude with foreign calls compiles to host object", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto bytePtr = Type::Ptr(Type::IntS8(), TypeSpace::Global());
  const Named capture("capture", bytePtr);
  const Named size("size", Type::IntU64());
  const Named remote("remote", Type::IntU64());

  std::vector<Term::Any> allocArgs{selectNamed(capture).widen(), selectNamed(size).widen(), Term::IntS32Const(0).widen()};
  Function prelude(Sym({"__polyregion_mirror_prelude"}), {}, std::optional<Arg>{}, {Arg(capture, {}), Arg(size, {})}, {}, {},
                   Type::IntU64(),
                   {
                       Var(remote, Expr::ForeignCall("polyrt_sma_alloc", allocArgs, Type::IntU64()).widen(), false).widen(),
                       Return(Expr::Alias(selectNamed(remote).widen()).widen()).widen(),
                   },
                   FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Host());

  Program p(prelude, {}, {}, PassPhase::Initial(), {});
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, {Target::Object_LLVM_HOST, "native"}, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("glcompute arena views do not demand fp16 for a float-only kernel", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named xField("x", Type::Float32());
  const StructDef capDef(Sym({"Cap"}), {}, {xField}, {});
  const Type::Struct capTpe(Sym({"Cap"}), {});
  const Named capture("#capture", Type::Ptr(capTpe, TypeSpace::Global()));

  std::vector<Stmt::Any> body{
      Stmt::Mut(Select({capture}, xField), Expr::Alias(Term::Float32Const(1.0f)).widen()).widen(),
      Stmt::Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
  };
  Function entry(Sym({"kernel"}), {}, std::optional<Arg>{}, {Arg(capture, {})}, {}, {}, Type::Unit0(), body, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {capDef}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV_GLCompute, ""};
  opts.pipelineSpec = "FullOpt;Anchor;ArenaView;VerifyAnchors(strict=true)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);
  CHECK(std::find(c.features.begin(), c.features.end(), "fp16") == c.features.end());
}

TEST_CASE("host-mirroring compile emits bitcode for the generated prelude", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto bytePtr = Type::Ptr(Type::IntS8(), TypeSpace::Global());
  Function entry(Sym({"_main"}), {}, std::optional<Arg>{}, {Arg(Named("capture", bytePtr), {})}, {}, {}, Type::Unit0(),
                 {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()}, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "Mirror";
  opts.hostMirroring = true;
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);
  REQUIRE(c.binary->size() >= 4);
  // LLVM bitcode magic 'B' 'C' 0xC0 0xDE
  CHECK(static_cast<unsigned char>((*c.binary)[0]) == 'B');
  CHECK(static_cast<unsigned char>((*c.binary)[1]) == 'C');
  CHECK(static_cast<unsigned char>((*c.binary)[2]) == 0xC0);
  CHECK(static_cast<unsigned char>((*c.binary)[3]) == 0xDE);
}
