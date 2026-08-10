#include <algorithm>

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"
#include "fmt/format.h"

#include "polyregion/env_keys.h"

#include "ast.h"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "scoped_env.h"

using namespace aspartame;
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

template <typename C> static const std::string &eventDataOf(const C &c, const std::string &name) {
  const auto event = c.events ^ find_cref([&](const auto &e) { return e.name == name; });
  REQUIRE(event);
  return event->get().data;
}

template <typename C> static const std::string &llvmIrOf(const C &c) { return eventDataOf(c, "ast_to_llvm_ir"); }

static Program arenaOffsetCastProgram() {
  using namespace polyregion::polyast::dsl;
  const Named off("off", Type::IntU64());
  const auto ptrTpe = Type::Ptr(Type::IntS8(), TypeSpace::Global());
  const Named ptr("p", ptrTpe);
  Function entry = mkFn("kernel", {}, Type::Unit0(),
                        {
                            Var(off, Expr::Alias(Term::IntU64Const(16).widen()).widen(), false).widen(),
                            Var(ptr, Expr::Cast(selectNamed(off).widen(), ptrTpe).widen(), false).widen(),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  return Program(entry, {}, {}, PassPhase::Initial(), {});
}

TEST_CASE("LLVM IR event payloads are opt-in", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv noDebug(polyregion::env::PolyregionDebug, std::nullopt);
  const ScopedEnv noVerbose(polyregion::env::PolycVerboseNames, std::nullopt);
  const Program p = arenaOffsetCastProgram();
  const polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};

  const auto normal = polyregion::compiler::compile(p, opts, OptLevel::O3);
  REQUIRE(normal.binary);
  CHECK(eventDataOf(normal, "ast_to_llvm_ir").empty());
  CHECK(eventDataOf(normal, "llvm_to_obj_opt").empty());

  {
    const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
    const auto diagnostic = polyregion::compiler::compile(p, opts, OptLevel::O3);
    REQUIRE(diagnostic.binary);
    CHECK_FALSE(eventDataOf(diagnostic, "ast_to_llvm_ir").empty());
    CHECK_FALSE(eventDataOf(diagnostic, "llvm_to_obj_opt").empty());
  }
}

TEST_CASE("C source bounds local names and retains source names on request", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Origin inputOrigin(SourcePosition("bounded.cpp", 4, 7), std::string("input"), Sym({"source"}));
  const Origin valueOrigin(SourcePosition("bounded.cpp", 5, 3), std::string("value"), Sym({"source"}));
  const Named input("an_internal_parameter_name_that_can_grow_without_bound", Type::IntS32(), inputOrigin);
  const Named value("an_internal_local_name_that_can_grow_without_bound", Type::IntS32(), valueOrigin);
  const Function entry = mkFn(
      "bounded_kernel_name", {Arg(input, {})}, Type::IntS32(),
      {Var(value, Expr::Alias(selectNamed(input).widen()).widen(), false).widen(), Return(Expr::Alias(selectNamed(value).widen()).widen())},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program p(entry, {}, {}, PassPhase::Initial(), {});
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  const auto emit = [&] {
    const auto result = polyregion::compiler::compile(p, opts, OptLevel::O0);
    REQUIRE(result.binary);
    return std::string(result.binary->begin(), result.binary->end());
  };

  const ScopedEnv noVerbose(polyregion::env::PolycVerboseNames, std::nullopt);
  const auto dense = emit();
  CHECK(dense == emit());
  CHECK_THAT(dense, Catch::Matchers::ContainsSubstring("bounded_kernel_name"));
  CHECK_THAT(dense, Catch::Matchers::ContainsSubstring("_v0"));
  CHECK_THAT(dense, Catch::Matchers::ContainsSubstring("_v1"));
  CHECK_THAT(dense, !Catch::Matchers::ContainsSubstring(input.symbol));
  CHECK_THAT(dense, !Catch::Matchers::ContainsSubstring(value.symbol));

  {
    const ScopedEnv verbose(polyregion::env::PolycVerboseNames, std::string("1"));
    const auto readable = emit();
    CHECK_THAT(readable, Catch::Matchers::ContainsSubstring("int input"));
    CHECK_THAT(readable, Catch::Matchers::ContainsSubstring("int value"));
    CHECK_THAT(readable, !Catch::Matchers::ContainsSubstring(input.symbol));
    CHECK_THAT(readable, !Catch::Matchers::ContainsSubstring(value.symbol));
  }
}

TEST_CASE("opencl source accepts configured subgroup emulation", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named value("value", Type::Float32());
  const Named result("result", Type::Float32());
  const Function entry =
      mkFn("kernel", {Arg(value, {})}, Type::Unit0(),
           {Var(result,
                Expr::SpecOp(Spec::GpuShuffleDown(selectNamed(value).widen(), Term::IntU32Const(1).widen(), Term::IntU32Const(7).widen(),
                                                  Term::IntU32Const(~uint32_t{0}).widen(), Type::Float32()))
                    .widen(),
                false)
                .widen(),
            Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program p(entry, {}, {}, PassPhase::Initial(), {});
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "SubgroupLower(width=8,maxGroupSize=256)";

  const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("[256]") != std::string::npos);
  CHECK(source.find("get_local_id(0)") != std::string::npos);
  CHECK(source.find("sub_group_barrier(CLK_LOCAL_MEM_FENCE)") != std::string::npos);
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
  const StructDef capDef(Sym({"Cap"}), {}, {xField}, {}, false);
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
  opts.pipelineSpec = "FullOpt;PartialEval(canonicaliseAddresses=true);ArenaView;VerifyAnchors(strict=true)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);
  CHECK(!(c.features ^ contains("fp16")));
}

TEST_CASE("by-value array initialisation copies contents on by-pointer targets", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto arrTpe = Type::Arr(Type::IntS32(), 3, TypeSpace::Global());
  const Named src("src", arrTpe);
  const Named dst("dst", arrTpe);
  const Named readBack("readBack", Type::IntS32());

  std::vector<Stmt::Any> body{
      Stmt::Var(src, std::optional<Expr::Any>{}, true).widen(),
      Stmt::Update(Select({}, src), Term::IntS32Const(0).widen(), Term::IntS32Const(7).widen()).widen(),
      Stmt::Var(dst, Expr::Alias(Select({}, src).widen()).widen(), false).widen(),
      Stmt::Var(readBack, Expr::Index(Select({}, dst).widen(), Term::IntS32Const(0).widen(), Type::IntS32()).widen(), false).widen(),
      Stmt::Update(Select({}, src), Term::IntS32Const(1).widen(), selectNamed(readBack).widen()).widen(),
      Stmt::Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
  };
  Function entry(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(), body, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV64_Kernel, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");

  CHECK(llvmIrOf(c).find("llvm.memcpy") != std::string::npos);
}

TEST_CASE("opencl source keeps scalar arena offset casts in the target pointer space", "[backend]") {
  polyregion::compiler::initialise();

  const Program p = arenaOffsetCastProgram();
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("global char* _v1 = ((global char*) _v0);") != std::string::npos);
  CHECK(source.find("private char* _v1") == std::string::npos);
}

TEST_CASE("C source zero-initialises struct locals", "[backend]") {
  polyregion::compiler::initialise();

  const auto stateSym = Sym({"State"});
  const auto stateTpe = Type::Struct(stateSym, {});
  const StructDef state(stateSym, {}, {Named("x", Type::IntS32()), Named("y", Type::IntS32())}, {}, false);
  const Named value("value", stateTpe);
  Function entry =
      mkFn("kernel", {}, Type::Unit0(),
           {Var(value, std::optional<Expr::Any>{}, true).widen(), Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  Program p(entry, {}, {state}, PassPhase::Initial(), {});

  for (const auto &[target, expected] : std::vector<std::pair<Target, std::string>>{
           {Target::Source_C_OpenCL1_1, "State _v0 = {0};"},
           {Target::Source_C_Metal1_0, "State _v0 = {};"},
       }) {
    polyregion::compiler::Options opts{target, ""};
    opts.pipelineSpec = "FullOpt(level=0)";
    const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
    INFO(repr(c));
    REQUIRE(c.binary != std::nullopt);
    const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
    CHECK(source.find(expected) != std::string::npos);
  }
}

TEST_CASE("C source omits values without a representation", "[backend]") {
  polyregion::compiler::initialise();

  const auto sourceOf = [](const Program &p) {
    polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
    opts.pipelineSpec = "FullOpt(level=0)";
    const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
    INFO(repr(c));
    REQUIRE(c.binary != std::nullopt);
    return std::string(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  };

  SECTION("function references") {
    const auto fnTpe = Type::FnRef(Sym({"dead"})).widen();
    const auto boxSym = Sym({"Box"});
    const auto boxTpe = Type::Struct(boxSym, {}).widen();
    const StructDef box(boxSym, {}, {Named("value", Type::IntS32()), Named("fn", fnTpe)}, {}, false);
    const Named fn("fn", fnTpe);
    const Named value("value", boxTpe);
    const auto poison = Expr::Alias(Term::Poison(fnTpe).widen()).widen();
    const Function entry = mkFn("kernel", {}, Type::Unit0(),
                                {Var(fn, poison, false).widen(), Var(value, std::optional<Expr::Any>{}, true).widen(),
                                 Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                                FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    const auto source = sourceOf(Program(entry, {}, {box}, PassPhase::Initial(), {}));
    CHECK(source.find("int value;") != std::string::npos);
    CHECK(source.find("fn;") == std::string::npos);
  }

  SECTION("aggregate poison") {
    const auto boxSym = Sym({"Box"});
    const auto boxTpe = Type::Struct(boxSym, {}).widen();
    const auto arrTpe = Type::Arr(Type::IntS32(), 2, TypeSpace::Global()).widen();
    const StructDef box(boxSym, {}, {Named("value", Type::IntS32())}, {}, false);
    const Named record("record", boxTpe);
    const Named values("values", arrTpe);
    const auto recordPoison = Expr::Alias(Term::Poison(boxTpe).widen()).widen();
    const auto valuesPoison = Expr::Alias(Term::Poison(arrTpe).widen()).widen();
    const Function entry = mkFn("kernel", {}, Type::Unit0(),
                                {Var(record, recordPoison, true).widen(), Mut(selectNamed(record), recordPoison).widen(),
                                 Var(values, valuesPoison, true).widen(), Mut(selectNamed(values), valuesPoison).widen(),
                                 Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                                FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    const auto source = sourceOf(Program(entry, {}, {box}, PassPhase::Initial(), {}));
    CHECK(source.find("Box _v0;") != std::string::npos);
    CHECK(source.find("int _v1[2];") != std::string::npos);
    CHECK(source.find("poison") == std::string::npos);
  }
}

TEST_CASE("C source emits every entry function", "[backend]") {
  polyregion::compiler::initialise();

  const auto body = std::vector<Stmt::Any>{Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()};
  Function first = mkFn("first_kernel", {}, Type::Unit0(), body, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  Function second = mkFn("second_kernel", {}, Type::Unit0(), body, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  Program p(first, {second}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("first_kernel") != std::string::npos);
  CHECK(source.find("second_kernel") != std::string::npos);
}

TEST_CASE("metal source does not emit zero-size empty marker members", "[backend]") {
  polyregion::compiler::initialise();

  const auto emptySym = Sym({"#empty"});
  const auto nestedSym = Sym({"#nested"});
  const auto middleSym = Sym({"#middle"});
  const auto ownerSym = Sym({"Owner"});
  const auto derivedSym = Sym({"Derived"});
  const StructDef empty(emptySym, {}, {}, {}, false);
  const StructDef nested(nestedSym, {}, {Named("inner", Type::Struct(emptySym, {}))}, {}, false);
  const StructDef middle(middleSym, {}, {Named("#base_nested", Type::Struct(nestedSym, {}))}, {Type::Struct(nestedSym, {})}, false);
  const StructDef owner(ownerSym, {},
                        {
                            Named("bytes", Type::Arr(Type::IntS8(), 23, TypeSpace::Global())),
                            Named("pad", Type::Struct(emptySym, {})),
                            Named("nest", Type::Struct(nestedSym, {})),
                            Named("tail", Type::IntU8()),
                        },
                        {}, false);
  const StructDef derived(derivedSym, {}, {Named("#base_middle", Type::Struct(middleSym, {}))},
                          {Type::Struct(middleSym, {}), Type::Struct(nestedSym, {})}, false);
  const Named value("value", Type::Struct(derivedSym, {}));
  const Named base("base", Type::Ptr(Type::Struct(nestedSym, {}), TypeSpace::Private()));
  Function entry(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(),
                 {
                     Var(value, std::optional<Expr::Any>{}, true).widen(),
                     Var(base,
                         Expr::RefTo(Term::Select(value, {PathStep::Field("#base_middle"), PathStep::Field("#base_nested")},
                                                  Type::Struct(nestedSym, {}))
                                         .widen(),
                                     {}, Type::Struct(nestedSym, {}), TypeSpace::Private(), Region::Opaque())
                             .widen(),
                         false)
                         .widen(),
                     Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
                 },
                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {empty, nested, middle, owner, derived}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("_empty pad") == std::string::npos);
  CHECK(source.find("_nested nest") == std::string::npos);
  CHECK(source.find("_v0._base_middle") == std::string::npos);
  CHECK(source.find("&(_v0)") != std::string::npos);
  CHECK(source.find("uint8_t tail;") != std::string::npos);
}

TEST_CASE("metal source canonicalises empty true branches", "[backend]") {
  polyregion::compiler::initialise();

  const Named flag("flag", Type::Bool1());
  const Named value("value", Type::IntS32());
  Function entry(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(),
                 {
                     Var(flag, Expr::Alias(Term::Bool1Const(false).widen()).widen(), true).widen(),
                     Var(value, Expr::Alias(Term::IntS32Const(0).widen()).widen(), true).widen(),
                     Cond(Term::Select(flag, {}, flag.tpe).widen(), {},
                          {Mut(Term::Select(value, {}, value.tpe), Expr::Alias(Term::IntS32Const(1).widen()).widen()).widen()})
                         .widen(),
                     Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
                 },
                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("if (!(_v0))") != std::string::npos);
  CHECK(source.find("if (_v0) {\n\n  } else") == std::string::npos);
}

TEST_CASE("opencl source escapes reserved words as whole identifiers only", "[backend]") {
  polyregion::compiler::initialise();

  const auto vecSym = Sym({"Vecs"});
  const StructDef vecs(vecSym, {}, {Named("long4", Type::IntS32()), Named("kernels", Type::IntS32())}, {}, false);
  Function entry(Sym({"my_kernel_agent"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(),
                 {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()}, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true, FunctionAffinity::Offload());
  Program p(entry, {}, {vecs}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("my_kernel_agent") != std::string::npos);
  CHECK(source.find("my__kernel_agent") == std::string::npos);
  CHECK(source.find("int _long4;") != std::string::npos);
  CHECK(source.find("int kernels;") != std::string::npos);
}

TEST_CASE("integer comparisons follow operand signedness", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const Named ua("ua", Type::IntU32()), ub("ub", Type::IntU32());
  const Named sa("sa", Type::IntS32()), sb("sb", Type::IntS32());
  Function entry = mkFn("kernel", {Arg(ua, {}), Arg(ub, {}), Arg(sa, {}), Arg(sb, {})}, Type::Unit0(),
                        {
                            let("ult") = IntrOp(LogicLt(selectNamed(ua), selectNamed(ub))),
                            let("ule") = IntrOp(LogicLte(selectNamed(ua), selectNamed(ub))),
                            let("ugt") = IntrOp(LogicGt(selectNamed(ua), selectNamed(ub))),
                            let("uge") = IntrOp(LogicGte(selectNamed(ua), selectNamed(ub))),
                            let("slt") = IntrOp(LogicLt(selectNamed(sa), selectNamed(sb))),
                            let("sle") = IntrOp(LogicLte(selectNamed(sa), selectNamed(sb))),
                            let("sgt") = IntrOp(LogicGt(selectNamed(sa), selectNamed(sb))),
                            let("sge") = IntrOp(LogicGte(selectNamed(sa), selectNamed(sb))),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  const auto &ir = llvmIrOf(c);
  for (const auto *predicate : {"icmp ult", "icmp ule", "icmp ugt", "icmp uge", //
                                "icmp slt", "icmp sle", "icmp sgt", "icmp sge"}) {
    INFO(predicate);
    CHECK(ir.find(predicate) != std::string::npos);
  }
}

TEST_CASE("integer division, remainder, min, max and shift follow operand signedness", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const Named ua("ua", Type::IntU32()), ub("ub", Type::IntU32());
  const Named sa("sa", Type::IntS32()), sb("sb", Type::IntS32());
  Function entry = mkFn("kernel", {Arg(ua, {}), Arg(ub, {}), Arg(sa, {}), Arg(sb, {})}, Type::Unit0(),
                        {
                            let("udiv") = IntrOp(Div(selectNamed(ua), selectNamed(ub), Type::IntU32())),
                            let("urem") = IntrOp(Rem(selectNamed(ua), selectNamed(ub), Type::IntU32())),
                            let("umin") = IntrOp(Min(selectNamed(ua), selectNamed(ub), Type::IntU32())),
                            let("umax") = IntrOp(Max(selectNamed(ua), selectNamed(ub), Type::IntU32())),
                            let("ushr") = IntrOp(BSR(selectNamed(ua), selectNamed(ub), Type::IntU32())),
                            let("sdiv") = IntrOp(Div(selectNamed(sa), selectNamed(sb), Type::IntS32())),
                            let("srem") = IntrOp(Rem(selectNamed(sa), selectNamed(sb), Type::IntS32())),
                            let("smin") = IntrOp(Min(selectNamed(sa), selectNamed(sb), Type::IntS32())),
                            let("smax") = IntrOp(Max(selectNamed(sa), selectNamed(sb), Type::IntS32())),
                            let("sshr") = IntrOp(BSR(selectNamed(sa), selectNamed(sb), Type::IntS32())),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  const auto &ir = llvmIrOf(c);
  for (const auto *op : {"udiv i32", "urem i32", "icmp ult", "lshr i32", //
                         "sdiv i32", "srem i32", "icmp slt", "ashr i32"}) {
    INFO(op);
    CHECK(ir.find(op) != std::string::npos);
  }
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

  llvm::LLVMContext context;
  const llvm::StringRef bytes(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  auto buffer = llvm::MemoryBuffer::getMemBuffer(bytes, "mirror.bc", /*RequiresNullTerminator*/ false);
  auto module = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  if (!module) {
    INFO(llvm::toString(module.takeError()));
    FAIL("host mirror output is not parseable LLVM bitcode");
  }

  CHECK((*module)->getFunction("__polyregion_mirror_prelude") != nullptr);
}

TEST_CASE("struct size and member offset agree on the target layout", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto sTpe = Type::Struct(Sym({"S"}), {});
  const StructDef sDef(Sym({"S"}), {}, {Named("a", Type::IntS8()), Named("b", Type::IntS64())}, {}, false);
  Function entry = mkFn("kernel", {}, Type::Unit0(),
                        {
                            let("size") = Expr::SizeOf(sTpe.widen()),
                            let("offset") = Expr::OffsetOf(sTpe.widen(), "b"),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {sDef}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  const auto &ir = llvmIrOf(c);
  CHECK(ir.find("store i64 16") != std::string::npos);
  CHECK(ir.find("store i64 8") != std::string::npos);
}

TEST_CASE("taking the address of a pointer variable yields its slot", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto ptrTpe = Type::Ptr(Type::IntS32(), TypeSpace::Global());
  const Named base("p", ptrTpe);
  const Named ref("pp", Type::Ptr(ptrTpe.widen(), TypeSpace::Global()));
  Function entry =
      mkFn("kernel", {}, Type::Unit0(),
           {
               Var(base, std::optional<Expr::Any>{}, true).widen(),
               Var(ref, Expr::RefTo(selectNamed(base).widen(), {}, ptrTpe.widen(), TypeSpace::Global(), Region::Opaque()).widen(), false)
                   .widen(),
               ret(),
           },
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  CHECK(llvmIrOf(c).find("load ptr") == std::string::npos);
}

TEST_CASE("a constant loop condition lowers to an unconditional branch", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto capTpe = Type::Struct(Sym({"Cap"}), {});
  const Named nField("n", Type::IntS32());
  const StructDef capDef(Sym({"Cap"}), {}, {nField}, {}, false);
  const Named capture("#capture", Type::Ptr(capTpe.widen(), TypeSpace::Global()));
  const Named done("done", Type::Bool1());

  Function entry = mkFn(
      "kernel", {Arg(capture, {})}, Type::Unit0(),
      {
          Stmt::While(Term::Bool1Const(true).widen(),
                      {
                          Var(done, Expr::IntrOp(LogicEq(Select({capture}, nField), Term::IntS32Const(0).widen())).widen(), false).widen(),
                          Stmt::Cond(selectNamed(done).widen(), {Stmt::Break().widen()}, {}).widen(),
                      })
              .widen(),
          ret(),
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {capDef}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV_GLCompute, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  CHECK(llvmIrOf(c).find("br i1 true") == std::string::npos);
}

TEST_CASE("integral to pointer casts lower to inttoptr", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));

  const Program p = arenaOffsetCastProgram();
  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  CHECK(llvmIrOf(c).find("inttoptr") != std::string::npos);
}

TEST_CASE("taking the address of a constant materialises an entry block slot", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const Named ref("p", Type::Ptr(Type::IntU32(), TypeSpace::Private()));
  Function entry = mkFn(
      "kernel", {}, Type::Unit0(),
      {
          Var(ref, Expr::RefTo(Term::IntU32Const(1).widen(), {}, Type::IntU32(), TypeSpace::Private(), Region::Opaque()).widen(), false)
              .widen(),
          ret(),
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");
  REQUIRE(c.binary != std::nullopt);

  const auto &ir = llvmIrOf(c);
  CHECK(ir.find("alloca i32") != std::string::npos);
  CHECK(ir.find("store i32 1") != std::string::npos);
}

TEST_CASE("a narrowing struct-to-struct cast reads the source members, not its address", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const Named a("a", SInt), b("b", SInt), c("c", SInt);
  const Named x("x", SInt), y("y", SInt);
  const Named r("r", SInt);
  const auto srcTpe = Type::Struct(Sym({"Src"}), {});
  const auto dstTpe = Type::Struct(Sym({"Dst"}), {});
  const auto outTpe = Type::Struct(Sym({"Out"}), {});
  const StructDef srcDef(Sym({"Src"}), {}, {a, b, c}, {}, false);
  const StructDef dstDef(Sym({"Dst"}), {}, {x, y}, {}, false);
  const StructDef outDef(Sym({"Out"}), {}, {r}, {}, false);

  const Named capture("#capture", Type::Ptr(outTpe.widen(), TypeSpace::Global()));
  const Named src("src", srcTpe.widen());
  const Named dst("dst", dstTpe.widen());

  // 305 is unique to reading x then y: swapping gives 503, shifting a slot gives 507.
  Function entry = mkFn("kernel", {Arg(capture, {})}, Unit,
                        {
                            Var(src, std::optional<Expr::Any>{}, true).widen(),
                            Mut(Select({src}, a), Expr::Alias(Term::IntS32Const(3).widen()).widen()).widen(),
                            Mut(Select({src}, b), Expr::Alias(Term::IntS32Const(5).widen()).widen()).widen(),
                            Mut(Select({src}, c), Expr::Alias(Term::IntS32Const(7).widen()).widen()).widen(),
                            Var(dst, Expr::Cast(selectNamed(src).widen(), dstTpe.widen()).widen(), false).widen(),
                            let("scaled") = IntrOp(Mul(Select({dst}, x).widen(), 100_(SInt), SInt)),
                            let("mixed") = IntrOp(Add("scaled"_(SInt), Select({dst}, y).widen(), SInt)),
                            Mut(Select({capture}, r), Expr::Alias("mixed"_(SInt)).widen()).widen(),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {srcDef, dstDef, outDef}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto compiled = polyregion::compiler::compile(p, opts, OptLevel::O3);
  INFO(repr(compiled));
  CHECK(compiled.messages == "");
  REQUIRE(compiled.binary != std::nullopt);

  using Catch::Matchers::ContainsSubstring;
  const auto &ir = llvmIrOf(compiled);
  CHECK_THAT(ir, ContainsSubstring("load %Dst, ptr %src_stack_ptr"));
  CHECK_THAT(ir, !ContainsSubstring("store ptr %src_stack_ptr, ptr %dst_stack_ptr"));
  CHECK_THAT(eventDataOf(compiled, "llvm_to_obj_opt"), ContainsSubstring("store i32 305"));
}

static Program absToCaptureProgram(const Type::Any &tpe, const Term::Any &input) {
  using namespace polyregion::polyast::dsl;

  const Named r("r", tpe);
  const StructDef outDef(Sym({"Out"}), {}, {r}, {}, false);
  const auto outTpe = Type::Struct(Sym({"Out"}), {});
  const Named capture("#capture", Type::Ptr(outTpe.widen(), TypeSpace::Global()));
  const Named x("x", tpe);

  Function entry = mkFn("kernel", {Arg(capture, {})}, Unit,
                        {
                            Var(x, Expr::Alias(input).widen(), true).widen(),
                            let("a") = MathOp(Math::Abs(selectNamed(x).widen(), tpe)),
                            Mut(Select({capture}, r), Expr::Alias("a"_(tpe)).widen()).widen(),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  return Program(entry, {}, {outDef}, PassPhase::Initial(), {});
}

TEST_CASE("integral abs emits llvm.abs with its is_int_min_poison operand", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;
  using Catch::Matchers::ContainsSubstring;

  struct Case {
    const char *label;
    Type::Any tpe;
    Term::Any input;
    const char *intrinsic;
    const char *folded;
  };
  const auto cases = std::vector<Case>{
      {"s32", SInt.widen(), Term::IntS32Const(-7).widen(), "@llvm.abs.i32", "store i32 7"},
      {"s64", Long.widen(), Term::IntS64Const(-7).widen(), "@llvm.abs.i64", "store i64 7"},
      {"s32-min", SInt.widen(), Term::IntS32Const(INT32_MIN).widen(), "@llvm.abs.i32", "store i32 -2147483648"},
      {"s64-min", Long.widen(), Term::IntS64Const(INT64_MIN).widen(), "@llvm.abs.i64", "store i64 -9223372036854775808"},
  };

  for (const auto &[label, tpe, input, intrinsic, folded] : cases) {
    INFO(label);
    const Program p = absToCaptureProgram(tpe, input);
    polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
    opts.pipelineSpec = "FullOpt(level=0)";
    auto compiled = polyregion::compiler::compile(p, opts, OptLevel::O3);
    INFO(repr(compiled));
    CHECK(compiled.messages == "");
    REQUIRE(compiled.binary != std::nullopt);

    const auto &ir = llvmIrOf(compiled);
    CHECK_THAT(ir, ContainsSubstring(intrinsic));
    CHECK_THAT(ir, ContainsSubstring(std::string(intrinsic) + "(") && ContainsSubstring("i1 false)"));
    CHECK_THAT(eventDataOf(compiled, "llvm_to_obj_opt"), ContainsSubstring(folded));
  }
}

TEST_CASE("a widening struct-to-struct cast is rejected", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named a("a", SInt);
  const Named x("x", SInt), y("y", SInt);
  const auto srcTpe = Type::Struct(Sym({"Src"}), {});
  const auto dstTpe = Type::Struct(Sym({"Dst"}), {});
  const StructDef srcDef(Sym({"Src"}), {}, {a}, {}, false);
  const StructDef dstDef(Sym({"Dst"}), {}, {x, y}, {}, false);

  const Named src("src", srcTpe.widen());
  const Named dst("dst", dstTpe.widen());

  Function entry = mkFn("kernel", {}, Unit,
                        {
                            Var(src, std::optional<Expr::Any>{}, true).widen(),
                            Mut(Select({src}, a), Expr::Alias(Term::IntS32Const(3).widen()).widen()).widen(),
                            Var(dst, Expr::Cast(selectNamed(src).widen(), dstTpe.widen()).widen(), false).widen(),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {srcDef, dstDef}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_HOST, "native"};
  opts.pipelineSpec = "FullOpt(level=0)";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(p, opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("would read past the source allocation"));
}
