#include <algorithm>
#include <cstring>

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"
#include "fmt/format.h"
#include "spirv/unified1/spirv.hpp"

#include "polyregion/env_keys.h"

#include "ast.h"
#include "backend/llvm.h"
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
  return Function(FunctionDecl(Sym({name}), {}, std::optional<Arg>{}, std::move(args), {}, {}, std::move(rtn), FunctionAffinity::Offload()),
                  std::move(body), std::move(visibility), std::move(fpMode), isEntry);
}

template <typename C> static const std::string &eventDataOf(const C &c, const std::string &name) {
  const auto event = c.events ^ find_cref([&](const auto &e) { return e.name == name; });
  REQUIRE(event);
  return event->get().data;
}

template <typename C> static const std::string &llvmIrOf(const C &c) { return eventDataOf(c, "ast_to_llvm_ir"); }

static polyregion::backend::LLVMBackend::Options llvmHostOptions() {
  using Backend = polyregion::backend::LLVMBackend;
  switch (polyregion::backend::llvmc::defaultHostTriple().getArch()) {
    case llvm::Triple::x86_64: return {Backend::Target::x86_64, "native"};
    case llvm::Triple::aarch64: return {Backend::Target::AArch64, "native"};
    case llvm::Triple::arm: return {Backend::Target::ARM, "native"};
    case llvm::Triple::riscv64: return {Backend::Target::RISCV64, "native"};
    case llvm::Triple::ppc64le: return {Backend::Target::PPC64LE, "native"};
    default: throw std::logic_error("Unsupported host architecture in LLVM backend test");
  }
}

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

TEST_CASE("SPIR-V normalises narrowed integer operands", "[backend][spirv]") {
  const std::vector<uint32_t> words{
      spv::MagicNumber,
      0x00010300u,
      0u,
      20u,
      0u,
      (4u << 16) | spv::OpTypeInt,
      1u,
      64u,
      0u,
      (4u << 16) | spv::OpTypeInt,
      2u,
      32u,
      0u,
      (5u << 16) | spv::OpConstant,
      1u,
      3u,
      7u,
      0u,
      (4u << 16) | spv::OpConstant,
      2u,
      4u,
      255u,
      (5u << 16) | spv::OpBitwiseAnd,
      2u,
      5u,
      3u,
      4u,
  };
  const std::string input(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
  const auto repaired = polyregion::backend::llvmc::normaliseSpirvNarrowIntegerOperands(input);
  REQUIRE(repaired.size() == input.size() + 4 * sizeof(uint32_t));
  const auto *out = reinterpret_cast<const uint32_t *>(repaired.data());
  CHECK(out[3] == 21u);
  const size_t convert = words.size() - 5;
  CHECK(out[convert] == ((4u << 16) | spv::OpUConvert));
  CHECK(out[convert + 1] == 2u);
  CHECK(out[convert + 2] == 20u);
  CHECK(out[convert + 3] == 3u);
  CHECK(out[convert + 4] == ((5u << 16) | spv::OpBitwiseAnd));
  CHECK(out[convert + 7] == 20u);
  CHECK(out[convert + 8] == 4u);

  auto wideningWords = words;
  wideningWords[23] = 1u;
  wideningWords[25] = 4u;
  wideningWords[26] = 3u;
  const std::string widening(reinterpret_cast<const char *>(wideningWords.data()), wideningWords.size() * sizeof(uint32_t));
  CHECK(polyregion::backend::llvmc::normaliseSpirvNarrowIntegerOperands(widening) == widening);

  for (const auto op : {spv::OpBitwiseOr, spv::OpBitwiseXor}) {
    auto variant = words;
    variant[convert] = (5u << 16) | op;
    const std::string blob(reinterpret_cast<const char *>(variant.data()), variant.size() * sizeof(uint32_t));
    const auto normalised = polyregion::backend::llvmc::normaliseSpirvNarrowIntegerOperands(blob);
    REQUIRE(normalised.size() == blob.size() + 4 * sizeof(uint32_t));
    const auto *normalisedWords = reinterpret_cast<const uint32_t *>(normalised.data());
    CHECK(normalisedWords[convert] == ((4u << 16) | spv::OpUConvert));
    CHECK(normalisedWords[convert + 4] == ((5u << 16) | op));
  }

  auto shiftWords = words;
  shiftWords[convert] = (5u << 16) | spv::OpShiftLeftLogical;
  const std::string shiftBlob(reinterpret_cast<const char *>(shiftWords.data()), shiftWords.size() * sizeof(uint32_t));
  const auto normalisedShift = polyregion::backend::llvmc::normaliseSpirvNarrowIntegerOperands(shiftBlob);
  REQUIRE(normalisedShift.size() == shiftBlob.size() + 4 * sizeof(uint32_t));
  const auto *normalisedShiftWords = reinterpret_cast<const uint32_t *>(normalisedShift.data());
  CHECK(normalisedShiftWords[convert] == ((4u << 16) | spv::OpUConvert));
  CHECK(normalisedShiftWords[convert + 4] == ((5u << 16) | spv::OpShiftLeftLogical));
  CHECK(normalisedShiftWords[convert + 7] == 20u);
  CHECK(normalisedShiftWords[convert + 8] == 4u);

  auto extendedWords = words;
  extendedWords.resize(convert);
  extendedWords[3] = 40u;
  const std::vector<uint32_t> vectorAndUnary{
      (4u << 16) | spv::OpTypeVector,
      6u,
      1u,
      2u,
      (4u << 16) | spv::OpTypeVector,
      7u,
      2u,
      2u,
      (3u << 16) | spv::OpUndef,
      6u,
      8u,
      (3u << 16) | spv::OpUndef,
      7u,
      9u,
      (5u << 16) | spv::OpBitwiseAnd,
      7u,
      10u,
      8u,
      9u,
      (4u << 16) | spv::OpNot,
      2u,
      11u,
      3u,
  };
  extendedWords.insert(extendedWords.end(), vectorAndUnary.begin(), vectorAndUnary.end());
  const std::string extendedBlob(reinterpret_cast<const char *>(extendedWords.data()), extendedWords.size() * sizeof(uint32_t));
  const auto normalisedExtended = polyregion::backend::llvmc::normaliseSpirvNarrowIntegerOperands(extendedBlob);
  REQUIRE(normalisedExtended.size() == extendedBlob.size() + 8 * sizeof(uint32_t));
  const auto *normalisedExtendedWords = reinterpret_cast<const uint32_t *>(normalisedExtended.data());
  CHECK(normalisedExtendedWords[extendedWords.size() - 9] == ((4u << 16) | spv::OpUConvert));
  CHECK(normalisedExtendedWords[extendedWords.size() - 8] == 7u);
  CHECK(normalisedExtendedWords[extendedWords.size() - 5] == ((5u << 16) | spv::OpBitwiseAnd));
  CHECK(normalisedExtendedWords[extendedWords.size()] == ((4u << 16) | spv::OpUConvert));
  CHECK(normalisedExtendedWords[extendedWords.size() + 1] == 2u);
  CHECK(normalisedExtendedWords[extendedWords.size() + 4] == ((4u << 16) | spv::OpNot));
}

TEST_CASE("CPU orchestration ABI follows the target pointer width", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const Named context("context", contextType);
  const Named bytes("bytes", Type::IntU64());
  const Named extent("extent", Type::IntU32());
  const Named remote("remote", contextType);
  const Named copied("copied", Type::Unit0());
  const Named launched("launched", Type::Unit0());
  const Named freed("freed", Type::Unit0());
  const auto kernel = Term::Poison(Type::FnRef(Sym({"kernel"}))).widen();
  const Function entry = mkFn(
      "orchestrate", {Arg(context, {}), Arg(bytes, {}), Arg(extent, {})}, Type::Unit0(),
      {Var(remote, Expr::SpecOp(Spec::RemoteAlloc(selectNamed(context).widen(), selectNamed(bytes).widen())).widen(), false).widen(),
       Var(copied,
           Expr::SpecOp(Spec::RemoteMemcpy(selectNamed(context).widen(), selectNamed(remote).widen(), selectNamed(remote).widen(),
                                           selectNamed(bytes).widen(), Direction::RemoteToRemote()))
               .widen(),
           false)
           .widen(),
       Var(launched,
           Expr::SpecOp(Spec::RemoteLaunch(/*context*/ selectNamed(context).widen(),
                                           /*kernel*/ kernel,
                                           /*tpeArgs*/ {},
                                           /*gridX*/ selectNamed(extent).widen(),
                                           /*gridY*/ selectNamed(extent).widen(),
                                           /*gridZ*/ selectNamed(extent).widen(),
                                           /*blockX*/ selectNamed(extent).widen(),
                                           /*blockY*/ selectNamed(extent).widen(),
                                           /*blockZ*/ selectNamed(extent).widen(),
                                           /*shmem*/ selectNamed(extent).widen(),
                                           /*args*/ {}))
               .widen(),
           false)
           .widen(),
       Var(freed, Expr::SpecOp(Spec::RemoteFree(selectNamed(context).widen(), selectNamed(remote).widen())).widen(), false).widen(), ret()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program p(entry, {}, {}, PassPhase::Initial(), {});
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));

  const auto checkAbi = [&](const auto &result, const std::string &sizeType) {
    REQUIRE(result.binary);
    const auto &ir = llvmIrOf(result);
    CHECK_THAT(ir, Catch::Matchers::ContainsSubstring(fmt::format("declare {} @polyrt_remote_malloc(ptr, {})", sizeType, sizeType)));
    CHECK_THAT(ir, Catch::Matchers::ContainsSubstring(
                       fmt::format("declare void @polyrt_remote_memcpy(ptr, {}, {}, {}, i32)", sizeType, sizeType, sizeType)));
    CHECK_THAT(ir, Catch::Matchers::ContainsSubstring(
                       fmt::format("declare void @polyrt_remote_launch(ptr, ptr, ptr, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}", sizeType)));
    CHECK_THAT(ir, Catch::Matchers::ContainsSubstring(fmt::format("declare void @polyrt_remote_free(ptr, {})", sizeType)));
    if (sizeType == "i64") CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("zext i32"));
  };

  checkAbi(polyregion::compiler::compile(p, {Target::Object_LLVM_HOST, "native"}, OptLevel::O0), sizeof(void *) == 4 ? "i32" : "i64");

  auto armTriple = llvm::Triple(llvm::sys::getProcessTriple());
  armTriple.setArch(llvm::Triple::arm);
  std::string targetError;
  if (llvm::TargetRegistry::lookupTarget("", armTriple, targetError)) {
    checkAbi(polyregion::compiler::compile(p, {Target::Object_LLVM_ARM, "cortex-a7"}, OptLevel::O0), "i32");
  } else WARN("ARM target is unavailable in this LLVM distribution: " << targetError);
}

TEST_CASE("CPU Bool1 boundaries use canonical byte values", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named out("out", Type::Ptr(Type::Bool1(), TypeSpace::Global()));
  const Function entry =
      mkFn("write_bool", {Arg(out, {})}, Type::Bool1(),
           {Stmt::Update(selectNamed(out), Term::IntS32Const(0), Term::Bool1Const(true)).widen(), ret(Term::Bool1Const(true))},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled =
      polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), {Target::Object_LLVM_HOST, "native"}, OptLevel::O0);
  REQUIRE(compiled.binary);
  const auto &ir = llvmIrOf(compiled);
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("store i8 1"));
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("ret i8 1"));
  CHECK_THAT(ir, !Catch::Matchers::ContainsSubstring("i8 -1"));
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
  CHECK(source ^ contains_slice("[256]"));
  CHECK(source ^ contains_slice("get_local_id(0)"));
  CHECK(source ^ contains_slice("barrier(CLK_LOCAL_MEM_FENCE)"));
}

TEST_CASE("C source shares one bounded workgroup region per kernel", "[backend]") {
  polyregion::compiler::initialise();

  const auto dynamicBytes = Type::Arr(Type::IntS8(), 0, TypeSpace::Local()).widen();
  const auto dynamicInts = Type::Arr(Type::IntS32(), 0, TypeSpace::Local()).widen();
  const auto staticFloats = Type::Arr(Type::Float32(), 64, TypeSpace::Local()).widen();
  const auto kernel = [&](const std::string &name) {
    const auto scratch = Var(Named("static", staticFloats), Expr::Alias(Term::Poison(staticFloats).widen()).widen(), false).widen();
    return mkFn(name, {}, Type::Unit0(),
                {scratch, scratch, Var(Named("bytes", dynamicBytes), std::optional<Expr::Any>{}, false).widen(),
                 Var(Named("ints", dynamicInts), std::optional<Expr::Any>{}, false).widen(),
                 Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  };
  const Program p(kernel("first"), {kernel("second")}, {}, PassPhase::Initial(), {});

  for (const auto target : {Target::Source_C_OpenCL1_1, Target::Source_C_Metal1_0}) {
    polyregion::compiler::Options opts{target, ""};
    opts.pipelineSpec = "Mirror";
    opts.workgroupMemoryBytes = 1024;
    const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
    INFO(repr(c));
    REQUIRE(c.binary != std::nullopt);
    const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
    const auto occurrences = [&](const std::string &needle) {
      return source | sliding(needle.size(), 1)
             | count([&](const auto &window) { return std::equal(window.begin(), window.end(), needle.begin()); });
    };
    CHECK(occurrences("[64]") == 2);
    CHECK(occurrences("[768]") == 2);
    CHECK(occurrences("_v2 = ((") == 2);
    CHECK_FALSE(source ^ contains_slice("[0]"));
  }
}

TEST_CASE("C source rejects workgroup storage beyond the configured capacity", "[backend]") {
  polyregion::compiler::initialise();

  const auto compile = [](uint32_t elements, uint32_t capacity = 128) {
    const auto local = Type::Arr(Type::IntU8(), elements, TypeSpace::Local()).widen();
    const Function entry = mkFn("kernel", {}, Type::Unit0(),
                                {Var(Named("storage", local), std::optional<Expr::Any>{}, false).widen(),
                                 Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                                FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
    opts.pipelineSpec = "Mirror";
    opts.workgroupMemoryBytes = capacity;
    return polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  };

  CHECK(compile(128).binary != std::nullopt);
  REQUIRE_THROWS_WITH(compile(129), Catch::Matchers::ContainsSubstring("workgroup storage exceeds configured capacity"));
  REQUIRE_THROWS_WITH(compile(0, 0), Catch::Matchers::ContainsSubstring("workgroup storage exceeds configured capacity"));
}

TEST_CASE("C source preserves hoisted workgroup array initialisation", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto fixed = Type::Arr(Type::IntS32(), 2, TypeSpace::Local()).widen();
  const auto dynamic = Type::Arr(Type::IntS32(), 0, TypeSpace::Local()).widen();
  const Named source("source", fixed), copy("copy", fixed), view("view", dynamic);
  const Function entry = mkFn(
      "kernel", {}, Type::Unit0(),
      {Var(source, std::optional<Expr::Any>{}, false).widen(), Var(copy, Expr::Alias(selectNamed(source).widen()).widen(), false).widen(),
       Var(view, std::optional<Expr::Any>{}, false).widen(), Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  opts.workgroupMemoryBytes = 64;

  const auto c = polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string sourceText(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(sourceText ^ contains_slice("local char _v3[48]"));
  CHECK(sourceText ^ contains_slice("local int* _v2 = ((local int*) _v3)"));
  CHECK(sourceText ^ contains_slice("_v1[_ac0] = _v0[_ac0]"));
}

TEST_CASE("C source accounts for workgroup struct storage", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto stateSym = Sym({"State"});
  const auto emptySym = Sym({"Empty"});
  const auto state = Type::Struct(stateSym, {}).widen();
  const auto empty = Type::Struct(emptySym, {}).widen();
  const StructDef stateDef(stateSym, {}, {Named("x", Type::IntS32()), Named("y", Type::IntS32())}, {}, false);
  const StructDef emptyDef(emptySym, {}, {}, {}, false);
  const auto fixedStates = Type::Arr(state, 2, TypeSpace::Local()).widen();
  const auto fixedEmpty = Type::Arr(empty, 1, TypeSpace::Local()).widen();
  const auto dynamic = Type::Arr(Type::IntS8(), 0, TypeSpace::Local()).widen();
  const auto statePtr = Type::Ptr(state, TypeSpace::Local()).widen();
  const Named states("states", fixedStates), empties("empties", fixedEmpty), storage("storage", dynamic), view("view", statePtr);
  const Function entry =
      mkFn("kernel", {}, Type::Unit0(),
           {Var(states, std::optional<Expr::Any>{}, false).widen(), Var(empties, std::optional<Expr::Any>{}, false).widen(),
            Var(storage, std::optional<Expr::Any>{}, false).widen(),
            Var(view, Expr::Cast(selectNamed(storage).widen(), statePtr).widen(), false).widen(),
            Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  opts.workgroupMemoryBytes = 64;

  const auto c = polyregion::compiler::compile(Program(entry, {}, {stateDef, emptyDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("2 * sizeof(State)"));
  CHECK(source ^ contains_slice("1 * sizeof(Empty)"));
  CHECK(source ^ contains_slice("sizeof(State) <= ("));
}

TEST_CASE("C source specialises pointer-bearing structs by address space", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const auto localStorage = Type::Arr(Type::IntS32(), 0, TypeSpace::Local()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const auto privateBoxPtrTpe = Type::Ptr(box, TypeSpace::Private()).widen();
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
      privateBoxPtr("privateBoxPtr", privateBoxPtrTpe), privateBoxPtrCopy("privateBoxPtrCopy", privateBoxPtrTpe),
      scratch("scratch", localStorage), localBox("localBox", box);
  const auto member = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const Function entry =
      mkFn("kernel", {Arg(input, {})}, Type::Unit0(),
           {
               Var(value, std::optional<Expr::Any>{}, true).widen(),
               Var(globalBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
               Mut(member(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
               Var(privateBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
               Mut(member(privateBox),
                   Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque())
                       .widen())
                   .widen(),
               Var(privateBoxPtr,
                   Expr::RefTo(Term::Select(privateBox, {}, box).widen(), {}, box, TypeSpace::Private(), Region::Opaque()).widen(), false)
                   .widen(),
               Var(privateBoxPtrCopy, Expr::Alias(Term::Select(privateBoxPtr, {}, privateBoxPtrTpe).widen()).widen(), false).widen(),
               Var(scratch, std::optional<Expr::Any>{}, true).widen(),
               Var(localBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
               Mut(member(localBox), Expr::RefTo(Term::Select(scratch, {}, localStorage).widen(), Term::IntS32Const(0).widen(),
                                                 Type::IntS32(), TypeSpace::Local(), Region::Opaque())
                                         .widen())
                   .widen(),
               Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
           },
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  opts.workgroupMemoryBytes = 64;
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("typedef struct Box_asp Box_asp;"));
  CHECK(source ^ contains_slice("typedef struct Box_asg Box_asg;"));
  CHECK(source ^ contains_slice("global int* ptr;"));
  CHECK(source ^ contains_slice("private int* ptr;"));
  CHECK(source ^ contains_slice("local int* ptr;"));
  CHECK(source ^ contains_slice("Box_asp _v3;"));
  CHECK(source ^ contains_slice("private Box_asp* _v4"));
  CHECK(source ^ contains_slice("private Box_asp* _v5"));
  CHECK(source ^ contains_slice("Box _v7;"));

  opts.target = Target::Source_C_Metal1_0;
  const auto metal = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(metal));
  REQUIRE(metal.binary != std::nullopt);
  const std::string metalSource(reinterpret_cast<const char *>(metal.binary->data()), metal.binary->size());
  CHECK(metalSource ^ contains_slice("thread int32_t* ptr;"));
  CHECK(metalSource ^ contains_slice("threadgroup int32_t* ptr;"));
  CHECK(metalSource ^ contains_slice("device int32_t* ptr;"));

  opts.target = Target::Source_C_C11;
  const auto c11 = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c11));
  REQUIRE(c11.binary != std::nullopt);
  const std::string c11Source(reinterpret_cast<const char *>(c11.binary->data()), c11.binary->size());
  CHECK_FALSE(c11Source ^ contains_slice("Box_as"));
}

TEST_CASE("C source propagates address-space specialisation through stored structs", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"}), wrapperSym = Sym({"Wrapper"});
  const auto box = Type::Struct(boxSym, {}).widen(), wrapper = Type::Struct(wrapperSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const StructDef wrapperDef(wrapperSym, {}, {Named("#base_Box", box)}, {Type::Struct(boxSym, {})}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
      globalWrapper("globalWrapper", wrapper), privateWrapper("privateWrapper", wrapper), privateWrapperCopy("privateWrapperCopy", wrapper),
      privateWrapperAssigned("privateWrapperAssigned", wrapper);
  const auto ptrMember = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const auto boxMember = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("#base_Box").widen()}, box); };
  const auto poison = [](const Type::Any &tpe) { return Expr::Alias(Term::Poison(tpe).widen()).widen(); };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {
          Var(value, std::optional<Expr::Any>{}, true).widen(),
          Var(globalBox, poison(box), true).widen(),
          Mut(ptrMember(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
          Var(privateBox, poison(box), true).widen(),
          Mut(ptrMember(privateBox),
              Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque())
                  .widen())
              .widen(),
          Var(globalWrapper, poison(wrapper), true).widen(),
          Mut(boxMember(globalWrapper), Expr::Alias(Term::Select(globalBox, {}, box).widen()).widen()).widen(),
          Var(privateWrapper, poison(wrapper), true).widen(),
          Mut(boxMember(privateWrapper), Expr::Alias(Term::Select(privateBox, {}, box).widen()).widen()).widen(),
          Var(privateWrapperCopy, Expr::Alias(Term::Select(privateWrapper, {}, wrapper).widen()).widen(), true).widen(),
          Var(privateWrapperAssigned, poison(wrapper), true).widen(),
          Mut(Term::Select(privateWrapperAssigned, {}, wrapper), Expr::Alias(Term::Select(privateWrapperCopy, {}, wrapper).widen()).widen())
              .widen(),
          Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef, wrapperDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("typedef struct Wrapper_asp Wrapper_asp;"));
  CHECK(source ^ contains_slice("Box_asp _base_Box;"));
  CHECK(source ^ contains_slice("Wrapper_asp _v5;"));
  CHECK(source ^ contains_slice("Wrapper_asp _v6;"));
  CHECK(source ^ contains_slice("Wrapper_asp _v7;"));

  opts.target = Target::Source_C_Metal1_0;
  const auto metal = polyregion::compiler::compile(Program(entry, {}, {boxDef, wrapperDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(metal));
  REQUIRE(metal.binary != std::nullopt);
  const std::string metalSource(reinterpret_cast<const char *>(metal.binary->data()), metal.binary->size());
  CHECK(metalSource ^ contains_slice("Wrapper_asp _v5;"));
  CHECK(metalSource ^ contains_slice("Wrapper_asp _v6 = _v5;"));
  CHECK(metalSource ^ contains_slice("Wrapper_asp _v7;"));
  CHECK(metalSource ^ contains_slice("_v7 = _v6;"));
}

TEST_CASE("C source combines nested struct specialisations deterministically", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"}), wrapperSym = Sym({"Wrapper"});
  const auto box = Type::Struct(boxSym, {}).widen(), wrapper = Type::Struct(wrapperSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const StructDef wrapperDef(wrapperSym, {}, {Named("left", box), Named("right", box)}, {}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
      mixed("mixed", wrapper), reversed("reversed", wrapper);
  const auto ptrMember = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const auto boxMember = [&](const Named &owner, const std::string &name) {
    return Term::Select(owner, {PathStep::Field(name).widen()}, box);
  };
  const auto poison = [](const Type::Any &tpe) { return Expr::Alias(Term::Poison(tpe).widen()).widen(); };
  const auto copy = [&](const Named &destination, const std::string &member, const Named &source) {
    return Mut(boxMember(destination, member), Expr::Alias(Term::Select(source, {}, box).widen()).widen()).widen();
  };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(globalBox, poison(box), true).widen(),
       Mut(ptrMember(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
       Var(privateBox, poison(box), true).widen(),
       Mut(ptrMember(privateBox),
           Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque()).widen())
           .widen(),
       Var(mixed, poison(wrapper), true).widen(), copy(mixed, "left", privateBox), copy(mixed, "right", globalBox),
       Var(reversed, poison(wrapper), true).widen(), copy(reversed, "left", globalBox), copy(reversed, "right", privateBox),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef, wrapperDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  const auto gp = source ^ index_of_slice("typedef struct Wrapper_asgp Wrapper_asgp;");
  const auto pg = source ^ index_of_slice("typedef struct Wrapper_aspg Wrapper_aspg;");
  CHECK(gp >= 0);
  CHECK(pg >= 0);
  CHECK(gp < pg);
  CHECK(source ^ contains_slice("Box_asp left;"));
  CHECK(source ^ contains_slice("Box_asp right;"));
}

TEST_CASE("C source rejects cross-space pointer merges before dereference", "[backend]") {
  polyregion::compiler::initialise();

  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const Named input("input", globalPtr), value("value", Type::IntS32()), merged("merged", globalPtr), result("result", Type::IntS32());
  const auto assign = [&](const Expr::Any &expr) { return Mut(Term::Select(merged, {}, globalPtr), expr).widen(); };
  const Function entry =
      mkFn("kernel", {Arg(input, {})}, Type::Unit0(),
           {
               Var(value, std::optional<Expr::Any>{}, true).widen(),
               Var(merged, std::optional<Expr::Any>{}, true).widen(),
               Stmt::Cond(Term::Bool1Const(true).widen(), {assign(Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen())},
                          {assign(Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(),
                                              Region::Opaque())
                                      .widen())})
                   .widen(),
               Var(result, Expr::Alias(Term::Select(merged, {PathStep::Deref().widen()}, Type::IntS32()).widen()).widen(), false).widen(),
               Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
           },
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("cross-address-space pointer merge"));
}

TEST_CASE("C source rejects cross-space pointer merges that escape", "[backend]") {
  polyregion::compiler::initialise();

  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const Named input("input", globalPtr), value("value", Type::IntS32()), merged("merged", globalPtr);
  const auto assign = [&](const Expr::Any &expr) { return Mut(Term::Select(merged, {}, globalPtr), expr).widen(); };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {
          Var(value, std::optional<Expr::Any>{}, true).widen(),
          Var(merged, std::optional<Expr::Any>{}, true).widen(),
          Stmt::Cond(Term::Bool1Const(true).widen(), {assign(Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen())},
                     {assign(Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(),
                                         Region::Opaque())
                                 .widen())})
              .widen(),
          Mut(Term::Select(merged, {PathStep::Deref().widen()}, Type::IntS32()), Expr::Alias(Term::IntS32Const(1).widen()).widen()).widen(),
          Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("cross-address-space pointer merge escapes read-only use"));
}

TEST_CASE("C source rejects indexed cross-space pointer merges", "[backend]") {
  polyregion::compiler::initialise();

  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const Named input("input", globalPtr), value("value", Type::IntS32()), merged("merged", globalPtr), result("result", Type::IntS32());
  const auto assign = [&](const Expr::Any &expr) { return Mut(Term::Select(merged, {}, globalPtr), expr).widen(); };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(merged, std::optional<Expr::Any>{}, true).widen(),
       Stmt::Cond(
           Term::Bool1Const(true).widen(), {assign(Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen())},
           {assign(Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque())
                       .widen())})
           .widen(),
       Var(result, Expr::Index(Term::Select(merged, {}, globalPtr), Term::IntS32Const(1).widen(), Type::IntS32()).widen(), false).widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("cross-address-space pointer merge escapes read-only use"));
}

TEST_CASE("C source rejects cross-space pointer demotion across mutation", "[backend]") {
  polyregion::compiler::initialise();

  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const Named input("input", globalPtr), value("value", Type::IntS32()), merged("merged", globalPtr), result("result", Type::IntS32());
  const auto assign = [&](const Expr::Any &expr) { return Mut(Term::Select(merged, {}, globalPtr), expr).widen(); };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(merged, std::optional<Expr::Any>{}, true).widen(),
       Stmt::Cond(
           Term::Bool1Const(true).widen(), {assign(Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen())},
           {assign(Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque())
                       .widen())})
           .widen(),
       Mut(Term::Select(value, {}, Type::IntS32()), Expr::Alias(Term::IntS32Const(7).widen()).widen()).widen(),
       Var(result, Expr::Alias(Term::Select(merged, {PathStep::Deref().widen()}, Type::IntS32()).widen()).widen(), false).widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("cross-address-space pointer merge escapes read-only use"));
}

TEST_CASE("C source traces reads from address-space-specialised fields", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
      loaded("loaded", globalPtr);
  const auto member = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(globalBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
       Mut(member(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
       Var(privateBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
       Mut(member(privateBox),
           Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque()).widen())
           .widen(),
       Var(loaded, Expr::Alias(member(privateBox).widen()).widen(), false).widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("private int* _v4 = _v3.ptr;"));
}

TEST_CASE("C source distinguishes struct specialisations by complete field signature", "[backend]") {
  polyregion::compiler::initialise();

  const auto pairSym = Sym({"Pair"});
  const auto pair = Type::Struct(pairSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef pairDef(pairSym, {}, {Named("a", globalPtr), Named("b", globalPtr)}, {}, false);
  const Named input("input", globalPtr), x("x", Type::IntS32()), y("y", Type::IntS32()), base("base", pair), left("left", pair),
      right("right", pair);
  const auto member = [&](const Named &owner, const std::string &name) {
    return Term::Select(owner, {PathStep::Field(name).widen()}, globalPtr);
  };
  const auto privateRef = [&](const Named &value) {
    return Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque()).widen();
  };
  const auto poison = Expr::Alias(Term::Poison(pair).widen()).widen();
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(x, std::optional<Expr::Any>{}, true).widen(), Var(y, std::optional<Expr::Any>{}, true).widen(), Var(base, poison, true).widen(),
       Mut(member(base, "a"), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
       Mut(member(base, "b"), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(), Var(left, poison, true).widen(),
       Mut(member(left, "a"), privateRef(x)).widen(), Var(right, poison, true).widen(), Mut(member(right, "b"), privateRef(y)).widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {pairDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("typedef struct Pair_aspg Pair_aspg;"));
  CHECK(source ^ contains_slice("typedef struct Pair_asgp Pair_asgp;"));
  CHECK(source ^ contains_slice("Pair_aspg _v4;"));
  CHECK(source ^ contains_slice("Pair_asgp _v5;"));
}

TEST_CASE("C source traces pointer fields through fixed array indices", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto boxesTpe = Type::Arr(box, 1, TypeSpace::Private()).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const Named value("value", Type::IntS32()), boxes("boxes", boxesTpe);
  const Term::Select member(boxes, {PathStep::Index(0).widen(), PathStep::Field("ptr").widen()}, globalPtr);
  const Function entry = mkFn(
      "kernel", {}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(boxes, std::optional<Expr::Any>{}, true).widen(),
       Mut(member,
           Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque()).widen())
           .widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("private int* ptr;"));
}

TEST_CASE("C source rejects conflicting pointer fields through fixed array indices", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto boxesTpe = Type::Arr(box, 1, TypeSpace::Private()).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), boxes("boxes", boxesTpe);
  const Term::Select direct(globalBox, {PathStep::Field("ptr").widen()}, globalPtr);
  const Term::Select indexed(boxes, {PathStep::Index(0).widen(), PathStep::Field("ptr").widen()}, globalPtr);
  const Function entry = mkFn(
      "kernel", {Arg(input, {})}, Type::Unit0(),
      {Var(value, std::optional<Expr::Any>{}, true).widen(), Var(globalBox, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(),
       Mut(direct, Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
       Var(boxes, std::optional<Expr::Any>{}, true).widen(),
       Mut(indexed,
           Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {}, Type::IntS32(), TypeSpace::Private(), Region::Opaque()).widen())
           .widen(),
       Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("cannot specialise indirect conflicting pointer field"));
}

TEST_CASE("C source keeps constant and global struct pointer fields distinct", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS8(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const Named input("input", globalPtr), globalBox("globalBox", box), constantBox("constantBox", box);
  const auto member = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const auto poison = Expr::Alias(Term::Poison(box).widen()).widen();
  const Function entry =
      mkFn("kernel", {Arg(input, {})}, Type::Unit0(),
           {Var(globalBox, poison, true).widen(),
            Mut(member(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
            Var(constantBox, poison, true).widen(), Mut(member(constantBox), Expr::Alias(Term::StringConst("x").widen()).widen()).widen(),
            Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("typedef struct Box_asc Box_asc;"));
  CHECK(source ^ contains_slice("constant char* ptr;"));
}

TEST_CASE("C source isolates struct specialisations between overloaded functions", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS8(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const auto worker = Sym({"worker"});
  const auto makeWorker = [&](const Type::Any &argTpe, const Expr::Any &value) {
    const Named arg("arg", argTpe), storage("storage", Type::IntS8()), slot("slot", box);
    const Term::Select member(slot, {PathStep::Field("ptr").widen()}, globalPtr);
    return mkFn(repr(worker), {Arg(arg, {})}, globalPtr,
                {Var(storage, std::optional<Expr::Any>{}, true).widen(),
                 Var(slot, Expr::Alias(Term::Poison(box).widen()).widen(), true).widen(), Mut(member, value).widen(),
                 Return(value).widen()});
  };
  const auto constantWorker = makeWorker(Type::IntS32(), Expr::Alias(Term::StringConst("x").widen()).widen());
  const Named privateStorage("storage", Type::IntS8());
  const auto privateWorker = makeWorker(Type::IntS64(), Expr::RefTo(Term::Select(privateStorage, {}, Type::IntS8()).widen(), {},
                                                                    Type::IntS8(), TypeSpace::Private(), Region::Opaque())
                                                            .widen());
  const Function entry = mkFn("kernel", {}, Type::Unit0(), {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {constantWorker, privateWorker}, {boxDef}, PassPhase::Initial(), {}), opts,
                                               OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("Box_asc _v2;"));
  CHECK(source ^ contains_slice("Box_asp _v2;"));
}

TEST_CASE("MSL lowers 32-bit atomic read-modify-write operations", "[backend][metal][atomic]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto compile = [&](const AtomicOp::Any &op, const MemOrder::Any &order, const Type::Any &tpe, const TypeSpace::Any &space) {
    const auto ptrTpe = Type::Ptr(tpe, space).widen();
    const Named ptr("ptr", ptrTpe), result("result", tpe);
    const auto value = tpe.template is<Type::IntU32>() ? Term::IntU32Const(1).widen() : Term::IntS32Const(1).widen();
    const Function entry = mkFn(
        "kernel", {Arg(ptr, {})}, Type::Unit0(),
        {Var(result, Expr::SpecOp(Spec::GpuAtomicRMW(op, selectNamed(ptr), value, MemScope::Device(), order, tpe)).widen(), false).widen(),
         Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
    opts.pipelineSpec = "Mirror";
    return polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  };

  const std::vector<std::pair<AtomicOp::Any, std::string>> operations{
      {AtomicOp::Xchg(), "atomic_exchange_explicit"}, {AtomicOp::Add(), "atomic_fetch_add_explicit"},
      {AtomicOp::Sub(), "atomic_fetch_sub_explicit"}, {AtomicOp::And(), "atomic_fetch_and_explicit"},
      {AtomicOp::Or(), "atomic_fetch_or_explicit"},   {AtomicOp::Xor(), "atomic_fetch_xor_explicit"},
      {AtomicOp::Min(), "atomic_fetch_min_explicit"}, {AtomicOp::Max(), "atomic_fetch_max_explicit"},
  };
  for (const auto &[op, name] : operations) {
    const auto c = compile(op, MemOrder::Relaxed(), Type::IntS32(), TypeSpace::Global());
    INFO(repr(c));
    REQUIRE(c.binary);
    const std::string source(c.binary->begin(), c.binary->end());
    CHECK(source ^ contains_slice(name + "((device atomic_int*)"));
  }

  const auto local = compile(AtomicOp::Add(), MemOrder::Relaxed(), Type::IntU32(), TypeSpace::Local());
  INFO(repr(local));
  REQUIRE(local.binary);
  const std::string source(local.binary->begin(), local.binary->end());
  CHECK(source ^ contains_slice("atomic_fetch_add_explicit((threadgroup atomic_uint*)"));
  CHECK(source ^ contains_slice("metal::memory_order_relaxed"));
  for (const auto &order : std::vector<MemOrder::Any>{MemOrder::Acquire(), MemOrder::Release(), MemOrder::AcqRel(), MemOrder::SeqCst()})
    REQUIRE_THROWS_WITH(compile(AtomicOp::Add(), order, Type::IntU32(), TypeSpace::Local()),
                        Catch::Matchers::ContainsSubstring("MSL atomic RMW supports only relaxed ordering"));
}

TEST_CASE("MSL rejects unsupported atomic types", "[backend][metal][atomic]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto compile = [&](const Type::Any &tpe, const Term::Any &value, const TypeSpace::Any &space = TypeSpace::Global().widen()) {
    const auto ptrTpe = Type::Ptr(tpe, space).widen();
    const Named ptr("ptr", ptrTpe), result("result", tpe);
    const Function entry =
        mkFn("kernel", {Arg(ptr, {})}, Type::Unit0(),
             {Var(result,
                  Expr::SpecOp(Spec::GpuAtomicRMW(AtomicOp::Add(), selectNamed(ptr), value, MemScope::Device(), MemOrder::Relaxed(), tpe))
                      .widen(),
                  false)
                  .widen(),
              Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
             FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
    opts.pipelineSpec = "Mirror";
    return polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  };

  REQUIRE_THROWS_WITH(compile(Type::IntS64(), Term::IntS64Const(1).widen()),
                      Catch::Matchers::ContainsSubstring("MSL supports only 32-bit integer atomic RMW"));
  REQUIRE_THROWS_WITH(compile(Type::Float32(), Term::Float32Const(1).widen()),
                      Catch::Matchers::ContainsSubstring("MSL supports only 32-bit integer atomic RMW"));
  REQUIRE_THROWS_WITH(compile(Type::IntS32(), Term::IntS32Const(1).widen(), TypeSpace::Private()),
                      Catch::Matchers::ContainsSubstring("MSL atomic RMW requires device or threadgroup storage"));
}

TEST_CASE("MSL lowers volatile aggregate access memberwise", "[backend][metal][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym innerName({"Inner"}), outerName({"Outer"});
  const auto inner = Type::Struct(innerName, {}).widen();
  const auto outer = Type::Struct(outerName, {}).widen();
  const StructDef innerDef(innerName, {}, {Named("x", Type::IntS32())}, {}, false);
  const StructDef outerDef(outerName, {}, {Named("inner", inner), Named("values", Type::Arr(Type::IntU16(), 2, TypeSpace::Private()))}, {},
                           false);
  const auto makeFnFor = [&](const std::string &name, const TypeSpace::Any &space) {
    const auto ptrTpe = Type::Ptr(outer, space).widen();
    const Named ptr("ptr", ptrTpe), loaded("loaded", outer), loadedAgain("loadedAgain", outer), stored("stored", Type::Unit0()),
        storedAgain("storedAgain", Type::Unit0());
    return mkFn(name, {Arg(ptr, {})}, Type::Unit0(),
                {Var(loaded, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), outer)).widen(), false).widen(),
                 Var(stored, Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), selectNamed(loaded))).widen(), false).widen(),
                 Var(loadedAgain, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), outer)).widen(), false).widen(),
                 Var(storedAgain, Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), selectNamed(loadedAgain))).widen(), false).widen(),
                 Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()});
  };
  const Function entry = mkFn("kernel", {}, Type::Unit0(), {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const auto global = makeFnFor("global_access", TypeSpace::Global());
  const auto local = makeFnFor("local_access", TypeSpace::Local());
  const auto priv = makeFnFor("private_access", TypeSpace::Private());
  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {global, local, priv}, {innerDef, outerDef}, PassPhase::Initial(), {}), opts,
                                               OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary);
  const std::string source(c.binary->begin(), c.binary->end());
  for (const auto &space : {"device", "threadgroup", "thread"}) {
    CHECK(source ^ contains_slice("_pr_vld_" + std::string(space) + "_Outer"));
    CHECK(source ^ contains_slice("_pr_vst_" + std::string(space) + "_Outer"));
  }
  CHECK(source ^ contains_slice("r.inner.x = (*p).inner.x;"));
  CHECK(source ^ contains_slice("for (int _vc1 = 0; _vc1 < 2; _vc1++)"));
  CHECK(source ^ contains_slice("(*p).values[_vc1] = v.values[_vc1];"));
  const auto loadNeedle = std::string("Outer _pr_vld_device_Outer(");
  const auto storeNeedle = std::string("void _pr_vst_device_Outer(");
  const auto loadDefinition = source ^ index_of_slice(loadNeedle);
  const auto storeDefinition = source ^ index_of_slice(storeNeedle);
  REQUIRE(loadDefinition >= 0);
  REQUIRE(storeDefinition >= 0);
  CHECK_FALSE((source | drop(loadDefinition + 1) | contains_slice(loadNeedle)));
  CHECK_FALSE((source | drop(storeDefinition + 1) | contains_slice(storeNeedle)));
  const auto again = polyregion::compiler::compile(Program(entry, {global, local, priv}, {innerDef, outerDef}, PassPhase::Initial(), {}),
                                                   opts, OptLevel::O0);
  REQUIRE(again.binary);
  CHECK(source == std::string(again.binary->begin(), again.binary->end()));
}

TEST_CASE("LLVM GPU targets lower volatile access", "[backend][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto ptrTpe = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const Named ptr("ptr", ptrTpe), loaded("loaded", Type::IntS32()), stored("stored", Type::Unit0());
  const Function entry =
      mkFn("kernel", {Arg(ptr, {})}, Type::Unit0(),
           {Var(loaded, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), Type::IntS32())).widen(), false).widen(),
            Var(stored, Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), selectNamed(loaded))).widen(), false).widen(),
            Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});

  for (const auto &[target, arch] : std::vector<std::pair<Target, std::string>>{
           {Target::Object_LLVM_NVPTX64, "sm_35"},
           {Target::Object_LLVM_AMDGCN, "gfx906"},
           {Target::Object_LLVM_SPIRV64_Kernel, ""},
       }) {
    INFO(arch);
    auto compiled = polyregion::compiler::compile(program, {target, arch}, OptLevel::O0);
    CHECK(compiled.messages == "");
    CHECK(compiled.binary != std::nullopt);
  }
}

TEST_CASE("LLVM aggregate volatile access keeps the pointee alignment", "[backend][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym pairName({"Pair"});
  const auto pairType = Type::Struct(pairName, {}).widen();
  const StructDef pair(pairName, {}, {Named("x", Type::IntS32()), Named("y", Type::IntS32())}, {}, false);
  const Named ptr("ptr", Type::Ptr(pairType, TypeSpace::Global())), loaded("loaded", pairType), stored("stored", Type::Unit0());
  const Function entry =
      mkFn("kernel", {Arg(ptr, {})}, Type::Unit0(),
           {Var(loaded, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), pairType)).widen(), false).widen(),
            Var(stored, Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), selectNamed(loaded))).widen(), false).widen(), ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  for (const auto &[target, arch] :
       std::vector<std::pair<Target, std::string>>{{Target::Object_LLVM_NVPTX64, "sm_60"}, {Target::Object_LLVM_SPIRV64_Kernel, ""}}) {
    INFO(arch);
    const auto compiled = polyregion::compiler::compile(Program(entry, {}, {pair}, PassPhase::Initial(), {}), {target, arch}, OptLevel::O0);
    REQUIRE(compiled.binary);
    CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("load volatile i64, ptr"));
    CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("store volatile i64"));
    CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring(", align 4"));
  }
}

TEST_CASE("Vulkan preserves typed aggregate volatile access", "[backend][volatile][vulkan]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym pairName({"Pair"});
  const auto pairType = Type::Struct(pairName, {}).widen();
  const StructDef pair(pairName, {}, {Named("x", Type::IntS32()), Named("y", Type::IntS32())}, {}, false);
  const Named value("value", pairType), ptr("ptr", Type::Ptr(pairType, TypeSpace::Private())), loaded("loaded", pairType),
      stored("stored", Type::Unit0());
  const Function entry =
      mkFn("kernel", {}, Type::Unit0(),
           {Var(value, std::optional<Expr::Any>{}, true).widen(),
            Var(ptr, Expr::RefTo(selectNamed(value).widen(), {}, pairType, TypeSpace::Private(), Region::Opaque()).widen(), false).widen(),
            Var(loaded, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), pairType)).widen(), false).widen(),
            Var(stored, Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), selectNamed(loaded))).widen(), false).widen(), ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {pair}, PassPhase::Initial(), {}),
                                                      {Target::Object_LLVM_SPIRV_GLCompute, ""}, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("load volatile %Pair"));
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("store volatile %Pair"));
  CHECK_THAT(llvmIrOf(compiled), !Catch::Matchers::ContainsSubstring("load volatile i64"));
  CHECK_THAT(llvmIrOf(compiled), !Catch::Matchers::ContainsSubstring("store volatile i64"));
}

TEST_CASE("Vulkan uses byte-backed workgroup booleans", "[backend][subgroup][vulkan]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto scratchTpe = Type::Arr(Type::Bool1(), 32, TypeSpace::Local());
  const Named scratch("scratch", scratchTpe), loaded("loaded", Type::Bool1());
  const Function entry =
      mkFn("kernel", {}, Type::Unit0(),
           {Var(scratch, std::optional<Expr::Any>{}, true).widen(),
            Update(selectNamed(scratch), Term::IntU32Const(0).widen(), Term::Bool1Const(true).widen()).widen(),
            Var(loaded, Expr::Index(selectNamed(scratch), Term::IntU32Const(0).widen(), Type::Bool1()).widen(), false).widen(), ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}),
                                                      {Target::Object_LLVM_SPIRV_GLCompute, ""}, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("@scratch_wg = internal addrspace(3) global [32 x i8]"));
  CHECK_THAT(llvmIrOf(compiled), !Catch::Matchers::ContainsSubstring("alloca [32 x i8]"));
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("load i8"));
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("store i8"));
}

TEST_CASE("Vulkan zeroes Boolean struct fields through their byte storage", "[backend][vulkan]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym flagName({"Flag"});
  const auto flagTpe = Type::Struct(flagName, {}).widen();
  const StructDef flag(flagName, {}, {Named("set", Type::Bool1())}, {}, false);
  const Named value("value", flagTpe);
  const Function entry = mkFn("kernel", {}, Type::Unit0(), {Var(value, std::optional<Expr::Any>{}, true).widen(), ret()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);

  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV_GLCompute, ""};
  opts.pipelineSpec = "StructuredExit";
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {flag}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("store i8 0"));
  CHECK_THAT(llvmIrOf(compiled), !Catch::Matchers::ContainsSubstring("store %Flag zeroinitializer"));
}

TEST_CASE("Vulkan retains multidimensional workgroup-array strides", "[backend][subgroup][vulkan]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto rowTpe = Type::Arr(Type::IntU32(), 8, TypeSpace::Local());
  const auto scratchTpe = Type::Arr(rowTpe, 4, TypeSpace::Local());
  const Named scratch("scratch", scratchTpe), loaded("loaded", Type::IntU32());
  const Term::Select row(scratch, {PathStep::IndexDyn(Term::IntU32Const(2).widen()).widen()}, rowTpe);
  const Function entry = mkFn("kernel", {}, Type::Unit0(),
                              {Var(scratch, std::optional<Expr::Any>{}, true).widen(),
                               Update(row, Term::IntU32Const(3).widen(), Term::IntU32Const(42).widen()).widen(),
                               Var(loaded, Expr::Index(row, Term::IntU32Const(3).widen(), Type::IntU32()).widen(), false).widen(), ret()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}),
                                                      {Target::Object_LLVM_SPIRV_GLCompute, ""}, OptLevel::O3);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("@scratch_wg = internal addrspace(3) global [32 x i32]"));
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("i64 19"));
}

TEST_CASE("Vulkan retains multidimensional private-array strides", "[backend][vulkan]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto rowTpe = Type::Arr(Type::IntU32(), 8, TypeSpace::Private());
  const auto scratchTpe = Type::Arr(rowTpe, 4, TypeSpace::Private());
  const Named scratch("scratch", scratchTpe), loaded("loaded", Type::IntU32());
  const Term::Select row(scratch, {PathStep::IndexDyn(Term::IntU32Const(2).widen()).widen()}, rowTpe);
  const Function entry = mkFn("kernel", {}, Type::Unit0(),
                              {Var(scratch, std::optional<Expr::Any>{}, true).widen(),
                               Update(row, Term::IntU32Const(3).widen(), Term::IntU32Const(42).widen()).widen(),
                               Var(loaded, Expr::Index(row, Term::IntU32Const(3).widen(), Type::IntU32()).widen(), false).widen(), ret()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}),
                                                      {Target::Object_LLVM_SPIRV_GLCompute, ""}, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("alloca [32 x i32]"));
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("i64 19"));
}

TEST_CASE("LLVM CUDA and HIP targets lower subgroup votes", "[backend][subgroup]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto mask = Term::IntU32Const(-1).widen();
  const auto predicate = Term::Bool1Const(true).widen();
  const Function entry = mkFn("kernel", {}, Type::Unit0(),
                              {Var(Named("ballot", Type::IntU32()), Expr::SpecOp(Spec::GpuBallot(mask, predicate)).widen(), false).widen(),
                               Var(Named("any", Type::Bool1()), Expr::SpecOp(Spec::GpuVoteAny(mask, predicate)).widen(), false).widen(),
                               Var(Named("all", Type::Bool1()), Expr::SpecOp(Spec::GpuVoteAll(mask, predicate)).widen(), false).widen(),
                               Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});

  for (const auto &[target, arch] : std::vector<std::pair<Target, std::string>>{
           {Target::Object_LLVM_NVPTX64, "sm_35"},
           {Target::Object_LLVM_AMDGCN, "gfx906"},
       }) {
    INFO(arch);
    auto compiled = polyregion::compiler::compile(program, {target, arch}, OptLevel::O0);
    CHECK(compiled.messages == "");
    CHECK(compiled.binary != std::nullopt);
  }
}

TEST_CASE("NVPTX subgroup barriers support legacy and synchronised warps", "[backend][subgroup]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Function entry = mkFn(
      "kernel", {}, Type::Unit0(),
      {Var(Named("barrier", Type::Unit0()), Expr::SpecOp(Spec::GpuSubgroupBarrier(Term::IntU32Const(-1).widen())).widen(), false).widen(),
       ret()},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));

  const auto legacy = polyregion::compiler::compile(program, {Target::Object_LLVM_NVPTX64, "sm_35"}, OptLevel::O0);
  REQUIRE(legacy.binary);
  CHECK_THAT(llvmIrOf(legacy), Catch::Matchers::ContainsSubstring("llvm.nvvm.membar.cta"));
  CHECK_THAT(llvmIrOf(legacy), !Catch::Matchers::ContainsSubstring("llvm.nvvm.bar.warp.sync"));

  const auto synchronised = polyregion::compiler::compile(program, {Target::Object_LLVM_NVPTX64, "sm_70"}, OptLevel::O0);
  REQUIRE(synchronised.binary);
  CHECK_THAT(llvmIrOf(synchronised), Catch::Matchers::ContainsSubstring("llvm.nvvm.bar.warp.sync"));
}

TEST_CASE("LLVM GPU shuffles pad narrow values without widening the result", "[backend][subgroup]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named value("value", Type::IntU8()), shuffled("shuffled", Type::IntU8());
  const Function entry =
      mkFn("kernel", {Arg(value, {})}, Type::Unit0(),
           {Var(shuffled,
                Expr::SpecOp(Spec::GpuShuffleDown(selectNamed(value).widen(), Term::IntU32Const(1).widen(), Term::IntU32Const(31).widen(),
                                                  Term::IntU32Const(-1).widen(), Type::IntU8()))
                    .widen(),
                false)
                .widen(),
            ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});

  for (const auto &[target, arch] : std::vector<std::pair<Target, std::string>>{
           {Target::Object_LLVM_NVPTX64, "sm_60"},
           {Target::Object_LLVM_AMDGCN, "gfx906"},
           {Target::Object_LLVM_SPIRV64_Kernel, ""},
       }) {
    INFO(arch);
    const auto compiled = polyregion::compiler::compile(program, {target, arch}, OptLevel::O0);
    CHECK(compiled.messages == "");
    CHECK(compiled.binary != std::nullopt);
    if (target == Target::Object_LLVM_SPIRV64_Kernel && compiled.binary) {
      REQUIRE(compiled.binary->size() >= 2 * sizeof(uint32_t));
      uint32_t spirvVersion = 0;
      std::memcpy(&spirvVersion, compiled.binary->data() + sizeof(uint32_t), sizeof(uint32_t));
      CHECK(spirvVersion == 0x00010300u);
    }
  }
}

TEST_CASE("LLVM GPU shuffles honour lane masks and physical subgroup bounds", "[backend][subgroup]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named value("value", Type::IntU32()), mask("mask", Type::IntU32()), shuffled("shuffled", Type::IntU32()),
      shuffledUp("shuffledUp", Type::IntU32());
  const Function entry =
      mkFn("kernel", {Arg(value, {}), Arg(mask, {})}, Type::Unit0(),
           {Var(shuffled,
                Expr::SpecOp(Spec::GpuShuffleIdx(selectNamed(value).widen(), Term::IntU32Const(0).widen(), Term::IntU32Const(7).widen(),
                                                 selectNamed(mask).widen(), Type::IntU32()))
                    .widen(),
                false)
                .widen(),
            Var(shuffledUp,
                Expr::SpecOp(Spec::GpuShuffleUp(selectNamed(value).widen(), Term::IntU32Const(1).widen(), Term::IntU32Const(7).widen(),
                                                selectNamed(mask).widen(), Type::IntU32()))
                    .widen(),
                false)
                .widen(),
            ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));

  const auto amd = polyregion::compiler::compile(program, {Target::Object_LLVM_AMDGCN, "gfx906"}, OptLevel::O0);
  REQUIRE(amd.binary);
  CHECK_THAT(llvmIrOf(amd), Catch::Matchers::ContainsSubstring("icmp ult i32"));
  CHECK_THAT(llvmIrOf(amd), Catch::Matchers::ContainsSubstring("and i32"));
  CHECK_THAT(llvmIrOf(amd), Catch::Matchers::ContainsSubstring("llvm.amdgcn.wavefrontsize"));

  const auto spirv = polyregion::compiler::compile(program, {Target::Object_LLVM_SPIRV64_Kernel, ""}, OptLevel::O0);
  REQUIRE(spirv.binary);
  CHECK_THAT(llvmIrOf(spirv), Catch::Matchers::ContainsSubstring("get_sub_group_size"));
  CHECK_THAT(llvmIrOf(spirv), Catch::Matchers::ContainsSubstring("icmp ult i32"));

  const auto nvLegacy = polyregion::compiler::compile(program, {Target::Object_LLVM_NVPTX64, "sm_60"}, OptLevel::O0);
  REQUIRE(nvLegacy.binary);
  CHECK_THAT(llvmIrOf(nvLegacy), Catch::Matchers::ContainsSubstring("vote.ballot.b32"));
  CHECK_THAT(llvmIrOf(nvLegacy), !Catch::Matchers::ContainsSubstring("vote.sync.ballot.b32"));
  CHECK_THAT(llvmIrOf(nvLegacy), !Catch::Matchers::ContainsSubstring("activemask.b32"));
  CHECK_THAT(llvmIrOf(nvLegacy), Catch::Matchers::ContainsSubstring("llvm.nvvm.shfl.idx.i32"));
  CHECK_THAT(llvmIrOf(nvLegacy), Catch::Matchers::ContainsSubstring("llvm.nvvm.shfl.up.i32"));
  CHECK_THAT(llvmIrOf(nvLegacy), Catch::Matchers::ContainsSubstring("i32 6151"));
  CHECK_THAT(llvmIrOf(nvLegacy), Catch::Matchers::ContainsSubstring("i32 6144"));

  const auto nvSync = polyregion::compiler::compile(program, {Target::Object_LLVM_NVPTX64, "sm_70"}, OptLevel::O0);
  REQUIRE(nvSync.binary);
  CHECK_THAT(llvmIrOf(nvSync), Catch::Matchers::ContainsSubstring("llvm.nvvm.shfl.sync.idx.i32"));
  CHECK_THAT(llvmIrOf(nvSync), Catch::Matchers::ContainsSubstring("llvm.nvvm.shfl.sync.up.i32"));
  CHECK_THAT(llvmIrOf(nvSync), Catch::Matchers::ContainsSubstring("activemask.b32"));
  CHECK_THAT(llvmIrOf(nvSync), Catch::Matchers::ContainsSubstring("i32 6151"));
  CHECK_THAT(llvmIrOf(nvSync), Catch::Matchers::ContainsSubstring("i32 6144"));
}

TEST_CASE("AMDGPU fences remain non-blocking memory fences", "[backend][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Function entry = mkFn("kernel", {}, Type::Unit0(),
                              {Var(Named("global", Type::Unit0()), Expr::SpecOp(Spec::GpuFenceGlobal()).widen(), false).widen(),
                               Var(Named("local", Type::Unit0()), Expr::SpecOp(Spec::GpuFenceLocal()).widen(), false).widen(),
                               Var(Named("all", Type::Unit0()), Expr::SpecOp(Spec::GpuFenceAll()).widen(), false).widen(), ret()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled =
      polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), {Target::Object_LLVM_AMDGCN, "gfx906"}, OptLevel::O0);
  REQUIRE(compiled.binary);
  const auto &ir = llvmIrOf(compiled);
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("fence syncscope(\"agent\") seq_cst"));
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("fence syncscope(\"workgroup\") seq_cst"));
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("fence seq_cst"));
  CHECK_THAT(ir, !Catch::Matchers::ContainsSubstring("llvm.amdgcn.s.barrier"));
}

TEST_CASE("AMDGPU isolates vendor reuse unions and clamps the affected optimisation", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym discontinuityName({"rocprim_block_discontinuity"});
  const auto discontinuityType = Type::Struct(discontinuityName, {}).widen();
  const StructDef discontinuity(discontinuityName, {}, {Named("tail", Type::IntU32())}, {}, false);
  const Sym reuseName({"rocprim_partition_kernel_impl_storage"});
  const auto reuseType = Type::Struct(reuseName, {}).widen();
  const StructDef reuse(reuseName, {}, {Named("exchange", Type::IntU32()), Named("flags", discontinuityType)}, {}, true);
  const auto localStorage = Type::Arr(reuseType, 1, TypeSpace::Local()).widen();
  const Named storage("storage", localStorage), temporary("temporary", Type::IntS32());
  const Function entry = mkFn("kernel", {}, Type::Unit0(),
                              {Var(storage, std::optional<Expr::Any>{}, true).widen(),
                               Var(temporary, Expr::Alias(Term::IntS32Const(1).widen()).widen(), false).widen(), ret()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  polyregion::backend::LLVMBackend backend({.target = polyregion::backend::LLVMBackend::Target::AMDGCN, .arch = "gfx906"});
  const auto compiled = backend.compileProgram(Program(entry, {}, {discontinuity, reuse}, PassPhase::Initial(), {}), OptLevel::O3);
  REQUIRE(compiled.binary);
  const auto layout = compiled.layouts ^ find_cref([&](const auto &x) { return x.name == repr(reuseName); });
  REQUIRE(layout);
  REQUIRE(layout->get().members.size() == 2);
  CHECK(layout->get().members[0].offsetInBytes == 0);
  CHECK(layout->get().members[1].offsetInBytes > 0);
  CHECK_THAT(eventDataOf(compiled, "llvm_to_obj_opt"), Catch::Matchers::ContainsSubstring("alloca i32"));
}

TEST_CASE("LLVM GPU kernels accept stateless callable parameters", "[backend][callable]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named callable("callable", Type::FnRef(Sym({"predicate"})));
  const Function entry =
      mkFn("kernel", {Arg(callable, {})}, Type::Unit0(), {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});

  for (const auto &[target, arch] : std::vector<std::pair<Target, std::string>>{
           {Target::Object_LLVM_NVPTX64, "sm_60"},
           {Target::Object_LLVM_AMDGCN, "gfx906"},
       }) {
    INFO(arch);
    const auto compiled = polyregion::compiler::compile(program, {target, arch}, OptLevel::O0);
    CHECK(compiled.messages == "");
    CHECK(compiled.binary != std::nullopt);
  }
}

TEST_CASE("LLVM forwards stateless callable parameters", "[backend][callable]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto callableType = Type::FnRef(Sym({"predicate"})).widen();
  const Named callable("callable", callableType);
  const Function sink = mkFn("sink", {Arg(callable, {})}, Type::Unit0(), {ret()}, FunctionVisibility::Internal());
  const Function forward = mkFn("forward", {Arg(callable, {})}, Type::Unit0(),
                                {ret(Expr::Invoke(Type::FnRef(sink.decl.name), {}, {}, {selectNamed(callable)}, Type::Unit0()).widen())},
                                FunctionVisibility::Internal());
  const Function entry =
      mkFn("entry", {}, Type::Unit0(),
           {ret(Expr::Invoke(Type::FnRef(forward.decl.name), {}, {}, {Term::Poison(callableType)}, Type::Unit0()).widen())},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {forward, sink}, {}, PassPhase::Initial(), {});

  const auto compiled = polyregion::compiler::compile(program, {Target::Object_LLVM_HOST, "native"}, OptLevel::O0);
  CHECK(compiled.messages == "");
  CHECK(compiled.binary != std::nullopt);
}

TEST_CASE("LLVM evaluates Unit0 return expressions", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Function entry = mkFn("entry", {}, Type::Unit0(), {ret(Expr::ForeignCall("unit_effect", {}, Type::Unit0()).widen())},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled =
      polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), {Target::Object_LLVM_HOST, "native"}, OptLevel::O0);
  REQUIRE(compiled.binary);
  CHECK_THAT(llvmIrOf(compiled), Catch::Matchers::ContainsSubstring("call void @unit_effect"));
}

TEST_CASE("LLVM internal functions do not shadow foreign symbols", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named bytes("bytes", Type::IntU64());
  const auto ptrType = Type::Ptr(Type::Unit0(), TypeSpace::Global()).widen();
  const Function wrapper = mkFn("malloc", {Arg(bytes, {})}, ptrType,
                                {ret(Expr::ForeignCall("malloc", {selectNamed(bytes)}, ptrType).widen())}, FunctionVisibility::Internal());
  const Function entry = mkFn("entry", {}, Type::Unit0(), {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);

  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  polyregion::backend::details::CodeGen cg(llvmHostOptions(), "foreign_collision");
  const auto [error, ir] = cg.transform(Program(entry, {wrapper}, {}, PassPhase::Initial(), {}), {});
  REQUIRE_FALSE(error);
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("define internal ptr @polyregion_internal_malloc"));
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("call ptr @malloc"));
}

TEST_CASE("LLVM gives stored callable references addressable bytes", "[backend][callable]") {
  polyregion::compiler::initialise();
  const auto callableType = Type::FnRef(Sym({"predicate"})).widen();
  const auto boxType = Type::Struct(Sym({"CallableBox"}), {}).widen();
  const StructDef box(Sym({"CallableBox"}), {}, {Named("callable", callableType)}, {}, false);
  const StructDef iterator(Sym({"Iterator"}), {}, {Named("input", Type::Ptr(Type::IntU8(), TypeSpace::Global())), Named("op", boxType)}, {},
                           false);
  const auto layouts = polyregion::backend::LLVMBackend(llvmHostOptions()).resolveLayouts({box, iterator});
  const auto boxLayout = layouts ^ find_cref([](const auto &x) { return x.name == "CallableBox"; });
  const auto iteratorLayout = layouts ^ find_cref([](const auto &x) { return x.name == "Iterator"; });
  REQUIRE(boxLayout);
  REQUIRE(iteratorLayout);
  CHECK(boxLayout->get().sizeInBytes == 1);
  CHECK(boxLayout->get().members[0].sizeInBytes == 1);
  REQUIRE(iteratorLayout->get().members.size() == 2);
  const auto pointerBytes = iteratorLayout->get().members[0].sizeInBytes;
  CHECK(pointerBytes == sizeof(void *));
  CHECK(iteratorLayout->get().sizeInBytes == pointerBytes * 2);
  CHECK(iteratorLayout->get().members[1].offsetInBytes == pointerBytes);
}

TEST_CASE("LLVM stores stateless callable fields at their storage width", "[backend][callable]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto callableType = Type::FnRef(Sym({"predicate"})).widen();
  const auto boxSym = Sym({"CallableBox"});
  const auto boxType = Type::Struct(boxSym, {}).widen();
  const StructDef box(boxSym, {}, {Named("callable", callableType)}, {}, false);
  const Named count("count", Type::IntS32()), value("value", boxType);
  const auto field = Term::Select(value, {PathStep::Field("callable").widen()}, callableType);
  const Function entry = mkFn("entry", {Arg(count, {})}, Type::IntS32(),
                              {Var(value, std::optional<Expr::Any>{}, true).widen(),
                               Mut(field, Expr::Alias(Term::Poison(callableType).widen()).widen()).widen(), ret(selectNamed(count))},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  const auto compiled = polyregion::compiler::compile(Program(entry, {}, {box}, PassPhase::Initial(), {}),
                                                      {Target::Object_LLVM_HOST, "native"}, OptLevel::O0);
  REQUIRE(compiled.binary);
  const auto &ir = llvmIrOf(compiled);
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("store i8 poison"));
  CHECK_THAT(ir, !Catch::Matchers::ContainsSubstring("store ptr null"));
}

TEST_CASE("LLVM stops emitting a function body after terminal branches", "[backend][control-flow]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Function entry = mkFn(
      "entry", {}, Type::IntS32(),
      {Stmt::Cond(Term::Bool1Const(true), {ret(Term::IntS32Const(1))}, {ret(Term::IntS32Const(2))}).widen(), ret(Term::IntS32Const(0))},
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});

  const auto compiled = polyregion::compiler::compile(program, {Target::Object_LLVM_HOST, "native"}, OptLevel::O0);
  CHECK(compiled.messages == "");
  CHECK(compiled.binary != std::nullopt);
}

TEST_CASE("volatile access respects C source address spaces", "[backend][metal][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym recordName({"Record"});
  const auto record = Type::Struct(recordName, {}).widen();
  const StructDef recordDef(recordName, {}, {Named("value", Type::IntS32())}, {}, false);
  const auto makeProgram = [&](const TypeSpace::Any &space, const bool store, const Type::Any &tpe = Type::IntS32().widen()) {
    const auto ptrTpe = Type::Ptr(tpe, space).widen();
    const Named ptr("ptr", ptrTpe), result("result", store ? Type::Unit0().widen() : tpe);
    const auto value = tpe.template is<Type::Struct>() ? Term::Poison(tpe).widen() : Term::IntS32Const(1).widen();
    const auto op = store ? Expr::SpecOp(Spec::GpuVolatileStore(selectNamed(ptr), value)).widen()
                          : Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), tpe)).widen();
    const Function entry = mkFn("kernel", {Arg(ptr, {})}, Type::Unit0(),
                                {Var(result, op, false).widen(), Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                                FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
    return Program(entry, {}, {recordDef}, PassPhase::Initial(), {});
  };
  const auto emit = [&](const Target target, const Program &program) {
    polyregion::compiler::Options opts{target, ""};
    opts.pipelineSpec = "Mirror";
    const auto c = polyregion::compiler::compile(program, opts, OptLevel::O0);
    REQUIRE(c.binary);
    return std::string(c.binary->begin(), c.binary->end());
  };

  const auto scalar = emit(Target::Source_C_Metal1_0, makeProgram(TypeSpace::Global(), false));
  CHECK(scalar ^ contains_slice("volatile device int32_t*"));
  CHECK_FALSE(scalar ^ contains_slice("_pr_vld_"));

  const auto constant = emit(Target::Source_C_Metal1_0, makeProgram(TypeSpace::Constant(), false, record));
  CHECK(constant ^ contains_slice("_pr_vld_constant_Record"));
  REQUIRE_THROWS_WITH(emit(Target::Source_C_Metal1_0, makeProgram(TypeSpace::Constant(), true, record)),
                      Catch::Matchers::ContainsSubstring("volatile store to constant storage is unsupported for MSL"));

  const auto opencl = emit(Target::Source_C_OpenCL1_1, makeProgram(TypeSpace::Global(), false, record));
  CHECK(opencl ^ contains_slice("volatile global Record*"));
  CHECK_FALSE(opencl ^ contains_slice("_pr_vld_"));
  const auto c11 = emit(Target::Source_C_C11, makeProgram(TypeSpace::Global(), false, record));
  CHECK(c11 ^ contains_slice("volatile Record*"));
  CHECK_FALSE(c11 ^ contains_slice("_pr_vld_"));
}

TEST_CASE("MSL rejects volatile union access", "[backend][metal][volatile]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Sym unionName({"Either"});
  const auto either = Type::Struct(unionName, {}).widen();
  const StructDef unionDef(unionName, {}, {Named("i", Type::IntS32()), Named("f", Type::Float32())}, {}, true);
  const auto ptrTpe = Type::Ptr(either, TypeSpace::Global()).widen();
  const Named ptr("ptr", ptrTpe), loaded("loaded", either);
  const Function entry = mkFn("kernel", {Arg(ptr, {})}, Type::Unit0(),
                              {Var(loaded, Expr::SpecOp(Spec::GpuVolatileLoad(selectNamed(ptr), either)).widen(), false).widen(),
                               Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()},
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "Mirror";
  REQUIRE_THROWS_WITH(polyregion::compiler::compile(Program(entry, {}, {unionDef}, PassPhase::Initial(), {}), opts, OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("volatile access to union Either is unsupported for MSL"));
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
  Function prelude(FunctionDecl(Sym({"__polyregion_mirror_prelude"}), {}, std::optional<Arg>{}, {Arg(capture, {}), Arg(size, {})}, {}, {},
                                Type::IntU64(), FunctionAffinity::Host()),
                   {
                       Var(remote, Expr::ForeignCall("polyrt_sma_alloc", allocArgs, Type::IntU64()).widen(), false).widen(),
                       Return(Expr::Alias(selectNamed(remote).widen()).widen()).widen(),
                   },
                   FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);

  Program p(prelude, {}, {}, PassPhase::Initial(), {});
  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, {Target::Object_LLVM_HOST, "native"}, OptLevel::O3);
  INFO(repr(c));
  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);
}

TEST_CASE("HostThreaded topology uses the task id and launch size", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const Named globalIdx("global_idx", Type::IntU32());
  const Named helperIdx("helper_idx", Type::IntU32());
  const Named globalSize("global_size", Type::IntU32());
  const auto dimension = Term::IntU32Const(0).widen();
  const Function helper = mkFn("index_helper", {}, Type::IntU32(),
                               {
                                   Var(globalIdx, Expr::SpecOp(Spec::GpuGlobalIdx(dimension)).widen(), false).widen(),
                                   Return(Expr::Alias(selectNamed(globalIdx).widen()).widen()).widen(),
                               },
                               FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), false);
  const Function entry = mkFn(
      "kernel", {}, Type::IntU32(),
      {
          Var(helperIdx, Expr::Invoke(Type::FnRef(helper.decl.name), {}, std::optional<Term::Any>{}, {}, Type::IntU32()).widen(), false)
              .widen(),
          Var(globalSize, Expr::SpecOp(Spec::GpuGlobalSize(dimension)).widen(), false).widen(),
          Return(Expr::IntrOp(Add(selectNamed(helperIdx).widen(), selectNamed(globalSize).widen(), Type::IntU32())).widen()).widen(),
      },
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  polyregion::backend::details::CodeGen codegen(llvmHostOptions(), "host_thread_topology");
  const auto [error, ir] = codegen.transform(Program(entry, {helper}, {}, PassPhase::Initial(), {}), {});
  REQUIRE_FALSE(error);
  CHECK(ir ^ contains_slice("@__polyregion_host_thread_global_idx"));
  CHECK(ir ^ contains_slice("@__polyregion_host_thread_global_size"));
  CHECK(ir ^ contains_slice("define internal i32 @polyregion_internal_index_helper()"));
  CHECK(ir ^ contains_slice("define i32 @kernel(i64"));
}

TEST_CASE("host orchestration lowers remote launches through the context ABI", "[backend]") {
  polyregion::compiler::initialise();
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const Named context("context", contextType);
  const Sym payloadName({"Payload"});
  const Type::Struct payloadType(payloadName, {});
  const StructDef payload(payloadName, {}, {Named("value", Type::IntS32())}, {}, false);
  const Named remote("remote", Type::Ptr(payloadType, TypeSpace::Global()));
  const Named closure("closure", payloadType);
  const auto kernel = mkFn("remote.kernel", {Arg(Named("remote", remote.tpe), {}), Arg(Named("closure", closure.tpe), {})}, Type::Unit0(),
                           {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const auto one = Term::IntU32Const(1).widen();
  const auto zero = Term::IntU32Const(0).widen();
  const auto launch = Spec::RemoteLaunch(/*context*/ selectNamed(context).widen(),
                                         /*kernel*/ Term::Poison(Type::FnRef(kernel.decl.name)).widen(),
                                         /*tpeArgs*/ {},
                                         /*gridX*/ one,
                                         /*gridY*/ one,
                                         /*gridZ*/ one,
                                         /*blockX*/ one,
                                         /*blockY*/ one,
                                         /*blockZ*/ one,
                                         /*shmem*/ zero,
                                         /*args*/ {selectNamed(remote).widen(), selectNamed(closure).widen()});
  const Named launched("launched", Type::Unit0());
  const Function driver(FunctionDecl(Sym({"driver"}), {}, {}, {Arg(context, {}), Arg(remote, {}), Arg(closure, {})}, {}, {}, Type::Unit0(),
                                     FunctionAffinity::Host()),
                        {Var(launched, Expr::SpecOp(launch).widen(), false).widen(), ret()}, FunctionVisibility::Exported(),
                        FunctionFpMode::Relaxed(), false);
  const Program program(driver, {kernel}, {payload}, PassPhase::Initial(), {});
  polyregion::compiler::Options options{Target::Object_LLVM_HOST, "native"};
  options.pipelineSpec = "FullOpt(level=0)";

  const auto compiled = polyregion::compiler::compile(program, options, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  const auto &ir = llvmIrOf(compiled);
  INFO(ir);
  CHECK(ir ^ contains_slice("polyrt_remote_launch"));
  const auto callCount = [&](const std::string &callee) {
    const auto needle = "@" + callee + "(";
    const auto lines = ir ^ split('\n');
    return lines ^ count([&](const auto &line) { return (line ^ contains_slice("call ")) && (line ^ contains_slice(needle)); });
  };
  // The by-value aggregate needs one temporary remote allocation; the pointer argument is forwarded unchanged.
  CHECK(callCount("polyrt_remote_malloc") == 1);
  CHECK(callCount("polyrt_remote_memcpy") == 1);
  CHECK(ir ^ contains_slice("i32 0)"));
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
  Function entry(
      FunctionDecl(Sym({"kernel"}), {}, std::optional<Arg>{}, {Arg(capture, {})}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()), body,
      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
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
  Function entry(FunctionDecl(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()), body,
                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV64_Kernel, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  CHECK(c.messages == "");

  CHECK(llvmIrOf(c) ^ contains_slice("llvm.memcpy"));
}

TEST_CASE("SPIR-V dynamic workgroup views do not initialise shared storage", "[backend][spirv]") {
  polyregion::compiler::initialise();
  const ScopedEnv captureIr(polyregion::env::PolyregionDebug, std::string("1"));
  using namespace polyregion::polyast::dsl;

  const auto dynamic = Type::Arr(Type::IntS8(), 0, TypeSpace::Local());
  const auto fixed = Type::Arr(Type::IntS8(), 16, TypeSpace::Local());
  const Named first("first", dynamic), second("second", dynamic), reserved("reserved", fixed);
  Function entry = mkFn("kernel", {}, Type::Unit0(),
                        {
                            Var(reserved, Expr::Alias(Term::Poison(fixed).widen()).widen(), true).widen(),
                            Var(first, Expr::Alias(Term::Poison(dynamic).widen()).widen(), true).widen(),
                            Var(second, Expr::Alias(Term::Poison(dynamic).widen()).widen(), true).widen(),
                            Update(selectNamed(first), Term::IntS32Const(0).widen(), Term::IntS8Const(7).widen()).widen(),
                            ret(),
                        },
                        FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Object_LLVM_SPIRV64_Kernel, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  opts.workgroupMemoryBytes = 64;
  const auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary);
  const auto &ir = llvmIrOf(c);
  CHECK(ir ^ contains_slice("@polyc_dyn_shared"));
  CHECK(ir ^ contains_slice("[48 x i8]"));
  CHECK(ir ^ contains_slice("store i8 7"));
  CHECK_FALSE(ir ^ contains_slice("store i8 poison"));
  REQUIRE(c.binary->size() >= 2 * sizeof(uint32_t));
  uint32_t spirvVersion = 0;
  std::memcpy(&spirvVersion, c.binary->data() + sizeof(uint32_t), sizeof(uint32_t));
  CHECK(spirvVersion == 0x00010200u);
}

TEST_CASE("LLVM GPU targets reject fixed workgroup storage beyond configured capacity", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto oversized = Type::Arr(Type::IntU8(), 129, TypeSpace::Local()).widen();
  const Function entry =
      mkFn("kernel", {}, Type::Unit0(), {Var(Named("storage", oversized), std::optional<Expr::Any>{}, false).widen(), ret()},
           FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const Program program(entry, {}, {}, PassPhase::Initial(), {});
  for (const auto &[target, arch] : std::vector<std::pair<Target, std::string>>{
           {Target::Object_LLVM_NVPTX64, "sm_60"},
           {Target::Object_LLVM_AMDGCN, "gfx906"},
           {Target::Object_LLVM_SPIRV64_Kernel, ""},
       }) {
    INFO(arch);
    polyregion::compiler::Options options{target, arch};
    options.pipelineSpec = "FullOpt(level=0)";
    options.workgroupMemoryBytes = 128;
    REQUIRE_THROWS_WITH(polyregion::compiler::compile(program, options, OptLevel::O0),
                        Catch::Matchers::ContainsSubstring("workgroup storage exceeds configured capacity of 128 bytes"));
  }
}

TEST_CASE("LLVM workgroup accounting includes reachable helper storage", "[backend][spirv]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto helperCall = [](const Function &helper) {
    return Expr::Invoke(Type::FnRef(helper.decl.name), {}, std::optional<Term::Any>{}, {}, Type::Unit0()).widen();
  };
  const auto local = [](const std::string &name, uint32_t bytes) {
    return Var(Named(name, Type::Arr(Type::IntU8(), bytes, TypeSpace::Local())), std::optional<Expr::Any>{}, false).widen();
  };

  const Function tooLargeHelper =
      mkFn("too_large_helper", {}, Type::Unit0(), {local("helper_storage", 80), ret()}, FunctionVisibility::Internal());
  const Function tooLargeEntry = mkFn("too_large_kernel", {}, Type::Unit0(), {local("entry_storage", 80), ret(helperCall(tooLargeHelper))},
                                      FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::backend::LLVMBackend constrained(
      {.target = polyregion::backend::LLVMBackend::Target::SPIRV64_Kernel, .arch = "", .workgroupMemoryBytes = 128});
  REQUIRE_THROWS_WITH(constrained.compileProgram(Program(tooLargeEntry, {tooLargeHelper}, {}, PassPhase::Initial(), {}), OptLevel::O0),
                      Catch::Matchers::ContainsSubstring("workgroup storage exceeds configured capacity of 128 bytes"));

  const Function dynamicHelper = mkFn("dynamic_helper", {}, Type::Unit0(), {local("dynamic", 0), ret()}, FunctionVisibility::Internal());
  const Function boundedEntry = mkFn("bounded_kernel", {}, Type::Unit0(), {local("reserved", 48), ret(helperCall(dynamicHelper))},
                                     FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  const ScopedEnv debug(polyregion::env::PolyregionDebug, std::string("1"));
  polyregion::backend::details::CodeGen codegen(
      {.target = polyregion::backend::LLVMBackend::Target::SPIRV64_Kernel, .arch = "", .workgroupMemoryBytes = 64}, "workgroup_accounting");
  const auto [error, ir] = codegen.transform(Program(boundedEntry, {dynamicHelper}, {}, PassPhase::Initial(), {}), {});
  REQUIRE_FALSE(error);
  CHECK_THAT(ir, Catch::Matchers::ContainsSubstring("[16 x i8]"));

  const std::string overloadName = "overloaded_helper";
  const Function dynamicOverload = mkFn(overloadName, {Arg(Named("value", Type::IntS32()), {})}, Type::Unit0(), {local("view", 0), ret()},
                                        FunctionVisibility::Internal());
  const Function unusedFixedOverload = mkFn(overloadName, {Arg(Named("value", Type::Float32()), {})}, Type::Unit0(),
                                            {local("unused_storage", 64), ret()}, FunctionVisibility::Internal());
  const auto invokeDynamic =
      Expr::Invoke(Type::FnRef(Sym({overloadName})), {}, std::optional<Term::Any>{}, {Term::IntS32Const(1).widen()}, Type::Unit0()).widen();
  const Function overloadEntry =
      mkFn("overload_kernel", {}, Type::Unit0(), {ret(invokeDynamic)}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::backend::details::CodeGen overloadCodegen(
      {.target = polyregion::backend::LLVMBackend::Target::SPIRV64_Kernel, .arch = "", .workgroupMemoryBytes = 128}, "overload_accounting");
  const auto [overloadError, overloadIr] =
      overloadCodegen.transform(Program(overloadEntry, {dynamicOverload, unusedFixedOverload}, {}, PassPhase::Initial(), {}), {});
  REQUIRE_FALSE(overloadError);
  CHECK_THAT(overloadIr, Catch::Matchers::ContainsSubstring("[128 x i8]"));
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
  CHECK(source ^ contains_slice("global char* _v1 = ((global char*) _v0);"));
  CHECK_FALSE(source ^ contains_slice("private char* _v1"));
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
    CHECK(source ^ contains_slice(expected));
  }
}

TEST_CASE("C source preserves erased callable member storage", "[backend][callable]") {
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
    CHECK(source ^ contains_slice("int value;"));
    CHECK(source ^ contains_slice("uchar fn;"));
    // A standalone function reference is still a compile-time-only value and has no C declaration.
    CHECK_FALSE(source ^ contains_slice("dead _v"));
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
    CHECK(source ^ contains_slice("Box _v0;"));
    CHECK(source ^ contains_slice("int _v1[2];"));
    CHECK_FALSE(source ^ contains_slice("poison"));
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
  CHECK(source ^ contains_slice("first_kernel"));
  CHECK(source ^ contains_slice("second_kernel"));
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
  Function entry(FunctionDecl(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
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
                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {empty, nested, middle, owner, derived}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK_FALSE(source ^ contains_slice("_empty pad"));
  CHECK_FALSE(source ^ contains_slice("_nested nest"));
  CHECK_FALSE(source ^ contains_slice("_v0._base_middle"));
  CHECK(source ^ contains_slice("&(_v0)"));
  CHECK(source ^ contains_slice("uint8_t tail;"));
}

TEST_CASE("metal source canonicalises empty true branches", "[backend]") {
  polyregion::compiler::initialise();

  const Named flag("flag", Type::Bool1());
  const Named value("value", Type::IntS32());
  Function entry(FunctionDecl(Sym({"kernel"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
                 {
                     Var(flag, Expr::Alias(Term::Bool1Const(false).widen()).widen(), true).widen(),
                     Var(value, Expr::Alias(Term::IntS32Const(0).widen()).widen(), true).widen(),
                     Cond(Term::Select(flag, {}, flag.tpe).widen(), {},
                          {Mut(Term::Select(value, {}, value.tpe), Expr::Alias(Term::IntS32Const(1).widen()).widen()).widen()})
                         .widen(),
                     Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
                 },
                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_Metal1_0, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("if (!(_v0))"));
  CHECK_FALSE(source ^ contains_slice("if (_v0) {\n\n  } else"));
}

TEST_CASE("opencl source escapes reserved words as whole identifiers only", "[backend]") {
  polyregion::compiler::initialise();

  const auto vecSym = Sym({"Vecs"});
  const StructDef vecs(vecSym, {}, {Named("long4", Type::IntS32()), Named("kernels", Type::IntS32())}, {}, false);
  Function entry(FunctionDecl(Sym({"my_kernel_agent"}), {}, std::optional<Arg>{}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
                 {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()}, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true);
  Program p(entry, {}, {vecs}, PassPhase::Initial(), {});

  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "FullOpt(level=0)";
  auto c = polyregion::compiler::compile(p, opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("my_kernel_agent"));
  CHECK_FALSE(source ^ contains_slice("my__kernel_agent"));
  CHECK(source ^ contains_slice("int _long4;"));
  CHECK(source ^ contains_slice("int kernels;"));
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
    CHECK(ir ^ contains_slice(predicate));
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
    CHECK(ir ^ contains_slice(op));
  }
}

TEST_CASE("host-mirroring compile emits bitcode for the generated prelude", "[backend]") {
  polyregion::compiler::initialise();
  using namespace polyregion::polyast::dsl;

  const auto bytePtr = Type::Ptr(Type::IntS8(), TypeSpace::Global());
  Function entry(FunctionDecl(Sym({"_main"}), {}, std::optional<Arg>{}, {Arg(Named("capture", bytePtr), {})}, {}, {}, Type::Unit0(),
                              FunctionAffinity::Offload()),
                 {Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen()}, FunctionVisibility::Exported(),
                 FunctionFpMode::Relaxed(), /*isEntry*/ true);
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
  CHECK(ir ^ contains_slice("store i64 16"));
  CHECK(ir ^ contains_slice("store i64 8"));
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

  CHECK_FALSE(llvmIrOf(c) ^ contains_slice("load ptr"));
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

  CHECK_FALSE(llvmIrOf(c) ^ contains_slice("br i1 true"));
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

  CHECK(llvmIrOf(c) ^ contains_slice("inttoptr"));
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
  CHECK(ir ^ contains_slice("alloca i32"));
  CHECK(ir ^ contains_slice("store i32 1"));
}

TEST_CASE("OpenCL source takes the address of a constant through a compound literal", "[backend]") {
  polyregion::compiler::initialise();
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
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source ^ contains_slice("private uint* _v0 = &((private uint){1})"));
  CHECK_FALSE(source ^ contains_slice("&(1 /*uint*/)"));
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
  const auto marker = std::string(" = alloca %Src");
  const auto markerPos = ir ^ index_of_slice(marker);
  REQUIRE(markerPos >= 0);
  const auto namePos = ir | take(markerPos) | last_index_of('%');
  REQUIRE(namePos >= 0);
  const auto srcAlloca = ir ^ slice(namePos, markerPos);
  CHECK_THAT(ir, ContainsSubstring("load %Dst, ptr " + srcAlloca));
  CHECK_THAT(ir, !ContainsSubstring("store ptr " + srcAlloca + ", ptr "));
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
