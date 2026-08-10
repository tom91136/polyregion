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
    CHECK(source.find("[0]") == std::string::npos);
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
  CHECK(sourceText.find("local char _v3[48]") != std::string::npos);
  CHECK(sourceText.find("local int* _v2 = ((local int*) _v3)") != std::string::npos);
  CHECK(sourceText.find("_v1[_ac0] = _v0[_ac0]") != std::string::npos);
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
  CHECK(source.find("2 * sizeof(State)") != std::string::npos);
  CHECK(source.find("1 * sizeof(Empty)") != std::string::npos);
  CHECK(source.find("sizeof(State) <= (") != std::string::npos);
}

TEST_CASE("C source specialises pointer-bearing structs by address space", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"});
  const auto box = Type::Struct(boxSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const auto localStorage = Type::Arr(Type::IntS32(), 0, TypeSpace::Local()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
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
  CHECK(source.find("typedef struct Box_asp Box_asp;") != std::string::npos);
  CHECK(source.find("typedef struct Box_asg Box_asg;") != std::string::npos);
  CHECK(source.find("global int* ptr;") != std::string::npos);
  CHECK(source.find("private int* ptr;") != std::string::npos);
  CHECK(source.find("local int* ptr;") != std::string::npos);
  CHECK(source.find("Box_asp _v3;") != std::string::npos);
  CHECK(source.find("Box _v5;") != std::string::npos);

  opts.target = Target::Source_C_Metal1_0;
  const auto metal = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(metal));
  REQUIRE(metal.binary != std::nullopt);
  const std::string metalSource(reinterpret_cast<const char *>(metal.binary->data()), metal.binary->size());
  CHECK(metalSource.find("thread int32_t* ptr;") != std::string::npos);
  CHECK(metalSource.find("threadgroup int32_t* ptr;") != std::string::npos);
  CHECK(metalSource.find("device int32_t* ptr;") != std::string::npos);

  opts.target = Target::Source_C_C11;
  const auto c11 = polyregion::compiler::compile(Program(entry, {}, {boxDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c11));
  REQUIRE(c11.binary != std::nullopt);
  const std::string c11Source(reinterpret_cast<const char *>(c11.binary->data()), c11.binary->size());
  CHECK(c11Source.find("Box_as") == std::string::npos);
}

TEST_CASE("C source propagates address-space specialisation through stored structs", "[backend]") {
  polyregion::compiler::initialise();

  const auto boxSym = Sym({"Box"}), wrapperSym = Sym({"Wrapper"});
  const auto box = Type::Struct(boxSym, {}).widen(), wrapper = Type::Struct(wrapperSym, {}).widen();
  const auto globalPtr = Type::Ptr(Type::IntS32(), TypeSpace::Global()).widen();
  const StructDef boxDef(boxSym, {}, {Named("ptr", globalPtr)}, {}, false);
  const StructDef wrapperDef(wrapperSym, {}, {Named("#base_Box", box)}, {Type::Struct(boxSym, {})}, false);
  const Named input("input", globalPtr), value("value", Type::IntS32()), globalBox("globalBox", box), privateBox("privateBox", box),
      globalWrapper("globalWrapper", wrapper), privateWrapper("privateWrapper", wrapper);
  const auto ptrMember = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("ptr").widen()}, globalPtr); };
  const auto boxMember = [&](const Named &owner) { return Term::Select(owner, {PathStep::Field("#base_Box").widen()}, box); };
  const auto poison = [](const Type::Any &tpe) { return Expr::Alias(Term::Poison(tpe).widen()).widen(); };
  const Function entry = mkFn("kernel", {Arg(input, {})}, Type::Unit0(),
                              {
                                  Var(value, std::optional<Expr::Any>{}, true).widen(),
                                  Var(globalBox, poison(box), true).widen(),
                                  Mut(ptrMember(globalBox), Expr::Alias(Term::Select(input, {}, globalPtr).widen()).widen()).widen(),
                                  Var(privateBox, poison(box), true).widen(),
                                  Mut(ptrMember(privateBox), Expr::RefTo(Term::Select(value, {}, Type::IntS32()).widen(), {},
                                                                         Type::IntS32(), TypeSpace::Private(), Region::Opaque())
                                                                 .widen())
                                      .widen(),
                                  Var(globalWrapper, poison(wrapper), true).widen(),
                                  Mut(boxMember(globalWrapper), Expr::Alias(Term::Select(globalBox, {}, box).widen()).widen()).widen(),
                                  Var(privateWrapper, poison(wrapper), true).widen(),
                                  Mut(boxMember(privateWrapper), Expr::Alias(Term::Select(privateBox, {}, box).widen()).widen()).widen(),
                                  Return(Expr::Alias(Term::Unit0Const().widen()).widen()).widen(),
                              },
                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
  polyregion::compiler::Options opts{Target::Source_C_OpenCL1_1, ""};
  opts.pipelineSpec = "Mirror";
  const auto c = polyregion::compiler::compile(Program(entry, {}, {boxDef, wrapperDef}, PassPhase::Initial(), {}), opts, OptLevel::O0);
  INFO(repr(c));
  REQUIRE(c.binary != std::nullopt);
  const std::string source(reinterpret_cast<const char *>(c.binary->data()), c.binary->size());
  CHECK(source.find("typedef struct Wrapper_asp Wrapper_asp;") != std::string::npos);
  CHECK(source.find("Box_asp _base_Box;") != std::string::npos);
  CHECK(source.find("Wrapper_asp _v5;") != std::string::npos);
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
  const auto gp = source.find("typedef struct Wrapper_asgp Wrapper_asgp;");
  const auto pg = source.find("typedef struct Wrapper_aspg Wrapper_aspg;");
  CHECK(gp != std::string::npos);
  CHECK(pg != std::string::npos);
  CHECK(gp < pg);
  CHECK(source.find("Box_asp left;") != std::string::npos);
  CHECK(source.find("Box_asp right;") != std::string::npos);
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
  CHECK(source.find("private int* _v4 = _v3.ptr;") != std::string::npos);
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
  CHECK(source.find("typedef struct Pair_aspg Pair_aspg;") != std::string::npos);
  CHECK(source.find("typedef struct Pair_asgp Pair_asgp;") != std::string::npos);
  CHECK(source.find("Pair_aspg _v4;") != std::string::npos);
  CHECK(source.find("Pair_asgp _v5;") != std::string::npos);
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
  CHECK(source.find("private int* ptr;") != std::string::npos);
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
  CHECK(source.find("typedef struct Box_asc Box_asc;") != std::string::npos);
  CHECK(source.find("constant char* ptr;") != std::string::npos);
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
  CHECK(source.find("Box_asc _v2;") != std::string::npos);
  CHECK(source.find("Box_asp _v2;") != std::string::npos);
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
    CHECK(source.find(name + "((device atomic_int*)") != std::string::npos);
  }

  const auto local = compile(AtomicOp::Add(), MemOrder::Relaxed(), Type::IntU32(), TypeSpace::Local());
  INFO(repr(local));
  REQUIRE(local.binary);
  const std::string source(local.binary->begin(), local.binary->end());
  CHECK(source.find("atomic_fetch_add_explicit((threadgroup atomic_uint*)") != std::string::npos);
  CHECK(source.find("metal::memory_order_relaxed") != std::string::npos);
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
    CHECK(source.find("_pr_vld_" + std::string(space) + "_Outer") != std::string::npos);
    CHECK(source.find("_pr_vst_" + std::string(space) + "_Outer") != std::string::npos);
  }
  CHECK(source.find("r.inner.x = (*p).inner.x;") != std::string::npos);
  CHECK(source.find("for (int _vc1 = 0; _vc1 < 2; _vc1++)") != std::string::npos);
  CHECK(source.find("(*p).values[_vc1] = v.values[_vc1];") != std::string::npos);
  const auto loadDefinition = source.find("Outer _pr_vld_device_Outer(");
  const auto storeDefinition = source.find("void _pr_vst_device_Outer(");
  REQUIRE(loadDefinition != std::string::npos);
  REQUIRE(storeDefinition != std::string::npos);
  CHECK(source.find("Outer _pr_vld_device_Outer(", loadDefinition + 1) == std::string::npos);
  CHECK(source.find("void _pr_vst_device_Outer(", storeDefinition + 1) == std::string::npos);
  const auto again = polyregion::compiler::compile(Program(entry, {global, local, priv}, {innerDef, outerDef}, PassPhase::Initial(), {}),
                                                   opts, OptLevel::O0);
  REQUIRE(again.binary);
  CHECK(source == std::string(again.binary->begin(), again.binary->end()));
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
  CHECK(scalar.find("volatile device int32_t*") != std::string::npos);
  CHECK(scalar.find("_pr_vld_") == std::string::npos);

  const auto constant = emit(Target::Source_C_Metal1_0, makeProgram(TypeSpace::Constant(), false, record));
  CHECK(constant.find("_pr_vld_constant_Record") != std::string::npos);
  REQUIRE_THROWS_WITH(emit(Target::Source_C_Metal1_0, makeProgram(TypeSpace::Constant(), true, record)),
                      Catch::Matchers::ContainsSubstring("volatile store to constant storage is unsupported for MSL"));

  const auto opencl = emit(Target::Source_C_OpenCL1_1, makeProgram(TypeSpace::Global(), false, record));
  CHECK(opencl.find("volatile global Record*") != std::string::npos);
  CHECK(opencl.find("_pr_vld_") == std::string::npos);
  REQUIRE_THROWS_WITH(emit(Target::Source_C_C11, makeProgram(TypeSpace::Global(), false, record)),
                      Catch::Matchers::ContainsSubstring("Spec::GpuVolatileLoad unsupported for C11"));
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
