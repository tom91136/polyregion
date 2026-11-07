#include <algorithm>
#include <memory>
#include <vector>

#include "aspartame/all.hpp"
#include "catch2/catch_all.hpp"
#include "fmt/format.h"

#include "polyinvoke/runtime.h"
#include "polyregion/env_keys.h"
#include "polyregion/group_native.h"

#include "ast.h"
#include "compiler.h"
#include "generated/polyast.h"
#include "scoped_env.h"

using namespace aspartame;
using namespace polyregion::polyast;
using namespace polyregion::compiletime;
using namespace Stmt;
using namespace Expr;
using namespace Intr;

namespace {

namespace invoke = polyregion::invoke;

constexpr int32_t TileLanes = 1024;

const Type::Any I32 = Type::IntS32(), U32 = Type::IntU32(), Unit = Type::Unit0(), Bool = Type::Bool1();
const Type::Any GlobalI32 = Type::Ptr(Type::IntS32(), TypeSpace::Global());

Term::Any c32(const int32_t v) { return Term::IntS32Const(v); }
Term::Any cu(const uint32_t v) { return Term::IntU32Const(v); }
Term::Any sel(const Named &n) { return selectNamed(n); }
Stmt::Any let(const Named &n, const Expr::Any &e, const bool mut = false) { return Var(n, e, mut); }
Stmt::Any decl(const Named &n) { return Var(n, std::optional<Expr::Any>{}, true); }
Stmt::Any assign(const Named &n, const Expr::Any &e) { return Mut(selectNamed(n), e); }
Expr::Any alias(const Term::Any &t) { return Alias(t); }
Expr::Any spec(const Spec::Any &s) { return SpecOp(s); }
Expr::Any op(const Intr::Any &i) { return IntrOp(i); }
Expr::Any at(const Named &arr, const Term::Any &i) { return Expr::Index(sel(arr), i, Type::IntS32()); }
Stmt::Any store(const Named &arr, const Term::Any &i, const Term::Any &v) { return Update(selectNamed(arr), i, v); }
Stmt::Any barrier() { return let(Named("#sync", Unit), spec(Spec::GpuBarrierLocal())); }
Stmt::Any leave() { return Return(alias(Term::Unit0Const())); }
Term::Any dim0() { return cu(0); }

struct Kernel {
  std::string name;
  std::vector<Stmt::Any> body;
  std::vector<Arg> args;
};

Program programOf(const Kernel &k) {
  auto body = k.body;
  body.emplace_back(leave());
  const Function entry(FunctionDecl(Sym({k.name}), {}, std::optional<Arg>{}, k.args, {}, {}, Unit, FunctionAffinity::Offload()), body,
                       FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  return Program(entry, {}, {}, PassPhase::Initial(), {});
}

const Named In("in", GlobalI32), Out("out", GlobalI32);
const Named Tile("tile", Type::Arr(Type::IntS32(), TileLanes, TypeSpace::Local()));
const Named Lid("lid", U32), Gid("gid", U32), Lanes("lanes", U32), Group("group", U32), Groups("groups", U32);
const Named LidI("lidI", I32), GidI("gidI", I32);

std::vector<Stmt::Any> join(std::vector<Stmt::Any> a, const std::vector<Stmt::Any> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

std::vector<Stmt::Any> ids() {
  return {let(Lid, spec(Spec::GpuLocalIdx(dim0()))),     let(Gid, spec(Spec::GpuGlobalIdx(dim0()))),
          let(Lanes, spec(Spec::GpuLocalSize(dim0()))),  let(Group, spec(Spec::GpuGroupIdx(dim0()))),
          let(Groups, spec(Spec::GpuGroupSize(dim0()))), let(LidI, Cast(sel(Lid), Type::IntS32())),
          let(GidI, Cast(sel(Gid), Type::IntS32()))};
}

Kernel flatKernel() {
  const Named v("v", I32), r("r", I32);
  return {"c11_flat",
          {let(Gid, spec(Spec::GpuGlobalIdx(dim0()))), let(v, at(In, sel(Gid))), let(r, op(Add(sel(v), sel(v), I32))),
           store(Out, sel(Gid), sel(r))},
          {Arg(In, {}), Arg(Out, {})}};
}

Kernel tidKernel() {
  const Named tid("__tid", Type::IntS64()), v("v", I32), r("r", I32);
  return {"c11_tid",
          {let(v, at(In, sel(tid))), let(r, op(Add(sel(v), sel(v), I32))), store(Out, sel(tid), sel(r))},
          {Arg(In, {}), Arg(Out, {})}};
}

Kernel minMaxKernel() {
  const Named v("v", I32), lo("lo", I32), hi("hi", I32), r("r", I32);
  return {"c11_minmax",
          {let(v, at(In, c32(0))), let(lo, op(Min(sel(v), c32(10), I32))), let(hi, op(Max(sel(v), c32(5), I32))),
           let(r, op(Add(sel(lo), sel(hi), I32))), store(Out, c32(0), sel(r))},
          {Arg(In, {}), Arg(Out, {})}};
}

Kernel idsKernel() {
  const Named base("base", U32), i1("i1", U32), i2("i2", U32), i3("i3", U32), groupI("groupI", I32), groupsI("groupsI", I32),
      lanesI("lanesI", I32);
  return {"c11_ids",
          join(ids(), {let(base, op(Mul(sel(Gid), cu(4), U32))), let(i1, op(Add(sel(base), cu(1), U32))),
                       let(i2, op(Add(sel(base), cu(2), U32))), let(i3, op(Add(sel(base), cu(3), U32))),
                       let(groupI, Cast(sel(Group), Type::IntS32())), let(groupsI, Cast(sel(Groups), Type::IntS32())),
                       let(lanesI, Cast(sel(Lanes), Type::IntS32())), store(Out, sel(base), sel(groupI)), store(Out, sel(i1), sel(groupsI)),
                       store(Out, sel(i2), sel(LidI)), store(Out, sel(i3), sel(lanesI))}),
          {Arg(Out, {})}};
}

Kernel reverseKernel() {
  const Named v("v", I32), last("last", U32), j("j", U32), w("w", I32);
  return {"c11_reverse",
          join(ids(),
               {decl(Tile), let(v, at(In, sel(Gid))), store(Tile, sel(Lid), sel(v)), barrier(), let(last, op(Sub(sel(Lanes), cu(1), U32))),
                let(j, op(Sub(sel(last), sel(Lid), U32))), let(w, at(Tile, sel(j))), store(Out, sel(Gid), sel(w))}),
          {Arg(In, {}), Arg(Out, {})}};
}

Kernel reduceKernel() {
  const Named v("v", I32), s("s", U32), more("more", Bool), active("active", Bool), hi("hi", U32), a("a", I32), b("b", I32),
      sum("sum", I32), leader("leader", Bool), total("total", I32);
  return {
      "c11_reduce",
      join(ids(), {decl(Tile), let(v, at(In, sel(Gid))), store(Tile, sel(Lid), sel(v)), barrier(),
                   let(s, op(Div(sel(Lanes), cu(2), U32)), true), let(more, op(LogicGt(sel(s), cu(0)))),
                   While(sel(more), {let(active, op(LogicLt(sel(Lid), sel(s)))),
                                     Cond(sel(active),
                                          {let(hi, op(Add(sel(Lid), sel(s), U32))), let(a, at(Tile, sel(Lid))), let(b, at(Tile, sel(hi))),
                                           let(sum, op(Add(sel(a), sel(b), I32))), store(Tile, sel(Lid), sel(sum))},
                                          {}),
                                     barrier(), assign(s, op(Div(sel(s), cu(2), U32))), assign(more, op(LogicGt(sel(s), cu(0))))}),
                   let(leader, op(LogicEq(sel(Lid), cu(0)))),
                   Cond(sel(leader), {let(total, at(Tile, cu(0))), store(Out, sel(Group), sel(total))}, {})}),
      {Arg(In, {}), Arg(Out, {})}};
}

Kernel continueKernel() {
  const Named acc("acc", I32), k("k", I32), kBase("kBase", I32), v("v", I32), next("next", U32), j("j", U32), n("n", I32),
      parity("parity", U32), even("even", Bool), second("second", Bool), skip("skip", Bool);
  return {"c11_continue",
          join(ids(), {decl(Tile), let(acc, alias(c32(0)), true),
                       ForRange(k, c32(0), c32(4), c32(1),
                                {let(kBase, op(Mul(sel(k), c32(100), I32))), let(v, op(Add(sel(LidI), sel(kBase), I32))),
                                 store(Tile, sel(Lid), sel(v)), barrier(), let(next, op(Add(sel(Lid), cu(1), U32))),
                                 let(j, op(Rem(sel(next), sel(Lanes), U32))), let(n, at(Tile, sel(j))), barrier(),
                                 let(parity, op(Rem(sel(Lid), cu(2), U32))), let(even, op(LogicEq(sel(parity), cu(0)))),
                                 let(second, op(LogicEq(sel(k), c32(1)))), let(skip, op(LogicAnd(sel(even), sel(second)))),
                                 Cond(sel(skip), {Cont()}, {}), assign(acc, op(Add(sel(acc), sel(n), I32)))}),
                       store(Out, sel(Gid), sel(acc))}),
          {Arg(Out, {})}};
}

Kernel earlyKernel() {
  const Named parity("parity", U32), odd("odd", Bool), v("v", I32), r("r", I32);
  return {"c11_early",
          join(ids(), {decl(Tile), let(parity, op(Rem(sel(Lid), cu(2), U32))), let(odd, op(LogicNeq(sel(parity), cu(0)))),
                       Cond(sel(odd), {leave()}, {}), store(Tile, sel(Lid), sel(GidI)), barrier(), let(v, at(Tile, sel(Lid))),
                       let(r, op(Add(sel(v), c32(1), I32))), store(Out, sel(Gid), sel(r))}),
          {Arg(Out, {})}};
}

Kernel stateKernel() {
  const Named priv("priv", Type::Arr(Type::IntS32(), 20, TypeSpace::Private())), i("i", I32), j("j", I32), cell("cell", I32),
      acc("acc", I32), x("x", I32), sum("sum", I32);
  return {"c11_state",
          join(ids(), {decl(priv),
                       ForRange(i, c32(0), c32(20), c32(1), {let(cell, op(Add(sel(LidI), sel(i), I32))), store(priv, sel(i), sel(cell))}),
                       barrier(), let(acc, alias(c32(0)), true),
                       ForRange(j, c32(0), c32(20), c32(1), {let(x, at(priv, sel(j))), assign(acc, op(Add(sel(acc), sel(x), I32)))}),
                       store(Out, sel(Gid), sel(acc))}),
          {Arg(Out, {})}};
}

Kernel zeroKernel() {
  const Named z("z", I32), mark("mark", I32);
  return {"c11_zero",
          join(ids(), {decl(Tile), let(z, at(Tile, sel(Lid))), store(Out, sel(Gid), sel(z)), let(mark, op(Add(sel(GidI), c32(1), I32))),
                       store(Tile, sel(Lid), sel(mark))}),
          {Arg(Out, {})}};
}

Kernel collectiveKernel() {
  const Named v("v", I32), total("total", I32), prefix("prefix", I32), r("r", I32), at4("at4", U32), at4b("at4b", U32);
  return {"c11_collective",
          join(ids(), {let(v, at(In, sel(Gid))), let(total, spec(Spec::GpuGroupReduce(AtomicOp::Add(), sel(v), Type::IntS32()))),
                       let(prefix, spec(Spec::GpuGroupInclusiveScan(AtomicOp::Add(), sel(v), Type::IntS32()))),
                       let(at4, op(Mul(sel(Gid), cu(2), U32))), let(at4b, op(Add(sel(at4), cu(1), U32))), store(Out, sel(at4), sel(total)),
                       store(Out, sel(at4b), sel(prefix))}),
          {Arg(In, {}), Arg(Out, {})}};
}

Kernel nestedKernel() {
  const Named acc("acc", I32), parity("parity", U32), odd("odd", Bool), small("small", Bool), half("half", U32), x("x", I32), y("y", I32),
      z("z", I32), uniform("uniform", Bool);
  return {"c11_nested",
          join(ids(), {decl(Tile), let(acc, alias(c32(0)), true), let(parity, op(Rem(sel(Lid), cu(2), U32))),
                       let(odd, op(LogicNeq(sel(parity), cu(0)))), let(uniform, op(LogicGt(sel(Lanes), cu(0)))),
                       Cond(sel(uniform),
                            {store(Tile, sel(Lid), sel(LidI)), barrier(),
                             Cond(sel(odd), {let(x, at(Tile, sel(Lid))), assign(acc, op(Add(sel(acc), sel(x), I32)))},
                                  {let(y, op(Add(sel(LidI), c32(7), I32))), assign(acc, op(Add(sel(acc), sel(y), I32)))}),
                             barrier(), let(half, op(Div(sel(Lanes), cu(2), U32))), let(small, op(LogicLt(sel(Lid), sel(half)))),
                             Cond(sel(small), {let(z, at(Tile, sel(Lid))), assign(acc, op(Add(sel(acc), sel(z), I32)))}, {})},
                            {}),
                       store(Out, sel(Gid), sel(acc))}),
          {Arg(Out, {})}};
}

std::string compileSource(const Kernel &k, const bool pipelineTail = true) {
  polyregion::compiler::initialise();
  polyregion::compiler::Options opts{Target::Source_C_C11, ""};
  opts.pipelineSpec =
      "FullOpt(level=0)" + (pipelineTail ? hostGroupPipelineTail("lowerGroups=true", false) : std::string(";StructuredExit"));
  const auto compiled = polyregion::compiler::compile(programOf(k), opts, OptLevel::O0);
  INFO(repr(compiled));
  REQUIRE(compiled.binary);
  return std::string(compiled.binary->begin(), compiled.binary->end());
}

struct Shape {
  size_t groups, lanes;
};

struct Host {
  std::unique_ptr<invoke::Platform> platform = invoke::Platform::maybe(invoke::Backend::SharedObject);
  std::unique_ptr<invoke::Device> device;
  std::unique_ptr<invoke::DeviceQueue> queue;
  Host() {
    REQUIRE(platform);
    auto devices = platform->enumerate();
    REQUIRE(!devices.empty());
    device = std::move(devices.front());
    queue = device->createQueue(std::chrono::seconds(30));
  }
  bool buildsC() { return device->moduleFormat() == invoke::ModuleFormat::Source; }
  void load(const Kernel &k) { device->loadModule(k.name, compileSource(k)); }

  std::vector<int32_t> run(const Kernel &k, const Shape shape, const std::vector<int32_t> &in, const size_t outCount,
                           const bool grouped = true) {
    std::vector<int32_t> out(outCount, 0);
    const void *inPtr = in.data(), *outPtr = out.data();
    invoke::ArgBuffer buffer;
    buffer.append(invoke::Type::IntS64, nullptr);
    if (!in.empty()) buffer.append(invoke::Type::Ptr, &inPtr);
    buffer.append(invoke::Type::Ptr, &outPtr);
    buffer.append(invoke::Type::Void, nullptr);
    const invoke::Policy policy{{shape.groups, 1, 1},
                                grouped ? std::optional{std::pair{invoke::Dim3{shape.lanes, 1, 1}, size_t{0}}} : std::nullopt};
    queue->enqueueInvokeAsync(k.name, k.name, buffer, policy, {});
    queue->enqueueWaitBlocking();
    return out;
  }
};

std::vector<int32_t> ramp(const size_t n, const int32_t scale = 3, const int32_t offset = 1) {
  std::vector<int32_t> xs(n);
  for (size_t i = 0; i < n; ++i)
    xs[i] = static_cast<int32_t>(i) * scale + offset;
  return xs;
}

const std::vector<Shape> AnyShapes{{1, 1}, {2, 13}, {3, 64}, {4, 256}, {2, 1024}};
const std::vector<Shape> PowerShapes{{1, 1}, {3, 64}, {4, 256}, {2, 1024}};

} // namespace

TEST_CASE("C11 entries take the host entry abi", "[c11]") {
  const auto source = compileSource(reverseKernel());
  INFO(source);
  CHECK_THAT(source, Catch::Matchers::ContainsSubstring("c11_reverse(int64_t __tid, int32_t* "));
  CHECK_THAT(source, Catch::Matchers::ContainsSubstring("c11_reverse__polyc_host_abi = 2;"));
  CHECK_THAT(source, Catch::Matchers::ContainsSubstring("c11_reverse__polyc_static_local_bytes = (int64_t) (4096);"));
  CHECK_THAT(source, Catch::Matchers::ContainsSubstring("_polyregion_host_table __polyc_host;"));
  CHECK_THAT(source, !Catch::Matchers::ContainsSubstring("((void)0)"));

  auto dotted = reverseKernel();
  dotted.name = "ns.dotted";
  CHECK_THAT(compileSource(dotted), Catch::Matchers::ContainsSubstring("ns_dotted__polyc_host_abi = 2;"));

  const auto flat = compileSource(flatKernel());
  CHECK_THAT(flat, Catch::Matchers::ContainsSubstring("c11_flat__polyc_host_abi = 1;"));
  CHECK_THAT(flat, Catch::Matchers::ContainsSubstring("c11_flat__polyc_static_local_bytes = (int64_t) (0);"));
}

TEST_CASE("C11 rejects work-group barriers in an entry that runs on the lane abi", "[c11]") {
  const ScopedEnv lane(polyregion::env::PolyregionHostAbi, std::string("lane"));
  CHECK_THROWS_WITH(compileSource(reverseKernel()), Catch::Matchers::ContainsSubstring("barrier or local id"));
  CHECK_THROWS_WITH(compileSource(reduceKernel(), false), Catch::Matchers::ContainsSubstring("c11_reduce"));
}

TEST_CASE("C11 images run through the shared object platform", "[c11]") {
  Host host;
  if (!host.buildsC()) SKIP("the shared object device found no dist clang to build C images");
  for (const auto &kernel : {flatKernel(), tidKernel(), minMaxKernel(), idsKernel(), reverseKernel(), reduceKernel(), continueKernel(),
                             earlyKernel(), stateKernel(), zeroKernel(), collectiveKernel(), nestedKernel()})
    host.load(kernel);

  SECTION("tid") {
    const auto out = host.run(tidKernel(), {37, 1}, ramp(37), 37, false);
    for (size_t i = 0; i < 37; ++i)
      REQUIRE(out[i] == 2 * (static_cast<int32_t>(i) * 3 + 1));
  }

  SECTION("min and max") {
    for (const int32_t x : {-3, 5, 7, 10, 40}) {
      const auto out = host.run(minMaxKernel(), {1, 1}, {x}, 1);
      REQUIRE(out[0] == std::min(x, 10) + std::max(x, 5));
    }
  }

  SECTION("flat") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(flatKernel(), shape, ramp(n), n);
      for (size_t i = 0; i < n; ++i)
        REQUIRE(out[i] == 2 * (static_cast<int32_t>(i) * 3 + 1));
    }
    const auto ungrouped = host.run(flatKernel(), {37, 1}, ramp(37), 37, false);
    for (size_t i = 0; i < 37; ++i)
      REQUIRE(ungrouped[i] == 2 * (static_cast<int32_t>(i) * 3 + 1));
  }

  SECTION("ids") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(idsKernel(), shape, {}, n * 4);
      for (size_t g = 0; g < shape.groups; ++g)
        for (size_t l = 0; l < shape.lanes; ++l) {
          const auto *p = &out[(g * shape.lanes + l) * 4];
          INFO(fmt::format("{}x{} group {} lane {}", shape.groups, shape.lanes, g, l));
          REQUIRE(std::vector<int32_t>(p, p + 4)
                  == std::vector<int32_t>{int32_t(g), int32_t(shape.groups), int32_t(l), int32_t(shape.lanes)});
        }
    }
  }

  SECTION("reverse through local memory") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto in = ramp(n);
      const auto out = host.run(reverseKernel(), shape, in, n);
      for (size_t g = 0; g < shape.groups; ++g)
        for (size_t l = 0; l < shape.lanes; ++l) {
          INFO(fmt::format("{}x{} group {} lane {}", shape.groups, shape.lanes, g, l));
          REQUIRE(out[g * shape.lanes + l] == in[g * shape.lanes + (shape.lanes - 1 - l)]);
        }
    }
  }

  SECTION("tree reduction") {
    for (const auto shape : PowerShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto in = ramp(n, 1, 0);
      const auto out = host.run(reduceKernel(), shape, in, shape.groups);
      for (size_t g = 0; g < shape.groups; ++g) {
        int32_t expected = 0;
        for (size_t l = 0; l < shape.lanes; ++l)
          expected += in[g * shape.lanes + l];
        INFO(fmt::format("{}x{} group {}", shape.groups, shape.lanes, g));
        REQUIRE(out[g] == expected);
      }
    }
  }

  SECTION("lane-varying continue in a loop with barriers") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(continueKernel(), shape, {}, n);
      for (size_t g = 0; g < shape.groups; ++g)
        for (size_t l = 0; l < shape.lanes; ++l) {
          int32_t expected = 0;
          for (int32_t k = 0; k < 4; ++k)
            if (!(l % 2 == 0 && k == 1)) expected += static_cast<int32_t>((l + 1) % shape.lanes) + 100 * k;
          INFO(fmt::format("{}x{} group {} lane {}", shape.groups, shape.lanes, g, l));
          REQUIRE(out[g * shape.lanes + l] == expected);
        }
    }
  }

  SECTION("early exit before a barrier") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(earlyKernel(), shape, {}, n);
      for (size_t i = 0; i < n; ++i) {
        INFO(fmt::format("{}x{} id {}", shape.groups, shape.lanes, i));
        REQUIRE(out[i] == ((i % shape.lanes) % 2 == 1 ? 0 : static_cast<int32_t>(i) + 1));
      }
    }
  }

  SECTION("per-lane array live across a barrier") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(stateKernel(), shape, {}, n);
      for (size_t i = 0; i < n; ++i) {
        INFO(fmt::format("{}x{} id {}", shape.groups, shape.lanes, i));
        REQUIRE(out[i] == 20 * static_cast<int32_t>(i % shape.lanes) + 190);
      }
    }
  }

  SECTION("group collectives") {
    for (const auto shape : PowerShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto in = ramp(n, 1, 0);
      const auto out = host.run(collectiveKernel(), shape, in, n * 2);
      for (size_t g = 0; g < shape.groups; ++g) {
        int32_t running = 0, total = 0;
        for (size_t l = 0; l < shape.lanes; ++l)
          total += in[g * shape.lanes + l];
        for (size_t l = 0; l < shape.lanes; ++l) {
          running += in[g * shape.lanes + l];
          const auto id = g * shape.lanes + l;
          INFO(fmt::format("{}x{} group {} lane {}", shape.groups, shape.lanes, g, l));
          REQUIRE(out[id * 2] == total);
          REQUIRE(out[id * 2 + 1] == running);
        }
      }
    }
  }

  SECTION("nested conditionals around barriers") {
    for (const auto shape : AnyShapes) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(nestedKernel(), shape, {}, n);
      for (size_t i = 0; i < n; ++i) {
        const auto l = static_cast<int32_t>(i % shape.lanes);
        const int32_t expected = (l % 2 == 1 ? l : l + 7) + (l < static_cast<int32_t>(shape.lanes / 2) ? l : 0);
        INFO(fmt::format("{}x{} id {}", shape.groups, shape.lanes, i));
        REQUIRE(out[i] == expected);
      }
    }
  }

  SECTION("local memory starts zeroed for every group") {
    for (const auto shape : {Shape{3, 64}, Shape{9, 1}, Shape{5, 1024}}) {
      const auto n = shape.groups * shape.lanes;
      const auto out = host.run(zeroKernel(), shape, {}, n);
      for (size_t i = 0; i < n; ++i)
        REQUIRE(out[i] == 0);
    }
  }
}
