#include <thread>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"
#include "polyinvoke/stream.hpp"
#include "polyregion/concurrency_utils.hpp"

#include "ast.h"
#include "compiler.h"
#include "generated/polyast.h"
#include "polytest/profile.hpp"

#ifndef POLYREGION_TEST_PROFILE_DIR
  #define POLYREGION_TEST_PROFILE_DIR ""
#endif

using namespace polyregion;
using namespace polyregion::concurrency_utils;
using namespace polyregion::polyast;
using namespace aspartame;

using namespace Stmt;
using namespace Expr;
using namespace Intr;
using namespace Math;
using namespace Spec;

using namespace polyregion::polyast::dsl;

struct StreamFunctions {
  Function copy, mul, add, triad, dot;
};

StreamFunctions mkStreamFunctions(std::string suffix, Type::Any type, bool gpu = false) {
  using Stmts = std::vector<Stmt::Any>;
  static auto empty = [](Term::Any, Term::Any) -> Stmts { return {}; };
  auto mkCpuStreamFn = [&](const std::string &name, std::vector<Arg> extraArgs, const std::function<Stmts(Term::Any, Term::Any)> &mkPrelude,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkLoopBody,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkEpilogue) {
    std::vector<Arg> args = {"a"_(Ptr(type))(), "b"_(Ptr(type))(), "c"_(Ptr(type))()};
    if (!gpu) {
      // XXX polyc auto-prepends `__tid` for CPU isEntry kernels; the user list omits it.
      // begin/end are the work-steal slice pointers, tid indexes into them.
      std::vector<Arg> cpuArgs = {"begin"_(Ptr(Long))(), "end"_(Ptr(Long))()};
      args.insert(args.begin(), cpuArgs.begin(), cpuArgs.end());
    }
    args ^= concat_inplace(extraArgs);

    Stmts stmts;

    if (!gpu) {
      stmts.push_back(var("i") = "begin"_(Ptr(Long))["__tid"_(Long)]); // long i = begin[__tid]

      stmts ^= concat_inplace(mkPrelude("__tid"_(Long), "i"_(Long)));

      auto loopBody = mkLoopBody("__tid"_(Long), "i"_(Long));
      loopBody.push_back(Mut("i"_(Long), call(Add("i"_(Long), 1_(Long), Long)))); // i++

      stmts ^= concat_inplace(whileLoop(
          {let("bound") = "end"_(Ptr(Long))["__tid"_(Long)], var("cont") = call(LogicLt("i"_(Long), "bound"_(Long)))}, "cont"_(Bool),
          loopBody)); // while(i < end[__tid])

      stmts ^= concat_inplace(mkEpilogue("__tid"_(Long), "i"_(Long)));
    } else {

      stmts.push_back(var("i") = call(GpuGlobalIdx(0_(UInt)))); // uint32_t i = global_id
      stmts.push_back(let("local_i") = call(GpuLocalIdx(0_(UInt))));

      stmts ^= concat_inplace(mkPrelude("local_i"_(UInt), "i"_(UInt)));
      stmts ^= concat_inplace(mkLoopBody("local_i"_(UInt), "i"_(UInt)));
      stmts ^= concat_inplace(mkEpilogue("local_i"_(UInt), "i"_(UInt)));
    }

    stmts.push_back(ret(Term::Unit0Const()));

    return function("stream_" + name + suffix, args, Unit, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true)(stmts);
  };

  auto copy = mkCpuStreamFn( //
      "copy", {}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ai") = "a"_(Ptr(type))[i],   // ai = a[i]
            "c"_(Ptr(type))[i] = "ai"_(type), // c[i] = ai
        };
      },
      empty);

  auto mul = mkCpuStreamFn( //
      "mul", {"scalar"_(type)()}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ci") = "c"_(Ptr(type))[i],                           // ci = b[i]
            let("r") = call(Mul("ci"_(type), "scalar"_(type), type)), // r = ci * scalar
            "b"_(Ptr(type))[i] = "r"_(type),                          // b[i] = result
        };
      },
      empty);

  auto add = mkCpuStreamFn( //
      "add", {}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ai") = "a"_(Ptr(type))[i],                       // ai = a[i]
            let("bi") = "b"_(Ptr(type))[i],                       // bi = b[i]
            let("r") = call(Add("ai"_(type), "bi"_(type), type)), // r = ai + bi
            "c"_(Ptr(type))[i] = "r"_(type),                      // c[i] = r
        };
      },
      empty);

  auto triad = mkCpuStreamFn( //
      "triad", {"scalar"_(type)()}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("bi") = "b"_(Ptr(type))[i],                            // bi = b[i]
            let("ci") = "c"_(Ptr(type))[i],                            // ci = b[i]
            let("r0") = call(Mul("ci"_(type), "scalar"_(type), type)), // r0 = ci * scalar
            let("r1") = call(Add("bi"_(type), "r0"_(type), type)),     // r1 = bi + r0
            "a"_(Ptr(type))[i] = "r1"_(type),                          // a[i] = r1
        };
      },
      empty);

  auto dot =
      !gpu ? //
          mkCpuStreamFn(
              "dot", {"sum"_(Ptr(type))()},
              [&](auto id, auto i) -> Stmts { //
                return {
                    var("acc") = 0_(type) // mutable accumulator across iterations
                };
              },
              [&](auto id, auto i) -> Stmts {
                return {
                    let("ai") = "a"_(Ptr(type))[i],                              // ai = a[i]
                    let("bi") = "b"_(Ptr(type))[i],                              // bi = b[i]
                    let("sumid") = "acc"_(type),                                 // sumid = acc
                    let("r0") = call(Mul("ai"_(type), "bi"_(type), type)),       // r0 = ai * bi
                    "acc"_(type) = call(Add("r0"_(type), "sumid"_(type), type)), // acc = r0 + sumid

                };
              },
              [&](auto id, auto i) -> Stmts {
                return {
                    "sum"_(Ptr(type))[id] = "acc"_(type) // sum[id] = acc
                };
              })
           : //
          mkCpuStreamFn(
              "dot", {"sum"_(Ptr(type))(), "wg_sum"_(Ptr(type, {}, Local))(), "array_size"_(UInt)()}, empty,
              [&](auto local_i, auto i) -> Stmts {
                Stmts body;
                body.push_back(let("global_size") = call(GpuGlobalSize(0_(UInt))));
                body.push_back("wg_sum"_(Ptr(type, {}, Local))[local_i] = 0_(type));
                body ^= concat_inplace(whileLoop({var("cont") = call(LogicLt("i"_(UInt), "array_size"_(UInt)))}, "cont"_(Bool),
                                                 {let("ai") = "a"_(Ptr(type))[i],                           // ai = a[i]
                                                  let("bi") = "b"_(Ptr(type))[i],                           // bi = b[i]
                                                  let("sumid") = "wg_sum"_(Ptr(type, {}, Local))[local_i],  // sumid = sum[local_i]
                                                  let("r0") = call(Mul("ai"_(type), "bi"_(type), type)),    // r0 = ai * bi
                                                  let("r1") = call(Add("r0"_(type), "sumid"_(type), type)), // r1 = r0 + sumid
                                                  "wg_sum"_(Ptr(type, {}, Local))[local_i] = "r1"_(type),   // a[i] = bi
                                                  ("i"_(UInt) = call(Add("i"_(UInt), "global_size"_(UInt), UInt)))})); // i += global_size
                body.push_back(var("offset") = call(GpuLocalSize(0_(UInt))));
                body.push_back("offset"_(UInt) = call(Div("offset"_(UInt), 2_(UInt), UInt))); // offset /= 2
                body ^= concat_inplace(
                    whileLoop({var("cont2") = call(LogicGt("offset"_(UInt), 0_(UInt)))}, "cont2"_(Bool),
                              {
                                  let("_") = call(GpuBarrierLocal()), let("__cond_lt") = call(LogicLt("local_i"_(UInt), "offset"_(UInt))),
                                  Cond("__cond_lt"_(Bool), //
                                       {
                                           let("new_offset") = call(Add("local_i"_(UInt), "offset"_(UInt), UInt)), // local_i + offset
                                           let("wg_sum_old") = "wg_sum"_(Ptr(type, {}, Local))[local_i],
                                           let("wg_sum_at_offset") = "wg_sum"_(Ptr(type, {}, Local))["new_offset"_(UInt)],
                                           "wg_sum_at_offset"_(type) = call(Add("wg_sum_at_offset"_(type), "wg_sum_old"_(type), type)),
                                           "wg_sum"_(Ptr(type, {}, Local))[local_i] = "wg_sum_at_offset"_(type),
                                       },
                                       {}),
                                  "offset"_(UInt) = call(Div("offset"_(UInt), 2_(UInt), UInt)) // offset /= 2
                              }));
                body.push_back(let("group_id") = call(GpuGroupIdx(0_(UInt))));
                body.push_back(let("__cond_eq") = call(LogicEq("local_i"_(UInt), 0_(UInt))));
                body.push_back(Cond("__cond_eq"_(Bool), //
                                    {
                                        let("wg_sum_old_1") = "wg_sum"_(Ptr(type, {}, Local))[local_i],
                                        "sum"_(Ptr(type))["group_id"_(UInt)] = "wg_sum_old_1"_(type),
                                    },
                                    {}));
                return body;
              },
              empty);

  //

  //  std::cout << repr(copy) << std::endl;
  //  std::cout << repr(mul) << std::endl;
  //  std::cout << repr(add) << std::endl;
  //  std::cout << repr(triad) << std::endl;
  //  std::cout << repr(dot) << std::endl;

  return StreamFunctions{copy, mul, add, triad, dot};
}

int main() {

  const auto resolved = polytest::resolveTestTargets(POLYREGION_TEST_PROFILE_DIR);
  if (resolved.empty()) {
    fmt::print(stderr, "babelstream: no test targets resolved (set POLYREGION_TEST_TARGETS or provide a profile under {})\n",
               POLYREGION_TEST_PROFILE_DIR);
    return EXIT_FAILURE;
  }

  for (const auto &r : resolved) {
    const auto backend = r.spec.runtime;
    const auto target = r.spec.codegen;
    const auto arch = r.arch;

    auto cpu = backend == invoke::Backend::RelocatableObject || backend == invoke::Backend::SharedObject;

    auto fns = mkStreamFunctions("_float", Float, !cpu);

    auto platform = polyregion::invoke::Platform::maybe(backend);
    if (!platform) throw std::runtime_error("Backend " + std::string(magic_enum::enum_name(backend)) + " failed to initialise");

    fmt::print("backend={} arch={}\n", magic_enum::enum_name(backend), arch);
    {

      compiler::initialise();
      auto compileOne = [&](const Function &fn) {
        auto c = compiler::compile(program(fn), {target, arch}, compiletime::OptLevel::Ofast);
        fmt::print(stderr, "{}\n", repr(c));
        std::fflush(stderr);
        if (!c.binary) throw std::logic_error("No binary produced for " + fn.name.fqn.front());
        return std::string(reinterpret_cast<char *>(c.binary->data()), c.binary->size());
      };
      auto images = stream::Kernels<std::string>{.copy = compileOne(fns.copy),
                                                 .mul = compileOne(fns.mul),
                                                 .add = compileOne(fns.add),
                                                 .triad = compileOne(fns.triad),
                                                 .dot = compileOne(fns.dot)};

      const auto relTolerance = 0.008f;

      for (auto &d : platform->enumerate()) {

        const auto isSpirvImage = target == compiletime::Target::Object_LLVM_SPIRV32_Kernel ||
                                  target == compiletime::Target::Object_LLVM_SPIRV64_Kernel ||
                                  target == compiletime::Target::Object_LLVM_SPIRV_GLCompute;
        if (backend == invoke::Backend::OpenCL) {
          const auto features = d->features();
          const auto hasSpirvKernel = std::find(features.begin(), features.end(), std::string("spirv_kernel")) != features.end();
          if (isSpirvImage != hasSpirvKernel) continue;
        }

        std::string suffix = "_float";

        d->loadModule("stream_copy" + suffix, images.copy);
        d->loadModule("stream_mul" + suffix, images.mul);
        d->loadModule("stream_add" + suffix, images.add);
        d->loadModule("stream_triad" + suffix, images.triad);
        d->loadModule("stream_dot" + suffix, images.dot);
        stream::Kernels<std::pair<std::string, std::string>> kernelSpecs = {.copy = {"stream_copy" + suffix, "stream_copy" + suffix},
                                                                            .mul = {"stream_mul" + suffix, "stream_mul" + suffix},
                                                                            .add = {"stream_add" + suffix, "stream_add" + suffix},
                                                                            .triad = {"stream_triad" + suffix, "stream_triad" + suffix},
                                                                            .dot = {"stream_dot" + suffix, "stream_dot" + suffix}};

        size_t size = 33554432;
        size_t times = 100;
        size_t groups = cpu ? std::thread::hardware_concurrency() : 256;

        auto fValidate = [](auto actual, auto tolerance) {
          if (actual >= tolerance) {
            fmt::print(stderr, "Tolerance ({}%) exceeded for array value delta: {}%\n", tolerance * 100, actual * 100);
          }
        };

        auto fValidateSum = [=](auto actual) {
          if (actual >= relTolerance) {
            fmt::print(stderr, "Tolerance ({}%) exceeded for dot value delta: {}%\n", relTolerance * 100, actual * 100);
          }
        };

        stream::runStream<float>(                              //
            invoke::Type::Float32, size, times, groups,        //
            "Explicit: " + platform->name(), platform->kind(), //
            *d, kernelSpecs, true, fValidate, fValidateSum);

        polyregion::stream::runStreamShared<float>(          //
            invoke::Type::Float32, size, times, groups,      //
            "Shared: " + platform->name(), platform->kind(), //
            *d, kernelSpecs, true, fValidate, fValidateSum);
      }

      //
    }
  }
}