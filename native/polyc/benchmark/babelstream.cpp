#include "ast.h"
#include "compiler.h"
#include "concurrency_utils.hpp"
#include "generated/polyast.h"
#include "polyrt/runtime.h"
#include "stream.hpp"
#include "utils.hpp"
#include <thread>

using namespace polyregion;
using namespace polyregion::concurrency_utils;
using namespace polyregion::polyast;

using namespace Stmt;
using namespace Term;
using namespace Expr;
using namespace Intr;
using namespace Math;
using namespace Spec;

using namespace polyregion::polyast::dsl;

Program mkStreamProgram(std::string suffix, Type::Any type, bool gpu = false) {
  using Stmts = std::vector<Stmt::Any>;
  static auto empty = [](auto, auto) -> Stmts { return {}; };
  auto mkCpuStreamFn = [&](const std::string &name, std::vector<Arg> extraArgs, const std::function<Stmts(Term::Any, Term::Any)> &mkPrelude,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkLoopBody,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkEpilogue) {
    std::vector<Arg> args = {"a"_(Ptr(type))(), "b"_(Ptr(type))(), "c"_(Ptr(type))()};
    if (!gpu) {
      std::vector<Arg> cpuArgs = {"id"_(Long)(), "begin"_(Ptr(Long))(), "end"_(Ptr(Long))()};
      args.insert(args.begin(), cpuArgs.begin(), cpuArgs.end());
    }
    args.insert(args.end(), extraArgs.begin(), extraArgs.end());

    Stmts stmts;

    if (!gpu) {
      stmts.push_back(let("i") = "begin"_(Ptr(Long))["id"_(Long)]); // long i = begin[id]

      auto prelude = mkPrelude("id"_(Long), "i"_(Long));
      stmts.insert(stmts.end(), prelude.begin(), prelude.end()); // ...

      auto loopBody = mkLoopBody("id"_(Long), "i"_(Long));
      loopBody.push_back(Mut("i"_(Long), invoke(Add("i"_(Long), 1_(Long), Long)), false)); // i++

      stmts.push_back(While({let("bound") = "end"_(Ptr(Long))["id"_(Long)], let("cont") = invoke(LogicLt("i"_(Long), "bound"_(Long)))},
                            "cont"_(Bool),
                            loopBody)); // while(i < end[id])

      auto epilogue = mkEpilogue("id"_(Long), "i"_(Long));
      stmts.insert(stmts.end(), epilogue.begin(), epilogue.end()); // ...
    } else {

      stmts.push_back(let("i") = invoke(GpuGlobalIdx(0_(UInt)))); // uint32+t i = begin[id]
      stmts.push_back(let("local_i") = invoke(GpuLocalIdx(0_(UInt))));

      auto prelude = mkPrelude("local_i"_(UInt), "i"_(UInt));
      stmts.insert(stmts.end(), prelude.begin(), prelude.end()); // ...

      auto loopBody = mkLoopBody("local_i"_(UInt), "i"_(UInt));
      stmts.insert(stmts.end(), loopBody.begin(), loopBody.end());

      auto epilogue = mkEpilogue("local_i"_(UInt), "i"_(UInt));
      stmts.insert(stmts.end(), epilogue.begin(), epilogue.end()); // ...
    }

    stmts.push_back(ret(Unit0Const()));

    return function("stream_" + name + suffix, args, Unit, FunctionKind::Exported())(stmts);
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
            let("r") = invoke(Mul("ci"_(type), "scalar"_(type), type)), // r = ci * scalar
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
            let("r") = invoke(Add("ai"_(type), "bi"_(type), type)), // r = ai + bi
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
            let("r0") = invoke(Mul("ci"_(type), "scalar"_(type), type)), // r0 = ci * scalar
            let("r1") = invoke(Add("bi"_(type), "r0"_(type), type)),     // r1 = bi + r0
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
                    let("acc") = 0_(type) // sum[id] = 0.0
                };
              },
              [&](auto id, auto i) -> Stmts {
                return {
                    let("ai") = "a"_(Ptr(type))[i],                                         // ai = a[i]
                    let("bi") = "b"_(Ptr(type))[i],                                         // bi = b[i]
                    let("sumid") = "acc"_(type),                                              // sumid = acc
                    let("r0") = invoke(Mul("ai"_(type), "bi"_(type), type)),                  // r0 = ai * bi
                    Mut("acc"_(type), invoke(Add("r0"_(type), "sumid"_(type), type)), false), // r1 = r0 + sumid

                };
              },
              [&](auto id, auto i) -> Stmts {
                return {
                    "sum"_(Ptr(type))[id] = "acc"_(type) // a[i] = bi
                };
              })
           : //
          mkCpuStreamFn(
              "dot", {"sum"_(Ptr(type))(), "wg_sum"_(Ptr(type, {}, Local))(), "array_size"_(UInt)()}, empty,
              [&](auto local_i, auto i) -> Stmts {
                return {
                    let("global_size") = invoke(GpuGlobalSize(0_(UInt))),
                    "wg_sum"_(Ptr(type, {}, Local))[local_i] = 0_(type),
                    While({let("cont") = invoke(LogicLt("i"_(UInt), "array_size"_(UInt)))}, "cont"_(Bool),
                          {let("ai") = "a"_(Ptr(type))[i],                                              // ai = a[i]
                           let("bi") = "b"_(Ptr(type))[i],                                              // bi = b[i]
                           let("sumid") = "wg_sum"_(Ptr(type, {}, Local))[local_i],                       // sumid = sum[local_i]
                           let("r0") = invoke(Mul("ai"_(type), "bi"_(type), type)),                       // r0 = ai * bi
                           let("r1") = invoke(Add("r0"_(type), "sumid"_(type), type)),                    // r1 = r0 + sumid
                           "wg_sum"_(Ptr(type, {}, Local))[local_i] = "r1"_(type),                        // a[i] = bi
                           Mut("i"_(UInt), invoke(Add("i"_(UInt), "global_size"_(UInt), UInt)), false)}), // i += array_size
                    let("offset") = invoke(GpuLocalSize(0_(UInt))),                                       // offset = get_local_size()
                    Mut("offset"_(UInt), invoke(Div("offset"_(UInt), 2_(UInt), UInt)), false),            // offset /= 2
                    While({let("cont2") = invoke(LogicGt("offset"_(UInt), 0_(UInt)))}, "cont2"_(Bool),
                          {
                              let("_") = invoke(GpuBarrierLocal()),
                              Cond({invoke(LogicLt("local_i"_(UInt), "offset"_(UInt)))}, //
                                   {
                                       let("new_offset") =
                                           invoke(Add("local_i"_(UInt), "offset"_(UInt), UInt)),   // new_offset = local_i + offset
                                       let("wg_sum_old") = "wg_sum"_(Ptr(type, {}, Local))[local_i], // wg_sum_old = wg_sum[local_i]
                                       let("wg_sum_at_offset") = "wg_sum"_(Ptr(type, {}, Local))["new_offset"_(UInt)], // wg_sum_at_offset =
                                                                                                                       // wg_sum[new_offset]
                                       Mut("wg_sum_at_offset"_(type), invoke(Add("wg_sum_at_offset"_(type), "wg_sum_old"_(type), type)),
                                           false),

                                       "wg_sum"_(Ptr(type, {}, Local))[local_i] = "wg_sum_at_offset"_(type), // a[i] = bi
                                   },
                                   {}),
                              Mut("offset"_(UInt), invoke(Div("offset"_(UInt), 2_(UInt), UInt)), false) // offset /= 2
                          }),
                    let("group_id") = invoke(GpuGroupIdx(0_(UInt))),
                    Cond({invoke(LogicEq("local_i"_(UInt), 0_(UInt)))}, //
                         {
                             let("wg_sum_old_1") = "wg_sum"_(Ptr(type, {}, Local))[local_i],
                             "sum"_(Ptr(type))["group_id"_(UInt)] = "wg_sum_old_1"_(type),
                         },

                         {}),
                };
              },
              empty);

  //

  //  std::cout << repr(copy) << std::endl;
  //  std::cout << repr(mul) << std::endl;
  //  std::cout << repr(add) << std::endl;
  //  std::cout << repr(triad) << std::endl;
  //  std::cout << repr(dot) << std::endl;

  auto entry = function("entry", {}, Unit)({ret(Unit0Const())});
  return Program(entry, {copy, mul, add, triad, dot}, {});
}

int main() {

  // x86-64 CMOV
  // x86-64-v2 CMPXCHG16B
  // x86-64-v3 AVX,AVX2
  // x86-64-v4 AVX512

  std::vector<std::tuple<runtime::Backend, compiletime::Target, std::string>> configs = {
      // CL runs everywhere
            {runtime::Backend::OpenCL, compiletime::Target::Source_C_OpenCL1_1, ""},
//      {runtime::Backend::Vulkan, compiletime::Target::Object_LLVM_SPIRV64, ""},
#ifdef __APPLE__
      {runtime::Backend::RELOCATABLE_OBJ, compiler::Target::Object_LLVM_AArch64, "apple-m1"},
      {runtime::Backend::Metal, compiler::Target::Source_C_Metal1_0, ""},
#else
      {runtime::Backend::CUDA, compiletime::Target::Object_LLVM_NVPTX64, "sm_60"},
//      {runtime::Backend::HIP, compiletime::Target::Object_LLVM_AMDGCN, "gfx1036"},
//      {runtime::Backend::HSA, compiletime::Target::Object_LLVM_AMDGCN, "gfx1036"},
      {runtime::Backend::RelocatableObject, compiletime::Target::Object_LLVM_x86_64, "x86-64-v3"},
#endif
  };

  for (auto [backend, target, arch] : configs) {

    auto cpu = backend == runtime::Backend::RelocatableObject || backend == runtime::Backend::SharedObject;

    auto p = mkStreamProgram("_float", Float, !cpu);
    //    std::cout << repr(p) << std::endl;

    auto platform = runtime::Platform::of(backend);
    std::cout << "backend=" << to_string(backend) << " arch=" << arch << std::endl;
    {

      polyregion::compiler::initialise();
      auto c = polyregion::compiler::compile(p, {target, arch}, compiletime::OptLevel::Ofast);
//          std::cerr << c << std::endl;
            std::cerr << repr(c) << std::endl;

      if (!c.binary) {
        throw std::logic_error("No binary produced");
      }

      std::string image(reinterpret_cast<char *>(c.binary->data()), c.binary->size());

      const auto relTolerance = 0.008f;

      for (auto &d : platform->enumerate()) {

        std::string suffix = "_float";

        polyregion::stream::Kernels<std::pair<std::string, std::string>> kernelSpecs;
        if (d->singleEntryPerModule() && false) {
          //      for (auto &[module_, data] : imageGroups)
          //        d->loadModule(module_, data);
          kernelSpecs = {.copy = {"stream_copy" + suffix, "main"},
                         .mul = {"stream_mul" + suffix, "main"},
                         .add = {"stream_add" + suffix, "main"},
                         .triad = {"stream_triad" + suffix, "main"},
                         .dot = {"stream_dot" + suffix, "main"}};
        } else {

          d->loadModule("module", image);
          kernelSpecs = {.copy = {"module", "stream_copy" + suffix},
                         .mul = {"module", "stream_mul" + suffix},
                         .add = {"module", "stream_add" + suffix},
                         .triad = {"module", "stream_triad" + suffix},
                         .dot = {"module", "stream_dot" + suffix}};
        }

        size_t size = 33554432;
        size_t times = 100;
        size_t groups = cpu ? std::thread::hardware_concurrency() : 256;

        auto fValidate = [](auto actual, auto tolerance) {
          if (actual >= tolerance) {
            std::cerr << "Tolerance (" << tolerance * 100 << "%) exceeded for array value delta: " << actual * 100 << "%" << std::endl;
          }
        };

        auto fValidateSum = [=](auto actual) {
          if (actual >= relTolerance) {
            std::cerr << "Tolerance (" << relTolerance * 100 << "%) exceeded for dot value delta: " << actual * 100 << "%" << std::endl;
          }
        };

        polyregion::stream::runStream<float>(                  //
            runtime::Type::Float32, size, times, groups,       //
            "Explicit: " + platform->name(), platform->kind(), //
            *d, kernelSpecs, true, fValidate, fValidateSum);

        polyregion::stream::runStreamShared<float>(          //
            runtime::Type::Float32, size, times, groups,     //
            "Shared: " + platform->name(), platform->kind(), //
            *d, kernelSpecs, true, fValidate, fValidateSum);
      }

      //
    }
  }
}