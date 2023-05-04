#include "stream.hpp"
#include "ast.h"
#include "compiler.h"
#include "concurrency_utils.hpp"
#include "generated/polyast.h"
#include "runtime.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace polyregion;
using namespace polyregion::concurrency_utils;
using namespace polyregion::polyast;

using namespace Stmt;
using namespace Term;
using namespace Expr;
using namespace BinaryIntrinsicKind;
using namespace NullaryIntrinsicKind;

using namespace polyregion::polyast::dsl;

TEST_CASE("GPU BabelStream") {
  //
}

Program mkStreamProgram(std::string suffix, Type::Any type, bool gpu = false) {
  using Stmts = std::vector<Stmt::Any>;
  static auto empty = [](auto, auto) -> Stmts { return {}; };
  auto mkCpuStreamFn = [&](const std::string &name, std::vector<Arg> extraArgs,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkPrelude,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkLoopBody,
                           const std::function<Stmts(Term::Any, Term::Any)> &mkEpilogue) {
    std::vector<Arg> args = {"a"_(Array(type))(), "b"_(Array(type))(), "c"_(Array(type))()};
    if (!gpu) {
      std::vector<Arg> cpuArgs = {"id"_(Long)(), "begin"_(Array(Long))(), "end"_(Array(Long))()};
      args.insert(args.begin(), cpuArgs.begin(), cpuArgs.end());
    }
    args.insert(args.end(), extraArgs.begin(), extraArgs.end());

    Stmts stmts;

    if (!gpu) {
      stmts.push_back(let("i") = "begin"_(Array(Long))["id"_(Long)]); // long i = begin[id]

      auto prelude = mkPrelude("id"_(Long), "i"_(Long));
      stmts.insert(stmts.end(), prelude.begin(), prelude.end()); // ...

      auto loopBody = mkLoopBody("id"_(Long), "i"_(Long));
      loopBody.push_back(Mut("i"_(Long), invoke(Add(), "i"_(Long), 1_(Long), Long), false)); // i++

      stmts.push_back(While({let("bound") = "end"_(Array(Long))["id"_(Long)],
                             let("cont") = BinaryIntrinsic("i"_(Long), "bound"_(Long), LogicLt(), Bool)},
                            "cont"_(Bool),
                            loopBody)); // while(i < end[id])

      auto epilogue = mkEpilogue("id"_(Long), "i"_(Long));
      stmts.insert(stmts.end(), epilogue.begin(), epilogue.end()); // ...
    } else {

      stmts.push_back(let("i") = invoke(GpuGlobalIdxX(), Int)); // long i = begin[id]
      stmts.push_back(let("local_i") = invoke(GpuLocalIdxX(), Int));

      auto prelude = mkPrelude("local_i"_(Int), "i"_(Int));
      stmts.insert(stmts.end(), prelude.begin(), prelude.end()); // ...

      auto loopBody = mkLoopBody("local_i"_(Int), "i"_(Int));
      stmts.insert(stmts.end(), loopBody.begin(), loopBody.end());

      auto epilogue = mkEpilogue("local_i"_(Int), "i"_(Long));
      stmts.insert(stmts.end(), epilogue.begin(), epilogue.end()); // ...
    }

    stmts.push_back(ret(UnitConst()));

    return function("stream_" + name + suffix, args, Unit)(stmts);
  };

  auto copy = mkCpuStreamFn( //
      "copy", {}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ai") = "a"_(Array(type))[i],   // ai = a[i]
            "c"_(Array(type))[i] = "ai"_(type), // c[i] = ai
        };
      },
      empty);

  auto mul = mkCpuStreamFn( //
      "mul", {"scalar"_(type)()}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ci") = "c"_(Array(type))[i],                             // ci = b[i]
            let("r") = invoke(Mul(), "ci"_(type), "scalar"_(type), type), // r = ci * scalar
            "b"_(Array(type))[i] = "r"_(type),                            // b[i] = result
        };
      },
      empty);

  auto add = mkCpuStreamFn( //
      "add", {}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("ai") = "a"_(Array(type))[i],                         // ai = a[i]
            let("bi") = "b"_(Array(type))[i],                         // bi = b[i]
            let("r") = invoke(Add(), "ai"_(type), "bi"_(type), type), // r = ai + bi
            "c"_(Array(type))[i] = "r"_(type),                        // c[i] = r
        };
      },
      empty);

  auto triad = mkCpuStreamFn( //
      "triad", {"scalar"_(type)()}, empty,
      [&](auto _, auto i) -> Stmts {
        return {
            let("bi") = "b"_(Array(type))[i],                              // bi = b[i]
            let("ci") = "c"_(Array(type))[i],                              // ci = b[i]
            let("r0") = invoke(Mul(), "ci"_(type), "scalar"_(type), type), // r0 = ci * scalar
            let("r1") = invoke(Add(), "bi"_(type), "r0"_(type), type),     // r1 = bi + r0
            "a"_(Array(type))[i] = "r1"_(type),                            // a[i] = r1
        };
      },
      empty);

  auto dot =
      !gpu ? //
          mkCpuStreamFn(
              "dot", {"sum"_(Array(type))()},
              [&](auto id, auto i) -> Stmts { return {"sum"_(Array(type))[id] = 0_(type)}; }, // sum[id] = 0.0
              [&](auto id, auto i) -> Stmts {
                return {
                    let("ai") = "a"_(Array(type))[i],                             // ai = a[i]
                    let("bi") = "b"_(Array(type))[i],                             // bi = b[i]
                    let("sumid") = "sum"_(Array(type))[id],                       // sumid = sum[id]
                    let("r0") = invoke(Mul(), "ai"_(type), "bi"_(type), type),    // r0 = ai * bi
                    let("r1") = invoke(Add(), "r0"_(type), "sumid"_(type), type), // r1 = r0 + sumid
                    "sum"_(Array(type))[id] = "r1"_(type),                        // a[i] = bi
                };
              },
              empty)
           : //
          mkCpuStreamFn(
              "dot", {"sum"_(Array(type))(), "wg_sum"_(Array(type, Local))(), "array_size"_(Int)()}, empty,
              [&](auto local_i, auto i) -> Stmts {
                return {
                    let("global_size") = invoke(GpuGlobalSizeX(), Int),
                    "wg_sum"_(Array(type, Local))[local_i] = 0_(type),
                    While(
                        {let("cont") = invoke(LogicLt(), "i"_(Int), "array_size"_(Int), Bool)}, "cont"_(Bool),
                        {let("ai") = "a"_(Array(type))[i],                             // ai = a[i]
                         let("bi") = "b"_(Array(type))[i],                             // bi = b[i]
                         let("sumid") = "wg_sum"_(Array(type, Local))[local_i],        // sumid = sum[local_i]
                         let("r0") = invoke(Mul(), "ai"_(type), "bi"_(type), type),    // r0 = ai * bi
                         let("r1") = invoke(Add(), "r0"_(type), "sumid"_(type), type), // r1 = r0 + sumid
                         "wg_sum"_(Array(type, Local))[local_i] = "r1"_(type),         // a[i] = bi
                         Mut("i"_(Int), invoke(Add(), "i"_(Int), "global_size"_(Int), Int), false)}), // i += array_size
                    let("offset") = invoke(GpuLocalSizeX(), Int), // offset = get_local_size()
                    Mut("offset"_(Int), invoke(Div(), "offset"_(Int), 2_(Int), Int), false), // offset /= 2
                    While({let("cont") = invoke(LogicGt(), "offset"_(Int), 0_(Int), Bool)}, "cont"_(Bool),
                          {
                              let("_") = invoke(GpuGroupBarrier(), Unit),
                              Cond({invoke(LogicLt(), "local_i"_(Int), "offset"_(Int), Bool)}, //
                                   {
                                       let("new_offset") = invoke(Add(), "local_i"_(Int), "offset"_(Int),
                                                                  Int), // new_offset = local_i + offset
                                       let("wg_sum_old") =
                                           "wg_sum"_(Array(type, Local))[local_i], // wg_sum_old = wg_sum[local_i]
                                       let("wg_sum_at_offset") =
                                           "wg_sum"_(Array(type, Local))["new_offset"_(Int)], // wg_sum_at_offset =
                                                                                              // wg_sum[new_offset]
                                       Mut("wg_sum_at_offset"_(type),
                                           invoke(Add(), "wg_sum_at_offset"_(type), "wg_sum_old"_(type), type), false),

                                       "wg_sum"_(Array(type, Local))[local_i] = "wg_sum_at_offset"_(type) // a[i] = bi
                                   },
                                   {}),
                              Mut("offset"_(Int), invoke(Div(), "offset"_(Int), 2_(Int), Int), false) // offset /= 2
                          }),
                    let("group_id") = invoke(GpuGroupIdxX(), Int),
                    Cond({invoke(LogicEq(), "local_i"_(Int), 0_(Int), Bool)}, //
                         {
                             let("wg_sum_old_1") = "wg_sum"_(Array(type, Local))[local_i],
                             "sum"_(Array(type))["group_id"_(Int)] = "wg_sum_old_1"_(type),
                         },
                         {})
                    //
                };
              },
              empty);

  //

  std::cout << repr(copy) << std::endl;
  std::cout << repr(mul) << std::endl;
  std::cout << repr(add) << std::endl;
  std::cout << repr(triad) << std::endl;
  std::cout << repr(dot) << std::endl;

  auto entry = function("entry", {}, Unit)({ret(UnitConst())});
  return Program(entry, {copy, mul, add, triad, dot}, {});
}

TEST_CASE("BabelStream") {

  polyregion::compiler::initialise();

  auto p = mkStreamProgram("_double", Double, true);

  INFO(repr(p));
  auto c = polyregion::compiler::compile(p, {polyregion::compiler::Target::Source_C_OpenCL1_1, "gfx1012"},
                                         polyregion::compiler::Opt::O3);

  //  auto c = polyregion::compiler::compile(p, {polyregion::compiler::Target::Object_LLVM_AMDGCN, "gfx1012"},
  //                                         polyregion::compiler::Opt::O3);

  //  auto c = polyregion::compiler::compile(p, {polyregion::compiler::Target::Object_LLVM_NVPTX64, "sm_60"},
  //                                         polyregion::compiler::Opt::O3);

  //      auto c = polyregion::compiler::compile(p, {polyregion::compiler::Target::Object_LLVM_x86_64, "znver3"},
  //                                             polyregion::compiler::Opt::O3);

  std::cout << c << std::endl;
  //
  std::string image(c.binary->data(), c.binary->size());

  auto platform = runtime::Platform::of(runtime::Backend::OpenCL);
  const auto relTolerance = 0.008f;
  for (auto &d : platform->enumerate()) {
    polyregion::stream::runStream<double>(
        runtime::Type::Double64, //
        "_double",               //
        33554432,                //
        100,                     //
        256,                     //
        std::move(d),            //
        image,                   //
        true,                    //
        [](auto actual, auto tolerance) {
          if (actual >= tolerance) {
            std::cerr << "Tolerance (" << tolerance << ") exceeded for value " << actual << std::endl;
          }
        }, //
        [=](auto actual) {
          if (actual >= relTolerance) {
            std::cerr << "Tolerance (" << relTolerance << ") exceeded for value " << actual << std::endl;
          }
        } //
    );
  }

  CHECK(c.messages == "");
  CHECK(c.binary != std::nullopt);

  //
}