#include <iostream>
#include <string>
#include <vector>

#include "ast.h"
#include "cl_platform.h"
#include "compiler.h"
#include "cuda_platform.h"
#include "hip_platform.h"
#include "hsa_platform.h"
#include "object_platform.h"
#include "runtime.h"
#include "utils.hpp"
#include <fstream>

void run() {

  using namespace polyregion;
  using namespace polyast::dsl;
  auto fn0 = function("lambda", {"xs"_(Array(Int)), "x"_(Int)}, Unit)({
      let("gid") = "x"_(Int), //  invoke(Fn0::GpuGlobalIdxX(), Int),
      let("xs_gid") = "xs"_(Array(Int))["gid"_(Int)],
      let("result") = invoke(Fn2::Add(), "xs_gid"_(Int), "gid"_(Int), Int),
      let("resultX2") = invoke(Fn2::Mul(), "result"_(Int), "x"_(Int), Int),
      "xs"_(Array(Int))["gid"_(Int)] = "resultX2"_(Int),
      ret(),
  });

  auto fn1 = function("lambda", {"xs"_(Array(Int)), "x"_(Int)}, Unit)({
      let("gid") = "x"_(Int), //  invoke(Fn0::GpuGlobalIdxX(), Int),
      let("xs_gid") = "xs"_(Array(Int))["gid"_(Int)],
//      let("resultX2_tan") = invoke(Fn1::Tanh(), "xs_gid"_(Int), Int),
      "xs"_(Array(Int))["gid"_(Int)] = "xs_gid"_(Int),
      ret(),
  });

  auto prog0 = program(fn0, {}, {});
  auto prog1 = program(fn1, {}, {});

  compiler::initialise();
  runtime::init();

  using namespace polyregion::runtime;
  using namespace polyregion::runtime::object;
  using namespace polyregion::runtime::cuda;
  using namespace polyregion::runtime::hip;
  using namespace polyregion::runtime::hsa;
  using namespace polyregion::runtime::cl;

  std::vector<std::unique_ptr<Platform>> rts;

  try {
    rts.push_back(std::make_unique<RelocatablePlatform>());
  } catch (const std::exception &e) {
    std::cerr << "[REL] " << e.what() << std::endl;
  }

  //      try {
  //        rts.push_back(std::make_unique<CudaPlatform>());
  //      } catch (const std::exception &e) {
  //        std::cerr << "[CUDA] " << e.what() << std::endl;
  //      }
  //
  //      try {
  //        rts.push_back(std::make_unique<ClPlatform>());
  //      } catch (const std::exception &e) {
  //        std::cerr << "[OCL] " << e.what() << std::endl;
  //      }
  //
      try {
        rts.push_back(std::make_unique<HsaPlatform>());
      } catch (const std::exception &e) {
        std::cerr << "[HSA] " << e.what() << std::endl;
      }
  //
    try {
      rts.push_back(std::make_unique<HipPlatform>());
    } catch (const std::exception &e) {
      std::cerr << "[HIP] " << e.what() << std::endl;
    }

  static std::vector<int> xs;

  for (auto &rt : rts) {
    std::cout << "RT=" << rt->name() << std::endl;
    auto devices = rt->enumerate();
    std::cout << "Found " << devices.size() << " devices" << std::endl;

    for (auto &d : devices) {
      auto features = polyregion::mk_string<std::string>(
          d->features(), [](auto &&s) { return s; }, ",");

      std::cout << "[Device " << d->id() << "]"
                << "name: `" << d->name() << "`; features: " << features << std::endl;

      xs.resize(4);
      std::fill(xs.begin(), xs.end(), 7);

      compiler::Options options;
      if (rt->name() == "CUDA") {
        options = {compiler::Target::Object_LLVM_NVPTX64, "sm_61"};
      } else if (rt->name() == "OpenCL") {
        options = {compiler::Target::Source_C_OpenCL1_1, {}};
      } else if (rt->name() == "HIP") {
        options = {compiler::Target::Object_LLVM_AMDGCN, "gfx1012"};
      } else if (rt->name() == "HSA") {
        options = {compiler::Target::Object_LLVM_AMDGCN, "gfx1012"};
      } else if (rt->name() == "CPU (RelocatableObject)") {
        options = {compiler::Target::Object_LLVM_HOST, {"native"}};
      } else {
        throw std::logic_error("?");
      }

      auto c0 = compiler::compile(prog0, options, compiler::Opt::O1);
      auto c1 = compiler::compile(prog1, options, compiler::Opt::O1);
      std::cout << c0 << std::endl;
      std::cout << c1 << std::endl;
      if (c0.binary) {

        if (options.target == compiler::Target::Source_C_OpenCL1_1) {
          std::ofstream outfile("bin_" + (options.arch.empty() ? "no_arch" : options.arch) + ".cl",
                                std::ios::out | std::ios::trunc);
          outfile.write(c0.binary->data(), c0.binary->size());
          outfile.close();
        } else {
          std::ofstream outfile("bin_" + (options.arch.empty() ? "no_arch" : options.arch) + ".so",
                                std::ios::out | std::ios::binary | std::ios::trunc);
          outfile.write(c0.binary->data(), c0.binary->size());
          outfile.close();
        }
      } else {
        std::cout << "No bin!" << std::endl;
      }

      std::string bin0(c0.binary->data(), c0.binary->size());
      std::string bin1(c1.binary->data(), c1.binary->size());
      try {
        for (int i = 0; i < 4; i++) {
          auto q1 = d->createQueue();

          if (!d->moduleLoaded("0")) {
            d->loadModule("0", bin0);
          }

          auto size = sizeof(decltype(xs)::value_type) * xs.size();
          auto ptr = d->malloc(size, Access::RW);
          q1->enqueueHostToDeviceAsync(xs.data(), ptr, size,
                                       [&]() { std::cout << "[" << i << "]  H->D ok" << std::endl; });

          int32_t x = 3;

          ArgBuffer buffer({
              {Type::Ptr, &ptr},
              {Type::Int32, &x},
              {Type::Void, nullptr},
          });

          q1->enqueueInvokeAsync("0", "lambda", buffer.types, buffer.data, {},
                                 [&]() { std::cout << "[" << i << "]  K 1 ok" << std::endl; });

          q1->enqueueDeviceToHostAsync(ptr, xs.data(), size, [&]() {
            std::cout << "[" << i << "]  D->H ok, r= "
                      << polyregion::mk_string<int>(
                             xs, [](auto x) { return std::to_string(x); }, ",")
                      << std::endl;
          });

          d->free(ptr);
        }

        for (int i = 0; i < 4; i++) {
          auto q1 = d->createQueue();

          if (!d->moduleLoaded("1")) {
            d->loadModule("1", bin1);
          }
          auto size = sizeof(decltype(xs)::value_type) * xs.size();
          auto ptr = d->malloc(size, Access::RW);
          q1->enqueueHostToDeviceAsync(xs.data(), ptr, size,
                                       [&]() { std::cout << "[" << i << "]  H->D ok" << std::endl; });

          int32_t x = 3;


          ArgBuffer buffer({
              {Type::Ptr, &ptr},
              {Type::Int32, &x},
              {Type::Void, nullptr},
          });

          x = 0;

          q1->enqueueInvokeAsync("1", "lambda", buffer.types, buffer.data, {},
                                 [&]() { std::cout << "[" << i << "]  K 2 ok" << std::endl; });
          q1->enqueueDeviceToHostAsync(ptr, xs.data(), size, [&]() {
            std::cout << "[" << i << "]  D->H ok, r= "
                      << polyregion::mk_string<int>(
                             xs, [](auto x) { return std::to_string(x); }, ",")
                      << std::endl;
          });
          d->free(ptr);
        }

      } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
      }

      std::cout << d->id() << " = Done" << std::endl;
    }
  }

  std::cout << "Done" << std::endl;
}

int main(int argc, char *argv[]) {

  run();

  return EXIT_SUCCESS;
}