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
  auto fn = function("foo", {"xs"_(Array(Int)), "x"_(Int)}, Unit)({
      let("gid") = invoke(Fn0::GpuGlobalIdxX(), Int),
      let("xs_gid") = "xs"_(Array(Int))["gid"_(Int)],
      let("result") = invoke(Fn2::Add(), "xs_gid"_(Int), "gid"_(Int), Int),
      let("resultX2") = invoke(Fn2::Mul(), "result"_(Int), "x"_(Int), Int),
      "xs"_(Array(Int))["gid"_(Int)] = "resultX2"_(Int),
      ret(),
  });

  auto prog = program(fn, {}, {});
  compiler::initialise();

  using namespace polyregion::runtime;
  using namespace polyregion::runtime::object;
  using namespace polyregion::runtime::cuda;
  using namespace polyregion::runtime::hip;
  using namespace polyregion::runtime::hsa;
  using namespace polyregion::runtime::cl;

  std::vector<std::unique_ptr<Platform>> rts;

  //    try {
  //      rts.push_back(std::make_unique<RelocatablePlatform>());
  //    } catch (const std::exception &e) {
  //      std::cerr << "[REL] " << e.what() << std::endl;
  //    }

  //    try {
  //      rts.push_back(std::make_unique<CudaPlatform>());
  //    } catch (const std::exception &e) {
  //      std::cerr << "[CUDA] " << e.what() << std::endl;
  //    }
  //
  //    try {
  //      rts.push_back(std::make_unique<ClPlatform>());
  //    } catch (const std::exception &e) {
  //      std::cerr << "[OCL] " << e.what() << std::endl;
  //    }

  //  try {
  //    rts.push_back(std::make_unique<HsaPlatform>());
  //  } catch (const std::exception &e) {
  //    std::cerr << "[HSA] " << e.what() << std::endl;
  //  }

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
      std::cout << d->id() << " = " << d->name() << std::endl;

      xs.resize(4);
      std::fill(xs.begin(), xs.end(), 7);

      compiler::Options options;
      if (rt->name() == "CUDA") {
        options = {compiler::Target::Object_LLVM_NVPTX64, "sm_61"};
      } else if (rt->name() == "OpenCL") {
        options = {compiler::Target::Source_C_OpenCL1_1, {}};
      } else if (rt->name() == "HIP") {
        options = {compiler::Target::Object_LLVM_AMDGCN, "gfx906"};
      } else if (rt->name() == "CPU (RelocatableObject)") {
        options = {compiler::Target::Object_LLVM_HOST, {}};
      } else {
        throw std::logic_error("?");
      }

      auto c = compiler::compile(prog, options, compiler::Opt::O3);
      std::cout << c << std::endl;
      if (c.binary) {

        if (options.target == compiler::Target::Source_C_OpenCL1_1) {
          std::ofstream outfile("bin_" + (options.arch.empty() ? "no_arch" : options.arch) + ".cl",
                                std::ios::out | std::ios::trunc);
          outfile.write(c.binary->data(), c.binary->size());
          outfile.close();
        } else {
          std::ofstream outfile("bin_" + (options.arch.empty() ? "no_arch" : options.arch) + ".so",
                                std::ios::out | std::ios::binary | std::ios::trunc);
          outfile.write(c.binary->data(), c.binary->size());
          outfile.close();
        }
      } else {
        std::cout << "No bin!" << std::endl;
      }

      try {
        for (int i = 0; i < 2; i++) {
          auto q1 = d->createQueue();
          if (!d->moduleLoaded("a")) {
            d->loadModule("a", std::string(c.binary->data(), c.binary->size()));
          }
          auto size = sizeof(decltype(xs)::value_type) * xs.size();
          auto ptr = d->malloc(size, Access::RW);
          q1->enqueueHostToDeviceAsync(xs.data(), ptr, size,
                                       [&]() { std::cout << "[" << i << "]  H->D ok" << std::endl; });

          int32_t x = 4;

          std::vector<Type> types{Type::Ptr, Type::Int32, Type::Void};
          std::vector<void *> args{&ptr, &x, nullptr};
          q1->enqueueInvokeAsync("a", "foo", types, args, {},
                                 [&]() { std::cout << "[" << i << "]  K 1 ok" << std::endl; });

          x = 5;

          q1->enqueueInvokeAsync("a", "foo", types, args, {},
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