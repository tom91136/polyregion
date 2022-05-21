#include <iostream>
#include <string>
#include <vector>

#include "utils.hpp"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "cl_runtime.h"
#include "cuda_runtime.h"
#include "hip_runtime.h"
#include "object_runtime.h"
#include "runtime.h"

static const char *clKernelSource{R"CLC(
__kernel void lambda(__global int* array, int x){
  int tid = get_global_id(0);
  array[tid] = array[tid] + tid + x;
}
)CLC"};

static const char *ptxKernelSource{R"PTX(
.version 7.4
.target sm_61
.address_size 64

        // .globl       lambda

.visible .entry lambda(
        .param .u64 lambda_param_0,
        .param .u32 lambda_param_1
)
{
        .reg .b32       %r<6>;
        .reg .b64       %rd<5>;


        ld.param.u64    %rd1, [lambda_param_0];
        ld.param.u32    %r1, [lambda_param_1];
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r2, %ctaid.x;
        mul.wide.s32    %rd3, %r2, 4;
        add.s64         %rd4, %rd2, %rd3;
        ld.global.u32   %r3, [%rd4];
        add.s32         %r4, %r2, %r1;
        add.s32         %r5, %r4, %r3;
        st.global.u32   [%rd4], %r5;
        ret;

}
)PTX"};

static const char *hsacoKernelSource{R"PTX(

)PTX"};

void run() {

  using namespace polyregion::runtime;
  using namespace polyregion::runtime::object;
  using namespace polyregion::runtime::cuda;
  using namespace polyregion::runtime::hip;
  using namespace polyregion::runtime::cl;

  std::vector<std::unique_ptr<Runtime>> rts;

  try {
    rts.push_back(std::make_unique<CudaRuntime>());
  } catch (const std::exception &e) {
    std::cerr << "[CUDA] " << e.what() << std::endl;
  }

    try {
      rts.push_back(std::make_unique<ClRuntime>());
    } catch (const std::exception &e) {
      std::cerr << "[OCL] " << e.what() << std::endl;
    }

  try {
    rts.push_back(std::make_unique<HipRuntime>());
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

      if (d->name().find("TITAN") != std::string::npos) {
        //        continue ;
      }

      xs.resize(4);
      std::fill(xs.begin(), xs.end(), 7);

      auto size = sizeof(decltype(xs)::value_type) * xs.size();
      auto ptr = d->malloc(size, Access::RW);

      std::string src;
      if (rt->name() == "CUDA") {
        src = ptxKernelSource;
      } else if (rt->name() == "OpenCL") {
        src = clKernelSource;
      } else if (rt->name() == "HIP") {
        src = ptxKernelSource;
      } else {
        throw std::logic_error("?");
      }

      d->loadModule("a", src);

      d->enqueueHostToDeviceAsync(xs.data(), ptr, size, []() { std::cout << "  H->D ok" << std::endl; });

      int32_t x = 4;
      d->enqueueInvokeAsync("a", "lambda", {{Type::Ptr, &ptr}, {Type::Int32, &x}}, {Type::Void, nullptr}, {},
                            []() { std::cout << "  K 1 ok" << std::endl; });

      x = 5;

      d->enqueueInvokeAsync("a", "lambda", {{Type::Ptr, &ptr}, {Type::Int32, &x}}, {Type::Void, nullptr}, {},
                            []() { std::cout << "  K 2 ok" << std::endl; });
      d->enqueueDeviceToHostAsync(ptr, xs.data(), size, [&]() {
        std::cout << "  D->H ok, r= "
                  << polyregion::mk_string<int>(
                         xs, [](auto x) { return std::to_string(x); }, ",")
                  << std::endl;
      });

      std::cout << d->id() << " = Done" << std::endl;

      d->free(ptr);
    }
  }

  std::cout << "Done" << std::endl;
}

int main(int argc, char *argv[]) {

  run();

  return EXIT_SUCCESS;
}