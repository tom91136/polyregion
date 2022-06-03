#include "ast.h"
#include "backend/llvmc.h"
#include "compiler.h"
#include "object_platform.h"
#include "utils.hpp"
#include "variants.hpp"

#include <iostream>

int main(int argc, char *argv[]) {

  using namespace polyregion;

  compiler::initialise();

  // "/home/tom/Nextcloud/vecAdd-cuda-nvptx64-nvidia-cuda-sm_61.ll"
  //"/home/tom/Nextcloud/vecAdd-cuda-nvptx64-nvidia-cuda-sm_61.ll"
  //  auto modExt = llvm::parseIRFile("/home/tom/Nextcloud/vecAdd-hip-amdgcn-amd-amdhsa-gfx906.ll", Err, *ctx,
  //                                  [&](llvm::StringRef DataLayoutTargetTriple) -> llvm::Optional<std::string> {
  //                                    std::cout << DataLayoutTargetTriple.str() << std::endl;
  //                                    return {};
  //                                  });
  //
  //

  using namespace polyast::dsl;
  auto fn = function("foo", {"xs"_(Array(Int)), "x"_(Int)}, Unit)({
      let("gid") = invoke(Fn0::GpuGlobalIdxX(), Int),
      let("xs@gid") = "xs"_(Array(Int))["gid"_(Int)],
      let("result") = invoke(Fn2::Add(), "xs@gid"_(Int), "gid"_(Int), Int),
      let("resultX2") = invoke(Fn2::Mul(), "result"_(Int), "x"_(Int), Int),
      "xs"_(Array(Int))["gid"_(Int)] = "resultX2"_(Int),
      ret(),
  });

  auto p = program(fn, {}, {});
  std::cout << repr(p) << std::endl;
//  compiler::Options options{compiler::Target::Object_LLVM_AMDGCN, "gfx906"};
    compiler::Options options{compiler::Target::Object_LLVM_NVPTX64, "sm_61"};
  auto c = compiler::compile(p, options, compiler::Opt::O3);
  std::cout << c << std::endl;

  if (c.binary) {
    std::ofstream outfile("bin_" + (options.arch.empty() ? "no_arch" : options.arch) + ".so",
                          std::ios::out | std::ios::binary | std::ios::trunc);
    outfile.write(c.binary->data(), c.binary->size());
    outfile.close();
  }

  auto simple =
      program(function("twice", {"x"_(Int)}, Int)({ret(invoke(Fn2::Mul(), "x"_(Int), 2_(Int), Int))}), {}, {});
  std::cout << repr(simple) << std::endl;
  auto c2 = compiler::compile(simple, {compiler::Target::Object_LLVM_x86_64, {}}, compiler::Opt::O3);
  std::cout << c2 << std::endl;
  if (c2.binary) {
    runtime::object::RelocatableDevice d;
    auto str = std::string(c2.binary->begin(), c2.binary->end());
    d.loadModule("", str);

    int a = 42;
    int actual = 0;
    std::vector<runtime::TypedPointer> args = {{runtime::Type::Int32, &a}, {runtime::Type::Int32, &actual}};
    std::vector<runtime::Type> types(args.size());
    std::vector<void *> pointers(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
      types[i] = args[i].first;
      pointers[i] = args[i].second;
    }

    d.createQueue()->enqueueInvokeAsync("", "twice", types, pointers, {}, {});

    std::cout << actual << "\n";
  }

  std::cout << "Done!" << std::endl;
  return EXIT_SUCCESS;
}
