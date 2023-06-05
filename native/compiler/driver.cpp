#include <iostream>
#include <string>
#include <vector>

#include "ast.h"
#include "compiler.h"
#include "fire.hpp"
#include "io.hpp"
#include "polyast_codec.h"
#include "utils.hpp"
#include <fstream>

using namespace polyregion;
using namespace polyast::dsl;
using namespace polyast::Intr;

static std::unordered_map<std::string, compiler::Target> table = {
    {"host", compiler::Target::Object_LLVM_HOST},
    {"x86_64", compiler::Target::Object_LLVM_x86_64},
    {"aarch64", compiler::Target::Object_LLVM_AArch64},
    {"arm", compiler::Target::Object_LLVM_ARM},

    {"nvptx64", compiler::Target::Object_LLVM_NVPTX64},
    {"amdgcn", compiler::Target::Object_LLVM_AMDGCN},
    {"spirv32", compiler::Target::Object_LLVM_SPIRV32},
    {"spirv64", compiler::Target::Object_LLVM_SPIRV64},

    {"c11", compiler::Target::Source_C_C11},
    {"opencl1_1", compiler::Target::Source_C_OpenCL1_1},
    {"metal1_0", compiler::Target::Source_C_Metal1_0},
};

// See https://stackoverflow.com/a/39758021/896997
std::vector<std::byte> readFromStdIn() {
  std::freopen(nullptr, "rb", stdin);
  if (std::ferror(stdin)) throw std::runtime_error(std::strerror(errno));
  std::size_t len;
  std::array<std::byte, 1024> buf{};
  std::vector<std::byte> input;
  while ((len = std::fread(buf.data(), sizeof(buf[0]), buf.size(), stdin)) > 0) {
    if (std::ferror(stdin) && !std::feof(stdin)) throw std::runtime_error(std::strerror(errno));
    input.insert(input.end(), buf.data(), buf.data() + len); // append to vector
  }
  return input;
}

int fired_main(fire::optional<std::string> maybePath = fire::arg({0, "<source>", "Input source, in either JSON or MessagePack format"}),
               std::string out = fire::arg({1, "<output>", "Output binary"}, "-"),
               std::string target = fire::arg({"-target", "<target>", "Target"}, "host"),
               std::string arch = fire::arg({"-arch", "<arch>", "Target architecture"}, "native"),
               int opt = fire::arg({"-opt", "<opt>", "Optimisation"}, 3),
               bool verbose = fire::arg({"-v", "<verbose>", "Verbose output"}, false)

) {

  auto bytes = maybePath ? polyregion::read_struct<std::byte>(maybePath.value()) : readFromStdIn();
  auto isJson = std::all_of(bytes.begin(), bytes.end(), [](auto &c) { return c <= std::byte{127}; });

  try {
    auto raw = isJson ? nlohmann::json::parse(bytes.begin(), bytes.end()) : nlohmann::json::from_msgpack(bytes.begin(), bytes.end());
    auto hash = polyast::hashed_from_json(raw);

    auto program = polyast::program_from_json(raw);

    auto compilation = compiler::compile(program, compiler::Options{compiler::Target::Object_LLVM_SPIRV64, ""}, compiler::Opt::O3);

    if (verbose) {
      std::cerr << compilation << std::endl;
    }
    if (!compilation.messages.empty()) {
      std::cerr << compilation.messages << std::endl;
    }
    if (compilation.binary) {
      if (out == "-") {
        std::freopen(nullptr, "wb", stdout);
        std::fwrite(compilation.binary->data(), compilation.binary->size(), sizeof(std::byte), stdout);
      } else {
        std::ofstream outStream(out, std::ios_base::binary | std::ios_base::out | std::ios_base::app);
        outStream.write(compilation.binary->data(), compilation.binary->size());
      }
    }

  } catch (nlohmann::json::exception &e) {
    throw std::logic_error("Unable to parse packed ast:" + std::string(e.what()));
  }

  std::cout << maybePath.value_or("???") << " = " << out << "";
  return EXIT_SUCCESS;
}

FIRE(fired_main)
