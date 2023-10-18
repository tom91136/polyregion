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

static std::unordered_map<std::string, polyast::Target> Targets = {
    {"host", polyast::Target::Object_LLVM_HOST},
    {"x86_64", polyast::Target::Object_LLVM_x86_64},
    {"aarch64", polyast::Target::Object_LLVM_AArch64},
    {"arm", polyast::Target::Object_LLVM_ARM},

    {"nvptx64", polyast::Target::Object_LLVM_NVPTX64},
    {"amdgcn", polyast::Target::Object_LLVM_AMDGCN},
    {"spirv32", polyast::Target::Object_LLVM_SPIRV32},
    {"spirv64", polyast::Target::Object_LLVM_SPIRV64},

    {"c11", polyast::Target::Source_C_C11},
    {"opencl1_1", polyast::Target::Source_C_OpenCL1_1},
    {"metal1_0", polyast::Target::Source_C_Metal1_0},
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

int fired_main(fire::optional<std::string> maybePath = fire::arg({0, "Input source, in either JSON or MessagePack format"}),
               std::string out = fire::arg({"-o", "--out", "Output binary"}, "-"),
               std::string rawTarget = fire::arg({"--target", "Target"}, "host"),
               std::string rawArch = fire::arg({"--arch", "Target architecture"}, "native"),
               int rawOpt = fire::arg({"--opt", "Optimisation"}, 3), bool verbose = fire::arg({"--verbose", "-v", "Verbose output"})

) {
  std::transform(rawTarget.begin(), rawTarget.end(), rawTarget.begin(), [](auto &c) { return std::tolower(c); });

  auto targetIt = Targets.find(rawTarget);
  if (targetIt == Targets.end()) {
    std::cerr << "Unknown target: " << rawTarget << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto target = targetIt->second;

  polyast::OptLevel opt;
  switch (rawOpt) {
    case 0: opt = polyast::OptLevel::O0; break;
    case 1: opt = polyast::OptLevel::O1; break;
    case 2: opt = polyast::OptLevel::O2; break;
    case 3: opt = polyast::OptLevel::O3; break;
    case 4: opt = polyast::OptLevel::Ofast; break;
    default: std::cerr << "Unknown optimisation level: " << std::to_string(rawOpt) << std::endl; std::exit(EXIT_FAILURE);
  }

  auto bytes = maybePath ? polyregion::read_struct<std::byte>(maybePath.value()) : readFromStdIn();
  auto isJson = std::all_of(bytes.begin(), bytes.end(), [](auto &c) { return c <= std::byte{127}; });

  try {
    auto raw = isJson ? nlohmann::json::parse(bytes.begin(), bytes.end()) : nlohmann::json::from_msgpack(bytes.begin(), bytes.end());
    auto program = polyast::program_from_json(polyast::hashed_from_json(raw));

    compiler::initialise();
    auto compilation = compiler::compile(program, compiler::Options{target, rawArch}, opt);
    if (verbose) std::cerr << repr(compilation) << std::endl;
    if (!compilation.messages.empty()) std::cerr << compilation.messages << std::endl;
    auto resultBytes = nlohmann::json::to_msgpack(compileresult_to_json(compilation));
    if (out == "-") {
      std::freopen(nullptr, "wb", stdout);
      std::fwrite(resultBytes.data(), resultBytes.size(), sizeof(std::byte), stdout);
    } else {
      std::ofstream outStream(out, std::ios_base::binary | std::ios_base::out | std::ios_base::app);
      outStream.write(reinterpret_cast<const char *>(resultBytes.data()), resultBytes.size());
    }
  } catch (nlohmann::json::exception &e) {
    throw std::logic_error("Unable to parse packed ast:" + std::string(e.what()));
  }
  return EXIT_SUCCESS;
}

FIRE(fired_main)
