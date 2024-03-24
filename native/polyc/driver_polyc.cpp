#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"

#include "ast.h"
#include "compiler.h"
#include "driver_polyc.h"
#include "fire.hpp"
#include "polyast_codec.h"
#include "polyregion/io.hpp"
#include "polyregion/utils.hpp"

using namespace polyregion;
using namespace aspartame;

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

static std::string targetDescription =
    "PolyAST to object code compiler.\nSupported targets:" //
    + (compiletime::targets() ^
       group_map_reduce([](auto, auto t) { return t; }, [](auto k, auto) { return k; }, [](auto l, auto r) { return l + "|" + r; }) //
       ^ to_vector()                                                                                                                //
       ^ sort_by([](auto k, auto) { return k; })                                                                                    //
       ^ mk_string("\n\t", "\n\t", "", [](compiletime::Target k, auto v) {
           switch (k) {
             case compiletime::Target::Object_LLVM_HOST: return v + ": \tObject (LLVM, HOST)";
             case compiletime::Target::Object_LLVM_x86_64: return v + ": \tObject (LLVM, x86_64)";
             case compiletime::Target::Object_LLVM_AArch64: return v + ": \tObject (LLVM, AArch64)";
             case compiletime::Target::Object_LLVM_ARM: return v + ": \tObject (LLVM, ARM)";
             case compiletime::Target::Object_LLVM_NVPTX64: return v + ": \tObject (LLVM, NVPTX64)";
             case compiletime::Target::Object_LLVM_AMDGCN: return v + ": \tObject (LLVM, AMDGCN)";
             case compiletime::Target::Object_LLVM_SPIRV32: return v + ": \tObject (LLVM, SPIRV32)";
             case compiletime::Target::Object_LLVM_SPIRV64: return v + ": \tObject (LLVM, SPIRV64)";
             case compiletime::Target::Source_C_C11: return v + ": \tSource (C, C11)";
             case compiletime::Target::Source_C_OpenCL1_1: return v + ": \tSource (C, OpenCL1_1)";
             case compiletime::Target::Source_C_Metal1_0: return v + ": \tSource (C, Metal1_0)";
           }
         }));

int fired_main(fire::optional<std::string> maybePath = // NOLINT(*-unnecessary-value-param)
               fire::arg({0, "Input source, in either JSON or MessagePack format. Format is auto detected based on ASCII ranges."}),
               std::string out = // NOLINT(*-unnecessary-value-param)
               fire::arg({"-o", "--out", "Output binary name"}, "-"),
               std::string rawTarget = //
               fire::arg({"-m", "--target", "Output target, see program description for a list of supported targets"}, "host"),
               std::string rawArch = //
               fire::arg({"-a", "--arch", "Target architecture (e.g sm_35, gfx906, skylake)"}, "native"),
               int rawOpt = //
               fire::arg({"-O", "--opt", "Optimisation level, from 0 (no optimisation) to 4 (unsafe optimisations)"}, 3),
               bool verbose = fire::arg({"--verbose", "-v", "Verbose output"})

) {
  return compiletime::targets() ^ get(rawTarget ^ to_lower()) ^
         fold(
             [&](const compiletime::Target &target) { //
               compiletime::OptLevel opt;
               switch (rawOpt) {
                 case 0: opt = compiletime::OptLevel::O0; break;
                 case 1: opt = compiletime::OptLevel::O1; break;
                 case 2: opt = compiletime::OptLevel::O2; break;
                 case 3: opt = compiletime::OptLevel::O3; break;
                 case 4: opt = compiletime::OptLevel::Ofast; break;
                 default: std::cerr << "Unknown optimisation level: " << std::to_string(rawOpt) << std::endl; return EXIT_FAILURE;
               }

               auto bytes = maybePath ? polyregion::read_struct<std::byte>(maybePath.value()) : readFromStdIn();
               auto isJson = bytes ^ forall([](auto c) { return c <= std::byte{127}; });

               try {
                 auto raw =
                     isJson ? nlohmann::json::parse(bytes.begin(), bytes.end()) : nlohmann::json::from_msgpack(bytes.begin(), bytes.end());
                 auto program = polyast::program_from_json(polyast::hashed_from_json(raw));

                 compiler::initialise();
                 std::cout << "[POLYC] Compiling program:\n";
                 for (auto &def : program.defs) {
                   std::cout << def << "\n";
                 }
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
                 std::cerr << "Unable to parse packed ast:" << e.what() << std::endl;
               }
               return EXIT_SUCCESS;
             },
             [&]() {
               std::cerr << "Unknown target: " << rawTarget << std::endl;
               return EXIT_FAILURE;
             });
}

int polyregion::polyc(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, fired_main, targetDescription);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, targetDescription));
  return FIRE_EXTRACT_1_PAD_(fired_main, targetDescription)();
}
