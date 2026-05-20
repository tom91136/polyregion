#include "driver_polyc.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/io.hpp"

#include "ast.h"
#include "compiler.h"
#include "fire.hpp"
#include "polyast_codec.h"

using namespace polyregion;
using namespace aspartame;

// See https://stackoverflow.com/a/39758021/896997
template <typename T = std::byte> std::vector<T> readFromStdIn() {
  std::freopen(nullptr, "rb", stdin);
  if (std::ferror(stdin)) throw std::runtime_error(std::strerror(errno));
  std::size_t len;
  std::array<T, 1024> buf{};
  std::vector<T> input;
  while ((len = std::fread(buf.data(), sizeof(buf[0]), buf.size(), stdin)) > 0) {
    if (std::ferror(stdin) && !std::feof(stdin)) throw std::runtime_error(std::strerror(errno));
    input.insert(input.end(), buf.data(), buf.data() + len); // append to vector
  }
  return input;
}

static std::string targetDescription =                                      //
    "PolyAST to object code compiler.\nSupported targets:" +                //
    (compiletime::TargetSpec::registry()                                    //
     ^ mk_string("\n\t", "\n\t", "", [](const compiletime::TargetSpec &s) { //
         std::string names(s.canonical);                                    //
         for (const auto &a : s.aliases)
           names += std::string("|") + std::string(a);                                     //
         return names + ": \t" + std::string(magic_enum::enum_name(s.codegen)) + " via " + //
                std::string(magic_enum::enum_name(s.runtime));                             //
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
  return compiletime::TargetSpec::findByName(rawTarget) ^
         fold(
             [&](const compiletime::TargetSpec &spec) {
               const auto target = spec.codegen;
               compiletime::OptLevel opt;
               switch (rawOpt) {
                 case 0: opt = compiletime::OptLevel::O0; break;
                 case 1: opt = compiletime::OptLevel::O1; break;
                 case 2: opt = compiletime::OptLevel::O2; break;
                 case 3: opt = compiletime::OptLevel::O3; break;
                 case 4: opt = compiletime::OptLevel::Ofast; break;
                 default: std::cerr << "Unknown optimisation level: " << std::to_string(rawOpt) << std::endl; return EXIT_FAILURE;
               }

               auto bytes = maybePath ? polyregion::read_struct<uint8_t>(maybePath.value()) : readFromStdIn<uint8_t>();
               auto isJson = bytes ^ forall([](auto c) { return c <= 127; });

               try {
                 auto program = [&]() {
                   if (isJson) {
                     auto raw = nlohmann::json::parse(bytes.begin(), bytes.end());
                     return polyast::program_from_json(polyast::hashed_from_json(raw));
                   }
                   return polyast::hashed_program_from_msgpack(bytes.data(), bytes.data() + bytes.size());
                 }();

                 compiler::initialise();
                 std::cout << "[POLYC] Compiling program:\n";
                 std::cout << "=================" << std::endl;
                 std::cout << repr(program) << "\n";
                 std::cout << "=================" << std::endl;

                 auto compilation = compiler::compile(program, compiler::Options{target, rawArch}, opt);
                 if (verbose) std::cerr << repr(compilation) << std::endl;
                 if (!compilation.messages.empty()) std::cerr << compilation.messages << std::endl;
                 auto resultBytes = compileresult_to_msgpack(compilation);
                 if (out == "-") {
                   std::freopen(nullptr, "wb", stdout);
                   std::fwrite(resultBytes.data(), resultBytes.size(), sizeof(std::byte), stdout);
                 } else {
                   std::ofstream outStream(out, std::ios_base::binary | std::ios_base::out | std::ios_base::app);
                   outStream.write(reinterpret_cast<const char *>(resultBytes.data()), resultBytes.size());
                 }
               } catch (const std::exception &e) {
                 std::cerr << "[POLYC] " << e.what() << std::endl;
                 return EXIT_FAILURE;
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
