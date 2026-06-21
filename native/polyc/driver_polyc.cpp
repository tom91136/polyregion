#include "driver_polyc.h"

#include <fstream>
#include <string>
#include <vector>

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/io.hpp"
#include "polyregion/polypass.h"

#include "ast.h"
#include "compiler.h"
#include "fire.hpp"
#include "generated/polypass_symbols.h"
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

static std::string targetDescription = [] {
  std::string targets =
      compiletime::TargetSpec::registry() ^ mk_string("\n\t", "\n\t", "", [](const compiletime::TargetSpec &s) {
        std::string names(s.canonical);
        for (const auto &a : s.aliases)
          names += std::string("|") + std::string(a);
        return names + ": \t" + std::string(magic_enum::enum_name(s.codegen)) + " via " + std::string(magic_enum::enum_name(s.runtime));
      });
  std::string env = std::string("\n\nEnvironment:\n  ") + polypass::abi::EnvPlugins +
                    " - PATH-separated list of PolyPass plugin paths (libpolypass.so / polypass.js). "
                    "Overrides the bundled default plugin.";
  return "PolyAST to object code compiler.\nSupported targets:" + targets + env;
}();

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
               std::string passes = //
               fire::arg({"-p", "--passes",
                          "PolyPass pipeline spec: `;`-separated `Name(k=v,k=v)` steps. "
                          "Empty selects the default."},
                         ""),
               bool hostMirroring = //
               fire::arg({"--host-mirroring", "Compile only the generated Host-affinity functions and emit LLVM bitcode"}),
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
                 default: fmt::print(stderr, "Unknown optimisation level: {}\n", rawOpt); return EXIT_FAILURE;
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
                 fmt::print(stderr, "[POLYC] Compiling program:\n=================\n{}\n=================\n", repr(program));

                 auto compilation = compiler::compile(program, compiler::Options{target, rawArch, passes, hostMirroring}, opt);
                 if (verbose) fmt::print(stderr, "{}\n", repr(compilation));
                 if (!compilation.messages.empty()) fmt::print(stderr, "{}\n", compilation.messages);
                 auto resultBytes = compileresult_to_msgpack(compilation);
                 if (out == "-") {
                   std::freopen(nullptr, "wb", stdout);
                   std::fwrite(resultBytes.data(), resultBytes.size(), sizeof(std::byte), stdout);
                 } else {
                   std::ofstream outStream(out, std::ios_base::binary | std::ios_base::out | std::ios_base::app);
                   outStream.write(reinterpret_cast<const char *>(resultBytes.data()), resultBytes.size());
                 }
               } catch (const std::exception &e) {
                 fmt::print(stderr, "[POLYC] {}\n", e.what());
                 return EXIT_FAILURE;
               }
               return EXIT_SUCCESS;
             },
             [&]() {
               fmt::print(stderr, "Unknown target: {}\n", rawTarget);
               return EXIT_FAILURE;
             });
}

int polyregion::polyc(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, fired_main, targetDescription);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, targetDescription));
  return FIRE_EXTRACT_1_PAD_(fired_main, targetDescription)();
}
