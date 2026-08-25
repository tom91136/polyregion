#include "driver_polyc.h"

#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/unordered_map.hpp"
#include "aspartame/vector.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/package_emit.hpp"
#include "polyregion/io.hpp"
#include "polyregion/polypass.h"

#include "ast.h"
#include "compiler.h"
#include "fire.hpp"
#include "generated/polypass_symbols.h"
#include "polyast_codec.h"

using namespace polyregion;
using namespace aspartame;

namespace {

std::optional<std::vector<uint8_t>> readBytes(const std::string &path) {
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "cannot read `" << path << "`\n";
    return {};
  }
  const auto bytes = (*buffer)->getBuffer();
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

int packageLinkMain(int argc, const char *argv[]) {
  std::set<std::string> capabilities;
  int first = 3;
  while (first < argc && std::string_view(argv[first]).starts_with("--capability=")) {
    capabilities.emplace(std::string_view(argv[first]).substr(std::string_view("--capability=").size()));
    ++first;
  }
  if (argc - first < 3) {
    llvm::errs() << "usage: polyc package link [--capability=<name>]... <interface.polyast> <output-root> "
                    "<program.polyast>...\n";
    return 2;
  }
  try {
    const auto interfaceBytes = readBytes(argv[first]);
    if (!interfaceBytes) return 3;
    const std::vector<std::string> paths(argv + first + 2, argv + argc);
    const auto programs = traverse(paths, [&](const auto &path) -> std::optional<polyast::Program> {
      const auto bytes = readBytes(path);
      return bytes ^ map([](const auto &value) { return polyast::hashed_program_from_msgpack(value); });
    });
    if (!programs) return 3;
    const auto interface = polyast::interface_from_msgpack(*interfaceBytes);
    const auto linked = compiler::linkPackage(
        polyast::PackageLinkRequest(interface, *programs, std::vector<std::string>(capabilities.begin(), capabilities.end())));
    if (!linked) {
      llvm::errs() << (linked.errors | mk_string("\n")) << '\n';
      return 4;
    }
    const auto published = polyfront::package::publishPackage(*linked.value, argv[first + 1]);
    if (!published) {
      llvm::errs() << (published.errors | mk_string("\n")) << '\n';
      return 4;
    }
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return 5;
  }
}

int packageCompileSymMain(int argc, const char *argv[]) {
  if (argc < 4) {
    llvm::errs() << "usage: polyc package compile-sym <request.msgpack> --out=<result.msgpack> --target=<host-target> "
                    "--arch=<host-arch> [--device=<target>@<arch>]... [--stack-depth=<n>]\n";
    return 2;
  }
  std::string output;
  std::string hostTargetName;
  std::string hostArch = "native";
  std::vector<std::pair<compiletime::Target, std::string>> devices;
  std::optional<int> stackDepth;
  for (int i = 4; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    const auto value = [&](const std::string_view prefix) -> std::optional<std::string_view> {
      return arg.starts_with(prefix) ? std::optional(arg.substr(prefix.size())) : std::nullopt;
    };
    if (const auto raw = value("--out=")) output = *raw;
    else if (const auto raw = value("--target=")) hostTargetName = *raw;
    else if (const auto raw = value("--arch=")) hostArch = *raw;
    else if (const auto raw = value("--stack-depth=")) {
      try {
        const auto parsed = std::stoi(std::string(*raw));
        if (parsed <= 0) throw std::invalid_argument("non-positive");
        stackDepth = parsed;
      } catch (const std::exception &) {
        llvm::errs() << "invalid stack depth: " << *raw << '\n';
        return 2;
      }
    } else if (const auto raw = value("--device=")) {
      const auto separator = raw->find('@');
      if (separator == std::string_view::npos) {
        llvm::errs() << "device target must be <target>@<arch>: " << *raw << '\n';
        return 2;
      }
      const auto target = compiletime::TargetSpec::findByName(raw->substr(0, separator));
      if (!target) {
        llvm::errs() << "unknown device target: " << raw->substr(0, separator) << '\n';
        return 2;
      }
      devices.emplace_back(target->codegen, raw->substr(separator + 1));
    } else {
      llvm::errs() << "unknown package compile-sym argument: " << arg << '\n';
      return 2;
    }
  }
  const auto hostTarget = compiletime::TargetSpec::findByName(hostTargetName);
  if (output.empty() || !hostTarget) {
    llvm::errs() << "package compile-sym requires a valid --out and --target\n";
    return 2;
  }
  try {
    const auto requestBytes = readBytes(argv[3]);
    if (!requestBytes) return 3;
    const auto request = polyast::packagesymrequest_from_msgpack(*requestBytes);
    const auto compiled = compiler::compilePackageSym(request, hostTarget->codegen, hostArch, devices, stackDepth);
    if (!compiled) {
      llvm::errs() << (compiled.errors | mk_string("\n")) << '\n';
      return 4;
    }
    const auto bytes = polyast::packagesymcompileresult_to_msgpack(*compiled.value);
    std::ofstream stream(output, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) {
      llvm::errs() << "cannot write `" << output << "`\n";
      return 3;
    }
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return 5;
  }
}

std::optional<int> packageMain(int argc, const char *argv[]) {
  if (argc < 2 || std::string_view(argv[1]) != "package") return {};
  if (argc >= 3 && std::string_view(argv[2]) == "link") return packageLinkMain(argc, argv);
  if (argc >= 3 && std::string_view(argv[2]) == "compile-sym") return packageCompileSymMain(argc, argv);
  llvm::errs() << "usage: polyc package <link|compile-sym> ...\n";
  return 2;
}

} // namespace

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
  std::string env = std::string("\n\nEnvironment:\n  ") + polypass::abi::EnvPlugins
                    + " - PATH-separated list of PolyPass plugin paths (libpolypass.so / polypass.js). "
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
               bool emitAst = //
               fire::arg({"--emit-ast", "Skip the backend and write the post-pass program as MessagePack polyAST"}),
               std::string exportName = //
               fire::arg({"--export", "With --emit-ast, narrow the export seed to this symbol so a pruning pass keeps only its "
                                      "closure; absent leaves every export seeded"},
                         ""),
               bool listExports = //
               fire::arg({"--list-exports", "Print the exported symbol names, one per line, and exit"}),
               bool verbose = fire::arg({"--verbose", "-v", "Verbose output"})

) {
  return compiletime::TargetSpec::findByName(rawTarget) //
         ^ fold(
             [&](const auto &spec) {
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
               auto isJson = bytes ^ forall([](const auto &c) { return c <= 127; });

               try {
                 auto program = [&]() {
                   if (isJson) {
                     auto raw = nlohmann::json::parse(bytes.begin(), bytes.end());
                     return polyast::program_from_json(polyast::hashed_from_json(raw));
                   }
                   return polyast::hashed_program_from_msgpack(bytes.data(), bytes.data() + bytes.size());
                 }();

                 const auto exportsOf = [](const polyast::Program &p) {
                   return p.functions ^ collect([](const auto &f) {
                            return f.visibility.template is<polyast::FunctionVisibility::Exported>() ? std::optional{fqcn(f.decl.name)}
                                                                                                     : std::nullopt;
                          });
                 };

                 if (listExports) {
                   for (auto &name : exportsOf(program) ^ sort())
                     fmt::print("{}\n", name);
                   return EXIT_SUCCESS;
                 }

                 const auto writeOut = [&](const auto &bytes, std::ios_base::openmode mode) {
                   if (out == "-") {
                     std::freopen(nullptr, "wb", stdout);
                     std::fwrite(bytes.data(), sizeof(std::byte), bytes.size(), stdout);
                   } else {
                     std::ofstream outStream(out, std::ios_base::binary | std::ios_base::out | mode);
                     outStream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
                   }
                 };

                 if (emitAst) {
                   if (!exportName.empty()) {
                     const auto known = exportsOf(program);
                     if (!(known ^ contains(exportName))) {
                       fmt::print(stderr, "[POLYC] Unknown export `{}`; the program exports: {}\n", exportName, known ^ mk_string(", "));
                       return EXIT_FAILURE;
                     }
                     for (auto &f : program.functions)
                       if (f.visibility.template is<polyast::FunctionVisibility::Exported>() && fqcn(f.decl.name) != exportName)
                         f.visibility = polyast::FunctionVisibility::Internal();
                   }
                   // an empty spec means no passes here, not the compile default: FullOpt on an entry-less library
                   // prunes it to nothing
                   const auto shaken = passes.empty() ? program : compiler::runPipeline(program, passes);
                   const auto astBytes = polyast::hashed_program_to_msgpack(shaken);
                   writeOut(astBytes, std::ios_base::trunc);
                   if (verbose)
                     fmt::print(stderr, "[POLYC] Wrote polyAST {} ({} functions, {} structs, {} bytes)\n", out, shaken.functions.size(),
                                shaken.defs.size(), astBytes.size());
                   return EXIT_SUCCESS;
                 }

                 compiler::initialise();
                 if (verbose) fmt::print(stderr, "[POLYC] Compiling program:\n=================\n{}\n=================\n", repr(program));

                 auto compilation = compiler::compile(program, compiler::Options{target, rawArch, passes, hostMirroring}, opt);
                 if (verbose) fmt::print(stderr, "{}\n", repr(compilation));
                 if (!compilation.messages.empty()) fmt::print(stderr, "{}\n", compilation.messages);
                 writeOut(compileresult_to_msgpack(compilation), std::ios_base::app);
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
  if (const auto result = packageMain(argc, argv)) return *result;
  PREPARE_FIRE_(argc, argv, false, fired_main, targetDescription);
  fire::_::logger.set_program_descr(FIRE_EXTRACT_2_PAD_(fired_main, targetDescription));
  return FIRE_EXTRACT_1_PAD_(fired_main, targetDescription)();
}
