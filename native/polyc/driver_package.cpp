#include "driver_package.h"

#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/string.hpp"
#include "aspartame/vector.hpp"

#include "polyfront/package_emit.hpp"

#include "ast.h"
#include "compiler.h"
#include "fire.hpp"
#include "package_compiler.hpp"
#include "polyast_codec.h"

using namespace polyregion;
using namespace aspartame;

namespace {

constexpr auto PackageUsage = "polyc package commands:\n"
                              "  link          Link program fragments and publish a package\n"
                              "  link-fragment Link one program fragment into an archive\n"
                              "  pack          Combine validated fragment archives\n"
                              "  compile       Compile a package import request\n";

constexpr int UsageError = 2;
constexpr int IoError = 3;
constexpr int PackageError = 4;
constexpr int UnexpectedError = 5;

std::optional<std::vector<uint8_t>> readBytes(const std::string &path) {
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "cannot read `" << path << "`\n";
    return {};
  }
  const auto bytes = (*buffer)->getBuffer();
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

template <typename T> bool writeBytes(const std::string &path, const std::vector<T> &bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (stream) return true;
  llvm::errs() << "cannot write `" << path << "`\n";
  return false;
}

std::vector<std::string> commaSeparated(const std::string &raw, const std::string_view option) {
  if (raw.empty()) return {};
  auto values = raw ^ split(',');
  if (values ^ exists([](const auto &value) { return value.empty(); }))
    throw std::invalid_argument(std::string(option) + " contains an empty value");
  return values;
}

std::set<std::string> capabilitySet(const std::string &raw) {
  const auto values = commaSeparated(raw, "--capabilities");
  return {values.begin(), values.end()};
}

compiler::package::Result<polyast::Package> linkPackageFragment(const polyast::Interface &interface, polyast::Program program,
                                                                const std::set<std::string> &capabilities, std::string fragmentIdentity) {
  std::vector<polyast::PackageFragment> fragments;
  fragments.emplace_back(std::move(fragmentIdentity), std::move(program));
  return compiler::package::link(polyast::PackageLinkRequest(interface, std::move(fragments),
                                                             std::vector<std::string>(capabilities.begin(), capabilities.end()), true));
}

polyfront::package::Checked<polyfront::package::ValidatedArchiveFragment> validatePackageFragment(polyfront::package::Archive archive) {
  auto fragment = polyfront::package::validateArchiveFragment(std::move(archive));
  if (!fragment) return fragment;
  std::vector<polyast::PackageFragment> fragments;
  fragments.reserve(fragment.value->implementations.size());
  for (size_t index = 0; index < fragment.value->implementations.size(); ++index)
    fragments.emplace_back(std::to_string(index), std::move(fragment.value->implementations[index]));
  auto linked = compiler::package::link(polyast::PackageLinkRequest(fragment.value->archive.interface, std::move(fragments), {}, true));
  if (!linked) return {{}, std::move(linked.errors)};
  if (linked.value->interface != fragment.value->archive.interface)
    return {{}, {"package fragment interface advertises declarations without compatible implementations"}};
  return fragment;
}

int packageLink(fire::optional<std::string> capabilities = fire::arg({"--capabilities", "Comma-separated implementation capabilities"}),
                std::vector<std::string> inputs = fire::arg({fire::variadic(),
                                                             "Arguments: <interface.polyast> <output-root> <program.polyast>..."})) {
  if (inputs.size() < 3) {
    llvm::errs() << "package link requires an interface, output root and at least one program\n";
    return UsageError;
  }
  try {
    const auto interfaceBytes = readBytes(inputs[0]);
    if (!interfaceBytes) return IoError;
    const auto interface = polyast::interface_from_msgpack(*interfaceBytes);
    const auto selectedCapabilities = capabilitySet(capabilities.value_or(""));
    polyfront::package::ArchiveBuilder archive(interface);
    for (size_t index = 2; index < inputs.size(); ++index) {
      const auto bytes = readBytes(inputs[index]);
      if (!bytes) return IoError;
      auto program = polyast::hashed_program_from_msgpack(*bytes);
      const auto linked = linkPackageFragment(interface, std::move(program), selectedCapabilities, std::to_string(index - 2));
      if (!linked) {
        llvm::errs() << (linked.errors | mk_string("\n")) << '\n';
        return PackageError;
      }
      const auto added = archive.addPackage(*linked.value);
      if (!added) {
        llvm::errs() << (added.errors | mk_string("\n")) << '\n';
        return PackageError;
      }
    }
    const auto published = archive.publish(inputs[1]);
    if (!published) {
      llvm::errs() << (published.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return UnexpectedError;
  }
}

int packageLinkFragment(std::string interfacePath = fire::arg({0, "<interface.polyast>", "Package interface"}),
                        std::string fragmentIdentity = fire::arg({1, "<fragment-identity>", "Stable fragment identity"}),
                        std::string programPath = fire::arg({2, "<program.polyast>", "Program fragment"}),
                        std::string outputPath = fire::arg({3, "<package.polyast>", "Output archive"}),
                        fire::optional<std::string> capabilities = fire::arg({"--capabilities",
                                                                              "Comma-separated implementation capabilities"})) {
  try {
    const auto interfaceBytes = readBytes(interfacePath);
    const auto programBytes = readBytes(programPath);
    if (!interfaceBytes || !programBytes) return IoError;
    const auto interface = polyast::interface_from_msgpack(*interfaceBytes);
    auto program = polyast::hashed_program_from_msgpack(*programBytes);
    const auto linked =
        linkPackageFragment(interface, std::move(program), capabilitySet(capabilities.value_or("")), std::move(fragmentIdentity));
    if (!linked) {
      llvm::errs() << (linked.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    polyfront::package::ArchiveBuilder archive(linked.value->interface);
    const auto added = archive.addPackage(*linked.value);
    if (!added) {
      llvm::errs() << (added.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    const auto published = archive.publishFile(outputPath);
    if (!published) {
      llvm::errs() << (published.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return UnexpectedError;
  }
}

int packagePack(std::vector<std::string> inputs = fire::arg({fire::variadic(),
                                                             "Arguments: <interface.polyast> <output-root> <package.polyast>..."})) {
  if (inputs.size() < 3) {
    llvm::errs() << "package pack requires an interface, output root and at least one fragment archive\n";
    return UsageError;
  }
  try {
    const auto interfaceBytes = readBytes(inputs[0]);
    if (!interfaceBytes) return IoError;
    const auto interface = polyast::interface_from_msgpack(*interfaceBytes);
    polyfront::package::ArchiveBuilder archive(interface);
    for (size_t index = 2; index < inputs.size(); ++index) {
      const auto fragment = polyfront::package::loadPackageFile(inputs[index]);
      if (!fragment) {
        llvm::errs() << (fragment.errors | mk_string("\n")) << '\n';
        return PackageError;
      }
      auto validated = validatePackageFragment(std::move(*fragment.value));
      if (!validated) {
        llvm::errs() << (validated.errors | mk_string("\n")) << '\n';
        return PackageError;
      }
      const auto added = archive.addArchive(*validated.value);
      if (!added) {
        llvm::errs() << (added.errors | mk_string("\n")) << '\n';
        return PackageError;
      }
    }
    const auto published = archive.publish(inputs[1]);
    if (!published) {
      llvm::errs() << (published.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return UnexpectedError;
  }
}

int packageCompile(std::string requestPath = fire::arg({0, "<request.msgpack>", "Package import request"}),
                   std::string outputPath = fire::arg({"-o", "--out", "Output compile bundle"}),
                   std::string hostTargetName = fire::arg({"-m", "--target", "Host code-generation target"}),
                   std::string hostArch = fire::arg({"-a", "--arch", "Host architecture"}, "native"),
                   fire::optional<std::string> devices =
                       fire::arg({"--devices", "Comma-separated <target>@<architecture> device profiles; architecture may be empty"}),
                   int stackDepth = fire::arg({"--stack-depth", "Positive device stack-depth bound; zero uses the target default"}, 0)) {
  try {
    if (stackDepth < 0) throw std::invalid_argument("--stack-depth must not be negative");
    std::vector<std::pair<compiletime::Target, std::string>> deviceTargets;
    for (const auto &raw : commaSeparated(devices.value_or(""), "--devices")) {
      const auto separator = raw.find('@');
      if (separator == std::string::npos || separator == 0)
        throw std::invalid_argument("device profile must be <target>@<architecture>: " + raw);
      const auto target = compiletime::TargetSpec::findByName(std::string_view(raw).substr(0, separator));
      if (!target) throw std::invalid_argument("unknown device target: " + raw.substr(0, separator));
      deviceTargets.emplace_back(target->codegen, raw.substr(separator + 1));
    }
    const auto hostTarget = compiletime::TargetSpec::findByName(hostTargetName);
    if (!hostTarget) throw std::invalid_argument("unknown host target: " + hostTargetName);
    const auto requestBytes = readBytes(requestPath);
    if (!requestBytes) return IoError;
    const auto request = polyast::programlinkrequest_from_msgpack(*requestBytes);
    const auto compiled = compiler::package::compile(request, hostTarget->codegen, hostArch, deviceTargets,
                                                     stackDepth == 0 ? std::optional<int>{} : std::optional<int>{stackDepth});
    if (!compiled) {
      llvm::errs() << (compiled.errors | mk_string("\n")) << '\n';
      return PackageError;
    }
    return writeBytes(outputPath, polyast::compilebundle_to_msgpack(*compiled.value)) ? EXIT_SUCCESS : IoError;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << '\n';
    return UnexpectedError;
  }
}

int runPackageLink(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, packageLink, "Link PolyAST program fragments into an indexed package archive.");
  fire::_::logger.set_program_descr("Link PolyAST program fragments into an indexed package archive.");
  return packageLink();
}

int runPackageLinkFragment(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, packageLinkFragment, "Link one PolyAST program fragment into an indexed archive.");
  fire::_::logger.set_program_descr("Link one PolyAST program fragment into an indexed archive.");
  return packageLinkFragment();
}

int runPackagePack(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, packagePack, "Combine validated package fragment archives without recompressing them.");
  fire::_::logger.set_program_descr("Combine validated package fragment archives without recompressing them.");
  return packagePack();
}

int runPackageCompile(int argc, const char *argv[]) {
  PREPARE_FIRE_(argc, argv, false, packageCompile, "Compile a package import request for its host and device profiles.");
  fire::_::logger.set_program_descr("Compile a package import request for its host and device profiles.");
  return packageCompile();
}

} // namespace

std::optional<int> polyregion::packageDriver(int argc, const char *argv[]) {
  if (argc < 2 || std::string_view(argv[1]) != "package") return {};
  if (argc < 3 || std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h") {
    llvm::outs() << PackageUsage;
    return argc < 3 ? EXIT_FAILURE : EXIT_SUCCESS;
  }

  const std::string_view command(argv[2]);
  std::string executable = std::string(argv[0]) + " package " + std::string(command);
  std::vector<const char *> commandArgs{executable.c_str()};
  commandArgs.insert(commandArgs.end(), argv + 3, argv + argc);
  const auto commandArgc = static_cast<int>(commandArgs.size());
  if (command == "link") return runPackageLink(commandArgc, commandArgs.data());
  if (command == "link-fragment") return runPackageLinkFragment(commandArgc, commandArgs.data());
  if (command == "pack") return runPackagePack(commandArgc, commandArgs.data());
  if (command == "compile") return runPackageCompile(commandArgc, commandArgs.data());
  llvm::errs() << "unknown package command `" << command << "`\n" << PackageUsage;
  return EXIT_FAILURE;
}
