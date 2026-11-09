#include "compiler.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/ThreadPool.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"
#include "nlohmann/json.hpp"

#include "polyregion/compat.h"
#include "polyregion/conventions.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/group_native.h"
#include "polyregion/io.hpp"
#include "polyregion/llvm_utils.hpp"
#include "polyregion/polypackage.h"
#include "polyregion/polypackage_symbols.h"
#include "polyregion/program_import.hpp"

#include "ast.h"
#include "backend/c_source.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "dso_runner.h"
#include "js_runner.h"
#include "package_compiler.hpp"
#include "polyast_codec.h"
#include "polypass_locate.h"

using namespace polyregion;
using namespace aspartame;

compiler::TimePoint compiler::nowMono() { return MonoClock::now(); }

int64_t compiler::elapsedNs(const TimePoint &a, const TimePoint &b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

int64_t compiler::nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void compiler::initialise() {
  static const bool initialised = [] {
    backend::llvmc::initialise();
    return true;
  }();
  (void)initialised;
}

static const uint8_t *bytesBegin(const polyast::Bytes &bytes) { return reinterpret_cast<const uint8_t *>(bytes.data()); }

static const uint8_t *bytesEnd(const polyast::Bytes &bytes) { return bytesBegin(bytes) + bytes.size(); }

static polyast::Program deserialiseProgram(const polyast::Bytes &astBytes) {
  try {
    return polyast::hashed_program_from_msgpack(bytesBegin(astBytes), bytesEnd(astBytes));
  } catch (const std::exception &e) {
    throw std::logic_error(fmt::format("Unable to parse packed ast: {}", e.what()));
  }
}

static std::vector<polyast::StructDef> deserialiseStructDefs(const polyast::Bytes &astBytes) {
  try {
    return polyast::hashed_structdefs_from_msgpack(bytesBegin(astBytes), bytesEnd(astBytes));
  } catch (const std::exception &e) {
    throw std::logic_error(fmt::format("Unable to parse packed struct defs: {}", e.what()));
  }
}

static backend::LLVMBackend::Options toLLVMBackendOptions(const compiler::Options &options) {

  auto validate = [&](llvm::Triple::ArchType arch) {
    if (!llvm_shared::isCPUTargetSupported(options.arch, arch)) {
      throw std::logic_error(fmt::format("Unsupported target CPU `{}` on `{}`", options.arch, llvm::Triple::getArchTypeName(arch).str()));
    }
  };

  auto opts = [&]() -> backend::LLVMBackend::Options {
    switch (options.target) {
      case compiletime::Target::Object_LLVM_HOST: {
        const auto host = backend::llvmc::defaultHostTriple();
        validate(host.getArch());
        switch (host.getArch()) {
          case llvm::Triple::ArchType::x86_64: return {.target = backend::LLVMBackend::Target::x86_64, .arch = options.arch};
          case llvm::Triple::ArchType::aarch64: return {.target = backend::LLVMBackend::Target::AArch64, .arch = options.arch};
          case llvm::Triple::ArchType::arm: return {.target = backend::LLVMBackend::Target::ARM, .arch = options.arch};
          case llvm::Triple::ArchType::riscv64: return {.target = backend::LLVMBackend::Target::RISCV64, .arch = options.arch};
          case llvm::Triple::ArchType::ppc64le: return {.target = backend::LLVMBackend::Target::PPC64LE, .arch = options.arch};
          default: throw std::logic_error(fmt::format("Unsupported host triplet: {}", host.str()));
        }
      }
      case compiletime::Target::Object_LLVM_x86_64:
        validate(llvm::Triple::ArchType::x86_64);
        return {.target = backend::LLVMBackend::Target::x86_64, .arch = options.arch};
      case compiletime::Target::Object_LLVM_AArch64:
        validate(llvm::Triple::ArchType::aarch64);
        return {.target = backend::LLVMBackend::Target::AArch64, .arch = options.arch};
      case compiletime::Target::Object_LLVM_ARM:
        validate(llvm::Triple::ArchType::arm);
        return {.target = backend::LLVMBackend::Target::ARM, .arch = options.arch};
      case compiletime::Target::Object_LLVM_NVPTX64:
        validate(llvm::Triple::ArchType::nvptx64);
        return {.target = backend::LLVMBackend::Target::NVPTX64, .arch = options.arch};
      case compiletime::Target::Object_LLVM_AMDGCN:
#if LLVM_VERSION_MAJOR >= 23
        validate(llvm::Triple::ArchType::amdgpu);
#else
        validate(llvm::Triple::ArchType::amdgcn);
#endif
        return {.target = backend::LLVMBackend::Target::AMDGCN, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
        return {.target = backend::LLVMBackend::Target::SPIRV32_Kernel, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
        return {.target = backend::LLVMBackend::Target::SPIRV64_Kernel, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
        return {.target = backend::LLVMBackend::Target::SPIRV_GLCompute, .arch = options.arch};
      case compiletime::Target::Source_C_OpenCL1_1: //
      case compiletime::Target::Source_C_OpenCL2_0: //
      case compiletime::Target::Source_C_Metal1_0:  //
      case compiletime::Target::Source_C_C11:       //
        throw std::logic_error("Not an object target");
      default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
    }
  }();
  opts.emitBitcode = options.hostMirroring;
  opts.workgroupMemoryBytes = options.workgroupMemoryBytes;
  return opts;
}

std::vector<polyast::StructLayout> compiler::layoutOf(const std::vector<polyast::StructDef> &defs, const Options &options) {
  switch (options.target) {
    case compiletime::Target::Object_LLVM_HOST: [[fallthrough]];
    case compiletime::Target::Object_LLVM_x86_64: [[fallthrough]];
    case compiletime::Target::Object_LLVM_AArch64: [[fallthrough]];
    case compiletime::Target::Object_LLVM_ARM: [[fallthrough]];
    case compiletime::Target::Object_LLVM_NVPTX64: [[fallthrough]];
    case compiletime::Target::Object_LLVM_AMDGCN: [[fallthrough]];
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel: [[fallthrough]];
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel: [[fallthrough]];
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute: return backend::LLVMBackend(toLLVMBackendOptions(options)).resolveLayouts(defs);
    case compiletime::Target::Source_C_C11: [[fallthrough]];
    case compiletime::Target::Source_C_OpenCL1_1: [[fallthrough]];
    case compiletime::Target::Source_C_OpenCL2_0: [[fallthrough]];
    case compiletime::Target::Source_C_Metal1_0:
      throw std::logic_error(fmt::format("Not available for source target {}", magic_enum::enum_name(options.target)));
    default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
  }
}

std::vector<polyast::StructLayout> compiler::layoutOf(const polyast::Bytes &bytes, const Options &options) {
  return layoutOf(deserialiseStructDefs(bytes), options);
}

namespace {

using FunctionIndex = std::unordered_map<polyast::Sym, std::vector<const polyast::Function *>>;

FunctionIndex indexFunctions(const polyast::Program &program) {
  FunctionIndex index;
  if (program.entry) index[program.entry->decl.name].emplace_back(&*program.entry);
  for (const auto &function : program.functions)
    index[function.decl.name].emplace_back(&function);
  return index;
}

using ReferenceGraph = std::unordered_map<polyast::Sym, std::vector<polyast::Sym>>;

ReferenceGraph referenceGraph(const polyast::Program &program) {
  ReferenceGraph graph;
  const auto add = [&](const polyast::Function &function) {
    auto &references = graph[function.decl.name];
    for (const auto &reference : function.collect_all<polyast::Type::FnRef>())
      references.emplace_back(reference.name);
  };
  if (program.entry) add(*program.entry);
  for (const auto &function : program.functions)
    add(function);
  return graph;
}

std::vector<polyast::Sym> referencesOf(const polyast::Function &function) {
  return function.collect_all<polyast::Type::FnRef>() | map([](const auto &reference) { return reference.name; }) | to_vector();
}

std::unordered_set<polyast::Sym> reachableFrom(const ReferenceGraph &graph, std::vector<polyast::Sym> frontier) {
  std::unordered_set<polyast::Sym> live;
  while (!frontier.empty()) {
    auto name = std::move(frontier.back());
    frontier.pop_back();
    if (!live.emplace(name).second) continue;
    if (const auto found = graph.find(name); found != graph.end())
      frontier.insert(frontier.end(), found->second.begin(), found->second.end());
  }
  return live;
}

struct PluginRegistry {
  std::vector<std::unique_ptr<polypass::PassRunner>> plugins;
  std::unordered_map<std::string, size_t> ownerByPass;
  std::vector<size_t> packageProviders;
};

PluginRegistry &sharedPlugins() {
  static PluginRegistry reg = [] {
    PluginRegistry result;
    std::string err;
    auto refs = polypass::resolvePlugins(err);
    if (!err.empty()) throw std::logic_error(fmt::format("polyc: {}", err));
    for (auto &ref : refs) {
      std::unique_ptr<polypass::PassRunner> runner =
          ref.kind == polypass::PluginKind::Js ? std::unique_ptr<polypass::PassRunner>(std::make_unique<polypass::JsPassRunner>(ref.path))
                                               : std::unique_ptr<polypass::PassRunner>(std::make_unique<polypass::DsoPassRunner>(ref.path));
      if (auto rerr = runner->load(); !rerr.empty())
        throw std::logic_error(fmt::format("polyc: failed to load PolyPass plugin {}: {}", ref.path, rerr));
      const size_t idx = result.plugins.size();
      for (const auto &name : runner->passNames()) {
        if (const auto it = result.ownerByPass.find(name); it != result.ownerByPass.end()) {
          fmt::print(stderr, "polyc: pass '{}' from {} overrides earlier definition from {}\n", name, runner->tag(),
                     result.plugins[it->second]->tag());
          it->second = idx;
        } else {
          result.ownerByPass.emplace(name, idx);
        }
      }
      if (runner->packageAbiVersion()) result.packageProviders.emplace_back(idx);
      result.plugins.push_back(std::move(runner));
    }
    return result;
  }();
  return reg;
}

std::vector<std::string> packageErrors(std::string diagnostic) {
  if (diagnostic.starts_with("PolyPackage "))
    if (const auto separator = diagnostic.find(": "); separator != std::string::npos) diagnostic.erase(0, separator + 2);
  std::vector<std::string> errors;
  for (size_t offset = 0; offset <= diagnostic.size();) {
    const auto end = diagnostic.find('\n', offset);
    const auto line = diagnostic.substr(offset, end - offset);
    if (!line.empty()) errors.emplace_back(line);
    if (end == std::string::npos) break;
    offset = end + 1;
  }
  return errors;
}

std::mutex &packageMutex() {
  static std::mutex mutex;
  return mutex;
}

template <typename T, typename Decode>
compiler::package::Result<T> invokePackage(const std::vector<uint8_t> &request, const std::string_view operation, Decode decode) {
  auto &registry = sharedPlugins();
  if (registry.packageProviders.size() != 1)
    return {.value = {}, .errors = {fmt::format("expected one package-service plugin, found {}", registry.packageProviders.size())}};
  auto &provider = *registry.plugins[registry.packageProviders.front()];
  const std::scoped_lock lock(packageMutex());
  const auto version = provider.packageAbiVersion();
  if (!version || *version != POLYPACKAGE_ABI_VERSION)
    return {.value = {},
            .errors = {fmt::format("PolyPackage ABI mismatch: service={}, polyc={}", version ? std::to_string(*version) : "<missing>",
                                   POLYPACKAGE_ABI_VERSION)}};
  String error;
  const auto output = provider.runPackage(operation, request, error);
  if (!error.empty()) return {.value = {}, .errors = packageErrors(std::move(error))};
  if (output.empty()) return {.value = {}, .errors = {"package service returned an empty successful result"}};
  try {
    return {.value = decode(output.data(), output.data() + output.size()), .errors = {}};
  } catch (const std::exception &error) {
    return {.value = {}, .errors = {std::string("cannot decode package-service result: ") + error.what()}};
  }
}

std::string bareName(const std::string &step) {
  const auto prefix = std::string_view(step) ^ take_while([](char c) { return c != '('; });
  return std::string(prefix ^ trim());
}

} // namespace

compiler::package::Result<polyast::Package> compiler::package::link(const polyast::PackageLinkRequest &request) {
  return invokePackage<polyast::Package>(
      polyast::packagelinkrequest_to_msgpack(request), polypackage::abi::LinkPackage,
      [](const uint8_t *begin, const uint8_t *end) { return polyast::package_service_result_from_msgpack(begin, end); });
}

static compiler::package::Result<polyast::Program> linkProgram(const std::vector<uint8_t> &request) {
  return invokePackage<polyast::Program>(request, polypackage::abi::LinkProgram, [](const uint8_t *begin, const uint8_t *end) {
    return polyast::program_service_result_from_msgpack(begin, end);
  });
}

compiler::package::Result<std::vector<compiler::package::Selection>> compiler::package::select(const polyast::ProgramLinkRequest &request) {
  auto selected = invokePackage<polyast::Program>(
      polyast::programlinkrequest_to_msgpack(request), polypackage::abi::SelectImplementations,
      [](const uint8_t *begin, const uint8_t *end) { return polyast::program_service_result_from_msgpack(begin, end); });
  if (!selected) return {.value = {}, .errors = std::move(selected.errors)};
  std::vector<Selection> selections;
  const auto add = [&](const polyast::Function &function) {
    if (!function.visibility.template is<polyast::FunctionVisibility::Exported>() || !function.implements) return;
    const auto key = conventions::LinkImplementationMeta + polyast::fqcn(function.decl.name);
    if (const auto found = selected.value->metadata ^ find([&](const auto &entry) { return entry.key == key; }))
      selections.push_back({.declaration = *function.implements, .implementation = found->value});
  };
  if (request.consumer.entry) add(*request.consumer.entry);
  for (const auto &function : request.consumer.functions)
    add(function);
  return {.value = std::move(selections), .errors = {}};
}

namespace {

std::string packageEntryPipeline(const compiletime::Target target, const std::string &arch, const std::optional<int> stackDepth) {
  const auto opt = stackDepth ? fmt::format("DeadFunctionElimination;Intrinsify;RecursionLower(maxDepth={});FnInline;Intrinsify;"
                                            "FullOpt(level=1)",
                                            *stackDepth)
                              : std::string("DeadFunctionElimination;Intrinsify;RecursionLower;FnInline;Intrinsify;FullOpt(level=1)");
  const auto groups = compiletime::groupLowerArgs(target, arch);
  switch (target) {
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
      return opt
             + fmt::format(";SubgroupLower(lowerSubgroups=false,{});StructuredExit;PartialEval(canonicaliseAddresses=true);ArenaView;"
                           "RegionRespace;VerifyAnchors(strict=true)",
                           groups);
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
      return opt + fmt::format(";SubgroupLower(lowerSubgroups=false,{});StructuredExit;RegionRespace;ArenaLower", groups);
    case compiletime::Target::Source_C_Metal1_0:
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_OpenCL2_0:
      return opt + fmt::format(";SubgroupLower(width=1,{});StructuredExit;RegionRespace;ArenaLower", groups);
    case compiletime::Target::Object_LLVM_NVPTX64:
      return opt + fmt::format(";SubgroupLower(lowerSubgroups=false,{});StructuredExit;RegionRespace", groups);
    case compiletime::Target::Object_LLVM_AMDGCN:
      return opt + fmt::format(";SubgroupLower(lowerSubgroups=false,{});StructuredExit", groups);
    default: return opt + compiletime::hostGroupPipelineTail(groups, false);
  }
}

unsigned compileJobs() { return polyregion::env::jobs(polyregion::env::PolycJobs); }

std::vector<polyast::Sym> consumerRoots(const polyast::ProgramLinkRequest &request) {
  std::vector<polyast::Sym> roots;
  const auto add = [&](const polyast::Function &function) {
    if (function.visibility.template is<polyast::FunctionVisibility::Exported>() && function.implements)
      roots.emplace_back(function.decl.name);
  };
  if (request.consumer.entry) add(*request.consumer.entry);
  for (const auto &function : request.consumer.functions)
    add(function);
  return roots;
}

std::vector<std::string> linkMetadata(const polyast::Program &program, const std::vector<polyast::Sym> &roots, const std::string &prefix) {
  return roots ^ collect([&](const auto &root) {
           const auto key = prefix + polyast::fqcn(root);
           return program.metadata ^ find([&](const auto &entry) { return entry.key == key; })
                  ^ map([](const auto &entry) { return entry.value; });
         });
}

std::vector<polyast::CompileRoot> compileRoots(const polyast::Program &program, const std::vector<polyast::Sym> &roots,
                                               const ReferenceGraph &graph) {
  std::unordered_map<polyast::Sym, std::string> moduleNames;
  for (const auto &function : program.functions)
    if (function.convention.is<polyast::CallConvention::OffloadEntry>())
      moduleNames.emplace(function.decl.name, backend::normaliseSymbol(function.decl.name));
  return roots | map([&](const auto &root) {
           std::vector<std::string> modules;
           for (const auto &name : reachableFrom(graph, {root}))
             if (const auto found = moduleNames.find(name); found != moduleNames.end()) modules.emplace_back(found->second);
           std::ranges::sort(modules);
           return polyast::CompileRoot(polyast::fqcn(root), std::move(modules),
                                       linkMetadata(program, {root}, conventions::LinkImplementationMeta),
                                       linkMetadata(program, {root}, conventions::LinkUnfitMeta));
         })
         | to_vector();
}

std::vector<std::string> validateImportedEntries(const polyast::ProgramLinkRequest &request, const polyast::Program &program) {
  std::vector<std::string> errors;
  const auto linked = indexFunctions(program);
  const auto validate = [&](const polyast::Function &root) {
    if (!root.visibility.template is<polyast::FunctionVisibility::Exported>() || !root.implements) return;
    if (program.metadata
        ^ exists([&](const auto &entry) { return entry.key == conventions::LinkUnfitMeta + polyast::fqcn(root.decl.name); }))
      return;
    const auto found = linked.find(root.decl.name);
    const auto count = found == linked.end() ? 0 : found->second.size();
    if (count != 1) {
      errors.emplace_back(fmt::format("linked consumer entry `{}` has {} definitions", polyast::fqcn(root.decl.name), count));
      return;
    }
    std::vector<polyast::Type::Any> expectedArgs{polyast::Type::Ptr(polyast::Type::IntU8(), polyast::TypeSpace::Global())};
    for (const auto &argument : root.decl.args) {
      switch (program::importArgumentMode(argument.named.tpe)) {
        case program::ImportArgumentMode::OmittedCallable: break;
        case program::ImportArgumentMode::DirectPointer: expectedArgs.emplace_back(argument.named.tpe); break;
        case program::ImportArgumentMode::AddressOfValue:
          expectedArgs.emplace_back(polyast::Type::Ptr(argument.named.tpe, polyast::TypeSpace::Global()));
          break;
      }
    }
    const auto hasResult = !root.decl.rtn.is<polyast::Type::Unit0>();
    if (hasResult) expectedArgs.emplace_back(polyast::Type::Ptr(root.decl.rtn, polyast::TypeSpace::Global()));
    const auto &actual = found->second.front()->decl;
    const auto argumentTypesMatch = [&] {
      if (actual.args.size() != expectedArgs.size()) return false;
      const auto valueArguments = expectedArgs.size() - (hasResult ? 1 : 0);
      for (size_t index = 0; index < valueArguments; ++index)
        if (actual.args[index].named.tpe != expectedArgs[index]) return false;
      if (!hasResult) return true;
      if (!root.decl.rtn.is<polyast::Type::Nothing>()) return actual.args.back().named.tpe == expectedArgs.back();
      const auto resolved = actual.args.back().named.tpe.get<polyast::Type::Ptr>();
      return resolved && resolved->space == polyast::TypeSpace::Global() && !resolved->comp.is<polyast::Type::Nothing>();
    }();
    if (!actual.tpeVars.empty() || actual.receiver || !actual.moduleCaptures.empty() || !actual.termCaptures.empty()
        || actual.rtn != polyast::Type::Unit0() || actual.affinity != polyast::FunctionAffinity::Host() || !argumentTypesMatch)
      errors.emplace_back(fmt::format("linked consumer entry `{}` has an invalid physical ABI", polyast::fqcn(root.decl.name)));
  };
  if (request.consumer.entry) validate(*request.consumer.entry);
  for (const auto &root : request.consumer.functions)
    validate(root);
  return errors;
}

std::vector<std::string> validateRemoteEntries(const polyast::Program &program) {
  std::vector<std::string> errors;
  const auto functions = indexFunctions(program);
  const auto launchTargets = program.collect_all<polyast::Spec::RemoteLaunch>()
                             | collect([](const auto &launch) { return launch.kernel.tpe().template get<polyast::Type::FnRef>(); })
                             | map([](const auto &reference) { return reference.name; }) | distinct() | to_vector();
  for (const auto &target : launchTargets) {
    const auto found = functions.find(target);
    const auto count = found == functions.end() ? 0 : found->second.size();
    if (count != 1) {
      errors.emplace_back(fmt::format("remote launch target `{}` has {} linked definitions", polyast::fqcn(target), count));
      continue;
    }
    if (!found->second.front()->convention.is<polyast::CallConvention::OffloadEntry>())
      errors.emplace_back(fmt::format("remote launch target `{}` is not an offload entry", polyast::fqcn(target)));
  }
  return errors;
}

polyast::Program namespaceRemoteEntries(polyast::Program program,
                                        const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                                        const std::optional<int> stackDepth) {
  const auto encoded = polyast::hashed_program_to_msgpack(program);
  llvm::SHA256 hash;
  hash.update(llvm::ArrayRef(encoded));
  std::vector<std::string> profiles;
  profiles.reserve(deviceTargets.size());
  for (const auto &[target, arch] : deviceTargets) {
    const auto pipeline = packageEntryPipeline(target, arch, stackDepth);
    profiles.emplace_back(fmt::format("{}:{}:{}:{}:{}", magic_enum::enum_integer(target), arch.size(), arch, pipeline.size(), pipeline));
  }
  std::ranges::sort(profiles);
  hash.update(fmt::format("|compile-profiles:{}|", profiles.size()));
  for (const auto &profile : profiles) {
    hash.update(fmt::format("{}:", profile.size()));
    hash.update(profile);
  }
  const auto identity = llvm::toHex(hash.final(), true);
  std::unordered_map<polyast::Sym, polyast::Sym> names;
  for (const auto &function : program.functions) {
    if (!function.convention.is<polyast::CallConvention::OffloadEntry>()) continue;
    names.emplace(function.decl.name, polyast::Sym({polyast::offloadEntrySymbol(function.decl.name), "module", identity}));
  }
  const std::function<polyast::Type::FnRef(const polyast::Type::FnRef &)> rename = [&](const polyast::Type::FnRef &reference) {
    const auto found = names.find(reference.name);
    return found == names.end() ? reference : reference.withName(found->second);
  };
  if (program.entry) program.entry = program.entry->modify_all<polyast::Type::FnRef>(rename);
  for (auto &function : program.functions)
    function = function.modify_all<polyast::Type::FnRef>(rename);
  for (auto &definition : program.defs)
    definition = definition.modify_all<polyast::Type::FnRef>(rename);
  if (program.entry) {
    if (const auto found = names.find(program.entry->decl.name); found != names.end()) program.entry->decl.name = found->second;
  }
  for (auto &function : program.functions) {
    if (const auto found = names.find(function.decl.name); found != names.end()) function.decl.name = found->second;
  }
  return program;
}

} // namespace

compiler::package::Result<polyast::CompileBundle>
compiler::package::compile(polyast::ProgramLinkRequest request, const compiletime::Target hostTarget, const std::string &hostArch,
                           const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                           const std::optional<int> stackDepth) {
  if (!(deviceTargets
        ^ forall([&](const auto &target) { return compiletime::logicalAddressing(target.first) == request.logicalAddressing; })))
    return {.value = {},
            .errors = {fmt::format("device targets do not all use {} addressing", request.logicalAddressing ? "logical" : "physical")}};
  request.stackDepth = stackDepth;
  const auto linkRequest = polyast::programlinkrequest_to_msgpack(request);
  request.packages = {};
  auto linked = linkProgram(linkRequest);
  if (!linked) return {.value = {}, .errors = std::move(linked.errors)};
  if (auto errors = validateImportedEntries(request, *linked.value); !errors.empty()) return {.value = {}, .errors = std::move(errors)};
  const auto roots = consumerRoots(request);
  auto implementations = linkMetadata(*linked.value, roots, conventions::LinkImplementationMeta);
  auto unfit = linkMetadata(*linked.value, roots, conventions::LinkUnfitMeta);
  if (implementations.empty() && !unfit.empty())
    return {.value = polyast::CompileBundle({}, {}, {}, std::move(unfit), compileRoots(*linked.value, roots, {})), .errors = {}};
  if (auto errors = validateRemoteEntries(*linked.value); !errors.empty()) return {.value = {}, .errors = std::move(errors)};
  initialise();
  std::unordered_set<polyast::Sym> consumerEntries;
  const auto retainConsumerEntry = [&](const polyast::Function &function) {
    if (function.visibility.template is<polyast::FunctionVisibility::Exported>() && function.implements)
      consumerEntries.emplace(function.decl.name);
  };
  if (request.consumer.entry) retainConsumerEntry(*request.consumer.entry);
  for (const auto &function : request.consumer.functions)
    retainConsumerEntry(function);
  auto program = namespaceRemoteEntries(std::move(*linked.value), deviceTargets, stackDepth);
  for (auto &function : program.functions) {
    const auto exported = function.convention.is<polyast::CallConvention::OffloadEntry>() || consumerEntries.contains(function.decl.name);
    if (exported) function.visibility = polyast::FunctionVisibility::Exported();
    else function.visibility = polyast::FunctionVisibility::Internal();
  }

  const auto graph = referenceGraph(program);
  auto compileRootInfo = compileRoots(program, roots, graph);
  {
    std::vector<polyast::Sym> seeds;
    if (program.entry) seeds = referencesOf(*program.entry);
    for (const auto &function : program.functions)
      if (function.visibility.template is<polyast::FunctionVisibility::Exported>()) seeds.emplace_back(function.decl.name);
    const auto live = reachableFrom(graph, std::move(seeds));
    std::erase_if(program.functions, [&](const auto &function) { return !live.contains(function.decl.name); });
  }

  llvm::StdThreadPool pool(llvm::hardware_concurrency(compileJobs()));
  const auto submit = [&](compiler::Staged staged, const compiletime::OptLevel opt) {
    return pool.async([staged = std::make_shared<compiler::Staged>(std::move(staged)), opt] {
      return std::make_shared<polyast::CompileResult>(compiler::emit(std::move(*staged), opt));
    });
  };

  struct EntryJob {
    std::string moduleName;
    polyast::Function entry;
    std::vector<polyast::Function> functions;
  };
  std::vector<EntryJob> entryJobs;
  for (const auto &entry : program.functions) {
    if (!entry.convention.is<polyast::CallConvention::OffloadEntry>()) continue;
    const auto live = reachableFrom(graph, referencesOf(entry));
    std::vector<polyast::Function> functions;
    for (const auto &function : program.functions) {
      if (function.decl.name == entry.decl.name || !live.contains(function.decl.name)) continue;
      auto internal = function.withVisibility(polyast::FunctionVisibility::Internal());
      if (internal.convention.is<polyast::CallConvention::OffloadEntry>())
        internal = internal.withConvention(polyast::CallConvention::RegularCall());
      functions.emplace_back(std::move(internal));
    }
    entryJobs.push_back({.moduleName = backend::normaliseSymbol(entry.decl.name), .entry = entry, .functions = std::move(functions)});
  }
  std::erase_if(program.functions,
                [](const auto &function) { return !function.decl.affinity.template is<polyast::FunctionAffinity::Host>(); });
  const auto sharedDefs = program.defs;
  const auto sharedPhase = program.phase;
  const auto sharedMetadata = program.metadata;

  auto hostStaged = compiler::stage(
      std::move(program), Options{.target = hostTarget, .arch = hostArch, .pipelineSpec = "StructuredExit", .hostMirroring = true});
  const auto hostCompile = submit(std::move(hostStaged), compiletime::OptLevel::O3);

  struct DeviceCompile {
    std::string moduleName;
    compiletime::Target target;
    std::shared_future<std::shared_ptr<polyast::CompileResult>> result;
  };
  std::vector<DeviceCompile> deviceCompiles;
  for (auto &job : entryJobs) {
    for (size_t index = 0; index < deviceTargets.size(); ++index) {
      const auto &[target, arch] = deviceTargets[index];
      const auto pipeline = packageEntryPipeline(target, arch, stackDepth);
      const auto last = index + 1 == deviceTargets.size();
      auto entryProgram = last ? polyast::Program(std::move(job.entry), std::move(job.functions), sharedDefs, sharedPhase, sharedMetadata)
                               : polyast::Program(job.entry, job.functions, sharedDefs, sharedPhase, sharedMetadata);
      deviceCompiles.push_back(
          {.moduleName = job.moduleName,
           .target = target,
           .result = submit(compiler::stage(std::move(entryProgram), Options{.target = target, .arch = arch, .pipelineSpec = pipeline}),
                            compiletime::OptLevel::O3)});
    }
  }

  const auto host = hostCompile.get();
  if (!host->binary) return {.value = {}, .errors = {"linked consumer program compilation failed: " + host->messages}};
  std::vector<polyast::CompileModule> remoteModules;
  for (const auto &[moduleName, target, result] : deviceCompiles) {
    const auto device = result.get();
    if (!device->binary)
      return {.value = {}, .errors = {"linked consumer entry compilation failed for " + moduleName + ": " + device->messages}};
    const auto format = runtime::moduleFormatOf(target);
    if (!format) continue;
    auto features = device->features;
    if (device->entryArgs ^ exists([](const auto &arg) { return arg.symbol == conventions::ErrorArg; }))
      features.emplace_back(conventions::ErrorArg);
    remoteModules.emplace_back(moduleName, static_cast<int32_t>(*format), static_cast<int32_t>(runtime::targetPlatformKind(target)),
                               std::move(features), std::move(*device->binary));
  }
  return {.value = polyast::CompileBundle(std::move(*host->binary), std::move(remoteModules), std::move(implementations), std::move(unfit),
                                          std::move(compileRootInfo)),
          .errors = {}};
}

static polyast::PassRunResult runPipelineChain(polyast::Program &&p, std::string_view rawSpec) {
  const std::string_view spec = rawSpec.empty() ? std::string_view{compiler::DefaultPipelineSpec} : rawSpec;
  const auto rootEpoch = compiler::nowMs();
  const auto rootStart = compiler::nowMono();
  const auto timed = [](const auto &name, const auto &data, const auto &f) {
    const auto epoch = compiler::nowMs();
    const auto start = compiler::nowMono();
    auto out = f();
    return std::pair{std::move(out), polyast::CompileEvent(epoch, compiler::elapsedNs(start), name, data, {})};
  };

  auto [bytes, serialiseEvent] = timed("polyast_msgpack_serialise_cpp", "", [&] { return polyast::program_to_msgpack(p); });
  serialiseEvent.data = fmt::format("bytes={}", bytes.size());
  static_cast<void>(polyast::Program(std::move(p)));

  auto &reg = sharedPlugins();

  auto ownerOf = [&](const std::string &step) {
    const auto bare = bareName(step);
    const auto it = reg.ownerByPass.find(bare);
    if (it == reg.ownerByPass.end()) throw std::logic_error(fmt::format("PolyPass: unknown pass '{}' in spec '{}'", bare, spec));
    return it->second;
  };

  const auto stepsWithOwner = std::string(spec)                                                       //
                                  ^ split(';')                                                        //
                              | map([](const auto &s) { return trim(s); })                            //
                              | filter([](const auto &s) { return !s.empty(); })                      //
                              | map([&](const auto &step) { return std::pair{ownerOf(step), step}; }) //
                              | to_vector();

  const auto groups = stepsWithOwner                                                              //
                      | chunk_by([](const auto &a, const auto &b) { return a.first == b.first; }) //
                      | map([](const auto &group) {                                               //
                          return std::pair{group.front().first, group ^ map([](const auto &, const auto &step) { return step; })};
                        }) //
                      | to_vector();
  if (groups.empty()) throw std::logic_error(fmt::format("PolyPass: empty pipeline spec '{}'", spec));

  std::vector<polyast::CompileEvent> items;
  items.push_back(std::move(serialiseEvent));

  std::optional<polyast::Program> currentProgram;
  for (const auto &[idx, stepStrings] : groups) {
    auto &runner = *reg.plugins[idx];
    std::string err;
    const std::string runnerTag(runner.tag());
    auto [out, runEvent] = timed(runnerTag, "", [&] { return runner.runPasses(stepStrings, bytes, err); });
    if (!err.empty()) throw std::logic_error(fmt::format("PolyPass {} ({}): {}", spec, runnerTag, err));
    runEvent.data = fmt::format("bytes={}", out.size());

    auto [result, decodeEvent] = timed("polyast_msgpack_deserialise_cpp", fmt::format("bytes={}", out.size()),
                                       [&] { return polyast::passrunresult_from_msgpack(out.data(), out.data() + out.size()); });

    runEvent.items = std::move(result.event.items);
    items.push_back(std::move(runEvent));
    items.push_back(std::move(decodeEvent));
    currentProgram = std::move(result.program);
    bytes = std::move(out);
  }

  return polyast::PassRunResult(std::move(*currentProgram), polyast::CompileEvent(rootEpoch, compiler::elapsedNs(rootStart), "PolyPass",
                                                                                  std::string(spec), std::move(items)));
}

polyast::Program compiler::runPipeline(const polyast::Program &program, const std::string &spec) {
  return runPipelineChain(polyast::Program(program), spec).program;
}

compiler::Staged compiler::stage(const polyast::Program &program, const Options &options) {
  return stage(polyast::Program(program), options);
}

compiler::Staged compiler::stage(polyast::Program &&program, const Options &options) {
  initialise();
  const bool hadEntry = program.entry.has_value();

  const bool isGpuTarget = runtime::targetPlatformKind(options.target) == runtime::PlatformKind::Managed;
  if (isGpuTarget && program.entry && !program.entry->decl.affinity.is<polyast::FunctionAffinity::Offload>())
    throw std::logic_error(
        fmt::format("GPU target {} requires an Offload-affinity entry function; got Host-affinity", magic_enum::enum_name(options.target)));

  std::vector<polyast::CompileEvent> preEvents;
  auto effective = std::move(program);
  {
    auto passRun = runPipelineChain(std::move(effective), options.pipelineSpec);
    effective = std::move(passRun.program);
    preEvents.emplace_back(std::move(passRun.event));
  }

  switch (options.target) {
    case compiletime::Target::Source_C_C11:
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_OpenCL2_0:
    case compiletime::Target::Source_C_Metal1_0: {
      const auto *const strict =
          options.target == compiletime::Target::Source_C_C11 || options.target == compiletime::Target::Source_C_OpenCL2_0 ? "false"
                                                                                                                           : "true";
      const auto *const metal = options.target == compiletime::Target::Source_C_Metal1_0 ? "true" : "false";
      auto passRun = runPipelineChain(std::move(effective), fmt::format("SourceStorageLegalise(elideRecursivelyEmptyAggregates={});"
                                                                        "SourcePointerLegalise(requiresConcreteSpaces={});"
                                                                        "SourceSelectionLegalise;SourceNameNormalise(metalKeywords={})",
                                                                        metal, strict, metal));
      effective = std::move(passRun.program);
      preEvents.emplace_back(std::move(passRun.event));
      break;
    }
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel: {
      if (compiletime::spirvKernelGeneric(options.arch)) break;
      auto passRun = runPipelineChain(std::move(effective), "SourcePointerLegalise(requiresConcreteSpaces=true)");
      effective = std::move(passRun.program);
      preEvents.emplace_back(std::move(passRun.event));
      break;
    }
    default: break;
  }

  if (options.hostMirroring) {
    if (!effective.entry)
      return {.program = {},
              .options = options,
              .hadEntry = hadEntry,
              .events = std::move(preEvents),
              .error = "hostMirroring: pipeline removed the Program entry"};
    auto hostFns = std::vector<polyast::Function>{*effective.entry}                                                       //
                   | concat(effective.functions)                                                                          //
                   | filter([](const auto &f) { return f.decl.affinity.template is<polyast::FunctionAffinity::Host>(); }) //
                   | to_vector();
    if (hostFns.empty())
      return {.program = {},
              .options = options,
              .hadEntry = hadEntry,
              .events = std::move(preEvents),
              .error = "hostMirroring: pipeline produced no Host-affinity functions"};
    effective = polyast::Program(hostFns.front(), std::vector<polyast::Function>(std::next(hostFns.begin()), hostFns.end()), effective.defs,
                                 effective.phase, effective.metadata);
  }

  return {.program = std::move(effective), .options = options, .hadEntry = hadEntry, .events = std::move(preEvents), .error = {}};
}

polyast::CompileResult compiler::emit(Staged staged, const compiletime::OptLevel &opt) {
  if (!staged.error.empty()) return {{}, {}, std::move(staged.events), {}, std::move(staged.error), {}};
  const auto &options = staged.options;
  const auto mkBackend = [&]() -> std::unique_ptr<backend::Backend> {
    switch (options.target) {
      case compiletime::Target::Object_LLVM_HOST:
      case compiletime::Target::Object_LLVM_x86_64:
      case compiletime::Target::Object_LLVM_AArch64:
      case compiletime::Target::Object_LLVM_ARM:
      case compiletime::Target::Object_LLVM_NVPTX64:
      case compiletime::Target::Object_LLVM_AMDGCN:
      case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
      case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
      case compiletime::Target::Object_LLVM_SPIRV_GLCompute:                                                           //
        return std::make_unique<backend::LLVMBackend>(toLLVMBackendOptions(options));                                  //
      case compiletime::Target::Source_C_OpenCL1_1:                                                                    //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::OpenCL1_1, options.workgroupMemoryBytes); //
      case compiletime::Target::Source_C_OpenCL2_0:
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::OpenCL2_0, options.workgroupMemoryBytes);
      case compiletime::Target::Source_C_Metal1_0:                                                                  //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::MSL1_0, options.workgroupMemoryBytes); //
      case compiletime::Target::Source_C_C11:                                                                       //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::C11, options.workgroupMemoryBytes);    //
      default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
    }
  };

  auto &effective = *staged.program;
  polyast::CompileResult c = mkBackend()->compileProgram(effective, opt);
  if (staged.hadEntry && !effective.entry) throw std::logic_error("pipeline removed the Program entry");
  if (effective.entry) c.entryArgs = effective.entry->decl.args ^ map([](const auto &a) { return a.named; });
  for (const auto &entry : effective.metadata)
    if (entry.key.starts_with(conventions::HostAbiNoteMeta)) c.messages += (c.messages.empty() ? "" : "\n") + entry.value;
  c.events ^= concat(staged.events);
  std::stable_sort(c.events.begin(), c.events.end(), [](const auto &l, const auto &r) { return l.epochMillis < r.epochMillis; });
  return c;
}

polyast::CompileResult compiler::compile(const polyast::Program &program, const Options &options, const compiletime::OptLevel &opt) {
  return emit(stage(program, options), opt);
}

polyast::CompileResult compiler::compile(const polyast::Bytes &astBytes, const Options &options, const compiletime::OptLevel &opt) {
  const auto astStart = nowMono();
  const auto program = deserialiseProgram(astBytes);
  // XXX `ast_deserialise` happens strictly before any event from compile(Program), so prepend
  // rather than re-running stable_sort over the merged list.
  auto c = compile(program, options, opt);
  c.events.insert(c.events.begin(), polyast::CompileEvent(nowMs(), elapsedNs(astStart), "ast_deserialise", "", {}));
  return c;
}
