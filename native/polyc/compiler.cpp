#include "compiler.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"
#include "nlohmann/json.hpp"

#include "polyregion/compat.h"
#include "polyregion/io.hpp"
#include "polyregion/llvm_utils.hpp"
#include "polyregion/polypackage.h"
#include "polyregion/polypackage_symbols.h"

#include "ast.h"
#include "backend/c_source.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "dso_runner.h"
#include "js_runner.h"
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
        auto host = backend::llvmc::defaultHostTriple();
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
        validate(llvm::Triple::ArchType::amdgcn);
        return {.target = backend::LLVMBackend::Target::AMDGCN, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
        return {.target = backend::LLVMBackend::Target::SPIRV32_Kernel, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
        return {.target = backend::LLVMBackend::Target::SPIRV64_Kernel, .arch = options.arch};
      case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
        return {.target = backend::LLVMBackend::Target::SPIRV_GLCompute, .arch = options.arch};
      case compiletime::Target::Source_C_OpenCL1_1: //
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
    case compiletime::Target::Source_C_Metal1_0:
      throw std::logic_error(fmt::format("Not available for source target {}", magic_enum::enum_name(options.target)));
    default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
  }
}

std::vector<polyast::StructLayout> compiler::layoutOf(const polyast::Bytes &bytes, const Options &options) {
  return layoutOf(deserialiseStructDefs(bytes), options);
}

namespace {

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
        if (auto it = result.ownerByPass.find(name); it != result.ownerByPass.end()) {
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
compiler::PackageResult<T> invokePackage(const std::vector<uint8_t> &request, const std::string_view operation, Decode decode) {
  auto &registry = sharedPlugins();
  if (registry.packageProviders.size() != 1)
    return {{}, {fmt::format("expected one package-service plugin, found {}", registry.packageProviders.size())}};
  auto &provider = *registry.plugins[registry.packageProviders.front()];
  std::lock_guard lock(packageMutex());
  const auto version = provider.packageAbiVersion();
  if (!version || *version != POLYPACKAGE_ABI_VERSION)
    return {{},
            {fmt::format("PolyPackage ABI mismatch: service={}, polyc={}", version ? std::to_string(*version) : "<missing>",
                         POLYPACKAGE_ABI_VERSION)}};
  String error;
  const auto output = provider.runPackage(operation, request, error);
  if (!error.empty()) return {{}, packageErrors(std::move(error))};
  if (output.empty()) return {{}, {"package service returned an empty successful result"}};
  try {
    return {{decode(output.data(), output.data() + output.size())}, {}};
  } catch (const std::exception &error) {
    return {{}, {std::string("cannot decode package-service result: ") + error.what()}};
  }
}

std::string bareName(const std::string &step) {
  const auto prefix = std::string_view(step) ^ take_while([](char c) { return c != '('; });
  return std::string(prefix ^ trim());
}

} // namespace

compiler::PackageResult<polyast::Package> compiler::linkPackage(const polyast::PackageLinkRequest &request) {
  return invokePackage<polyast::Package>(
      polyast::packagelinkrequest_to_msgpack(request), polypackage::abi::LinkPackage,
      [](const uint8_t *begin, const uint8_t *end) { return polyast::package_service_result_from_msgpack(begin, end); });
}

compiler::PackageResult<polyast::PackageSymResolvedProgram> compiler::resolvePackageSym(const polyast::PackageSymRequest &request) {
  return invokePackage<polyast::PackageSymResolvedProgram>(
      polyast::packagesymrequest_to_msgpack(request), polypackage::abi::ResolveSym,
      [](const uint8_t *begin, const uint8_t *end) { return polyast::resolvedsymprogram_from_msgpack(begin, end); });
}

namespace {

std::string packageEntryPipeline(const compiletime::Target target, const std::optional<int> stackDepth) {
  const auto opt = stackDepth ? fmt::format("DeadFunctionElimination;Intrinsify;RecursionLower(maxDepth={});FnInline;Intrinsify;"
                                            "KernelCaptureFlatten;FullOpt(level=1)",
                                            *stackDepth)
                              : std::string("DeadFunctionElimination;Intrinsify;RecursionLower;FnInline;Intrinsify;KernelCaptureFlatten;"
                                            "FullOpt(level=1)");
  switch (target) {
    case compiletime::Target::Object_LLVM_SPIRV_GLCompute:
      return opt
             + ";StructuredExit;PartialEval(canonicaliseAddresses=true);ArenaView;RegionRespace;"
               "VerifyAnchors(strict=true)";
    case compiletime::Target::Object_LLVM_SPIRV32_Kernel:
    case compiletime::Target::Object_LLVM_SPIRV64_Kernel:
    case compiletime::Target::Source_C_OpenCL1_1:
    case compiletime::Target::Source_C_Metal1_0: return opt + ";SubgroupLower;StructuredExit;RegionRespace;ArenaLower";
    default: return opt + ";StructuredExit";
  }
}

} // namespace

compiler::PackageResult<polyast::PackageSymCompileResult>
compiler::compilePackageSym(const polyast::PackageSymRequest &request, const compiletime::Target hostTarget, const std::string &hostArch,
                            const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                            const std::optional<int> stackDepth) {
  auto resolved = resolvePackageSym(request);
  if (!resolved) return {{}, std::move(resolved.errors)};
  initialise();
  auto program = resolved.value->program;
  for (auto &function : program.functions)
    if (function.convention.is<polyast::CallConvention::OffloadEntry>()) function.visibility = polyast::FunctionVisibility::Exported();

  const auto host = compile(program, Options{hostTarget, hostArch, "FullOpt(level=0)", true}, compiletime::OptLevel::O3);
  if (!host.binary) return {{}, {"resolved Sym program compilation failed: " + host.messages}};

  std::vector<polyast::PackageSymCompiledObject> remoteObjects;
  for (const auto &entry : program.functions) {
    if (!entry.convention.is<polyast::CallConvention::OffloadEntry>()) continue;
    auto moduleName = polyast::fqcn(entry.decl.name);
    for (auto &c : moduleName)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    auto entryProgram = program.withEntry(entry);
    std::vector<polyast::Function> functions;
    functions.reserve(program.functions.size() - 1);
    for (const auto &function : program.functions) {
      if (function.decl.name == entry.decl.name) continue;
      functions.emplace_back(
          function.convention.is<polyast::CallConvention::OffloadEntry>()
              ? function.withVisibility(polyast::FunctionVisibility::Internal()).withConvention(polyast::CallConvention::RegularCall())
              : function);
    }
    entryProgram.functions = std::move(functions);
    for (const auto &[target, arch] : deviceTargets) {
      const auto pipeline = packageEntryPipeline(target, stackDepth);
      const auto device = compile(entryProgram, Options{target, arch, pipeline}, compiletime::OptLevel::O3);
      if (!device.binary) return {{}, {"package entry compilation failed for " + moduleName + ": " + device.messages}};
      const auto format = runtime::moduleFormatOf(target);
      if (!format) continue;
      remoteObjects.emplace_back(moduleName, static_cast<int32_t>(*format), static_cast<int32_t>(runtime::targetPlatformKind(target)),
                                 device.features, *device.binary);
    }
  }
  return {{polyast::PackageSymCompileResult(std::move(*resolved.value), std::move(*host.binary), std::move(remoteObjects))}, {}};
}

static polyast::PassRunResult runPipelineChain(const polyast::Program &p, std::string_view rawSpec) {
  const std::string_view spec = rawSpec.empty() ? std::string_view{compiler::DefaultPipelineSpec} : rawSpec;
  const auto rootEpoch = compiler::nowMs();
  const auto rootStart = compiler::nowMono();
  auto timed = [](const auto &name, const auto &data, const auto &f) {
    const auto epoch = compiler::nowMs();
    const auto start = compiler::nowMono();
    auto out = f();
    return std::pair{std::move(out), polyast::CompileEvent(epoch, compiler::elapsedNs(start), name, data, {})};
  };

  auto [bytes, serialiseEvent] = timed("polyast_msgpack_serialise_cpp", "", [&] { return polyast::program_to_msgpack(p); });
  serialiseEvent.data = fmt::format("bytes={}", bytes.size());

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

  polyast::Program currentProgram = p;
  for (auto &[idx, stepStrings] : groups) {
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

  return polyast::PassRunResult(std::move(currentProgram), polyast::CompileEvent(rootEpoch, compiler::elapsedNs(rootStart), "PolyPass",
                                                                                 std::string(spec), std::move(items)));
}

polyast::Program compiler::runPipeline(const polyast::Program &program, const std::string &spec) {
  return runPipelineChain(program, spec).program;
}

polyast::CompileResult compiler::compile(const polyast::Program &program, const Options &options, const compiletime::OptLevel &opt) {
  initialise();
  const bool hadEntry = program.entry.has_value();
  auto mkBackend = [&]() -> std::unique_ptr<backend::Backend> {
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
      case compiletime::Target::Source_C_Metal1_0:                                                                     //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::MSL1_0, options.workgroupMemoryBytes);    //
      case compiletime::Target::Source_C_C11:                                                                          //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::C11, options.workgroupMemoryBytes);       //
      default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
    }
  };

  const bool isGpuTarget = runtime::targetPlatformKind(options.target) == runtime::PlatformKind::Managed;
  if (isGpuTarget && program.entry && !program.entry->decl.affinity.is<polyast::FunctionAffinity::Offload>())
    throw std::logic_error(
        fmt::format("GPU target {} requires an Offload-affinity entry function; got Host-affinity", magic_enum::enum_name(options.target)));

  std::vector<polyast::CompileEvent> preEvents;
  auto effective = program;
  {
    auto passRun = runPipelineChain(effective, options.pipelineSpec);
    effective = std::move(passRun.program);
    preEvents.emplace_back(std::move(passRun.event));
  }

  if (options.hostMirroring) {
    if (!effective.entry) return {{}, {}, preEvents, {}, "hostMirroring: pipeline removed the Program entry", {}};
    auto hostFns = std::vector<polyast::Function>{*effective.entry}                                                       //
                   | concat(effective.functions)                                                                          //
                   | filter([](const auto &f) { return f.decl.affinity.template is<polyast::FunctionAffinity::Host>(); }) //
                   | to_vector();
    if (hostFns.empty()) return {{}, {}, preEvents, {}, "hostMirroring: pipeline produced no Host-affinity functions", {}};
    effective = polyast::Program(hostFns.front(), std::vector<polyast::Function>(std::next(hostFns.begin()), hostFns.end()), effective.defs,
                                 effective.phase, effective.metadata);
  }

  polyast::CompileResult c = mkBackend()->compileProgram(effective, opt);
  if (hadEntry && !effective.entry) throw std::logic_error("pipeline removed the Program entry");
  if (effective.entry) c.entryArgs = effective.entry->decl.args ^ map([](const auto &a) { return a.named; });
  c.events ^= concat(preEvents);
  std::stable_sort(c.events.begin(), c.events.end(), [](const auto &l, const auto &r) { return l.epochMillis < r.epochMillis; });
  return c;
}

polyast::CompileResult compiler::compile(const polyast::Bytes &astBytes, const Options &options, const compiletime::OptLevel &opt) {
  auto astStart = nowMono();
  auto program = deserialiseProgram(astBytes);
  // XXX `ast_deserialise` happens strictly before any event from compile(Program), so prepend
  // rather than re-running stable_sort over the merged list.
  auto c = compile(program, options, opt);
  c.events.insert(c.events.begin(), polyast::CompileEvent(nowMs(), elapsedNs(astStart), "ast_deserialise", "", {}));
  return c;
}
