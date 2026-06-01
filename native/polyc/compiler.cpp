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
  static std::once_flag flag;
  std::call_once(flag, []() { backend::llvmc::initialise(); });
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

  switch (options.target) {
    case compiletime::Target::Object_LLVM_HOST: {
      auto host = backend::llvmc::defaultHostTriple();
      validate(host.getArch());
      switch (host.getArch()) {
        case llvm::Triple::ArchType::x86_64: return {.target = backend::LLVMBackend::Target::x86_64, .arch = options.arch};
        case llvm::Triple::ArchType::aarch64: return {.target = backend::LLVMBackend::Target::AArch64, .arch = options.arch};
        case llvm::Triple::ArchType::arm: return {.target = backend::LLVMBackend::Target::ARM, .arch = options.arch};
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
};

PluginRegistry &sharedPlugins() {
  static PluginRegistry reg;
  static std::once_flag loaded;
  std::call_once(loaded, [] {
    std::string err;
    auto refs = polypass::resolvePlugins(err);
    if (!err.empty()) throw std::logic_error(fmt::format("polyc: {}", err));
    for (auto &ref : refs) {
      std::unique_ptr<polypass::PassRunner> runner =
          ref.kind == polypass::PluginKind::Js ? std::unique_ptr<polypass::PassRunner>(std::make_unique<polypass::JsPassRunner>(ref.path))
                                               : std::unique_ptr<polypass::PassRunner>(std::make_unique<polypass::DsoPassRunner>(ref.path));
      if (auto rerr = runner->load(); !rerr.empty())
        throw std::logic_error(fmt::format("polyc: failed to load PolyPass plugin {}: {}", ref.path, rerr));
      const size_t idx = reg.plugins.size();
      for (const auto &name : runner->passNames()) {
        if (auto it = reg.ownerByPass.find(name); it != reg.ownerByPass.end()) {
          fmt::print(stderr, "polyc: pass '{}' from {} overrides earlier definition from {}\n", name, runner->tag(),
                     reg.plugins[it->second]->tag());
          it->second = idx;
        } else {
          reg.ownerByPass.emplace(name, idx);
        }
      }
      reg.plugins.push_back(std::move(runner));
    }
  });
  return reg;
}

std::string bareName(const std::string &step) {
  const auto paren = step.find('(');
  return paren == std::string::npos ? step : trim(step.substr(0, paren));
}

} // namespace

static polyast::PassRunResult runPipelineChain(const polyast::Program &p, std::string_view spec) {
  const auto rootEpoch = compiler::nowMs();
  const auto rootStart = compiler::nowMono();
  auto timed = [](auto &&name, auto &&data, auto &&f) {
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

  const auto stepsWithOwner = (std::string(spec) ^ split(";"))                                  //
                              | map([](auto &s) { return trim(s); })                            //
                              | filter([](auto &s) { return !s.empty(); })                      //
                              | map([&](auto &step) { return std::pair{ownerOf(step), step}; }) //
                              | to_vector();

  std::vector<std::pair<size_t, std::vector<std::string>>> groups;
  for (const auto &[owner, step] : stepsWithOwner) {
    if (groups.empty() || groups.back().first != owner) groups.emplace_back(owner, std::vector<std::string>{});
    groups.back().second.emplace_back(step);
  }
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

polyast::CompileResult compiler::compile(const polyast::Program &program, const Options &options, const compiletime::OptLevel &opt) {
  initialise();
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
      case compiletime::Target::Object_LLVM_SPIRV_GLCompute:                             //
        return std::make_unique<backend::LLVMBackend>(toLLVMBackendOptions(options));    //
      case compiletime::Target::Source_C_OpenCL1_1:                                      //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::OpenCL1_1); //
      case compiletime::Target::Source_C_Metal1_0:                                       //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::MSL1_0);    //
      case compiletime::Target::Source_C_C11:                                            //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::C11);       //
      default: throw std::logic_error(fmt::format("Unknown target: {}", magic_enum::enum_name(options.target)));
    }
  };

  std::vector<polyast::CompileEvent> preEvents;
  auto effective = program;
  {
    const std::string_view spec =
        options.pipelineSpec.empty() ? std::string_view{DefaultPipelineSpec} : std::string_view{options.pipelineSpec};
    auto passRun = runPipelineChain(effective, spec);
    effective = std::move(passRun.program);
    preEvents.emplace_back(std::move(passRun.event));
  }

  polyast::CompileResult c = mkBackend()->compileProgram(effective, opt);
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
