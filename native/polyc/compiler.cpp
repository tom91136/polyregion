#include "compiler.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>

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
#include "polyast_codec.h"
#include "qjs_runner.h"

#ifndef POLYPASS_JS_DEV_PATH
  #define POLYPASS_JS_DEV_PATH ""
#endif

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
[[maybe_unused]] void polypassJsAnchor() {}

// Resolution order: $POLYPASS_JS env, <exe-dir>/polypass.js, <exe-dir>/../lib/polypass.js,
// then the build-time POLYPASS_JS_DEV_PATH baked in by CMake. Returns "" if unfound.
std::string findPolypassJs() {
  namespace fs = llvm::sys::fs;
  namespace path = llvm::sys::path;
  if (auto env = std::getenv("POLYPASS_JS"); env && *env && fs::exists(env)) return env;
  const auto exe = fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&polypassJsAnchor));
  if (!exe.empty()) {
    const auto dir = path::parent_path(exe);
    for (const llvm::StringRef rel : {"polypass.js", "../lib/polypass.js"}) {
      llvm::SmallString<256> candidate(dir);
      path::append(candidate, rel);
      if (!fs::exists(candidate)) continue;
      llvm::SmallString<256> resolved;
      if (fs::real_path(candidate, resolved)) return candidate.str().str();
      return resolved.str().str();
    }
  }
  if (constexpr std::string_view dev = POLYPASS_JS_DEV_PATH; !dev.empty() && fs::exists(dev)) return std::string(dev);
  return {};
}
} // namespace

static polypass::JsPassRunner &sharedJsRunner() {
  static polypass::JsPassRunner runner;
  static std::once_flag loaded;
  std::call_once(loaded, [] {
    const auto path = findPolypassJs();
    if (path.empty()) throw std::logic_error("polyc: polypass.js not found (set $POLYPASS_JS or install the dist)");
    if (auto err = runner.loadModule(read_string(path), path); !err.empty())
      throw std::logic_error("polyc: failed to load polypass.js: " + err);
  });
  return runner;
}

static polyast::PassRunResult runJsPipeline(const polyast::Program &p, std::string_view spec) {
  const auto rootEpoch = compiler::nowMs();
  const auto rootStart = compiler::nowMono();
  auto timed = [](auto &&name, auto &&data, auto &&f) {
    const auto epoch = compiler::nowMs();
    const auto start = compiler::nowMono();
    auto out = f();
    return std::pair{std::move(out), polyast::CompileEvent(epoch, compiler::elapsedNs(start), name, data, {})};
  };

  auto [in, serialiseEvent] = timed("polyast_msgpack_serialise_cpp", "", [&] { return polyast::program_to_msgpack(p); });
  serialiseEvent.data = fmt::format("bytes={}", in.size());

  std::string err;
  auto [out, jsEvent] = timed("polypass_js", "", [&] { return sharedJsRunner().runPipeline(spec, in, err); });
  if (!err.empty()) throw std::logic_error(fmt::format("polypass {}: {}", spec, err));
  jsEvent.data = fmt::format("bytes={}", out.size());

  auto [result, deserialiseEvent] = timed("polyast_msgpack_deserialise_cpp", fmt::format("bytes={}", out.size()),
                                          [&] { return polyast::passrunresult_from_msgpack(out.data(), out.data() + out.size()); });

  jsEvent.items = std::move(result.event.items);
  std::vector<polyast::CompileEvent> items{std::move(serialiseEvent), std::move(jsEvent), std::move(deserialiseEvent)};
  result.event = polyast::CompileEvent(rootEpoch, compiler::elapsedNs(rootStart), "polypass", std::string(spec), std::move(items));
  return result;
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
    auto passRun = runJsPipeline(effective, "FullOpt");
    effective = std::move(passRun.program);
    preEvents.emplace_back(std::move(passRun.event));
  }

  polyast::CompileResult c = mkBackend()->compileProgram(effective, opt);
  c.events ^= concat_inplace(preEvents);
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
