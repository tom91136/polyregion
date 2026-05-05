#include "polyregion/compat.h"

#include <mutex>

#include "ast.h"
#include "compiler.h"

#include "backend/c_source.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "qjs_runner.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"
#include "nlohmann/json.hpp"
#include "polyregion/io.hpp"
#include "polyregion/llvm_utils.hpp"

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

static json deserialiseAst(const polyast::Bytes &astBytes) {
  try {
    const auto json = nlohmann::json::from_msgpack(astBytes.data(), astBytes.data() + astBytes.size());
    // the JSON comes in versioned with the hash
    return polyast::hashed_from_json(json);
  } catch (nlohmann::json::exception &e) {
    throw std::logic_error(fmt::format("Unable to parse packed ast: {}", e.what()));
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
  const json json = deserialiseAst(bytes);
  const auto structs = json | map([](auto &x) { return polyast::structdef_from_json(x); }) | to_vector();
  return layoutOf(structs, options);
}

static void sortEvents(polyast::CompileResult &c) {
  c.events = c.events ^ sort_by([](auto &x) { return x.epochMillis; });
}

static polypass::JsPassRunner &sharedJsRunner() {
  static polypass::JsPassRunner runner;
  static std::once_flag loaded;
  std::call_once(loaded, [] {
    const auto path = polypass::JsPassRunner::findBundle();
    if (path.empty()) throw std::logic_error("polyc: polypass.js not found (set $POLYPASS_JS or install the dist)");
    if (auto err = runner.loadModule(read_string(path), path); !err.empty())
      throw std::logic_error("polyc: failed to load polypass.js: " + err);
  });
  return runner;
}

static polyast::Program runJsPass(const polyast::Program &p, std::string_view passName) {
  auto bytes = nlohmann::json::to_msgpack(polyast::hashed_to_json(polyast::program_to_json(p)));
  std::vector<uint8_t> in(bytes.begin(), bytes.end());
  std::string err;
  auto out = sharedJsRunner().runPass(passName, in, err);
  // auto out = in;
  if (!err.empty()) throw std::logic_error(fmt::format("polypass {}: {}", passName, err));
  const auto outJson = polyast::hashed_from_json(nlohmann::json::from_msgpack(out.data(), out.data() + out.size()));
  return polyast::program_from_json(outJson);
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
    const auto t0 = nowMono();
    effective = runJsPass(effective, "DeadStructElimination");
    preEvents.emplace_back(nowMs(), elapsedNs(t0), "polypass_DeadStructElimination", "");
  }

  polyast::CompileResult c = mkBackend()->compileProgram(effective, opt);
  c.events ^= concat_inplace(preEvents);
  sortEvents(c);
  return c;
}

polyast::CompileResult compiler::compile(const polyast::Bytes &astBytes, const Options &options, const compiletime::OptLevel &opt) {

  std::vector<polyast::CompileEvent> events;

  auto jsonStart = nowMono();
  json json = deserialiseAst(astBytes);
  events.emplace_back(nowMs(), elapsedNs(jsonStart), "ast_deserialise", "");

  auto astLift = nowMono();
  auto program = polyast::program_from_json(json);
  events.emplace_back(nowMs(), elapsedNs(astLift), "ast_lift", "");

  auto c = compile(program, options, opt);
  c.events ^= concat_inplace(events);
  sortEvents(c);
  return c;
}
