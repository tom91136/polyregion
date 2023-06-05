#include <atomic>
#include <iostream>

#include "ast.h"
#include "backend/c_source.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "compiler.h"
#include "generated/polyast_codec.h"
#include "json.hpp"
#include "llvm_utils.hpp"
#include "utils.hpp"

using namespace polyregion;

static std::atomic_bool init = false;

compiler::TimePoint compiler::nowMono() { return MonoClock::now(); }

int64_t compiler::elapsedNs(const TimePoint &a, const TimePoint &b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

int64_t compiler::nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::optional<compiler::Target> compiler::targetFromOrdinal(std::underlying_type_t<compiler::Target> ordinal) {
  auto target = static_cast<Target>(ordinal);
  switch (target) {
    case Target::Object_LLVM_HOST:
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_ARM:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN:
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_C11:
    case Target::Source_C_Metal1_0:
    case Target::Object_LLVM_SPIRV32:
    case Target::Object_LLVM_SPIRV64:
      return target;
      // XXX do not add default here, see  -Werror=switch
  }
}

std::optional<compiler::Opt> compiler::optFromOrdinal(std::underlying_type_t<compiler::Opt> ordinal) {
  auto target = static_cast<Opt>(ordinal);
  switch (target) {
    case Opt::O0:
    case Opt::O1:
    case Opt::O2:
    case Opt::O3:
    case Opt::Ofast:
      return target;
      // XXX do not add default here, see  -Werror=switch
  }
}

std::ostream &compiler::operator<<(std::ostream &os, const compiler::Compilation &compilation) {
  os << "Compilation {"                                                                                            //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)") //
     << "\n  messages: `" << compilation.messages << "`"                                                           //
     << "\n  events:\n";

  for (auto &e : compilation.events) {
    os << "    [" << e.epochMillis << ", +" << (double(e.elapsedNanos) / 1e6) << "ms] " << e.name;
    if (e.data.empty()) continue;
    os << ":\n";
    std::stringstream ss(e.data);
    std::string l;
    size_t ln = 0;
    while (std::getline(ss, l, '\n')) {
      ln++;
      os << "    " << std::setw(3) << ln << "│" << l << '\n';
    }
    os << "       ╰───\n";
  }
  os << "\n}";
  return os;
}

std::ostream &compiler::operator<<(std::ostream &os, const compiler::Member &member) {
  return os << "Member { "                                 //
            << "name: " << member.name                     //
            << ", offsetInBytes: " << member.offsetInBytes //
            << ", sizeInBytes: " << member.sizeInBytes     //
            << "}";
}

std::ostream &compiler::operator<<(std::ostream &os, const compiler::Layout &layout) {
  os << "Layout { "                               //
     << "\n  sizeInBytes: " << layout.sizeInBytes //
     << "\n  alignment: " << layout.alignment     //
     << "\n  members: ";                          //
  for (auto &&l : layout.members)
    os << "\n    " << l;
  return os << "}";
}

void compiler::initialise() {
  if (!init) {
    init = true;
    backend::llvmc::initialise();
  }
}

static json deserialiseAst(const compiler::Bytes &astBytes) {
  try {
    auto json = nlohmann::json::from_msgpack(astBytes.data(), astBytes.data() + astBytes.size());
    // the JSON comes in versioned with the hash
    return polyast::hashed_from_json(json);
  } catch (nlohmann::json::exception &e) {
    throw std::logic_error("Unable to parse packed ast:" + std::string(e.what()));
  }
}

static backend::LLVMBackend::Options toLLVMBackendOptions(const compiler::Options &options) {

  auto validate = [&](llvm::Triple::ArchType arch) {
    if (!llvm_shared::isCPUTargetSupported(options.arch, arch)) {
      throw std::logic_error("Unsupported target CPU `" + options.arch + "` on `" + llvm::Triple::getArchTypeName(arch).str() + "`");
    }
  };

  switch (options.target) {
    case compiler::Target::Object_LLVM_HOST: {
      auto host = backend::llvmc::defaultHostTriple();
      validate(host.getArch());
      switch (host.getArch()) {
        case llvm::Triple::ArchType::x86_64: return {.target = backend::LLVMBackend::Target::x86_64, .arch = options.arch};
        case llvm::Triple::ArchType::aarch64: return {.target = backend::LLVMBackend::Target::AArch64, .arch = options.arch};
        case llvm::Triple::ArchType::arm: return {.target = backend::LLVMBackend::Target::ARM, .arch = options.arch};
        default: throw std::logic_error("Unsupported host triplet: " + host.str());
      }
    }
    case compiler::Target::Object_LLVM_x86_64:
      validate(llvm::Triple::ArchType::x86_64);
      return {.target = backend::LLVMBackend::Target::x86_64, .arch = options.arch};
    case compiler::Target::Object_LLVM_AArch64:
      validate(llvm::Triple::ArchType::aarch64);
      return {.target = backend::LLVMBackend::Target::AArch64, .arch = options.arch};
    case compiler::Target::Object_LLVM_ARM:
      validate(llvm::Triple::ArchType::arm);
      return {.target = backend::LLVMBackend::Target::ARM, .arch = options.arch};
    case compiler::Target::Object_LLVM_NVPTX64:
      validate(llvm::Triple::ArchType::nvptx64);
      return {.target = backend::LLVMBackend::Target::NVPTX64, .arch = options.arch};
    case compiler::Target::Object_LLVM_AMDGCN:
      validate(llvm::Triple::ArchType::amdgcn);
      return {.target = backend::LLVMBackend::Target::AMDGCN, .arch = options.arch};
    case compiler::Target::Object_LLVM_SPIRV32: return {.target = backend::LLVMBackend::Target::SPIRV32, .arch = options.arch};
    case compiler::Target::Object_LLVM_SPIRV64: return {.target = backend::LLVMBackend::Target::SPIRV64, .arch = options.arch};
    case compiler::Target::Source_C_OpenCL1_1: //
    case compiler::Target::Source_C_Metal1_0:  //
    case compiler::Target::Source_C_C11:       //
      throw std::logic_error("Not an object target");
  }
}

std::vector<compiler::Layout> compiler::layoutOf(const std::vector<polyast::StructDef> &defs, const Options &options) {

  switch (options.target) {
    case Target::Object_LLVM_HOST:
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_ARM:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN:
    case Target::Object_LLVM_SPIRV32:
    case Target::Object_LLVM_SPIRV64: {

      auto llvmOptions = toLLVMBackendOptions(options);
      auto dataLayout = llvmOptions.targetInfo().resolveDataLayout();

      llvm::LLVMContext c;
      backend::LLVMBackend::AstTransformer xform(llvmOptions, c);
      std::vector<compiler::Layout> layouts;
      xform.addDefs(defs);
      std::unordered_map<polyast::Sym, polyast::StructDef> lut(defs.size());
      for (auto &d : defs)
        lut.emplace(d.name, d);
      for (auto &[sym, structTy] : xform.getStructTypes()) {
        if (auto it = lut.find(sym); it != lut.end()) {
          auto layout = dataLayout.getStructLayout(structTy);
          std::vector<compiler::Member> members;
          for (size_t i = 0; i < it->second.members.size(); ++i) {
            members.emplace_back(it->second.members[i].named,                             //
                                 layout->getElementOffset(i),                             //
                                 dataLayout.getTypeAllocSize(structTy->getElementType(i)) //
            );
          }
          layouts.emplace_back(sym, layout->getSizeInBytes(), layout->getAlignment().value(), members);
        } else
          throw std::logic_error("Cannot find symbol " + to_string(sym) + " from domain");
      }
      return layouts;
    }
    case Target::Source_C_C11:
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_Metal1_0:
      // TODO we need to submit a kernel and execute it to get the offsets
      throw std::logic_error("Not available for source targets");
  }
}

std::vector<compiler::Layout> compiler::layoutOf(const Bytes &sdef, const Options &options) {
  json json = deserialiseAst(sdef);
  std::vector<polyast::StructDef> sdefs;
  std::transform(json.begin(), json.end(), std::back_inserter(sdefs), &polyast::structdef_from_json);
  return layoutOf(sdefs, options);
}

static void sortEvents(compiler::Compilation &c) {
  std::sort(c.events.begin(), c.events.end(), [](const auto &l, const auto &r) { return l.epochMillis < r.epochMillis; });
}

compiler::Compilation compiler::compile(const polyast::Program &program, const Options &options, const Opt &opt) {
  if (!init) {
    return Compilation{"initialise was not called before"};
  }

  auto mkBackend = [&]() -> std::unique_ptr<backend::Backend> {
    switch (options.target) {
      case Target::Object_LLVM_HOST:
      case Target::Object_LLVM_x86_64:
      case Target::Object_LLVM_AArch64:
      case Target::Object_LLVM_ARM:
      case Target::Object_LLVM_NVPTX64:
      case Target::Object_LLVM_AMDGCN:
      case Target::Object_LLVM_SPIRV32:
      case Target::Object_LLVM_SPIRV64:                                                  //
        return std::make_unique<backend::LLVMBackend>(toLLVMBackendOptions(options));    //
      case Target::Source_C_OpenCL1_1:                                                   //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::OpenCL1_1); //
      case Target::Source_C_Metal1_0:                                                    //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::MSL1_0);    //
      case Target::Source_C_C11:                                                         //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::C11);       //
    }
  };

  Compilation c;
  //  try {
  c = mkBackend()->compileProgram(program, opt);
  //  } catch (const std::exception &e) {
  //    c.messages = e.what();
  //    throw e;
  //  }
  sortEvents(c);
  return c;
}

compiler::Compilation compiler::compile(const Bytes &astBytes, const Options &options, const Opt &opt) {

  std::vector<Event> events;

  //  std::cout << "[polyregion-native] Len  : " << astBytes.size() << std::endl;
  auto jsonStart = nowMono();
  json json = deserialiseAst(astBytes);
  events.emplace_back(nowMs(), elapsedNs(jsonStart), "ast_deserialise", "");
  //  std::cout << "[polyregion-native] JSON :" << json << std::endl;

  auto astLift = nowMono();
  auto program = polyast::program_from_json(json);
  events.emplace_back(nowMs(), elapsedNs(astLift), "ast_lift", "");

  //  std::cout << "[polyregion-native] AST  :" << program << std::endl;
  //  std::cout << "[polyregion-native] Repr :" << polyast::repr(program) << std::endl;

  auto c = compile(program, options, opt);
  c.events.insert(c.events.end(), events.begin(), events.end());
  sortEvents(c);
  return c;
}
