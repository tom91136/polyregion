#include <atomic>
#include <iostream>

#include "ast.h"
#include "backend/c_source.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "compiler.h"
#include "generated/polyast_codec.h"
#include "json.hpp"
#include "utils.hpp"

using namespace polyregion;

static std::atomic_bool init = false;

compiler::TimePoint compiler::nowMono() { return MonoClock::now(); }

int64_t compiler::elapsedNs(const TimePoint &a, const TimePoint &b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

int64_t compiler::nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::optional<compiler::Target> compiler::targetFromOrdinal(std::underlying_type_t<compiler::Target> ordinal) {
  auto target = static_cast<Target>(ordinal);
  switch (target) {
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN:
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_C11: return target;
    default: return {};
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
    std::cout << "Init LLVM..." << std::endl;
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

static backend::LLVM::Options toLLVMBackendOptions(const compiler::Options &options) {
  switch (options.target) {
    case compiler::Target::Object_LLVM_x86_64: return {.target = backend::LLVM::Target::x86_64, .arch = options.arch};
    case compiler::Target::Object_LLVM_AArch64: return {.target = backend::LLVM::Target::AArch64, .arch = options.arch};
    case compiler::Target::Object_LLVM_NVPTX64: return {.target = backend::LLVM::Target::NVPTX64, .arch = options.arch};
    case compiler::Target::Object_LLVM_AMDGCN: return {.target = backend::LLVM::Target::AMDGCN, .arch = options.arch};
    case compiler::Target::Source_C_OpenCL1_1: //
    case compiler::Target::Source_C_C11:       //
      throw std::logic_error("Not an object target");
  }
}

compiler::Layout compiler::layoutOf(const polyast::StructDef &def, const Options &options) {

  switch (options.target) {
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN: {
      auto llvmOptions = toLLVMBackendOptions(options);
      auto dataLayout = backend::llvmc::targetMachineFromTarget(llvmOptions.toTargetInfo())->createDataLayout();
      ;

      llvm::LLVMContext c;
      backend::LLVM::AstTransformer xform(llvmOptions, c);
      auto [structTy, _] = xform.mkStruct(def);

      auto layout = dataLayout.getStructLayout(structTy);
      std::vector<compiler::Member> members;
      for (size_t i = 0; i < def.members.size(); ++i) {
        members.emplace_back(def.members[i],                                          //
                             layout->getElementOffset(i),                             //
                             dataLayout.getTypeAllocSize(structTy->getElementType(i)) //
        );
      }

      return compiler::Layout{.name = def.name,
                              .sizeInBytes = layout->getSizeInBytes(),
                              .alignment = layout->getAlignment().value(),
                              .members = members};
    }
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_C11: throw std::logic_error("Not available for source targets");
  }
}

compiler::Layout compiler::layoutOf(const Bytes &sdef, const Options &backend) {
  json json = deserialiseAst(sdef);
  auto def = polyast::structdef_from_json(json);
  return layoutOf(def, backend);
}

static void sortEvents(compiler::Compilation &c) {
  std::sort(c.events.begin(), c.events.end(),
            [](const auto &l, const auto &r) { return l.epochMillis < r.epochMillis; });
}

compiler::Compilation compiler::compile(const polyast::Program &program, const Options &options) {
  if (!init) {
    return Compilation{"initialise was not called before"};
  }

  auto mkBackend = [&]() -> std::unique_ptr<backend::Backend> {
    switch (options.target) {
      case Target::Object_LLVM_x86_64:
      case Target::Object_LLVM_AArch64:
      case Target::Object_LLVM_NVPTX64:
      case Target::Object_LLVM_AMDGCN:                                                   //
        return std::make_unique<backend::LLVM>(toLLVMBackendOptions(options));           //
      case Target::Source_C_OpenCL1_1:                                                   //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::OpenCL1_1); //
      case Target::Source_C_C11:                                                         //
        return std::make_unique<backend::CSource>(backend::CSource::Dialect::C11);       //
    }
  };

  Compilation c;
  try {
    c = mkBackend()->run(program);
  } catch (const std::exception &e) {
    c.messages = e.what();
  }
  sortEvents(c);
  return c;
}

compiler::Compilation compiler::compile(const Bytes &astBytes, const Options &options) {

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

  auto c = compile(program, options);
  c.events.insert(c.events.end(), events.begin(), events.end());
  sortEvents(c);
  return c;
}
