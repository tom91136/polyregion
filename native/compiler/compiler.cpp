#include <atomic>
#include <iostream>

#include "ast.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "backend/opencl.h"
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

std::ostream &compiler::operator<<(std::ostream &os, const compiler::Compilation &compilation) {
  auto events = mk_string<Event>(
      compilation.events,
      [](auto &e) {
        return "[" + std::to_string(e.epochMillis) + "] " + e.name + ": " + std::to_string(e.elapsedNanos) + "ns";
      },
      ", ");
  os << "Compilation {"                                                                                 //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) : "(empty)") //
     << "\n  disassembly:\n`" << compilation.disassembly.value_or("(empty)") << "`"                     //
     << "\n  events: " << events                                                                        //
     << "\n  messages: `" << compilation.messages << "`"                                                //
     << "\n}";
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

compiler::Layout compiler::layoutOf(const polyast::StructDef &def) {

  llvm::LLVMContext c;
  backend::LLVMAstTransformer xform(c);
  auto [structTy, _] = xform.mkStruct(def);

  auto layout = backend::llvmc::targetMachine().createDataLayout();
  auto structLayout = layout.getStructLayout(structTy);
  std::vector<compiler::Member> out;
  for (size_t i = 0; i < def.members.size(); ++i) {
    out.emplace_back(def.members[i],                                      //
                     structLayout->getElementOffset(i),                   //
                     layout.getTypeAllocSize(structTy->getElementType(i)) //
    );
  }

  return {def.name, structLayout->getSizeInBytes(), structLayout->getAlignment().value(), out};
}

compiler::Layout compiler::layoutOf(const Bytes &structDef) {
  json json = deserialiseAst(structDef);
  auto def = polyast::structdef_from_json(json);
  return layoutOf(def);
}

compiler::Compilation compiler::compile(const polyast::Program &program) {
  if (!init) {
    return Compilation{"initialise was not called before"};
  }
  backend::OpenCL oclGen;
  oclGen.run(program);

  backend::LLVM gen;
  Compilation c;
  try {
    c = gen.run(program);
  } catch (const std::exception &e) {
    c.messages = e.what();
  }
  return c;
}

compiler::Compilation compiler::compile(const Bytes &astBytes) {

  std::vector<Event> events;

  std::cout << "[polyregion-native] Len  : " << astBytes.size() << std::endl;
  auto jsonStart = nowMono();
  json json = deserialiseAst(astBytes);
  events.emplace_back(nowMs(), "ast_deserialise", elapsedNs(jsonStart));
  std::cout << "[polyregion-native] JSON :" << json << std::endl;

  auto astLift = nowMono();
  auto program = polyast::program_from_json(json);
  events.emplace_back(nowMs(), "ast_lift", elapsedNs(astLift));

  std::cout << "[polyregion-native] AST  :" << program << std::endl;
  std::cout << "[polyregion-native] Repr :" << polyast::repr(program) << std::endl;

  auto r = compile(program);
  r.events.insert(r.events.end(), events.begin(), events.end());

  return r;
}
