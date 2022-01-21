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

void compiler::initialise() {
  if (!init) {
    init = true;
    std::cout << "Init LLVM..." << std::endl;
    backend::llvmc::initialise();
  }
}

compiler::Compilation compiler::compile(std::vector<uint8_t> astBytes) {
  if (!init) {
    return Compilation{"initialise was not called before"};
  }

  std::vector<Event> events;

  std::cout << "[polyregion-native] Len  : " << astBytes.size() << std::endl;

  auto jsonStart = nowMono();
  json json;
  try {
    json = nlohmann::json::from_msgpack(astBytes.data(), astBytes.data() + astBytes.size());
  } catch (nlohmann::json::exception &e) {
    return Compilation("Unable to parse packed ast:" + std::string(e.what()));
  }
  events.emplace_back(nowMs(), "ast_deserialise", elapsedNs(jsonStart));

  std::cout << "[polyregion-native] JSON :" << json << std::endl;

  auto astLift = nowMono();
  auto ast = polyast::function_from_json(json);
  events.emplace_back(nowMs(), "ast_lift", elapsedNs(astLift));

  std::cout << "[polyregion-native] AST  :" << ast << std::endl;
  std::cout << "[polyregion-native] Repr :" << polyast::repr(ast) << std::endl;

  backend::OpenCL oclGen;
  oclGen.run(ast);

  backend::LLVM gen;
  auto c = gen.run(ast);
  c.events.insert(c.events.end(), events.begin(), events.end());

  return c;
}
