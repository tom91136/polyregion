#include <atomic>
#include <iostream>

#include "ast.h"
#include "backend/llvm.h"
#include "backend/llvmc.h"
#include "backend/opencl.h"
#include "compiler.h"
#include "generated/polyast_codec.h"
#include "json.hpp"

using namespace polyregion;

std::atomic_bool init = false;

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;
static uint64_t elapsedNs(const TimePoint &a, const TimePoint &b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
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

  Compilation c;

  std::cout << "[polyregion-native] Len  : " << astBytes.size() << std::endl;

  auto jsonStart = Clock::now();
  json json;
  try {
    json = nlohmann::json::from_msgpack(astBytes.data(), astBytes.data() + astBytes.size());
  } catch (nlohmann::json::exception &e) {
    return Compilation("Unable to parse packed ast:" + std::string(e.what()));
  }
  c.elapsed.emplace_back("ast_deserialise", elapsedNs(Clock::now(), jsonStart));

  std::cout << "[polyregion-native] JSON :" << json << std::endl;

  auto astLift = Clock::now();
  auto ast = polyast::function_from_json(json);
  c.elapsed.emplace_back("ast_lift", elapsedNs(Clock::now(), astLift));

  std::cout << "[polyregion-native] AST  :" << ast << std::endl;
  std::cout << "[polyregion-native] Repr :" << polyast::repr(ast) << std::endl;

  backend::OpenCL oclGen;
  oclGen.run(ast);
  backend::LLVM gen;
  gen.run(ast);

  return c;
}
