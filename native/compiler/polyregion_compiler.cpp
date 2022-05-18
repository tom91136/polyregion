#include <chrono>
#include <unordered_map>

#include "compiler.h"
#include "polyregion_compiler.h"
#include "utils.hpp"

static_assert(                                               //
    std::is_same_v<                                          //
        decltype(polyregion_backend::ordinal),               //
        std::underlying_type_t<polyregion::compiler::Target> //
        >);

using polyregion::compiler::Target;

const polyregion_backend OBJECT_LLVM_X86 = {polyregion::to_underlying(Target::Object_LLVM_x86_64)};
const polyregion_backend OBJECT_LLVM_AArch64 = {polyregion::to_underlying(Target::Object_LLVM_AArch64)};
const polyregion_backend OBJECT_LLVM_NVPTX64 = {polyregion::to_underlying(Target::Object_LLVM_NVPTX64)};
const polyregion_backend OBJECT_LLVM_AMDGCN = {polyregion::to_underlying(Target::Object_LLVM_AMDGCN)};
const polyregion_backend SOURCE_C_OPENCL1_1 = {polyregion::to_underlying(Target::Source_C_OpenCL1_1)};
const polyregion_backend SOURCE_C_C11 = {polyregion::to_underlying(Target::Source_C_C11)};

void polyregion_initialise() { polyregion::compiler::initialise(); }

static_assert(sizeof(bool) == 1);

polyregion_compilation *polyregion_compile(const polyregion_buffer *ast, bool emitDisassembly,
                                           polyregion_backend backend) {

  // FIXME update signature
  auto compilation = polyregion::compiler::compile(std::vector<char>(ast->data, ast->data + ast->size),
                                                   polyregion::compiler::Options{Target::Object_LLVM_x86_64});
  auto bin = compilation.binary ? polyregion_buffer{compilation.binary->data(), compilation.binary->size()}
                                : polyregion_buffer{nullptr, 0};

  auto elapsed = new polyregion_event[compilation.events.size()];
  std::transform(compilation.events.begin(), compilation.events.end(), elapsed, [](auto &e) {
    return polyregion_event{e.epochMillis, e.elapsedNanos, polyregion::new_str(e.name), polyregion::new_str(e.data)};
  });

  return new polyregion_compilation{
      bin,                                       //
      polyregion::new_str(compilation.messages), //
      elapsed,                                   //
      compilation.events.size()                  //
  };
}

void polyregion_release_compile(polyregion_compilation *buffer) {
  if (buffer) {
    polyregion::free_str(buffer->messages);
    if (buffer->events) {
      for (size_t i = 0; i < buffer->elapsed_size; ++i) {
        polyregion::free_str(buffer->events[i].name);
        polyregion::free_str(buffer->events[i].data);
      }
    }
    delete[] buffer->events;
  }
  delete buffer;
}