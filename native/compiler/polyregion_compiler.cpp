#include <chrono>
#include <unordered_map>

#include "compiler.h"
#include "polyregion_compiler.h"
#include "utils.hpp"

static_assert(                                                //
    std::is_same_v<                                           //
        decltype(polyregion_backend::ordinal),                //
        std::underlying_type_t<polyregion::compiler::Backend> //
        >);

const polyregion_backend POLYREGION_BACKEND_LLVM = {polyregion::to_underlying(polyregion::compiler::Backend::LLVM)};
const polyregion_backend POLYREGION_BACKEND_OPENCL = {polyregion::to_underlying(polyregion::compiler::Backend::OpenCL)};
const polyregion_backend POLYREGION_BACKEND_CUDA = {polyregion::to_underlying(polyregion::compiler::Backend::CUDA)};

void polyregion_initialise() { polyregion::compiler::initialise(); }

static_assert(sizeof(bool) == 1);

polyregion_compilation *polyregion_compile(const polyregion_buffer *ast, bool emitDisassembly,
                                           polyregion_backend backend) {

  auto compilation = polyregion::compiler::compile(std::vector<uint8_t>(ast->data, ast->data + ast->size));
  auto bin = compilation.binary ? polyregion_buffer{compilation.binary->data(), compilation.binary->size()}
                                : polyregion_buffer{nullptr, 0};

  auto dis = compilation.disassembly ? polyregion::new_str(*compilation.disassembly) : nullptr;

  auto elapsed = new polyregion_event[compilation.events.size()];
  std::transform(compilation.events.begin(), compilation.events.end(), elapsed, [](auto &e) {
    return polyregion_event{e.epochMillis, polyregion::new_str(e.name), e.elapsedNanos};
  });

  return new polyregion_compilation{
      bin, dis,                                           //
      polyregion::new_str(compilation.messages), elapsed, //
      compilation.events.size()                           //
  };
}

void polyregion_release_compile(polyregion_compilation *buffer) {
  if (buffer) {
    polyregion::free_str(buffer->messages);
    polyregion::free_str(buffer->disassembly);
    if (buffer->elapsed) {
      for (size_t i = 0; i < buffer->elapsed_size; ++i) {
        polyregion::free_str(buffer->elapsed[i].name);
      }
    }
    delete[] buffer->elapsed;
  }
  delete buffer;
}