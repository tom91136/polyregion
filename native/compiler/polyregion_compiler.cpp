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

template <typename E> constexpr auto to_underlying(E e) noexcept { return static_cast<std::underlying_type_t<E>>(e); }

const polyregion_backend POLYREGION_BACKEND_LLVM = {to_underlying(polyregion::compiler::Backend::LLVM)};
const polyregion_backend POLYREGION_BACKEND_OPENCL = {to_underlying(polyregion::compiler::Backend::OpenCL)};
const polyregion_backend POLYREGION_BACKEND_CUDA = {to_underlying(polyregion::compiler::Backend::CUDA)};

void polyregion_initialise() { polyregion::compiler::initialise(); }

static_assert(sizeof(bool) == 1);

polyregion_compilation *polyregion_compile(const polyregion_buffer *ast, bool emitDisassembly,
                                           polyregion_backend backend) {

  auto compilation = polyregion::compiler::compile(std::vector<uint8_t>(ast->data, ast->data + ast->size));
  auto bin = compilation.binary ? polyregion_buffer{compilation.binary->data(), compilation.binary->size()}
                                : polyregion_buffer{nullptr, 0};

  auto dis = compilation.disassembly ? polyregion::new_str(*compilation.disassembly) : nullptr;

  auto elapsed = new polyregion_elapsed[compilation.elapsed.size()];
  std::transform(compilation.elapsed.begin(), compilation.elapsed.end(), elapsed, [](auto &e) {
    return polyregion_elapsed{polyregion::new_str(e.first), e.second};
  });

  return new polyregion_compilation{
      bin, dis,                                           //
      polyregion::new_str(compilation.messages), elapsed, //
      compilation.elapsed.size()                          //
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