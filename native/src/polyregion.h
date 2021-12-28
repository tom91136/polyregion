#pragma once
#include "variants.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include "foo.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t size;
  uint8_t *data;
} Buffer;

typedef struct {
  Buffer program;
  char *disassembly;
} Program;

Program compile(Buffer *polyast_proto);

void release(Program *buffer);

#ifdef __cplusplus
}

namespace aaa {

static void x() {

  using namespace foo;

//  Alt l = A();
//  T1ALeaf xx ({}, {"a", "b"}, 23, T1BLeaf());




//  std::cout << "[] = "  << xx;
}

// ======================================================================================

} // namespace aaa

#endif