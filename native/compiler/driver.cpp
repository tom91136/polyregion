#include "compiler.h"
#include "utils.hpp"

int main(int argc, char *argv[]) {

  polyregion::compiler::initialise();

  std::vector<uint8_t> xs = polyregion::read_struct<uint8_t>("../ast.bin");

  auto c = polyregion::compiler::compile(xs);

  // TODO

  return EXIT_SUCCESS;
}
