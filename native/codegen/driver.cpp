#include "polyregion_codegen.h"
#include "utils.hpp"

int main(int argc, char *argv[]) {

  polyregion_initialise();

  std::vector<uint8_t> xs = polyregion::read_struct<uint8_t>("../ast.bin");

  polyregion_buffer buffer{xs.data(), xs.size()};
  polyregion_compile(&buffer);

  return EXIT_SUCCESS;
}
