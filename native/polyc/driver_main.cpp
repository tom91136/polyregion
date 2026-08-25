#include <string_view>

#include "driver_polyc.h"

int main(int argc, const char *argv[]) {
  if (argc > 1 && std::string_view(argv[1]) == "--polyc") return polyregion::polyc(argc - 1, argv + 1);
  return polyregion::polyc(argc, argv);
}
