#include <iostream>
#include <string>
#include <vector>

#include "polyregion_runtime.h"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) {
    std::cout << "runtime-drv: read and enumerate executable symbols in objects\n"
                 "usage: \n"
              << argv[0] << " [<obj file path>...]\n"
              << std::endl;
  } else {
    for (auto &arg : args) {
      std::cout << "[" << arg << "]" << std::endl;
      try {
        auto data = polyregion::read_struct<uint8_t>(arg);
        auto ref = polyregion_load_object(data.data(), data.size());
        if (ref->message) {
          std::cout << (!ref->object ? "FATAL" : "WARN") << ":" << ref->message << std::endl;
        }
        if (ref->object) {
          auto table = polyregion_enumerate(ref->object);
          for (size_t i = 0; i < table->size; ++i) {
            auto &sym = table->symbols[i];
            std::cout << "  "
                      << "`" << sym.name << "`"
                      << "(0x" << std::hex << sym.address << ")" << std::endl;
          }
          auto r = polyregion_invoke(ref->object, "lambda", nullptr, 0, nullptr);
          if (r) {
            std::cerr << r << std::endl;
          }
          polyregion_release_enumerate(table);
        }
        polyregion_release_object(ref);
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
      }
    }
  }
  return EXIT_SUCCESS;
}