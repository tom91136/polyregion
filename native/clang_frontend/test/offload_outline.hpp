#include "offload.hpp"


TEST_CASE("Invoke outline") {
  int a = 42;
  assertOffload<int>([&]() { return a; });
}

// list(FILES  offload_arith.hpp)
// offload_arith.cpp*