#include <execution>
#include <stdlib.h>

void foo(){
  auto xs = (short *)malloc(42);
  auto ys = (short *)malloc(42);
  int hello = 42;
  int bar = 9;
  std::transform(std::execution::par, xs , xs+ 10, ys , [&](auto &x) {
    return 0;
  });
}