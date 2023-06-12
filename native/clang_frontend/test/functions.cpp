#include <execution>
#include <stdlib.h>

int main() {
  //  std::vector<int> xs{1, 2, 3};
  //  std::vector<int> ys(xs.size());

  struct A {
    int memberA = 42;

    void run() {
      auto xs = (int *)malloc(42);
      auto ys = (int *)malloc(42);

      int hello = 42;
      int bar = 9;
      auto &u = bar;

      std::transform(std::execution::par, xs, xs + 10, ys, //
                     [&](auto &x) { return x + hello + bar + u + this->memberA; });
    }
  };
}