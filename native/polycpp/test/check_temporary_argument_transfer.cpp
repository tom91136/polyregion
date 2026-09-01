#pragma region case: temporary-argument-transfer
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1

#include <cstdio>
#include <tuple>
#include <utility>

#include "test_utils.h"

struct Owner {
  int *counter;
  bool owns;

  explicit Owner(int *counter) : counter(counter), owns(true) {}
  Owner(const Owner &) = delete;
  Owner(Owner &&other) : counter(other.counter), owns(other.owns) { other.owns = false; }
  ~Owner() {
    if (owns) ++*counter;
  }
};

struct Future {
  Owner owner;

  explicit Future(Owner owner) : owner(static_cast<Owner &&>(owner)) {}
};

static Future makeFuture(int *counter) { return Future(Owner(counter)); }

static std::tuple<Owner> makeTuple(int *counter) { return std::tuple<Owner>(Owner(counter)); }

int main() {
  int *counter = new int(0);
  const int result = __polyregion_offload_f1__([=]() {
    {
      Future future = makeFuture(counter);
      if (*counter != 0) return 10 + *counter;
    }
    if (*counter != 1) return 40 + *counter;
    {
      auto &&[owner] = makeTuple(counter);
      if (*counter != 1) return 50 + *counter;
      Owner moved(std::move(owner));
      if (*counter != 1) return 60 + *counter;
    }
    if (*counter != 2) return 70 + *counter;
    return *counter - 1;
  });
  std::printf("%d", result);
  return 0;
}
