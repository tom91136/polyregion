#pragma region case: nested base member constructor
#pragma region using: capture=&,=
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_CAPTURE={capture} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

struct Predicate {
  int value;
};

struct Walker {
  Predicate predicate;
};

struct Indexed {
  Walker walker;
};

struct Matcher : Indexed {
  explicit Matcher(Predicate predicate) : Indexed{Walker{predicate}} {}
};

int main() {
  int result = __polyregion_offload_f1__([CHECK_CAPTURE]() {
    Matcher matcher{Predicate{42}};
    return matcher.walker.predicate.value;
  });
  std::printf("%d", result);
  return 0;
}
