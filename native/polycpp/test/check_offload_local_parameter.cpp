#pragma region case: offload-local-parameter
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 42

#include <cstdio>

#include "test_utils.h"

static int read_local([[clang::annotate("__polyregion_local")]] int *p) { return p[0]; }

template <typename T> using local_ptr = T __attribute__((address_space(3))) *;

template <typename P> static int read_template_pointer(const P &p) { return p[0]; }

[[clang::annotate("__polyregion_local")]] static int *return_local([[clang::annotate("__polyregion_local")]] int *p) { return p; }

struct local_holder {
  [[clang::annotate("__polyregion_local")]] int *p;
};

struct typed_local_holder {
  local_ptr<int> p;
};

int main() {
  const int r = __polyregion_offload_f1__([]() -> int {
    [[clang::annotate("__polyregion_local")]] int scratch[1];
    scratch[0] = 40;
    local_holder h{scratch};
    typed_local_holder th{(local_ptr<int>)scratch};
    int *p = return_local(h.p);
    return read_local(p) + read_template_pointer(th.p) - 38;
  });
  std::printf("%d", r);
  return 0;
}
