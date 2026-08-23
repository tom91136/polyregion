#pragma region case: rocprim-sleep-query
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 20

#include <cstdio>

#include "test_utils.h"

namespace rocprim::detail {

int is_sleep_scan_state_used(void *, bool &use_sleep) {
  use_sleep = true;
  return 0;
}

} // namespace rocprim::detail

int main() {
  const int result = __polyregion_offload_f1__([=]() {
    bool use_sleep = true;
    const int status = rocprim::detail::is_sleep_scan_state_used(nullptr, use_sleep);
    return status == 0 && !use_sleep ? 20 : 0;
  });
  std::printf("%d", result);
  return 0;
}
