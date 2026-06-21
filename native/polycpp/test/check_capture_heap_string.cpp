#pragma region case: capture_heap_string
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstddef>
#include <cstdio>
#include <string>

#include "test_utils.h"

int main() {
  const std::string s = "the quick brown fox jumps over the lazy dog 0123456789";

  const long remote = __polyregion_offload_f1__([&]() {
    long sum = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
      sum += static_cast<long>(s[i]);
    return sum;
  });

  long local = 0; // re-read after offload
  for (std::size_t i = 0; i < s.size(); ++i)
    local += static_cast<long>(s[i]);

  const bool ok = remote == local && local > 0;
  std::printf(ok ? "pass" : "fail (dev=%ld host=%ld)", remote, local);
  return ok ? 0 : 1;
}
