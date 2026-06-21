#pragma region case: capture_const_readonly
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.h"

int main() {
  const std::vector<std::string> db = {std::string(20, 'A'), std::string(33, 'C'), //
                                       std::string(47, 'G'), std::string(58, 'T')};

  const int remote = __polyregion_offload_f1__([&]() {
    int acc = 0;
    for (int i = 0; i < 4; ++i) {
      const std::string &s = db[static_cast<std::size_t>(i)];
      acc += static_cast<int>(s.size()); // _M_string_length
      acc += static_cast<int>(s[0]);     // heap _M_p[0]
    }
    return acc;
  });

  int local = 0; // re-read after offload
  for (int i = 0; i < 4; ++i) {
    const std::string &s = db[static_cast<std::size_t>(i)];
    local += static_cast<int>(s.size());
    local += static_cast<int>(s[0]);
  }

  const int expected = (20 + 33 + 47 + 58) + ('A' + 'C' + 'G' + 'T');
  const bool ok = remote == local && local == expected;
  std::printf(ok ? "pass" : "fail (dev=%d host=%d expected=%d)", remote, local, expected);
  return ok ? 0 : 1;
}
