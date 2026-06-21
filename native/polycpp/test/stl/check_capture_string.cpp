#pragma region case: capture_string
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

int main() {
  const int N = 64;
  std::vector<int> out(N, -1);
  const std::string a = "AB";
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [&](int i) { out[i] = static_cast<int>(a.size()) * 100 + (a[0] - 'A') + i; });
  bool ok = true;
  for (int i = 0; i < N; ++i)
    ok &= out[i] == 200 + i;
  std::printf(ok ? "pass" : "fail (out[0]=%d)", out[0]);
  return ok ? 0 : 1;
}
