#pragma region case: capture_ptr_strings
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input} {libm}
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {
double score(const std::string_view &s, const std::string_view &t) {
  int m = 0;
  for (std::size_t a = 0; a < s.size(); ++a)
    for (std::size_t b = 0; b < t.size(); ++b)
      m += (s[a] == t[b]) ? 1 : 0;
  return static_cast<double>(m);
}
std::vector<std::string> make_db(int n) {
  std::vector<std::string> db;
  db.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    db.push_back(std::string(static_cast<std::size_t>(8 + (i % 50)), static_cast<char>('A' + (i % 4))));
  return db;
}
} // namespace

int main() {
  const int N = 128;
  const std::string needle = "GATTACAGATTACA";
  const std::vector<std::string> db = make_db(N);
  std::vector<double> scores(static_cast<std::size_t>(N), 0.0);
  std::vector<int> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);

  double *scoresPtr = scores.data();
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [&](int i) { scoresPtr[i] = score(needle, db[static_cast<std::size_t>(i)]); });

  double err = 0;
  for (int i = 0; i < N; ++i)
    err += std::fabs(scores[static_cast<std::size_t>(i)] - score(needle, db[static_cast<std::size_t>(i)]));
  const bool ok = err <= 1e-9;
  std::printf(ok ? "pass" : "fail (err=%g)", err);
  return ok ? 0 : 1;
}
