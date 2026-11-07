#pragma region case: capture_overaligned_vector
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input} {libm}
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {
template <typename T> struct OverAlignedAllocator {
  using value_type = T;

  OverAlignedAllocator() = default;
  template <typename U> OverAlignedAllocator(const OverAlignedAllocator<U> &) {}

  T *allocate(std::size_t n) {
    const std::size_t bytes = n * sizeof(T);
    if (bytes < 4096) return static_cast<T *>(std::malloc(bytes));
    char *raw = static_cast<char *>(std::malloc(bytes + 39));
    const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(raw) + 8 + 31) & ~std::uintptr_t(31);
    reinterpret_cast<char **>(aligned)[-1] = raw;
    return reinterpret_cast<T *>(aligned);
  }
  void deallocate(T *p, std::size_t n) {
    if (n * sizeof(T) < 4096) std::free(p);
    else std::free(reinterpret_cast<char **>(p)[-1]);
  }
  template <typename U> bool operator==(const OverAlignedAllocator<U> &) const { return true; }
  template <typename U> bool operator!=(const OverAlignedAllocator<U> &) const { return false; }
};

using Db = std::vector<std::string, OverAlignedAllocator<std::string>>;

double score(const std::string_view &s, const std::string_view &t) {
  int m = 0;
  for (std::size_t a = 0; a < s.size(); ++a)
    for (std::size_t b = 0; b < t.size(); ++b)
      m += (s[a] == t[b]) ? 1 : 0;
  return static_cast<double>(m);
}
Db make_db(int n) {
  Db db;
  db.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    db.push_back(std::string(static_cast<std::size_t>(8 + (i % 50)), static_cast<char>('A' + (i % 4))));
  return db;
}
} // namespace

int main() {
  const int N = 128;
  const std::string needle = "GATTACAGATTACA";
  const Db db = make_db(N);
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
