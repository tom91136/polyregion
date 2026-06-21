#pragma region case: swg
#pragma region using: strty=std::string_view,std::string
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSTRING_TYPE={strty} -o {output} {input} {libm}
#pragma region do: {output} 128

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

#ifndef STRING_TYPE
  #define STRING_TYPE std::string_view
#endif

namespace {

constexpr int MaxLen = 64;
constexpr int Match = 1;
constexpr int Mismatch = -2;
constexpr int Gap = -1;
constexpr char Alphabet[4] = {'A', 'C', 'G', 'T'};

double swg(const STRING_TYPE &s, const STRING_TYPE &t) {
  const int slen = static_cast<int>(s.size());
  const int tlen = static_cast<int>(t.size());
  if (slen == 0 || tlen == 0) return 0.0;
  int v0[MaxLen], v1[MaxLen];
  int m = std::max(0, std::max(Gap, s[0] == t[0] ? Match : Mismatch));
  v0[0] = m;
  for (int j = 1; j < tlen; ++j) {
    int c = s[0] == t[j] ? Match : Mismatch;
    v0[j] = std::max(0, std::max(v0[j - 1] + Gap, c));
  }
  for (int j = 1; j < tlen; ++j)
    m = std::max(m, v0[j]);
  for (int i = 1; i < slen; ++i) {
    int c0 = s[i] == t[0] ? Match : Mismatch;
    v1[0] = std::max(0, std::max(v0[0] + Gap, c0));
    m = std::max(m, v1[0]);
    for (int j = 1; j < tlen; ++j) {
      int c = s[i] == t[j] ? Match : Mismatch;
      v1[j] = std::max(std::max(0, v0[j] + Gap), std::max(v1[j - 1] + Gap, v0[j - 1] + c));
    }
    for (int j = 1; j < tlen; ++j)
      m = std::max(m, v1[j]);
    for (int j = 0; j < tlen; ++j)
      v0[j] = v1[j];
  }
  return static_cast<double>(m) / static_cast<double>(std::min(slen, tlen) * std::max(Match, Gap));
}

std::vector<std::string> make_database(int n) {
  std::vector<std::string> db;
  db.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    std::uint32_t rng = static_cast<std::uint32_t>(i) * 2654435761u + 12345u;
    auto next = [&] {
      rng = rng * 1664525u + 1013904223u;
      return rng;
    };
    const int len = 8 + static_cast<int>(next() % static_cast<std::uint32_t>(MaxLen - 8 + 1));
    std::string s;
    s.reserve(static_cast<std::size_t>(len));
    for (int j = 0; j < len; ++j)
      s.push_back(Alphabet[next() & 3u]);
    db.push_back(std::move(s));
  }
  return db;
}

} // namespace

int main(int argc, const char *argv[]) {
  int size = 8192;
  if (argc - 1 >= 1) {
    char *endp = nullptr;
    long v = std::strtol(argv[1], &endp, 10);
    if (endp && *endp == '\0' && v > 0) size = static_cast<int>(v);
  }

  const std::string needle = "GATTACAGATTACA";
  const std::vector<std::string> db = make_database(size);
  std::vector<double> scores(static_cast<std::size_t>(size));
  std::vector<int> idx(static_cast<std::size_t>(size));
  std::iota(idx.begin(), idx.end(), 0);

  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int i) { scores[i] = swg(needle, db[i]); });

  double err = 0.0, total = 0.0;
  for (int i = 0; i < size; ++i) {
    const double ref = swg(needle, db[static_cast<std::size_t>(i)]);
    err += std::fabs(scores[static_cast<std::size_t>(i)] - ref);
    total += scores[static_cast<std::size_t>(i)];
  }
  const bool ok = err / size <= 1e-9;
  if (!ok) std::fprintf(stderr, "Validation failed, average error = %g\n", err / size);

  std::printf("Entries: %d\n", size);
  std::printf("Total similarity: %.6f\n", total);
  std::printf("Done\n");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
