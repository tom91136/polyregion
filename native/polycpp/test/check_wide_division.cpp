#pragma region case: wide-division
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: DIVISOR=384 {output}
#pragma region requires: 0

#pragma region case: wide-division-large-divisor
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: DIVISOR=6000000000 {output}
#pragma region requires: 0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

int main() {
  const uint64_t d = std::strtoull(std::getenv("DIVISOR"), nullptr, 10);
  const int64_t sd = static_cast<int64_t>(d);
  const size_t n = 4099;
  const uint64_t bases[] = {0, 1000003, 0xFFFFFFFFull, 0x100000000ull, 0x123456789ABCull, 0xFFFFFFFFFFFFFFF0ull};
  std::vector<uint64_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<uint64_t> uq(n), ur(n);
  std::vector<int64_t> sq(n), sr(n);
  uint64_t *uqp = uq.data(), *urp = ur.data();
  int64_t *sqp = sq.data(), *srp = sr.data();
  std::for_each(std::execution::par_unseq, ids.begin(), ids.end(), [=](uint64_t t) {
    for (uint64_t k = 0; k < 6; ++k) {
      const uint64_t u = bases[k] + t;
      const int64_t s = (t & 1 ? -1 : 1) * static_cast<int64_t>((bases[k] >> 2) + t);
      uqp[t] += u / d;
      urp[t] += u % d;
      sqp[t] += s / sd;
      srp[t] += s % sd;
    }
  });
  size_t bad = 0;
  for (uint64_t t = 0; t < n; ++t) {
    uint64_t eq = 0, er = 0;
    int64_t esq = 0, esr = 0;
    for (uint64_t k = 0; k < 6; ++k) {
      const uint64_t u = bases[k] + t;
      const int64_t s = (t & 1 ? -1 : 1) * static_cast<int64_t>((bases[k] >> 2) + t);
      eq += u / d;
      er += u % d;
      esq += s / sd;
      esr += s % sd;
    }
    bad += (uq[t] != eq) + (ur[t] != er) + (sq[t] != esq) + (sr[t] != esr);
  }
  std::printf("%zu", bad);
  return 0;
}
