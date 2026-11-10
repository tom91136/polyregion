#pragma region case: deque
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 2 bound to spectra
#pragma region do: {output}
#pragma region requires: pass
#pragma region do: POLYRT_DEBUG=2 {output} multi
#pragma region stderr-count: 0 grid=(
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <execution>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  const auto par = std::execution::par_unseq;
  const auto sizes = argc > 1 ? std::vector<std::size_t>{10000} : std::vector<std::size_t>{0, 1, 31, 32, 33};
  std::string failures;
  for (const std::size_t n : sizes) {
    std::deque<int> d(n);
    for (std::size_t i = 0; i < n; ++i)
      d[i] = static_cast<int>((i * 7919) % 1009) - 500;

    std::deque<int> p = d, s = d;
    std::reverse(par, p.begin(), p.end());
    std::reverse(s.begin(), s.end());
    if (p != s) failures += failures.empty() ? "reverse" : " reverse";

    p = d;
    s = d;
    std::sort(par, p.begin(), p.end());
    std::sort(s.begin(), s.end());
    if (p != s) failures += failures.empty() ? "sort" : " sort";
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
