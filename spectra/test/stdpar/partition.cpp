#pragma region case: partition
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 2 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: partition-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

namespace {
std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}
}

int main() {
  const auto par = std::execution::par_unseq;
  const auto negative = [](const int x) { return x < 0; };
  for (const std::size_t n : {0u, 1u, 2u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;
    std::vector<int> none(n, 1), all(n, -1);

    std::vector<int> sorted = a;
    std::sort(sorted.begin(), sorted.end());

    std::vector<int> ref = a;
    const auto pivot = std::stable_partition(ref.begin(), ref.end(), negative) - ref.begin();

    std::vector<int> p = a;
    const auto pp = std::partition(par, p.begin(), p.end(), negative);
    expect("partition_pivot", pp - p.begin() == pivot);
    expect("partition_split", std::all_of(p.begin(), pp, negative) && std::none_of(pp, p.end(), negative));
    std::sort(p.begin(), p.end());
    expect("partition_permutation", p == sorted);

    p = a;
    const auto sp = std::stable_partition(par, p.begin(), p.end(), negative);
    expect("stable_partition_pivot", sp - p.begin() == pivot);
    expect("stable_partition_order", p == ref);

    std::vector<int> q = none;
    expect("partition_none", std::partition(par, q.begin(), q.end(), negative) == q.begin());
    q = all;
    expect("partition_all", std::partition(par, q.begin(), q.end(), negative) == q.end());
    q = none;
    expect("stable_partition_none", std::stable_partition(par, q.begin(), q.end(), negative) == q.begin());
    q = all;
    expect("stable_partition_all", std::stable_partition(par, q.begin(), q.end(), negative) == q.end());
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
