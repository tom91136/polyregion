#pragma region case: mutate
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 10 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: mutate-cxx17
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -std=c++17 -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 10 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#pragma region case: mutate-baseline
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
  const auto positive = [](const int x) { return x > 0; };
  const auto seven = [] { return 7; };
  const auto half = [] { return 1.5f; };
  for (const std::size_t n : {0u, 1u, 31u, 32u, 33u, 1u << 20}) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>((i * 7919) % 1009) - 500;

    std::vector<int> p = a, s = a;
    std::replace(par, p.begin(), p.end(), 3, 7);
    std::replace(s.begin(), s.end(), 3, 7);
    expect("replace", p == s);

    p = a;
    s = a;
    std::replace_if(par, p.begin(), p.end(), positive, 0);
    std::replace_if(s.begin(), s.end(), positive, 0);
    expect("replace_if", p == s);

    std::vector<int> pc(n), sc(n);
    std::replace_copy(par, a.begin(), a.end(), pc.begin(), 3, 7);
    std::replace_copy(a.begin(), a.end(), sc.begin(), 3, 7);
    expect("replace_copy", pc == sc);

    std::replace_copy_if(par, a.begin(), a.end(), pc.begin(), positive, 0);
    std::replace_copy_if(a.begin(), a.end(), sc.begin(), positive, 0);
    expect("replace_copy_if", pc == sc);

    p = a;
    s = a;
    std::reverse(par, p.begin(), p.end());
    std::reverse(s.begin(), s.end());
    expect("reverse", p == s);

    std::reverse_copy(par, a.begin(), a.end(), pc.begin());
    std::reverse_copy(a.begin(), a.end(), sc.begin());
    expect("reverse_copy", pc == sc);

    std::vector<int> x = a, y(n, 9), xs = a, ys(n, 9);
    std::swap_ranges(par, x.begin(), x.end(), y.begin());
    std::swap_ranges(xs.begin(), xs.end(), ys.begin());
    expect("swap_ranges", x == xs && y == ys);

    std::copy_n(par, a.begin(), n, pc.begin());
    std::copy_n(a.begin(), n, sc.begin());
    expect("copy_n", pc == sc);

    std::vector<int> g(n), gs(n);
    std::generate(par, g.begin(), g.end(), seven);
    std::generate(gs.begin(), gs.end(), seven);
    expect("generate", g == gs);

    std::vector<float> f(n);
    std::generate_n(par, f.begin(), n, half);
    expect("generate_n", f == std::vector<float>(n, 1.5f));
  }
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
