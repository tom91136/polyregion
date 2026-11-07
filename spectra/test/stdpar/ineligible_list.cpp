#pragma region case: list
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 0 bound to spectra
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <cstdio>
#include <execution>
#include <list>
#include <string>

int main() {
  const auto par = std::execution::par_unseq;
  const auto positive = [](const int x) { return x > 0; };
  std::list<int> l{5, 3, 9, 3, 1};
  std::replace(par, l.begin(), l.end(), 3, 7);
  std::string failures;
  if (l != std::list<int>{5, 7, 9, 7, 1}) failures += "replace";
  if (!std::all_of(par, l.begin(), l.end(), positive)) failures += failures.empty() ? "all_of" : " all_of";
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
