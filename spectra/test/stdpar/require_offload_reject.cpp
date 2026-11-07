#pragma region case: require-offload-rejected
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region compile-fails: -fstdpar-require-offload rejects
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-require-offload -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}

#include <algorithm>
#include <execution>
#include <vector>

struct wide {
  double value[9];
};

int main() {
  std::vector<wide> v(4);
  std::reverse(std::execution::par_unseq, v.begin(), v.end());
  return v.size() == 4 ? 0 : 1;
}
