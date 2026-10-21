#pragma region case: core-fallback
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3 600

#pragma region case: core-package
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 3 600

#pragma region case: core-package-specialisations
#pragma region offload-only
#pragma region do: {package_fixture} {output}.packages
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DSPECIALISATIONS -fstdpar-library-path={output}.packages -o {output} {input}
#pragma region do: {output}
#pragma region requires: 60 6100 400.0

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

#ifdef SPECIALISATIONS
template <class T> void scale(std::vector<T> &xs, T factor) {
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [factor](T &x) { x *= factor; });
}

int main() {
  std::vector<int> xs(100, 1);
  std::vector<double> ys(100, 0.5);
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x += 1; });
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x *= 3; });
  scale(xs, 2);
  scale(xs, 5);
  scale(ys, 4.0);
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x + 1; });
  const double total =
      std::transform_reduce(std::execution::par_unseq, ys.begin(), ys.end(), 0.0, std::plus<>(), [](double y) { return y * 2; });
  std::printf("%d %d %.1f", xs[7], sum, total);
  return 0;
}
#else
int main() {
  std::vector<int> xs(100, 1);
  const int bias = 2;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [bias](int &x) { x += bias; });
  const int sum = std::transform_reduce(std::execution::par_unseq, xs.begin(), xs.end(), 0, std::plus<>(), [](int x) { return x * 2; });
  std::printf("%d %d", xs[7], sum);
  return 0;
}
#endif
