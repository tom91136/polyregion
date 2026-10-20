#pragma region case: string_stream
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_COUT=0 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 90

#pragma region case: standard_output
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_COUT=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 90

#include <algorithm>
#include <cstdio>
#include <execution>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  std::vector<int> xs(10);
  for (int i = 0; i < 10; ++i)
    xs[i] = i;
  std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x *= 2; });
  int sum = 0;
  for (const int x : xs)
    sum += x;
#if CHECK_COUT
  std::cout << sum << std::flush;
#else
  std::ostringstream out;
  out << sum;
  std::printf("%s", out.str().c_str());
#endif
  return 0;
}
