#pragma region case: reduce-struct-accumulator
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}

#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

struct Sum {
  int a = 0;
  int b = 0;
  Sum operator+(const Sum &s) const { return {a + s.a, b + s.b}; }
};

int main() {
  const int N = 1024;
  std::vector<int> va(N, 2), vb(N, 3);
  int *pa = va.data();
  int *pb = vb.data();
  std::vector<int> r(N);
  for (int i = 0; i < N; i++)
    r[i] = i;
  Sum s = std::transform_reduce(std::execution::par_unseq, r.begin(), r.end(), Sum{}, std::plus<>(),
                                [pa, pb](int i) { return Sum{pa[i], pb[i]}; });
  std::printf("a=%d b=%d\n", s.a, s.b);
  if (s.a != 2048 || s.b != 3072) {
    std::fprintf(stderr, "FAIL: expected a=2048 b=3072\n");
    return 1;
  }
  return 0;
}
