// #CASE: general
// #MATRIX: platform=cuda
// #MATRIX: ptr=0,1
// #RUN: polycpp -O3 -Wall -Wextra -pedantic -g3 -fstdpar -fstdpar-arch=cuda@sm_60 -DUSE_PTR={ptr} -o {output} {input}
// #RUN: POLYSTL_PLATFORM={platform} SIZE=1024 POLYSTL_HOST_FALLBACK=0 {output}
//   #EXPECT: 20.480000

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

#if USE_PTR == 0
  #define BEGIN(x) (x.begin())
  #define END(x, size) (x.end())
#elif USE_PTR == 1
  #define BEGIN(x) (x.data())
  #define END(x, size) (x.data() + size)
#else
  #error "Bad USE_PTR value"
#endif

int main() {

  int size = 1024;
  if (auto sizeEnv = std::getenv("SIZE"); sizeEnv) size = std::stoi(sizeEnv);

  std::vector<double> a(size);
  std::fill(a.begin(), a.end(), 0.1);

  std::vector<double> b(size);
  std::fill(b.begin(), b.end(), 0.2);

  // a = b + scalar * c
  auto checksum = std::transform_reduce( //
      std::execution::par_unseq, BEGIN(a), END(a, size), BEGIN(b), 0.0);

  printf("%f", checksum);
  return EXIT_SUCCESS;
}
