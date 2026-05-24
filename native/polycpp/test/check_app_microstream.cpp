#pragma region case: microstream
#pragma region using: num_type=double,float
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_NUM_TYPE={num_type} -o {output} {input} -lm
#pragma region do: {output} 1024 10

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#ifndef CHECK_NUM_TYPE
  #define CHECK_NUM_TYPE double
#endif

namespace {
constexpr double startA = 0.1;
constexpr double startB = 0.2;
constexpr double startC = 0.0;
constexpr double startScalar = 0.4;

template <typename T> bool failed(const std::vector<T> &xs, double epsi, T gold, const char *name) {
  if (xs.empty()) return false;
  double err = 0.0;
  for (auto v : xs)
    err += std::fabs(static_cast<double>(v - gold));
  if ((err / static_cast<double>(xs.size())) > epsi) {
    std::fprintf(stderr, "Validation failed on %s, average error = %f", name, err);
    return true;
  }
  return false;
}

template <typename T> bool run(int size, int times) {
  const std::size_t bytes = sizeof(T) * static_cast<std::size_t>(size);
  std::printf("Running kernels %d times\n", times);
  std::printf("Number of elements: %d\n", size);
  std::printf("Precision: %zu-byte\n", sizeof(T));
  std::printf("Array size: %.1f MB (= %.1f GB)\n", static_cast<double>(bytes) * 1.0e-6, static_cast<double>(bytes) * 1.0e-9);
  std::printf("Total size: %.1f MB (= %.1f GB)\n", 3.0 * static_cast<double>(bytes) * 1.0e-6, 3.0 * static_cast<double>(bytes) * 1.0e-9);

  using clock = std::chrono::steady_clock;
  auto seconds_since = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };

  std::vector<T> a(size), b(size), c(size);
  std::fill(std::execution::par_unseq, a.begin(), a.end(), static_cast<T>(startA));
  std::fill(std::execution::par_unseq, b.begin(), b.end(), static_cast<T>(startB));
  std::fill(std::execution::par_unseq, c.begin(), c.end(), static_cast<T>(startC));

  const T scalar = static_cast<T>(startScalar);
  T sum = static_cast<T>(0);
  std::vector<double> timings[5];
  for (auto &t : timings)
    t.reserve(times);

  for (int k = 0; k < times; ++k) {
    auto t = clock::now();
    std::copy(std::execution::par_unseq, a.begin(), a.end(), c.begin());
    timings[0].push_back(seconds_since(t));

    t = clock::now();
    std::transform(std::execution::par_unseq, c.begin(), c.end(), b.begin(), [scalar](T x) { return scalar * x; });
    timings[1].push_back(seconds_since(t));

    t = clock::now();
    std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(), c.begin(), std::plus<T>{});
    timings[2].push_back(seconds_since(t));

    t = clock::now();
    std::transform(std::execution::par_unseq, b.begin(), b.end(), c.begin(), a.begin(), [scalar](T x, T y) { return x + scalar * y; });
    timings[3].push_back(seconds_since(t));

    t = clock::now();
    sum = std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(), b.begin(), T{}, std::plus<T>{}, std::multiplies<T>{});
    timings[4].push_back(seconds_since(t));
  }

  std::printf("%-12s%-12s%-12s%-12s%-12s\n", "Function", "MBytes/sec", "Min(s)", "Max(s)", "Avg(s)");
  const char *names[5] = {"Copy", "Mul", "Add", "Triad", "Dot"};
  std::size_t bytes_per[5] = {2 * bytes, 2 * bytes, 3 * bytes, 3 * bytes, 2 * bytes};
  for (int i = 0; i < 5; ++i) {
    auto &ts = timings[i];
    if (ts.size() <= 1) continue;
    auto begin = ts.begin() + 1, end = ts.end();
    double minv = *std::min_element(begin, end);
    double maxv = *std::max_element(begin, end);
    double avg = std::accumulate(begin, end, 0.0) / static_cast<double>(end - begin);
    std::printf("%-12s%-12.3f%-12.5f%-12.5f%-12.5f\n", names[i], 1.0e-6 * static_cast<double>(bytes_per[i]) / minv, minv, maxv, avg);
  }

  T goldA = static_cast<T>(startA), goldB = static_cast<T>(startB), goldC = static_cast<T>(startC);
  for (int i = 0; i < times; ++i) {
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }
  T goldSum = goldA * goldB * static_cast<T>(size);
  const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
  bool errA = failed(a, eps * 100.0, goldA, "a[]");
  bool errB = failed(b, eps * 100.0, goldB, "b[]");
  bool errC = failed(c, eps * 100.0, goldC, "c[]");
  double errSum = std::fabs(static_cast<double>((sum - goldSum) / goldSum));
  bool sumErr = errSum > eps * static_cast<double>(size);
  if (sumErr) {
    std::fprintf(stderr, "Validation failed on sum. Error %g\n", errSum);
    std::fprintf(stderr, "Sum was %.15f but should be %.15f\n", static_cast<double>(sum), static_cast<double>(goldSum));
  }
  return !errA && !errB && !errC && !sumErr;
}
} // namespace

int main(int argc, const char *argv[]) {
  int size = 33554432;
  int times = 100;
  if (argc - 1 >= 1) {
    char *endp = nullptr;
    long v = std::strtol(argv[1], &endp, 10);
    if (endp && *endp == '\0' && v > 0) size = static_cast<int>(v);
  }
  if (argc - 1 >= 2) {
    char *endp = nullptr;
    long v = std::strtol(argv[2], &endp, 10);
    if (endp && *endp == '\0' && v > 0) times = static_cast<int>(v);
  }
  int code = run<CHECK_NUM_TYPE>(size, times) ? EXIT_SUCCESS : EXIT_FAILURE;
  std::printf("Done\n");
  return code;
}
