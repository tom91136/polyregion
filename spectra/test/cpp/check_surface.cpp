#include <functional>

#include <polyregion/spectra_api.hpp>

void check(int *xs, const int *ys, const int *map) {
  float fs[4]{};
  double ds[4]{};
  const auto unary = [](const int &x) { return x + 1; };
  const auto binary = [](const int &x, const int &y) { return x + y; };
  const auto predicate = [](const int &x) { return x > 0; };
  const auto compare = [](const int &x, const int &y) { return x < y; };
  const auto valueOp = [](const float &x, const float &y) { return x + y; };

  spectra::transform(ys, fs, 4, [](const int &x) { return float(x + 1); });
  spectra::transform_binary(ys, fs, ds, 4, [](const int &x, const float &y) { return double(x) + y; });
  (void)spectra::reduce(ys, 4, 0, binary);
  spectra::exclusive_scan(ys, xs, 4, 0, binary);
  spectra::transform_inclusive_scan(ys, ds, 4, [](const int &x) { return double(x); }, std::plus<>{});
  spectra::transform_exclusive_scan(ys, ds, 4, 0.0, [](const int &x) { return double(x); }, std::plus<>{});
  spectra::adjacent_difference(ys, xs, 4, [](const int &x, const int &y) { return x - y; });
  spectra::for_each(xs, 4, unary);
  spectra::generate(xs, 4, [] { return 1; });
  spectra::copy(ys, 4, xs);
  spectra::gather(map, 4, ys, 16, xs);
  spectra::scatter(ys, 4, map, xs, 16);
  (void)spectra::all_of(ys, 4, predicate);
  (void)spectra::count(ys, 4, 1);
  (void)spectra::inner_product(ys, fs, 4, 0.0, std::plus<>{}, [](const int &x, const float &y) { return double(x) * y; });
  (void)spectra::transform_reduce(ys, 4, 0.0, [](const int &x) { return double(x); }, std::plus<>{});
  (void)spectra::set_intersection(ys, 4, ys, 4, xs, 4, compare);
  spectra::sort_by_key(xs, fs, 4, compare);
  (void)spectra::reduce_by_key(ys, fs, xs, fs, 4, compare, valueOp);
}
