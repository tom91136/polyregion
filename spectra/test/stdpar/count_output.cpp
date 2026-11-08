#pragma region case: count-output-cxx20
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 8 bound to spectra
#pragma region do: {output} exact
#pragma region requires: pass
#pragma region do: {output} empty
#pragma region requires: pass

#pragma region case: count-output-cxx17
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -std=c++17 -O3 -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 bound to spectra
#pragma region do: {output} exact
#pragma region requires: pass
#pragma region do: {output} empty
#pragma region requires: pass

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

namespace {
using Vec = std::vector<int>;

std::string failures;

void expect(const char *name, const bool ok) {
  if (!ok) failures += failures.empty() ? name : std::string(" ") + name;
}

const auto par = std::execution::par_unseq;
const auto positive = [](const int x) { return x > 0; };

template <class Par, class Seq> void check(const char *name, const Vec &x, const Vec &y, const bool empty, Par par_op, Seq seq_op) {
  Vec buf(x.size() + y.size());
  const auto count = static_cast<std::size_t>(seq_op(x, y, buf.begin()) - buf.begin());
  Vec ref(count), got(count);
  expect(name, (count == 0) == empty && seq_op(x, y, ref.begin()) == ref.end() && par_op(x, y, got.begin()) == got.end() && got == ref);
}

void check_raw(const Vec &x, const bool empty) {
  Vec ref(x.size());
  ref.resize(static_cast<std::size_t>(std::copy_if(x.begin(), x.end(), ref.begin(), positive) - ref.begin()));
  Vec got(ref.size());
  const int *end = std::copy_if(par, x.begin(), x.end(), got.data(), positive);
  expect("copy_if_pointer", ref.empty() == empty && end == got.data() + got.size() && got == ref);
}
} // namespace

int main(int argc, char **argv) {
  const bool empty = argc > 1 && std::string(argv[1]) == "empty";

  Vec a(64), b(40), negative(40, -3), repeated(20, 7), ascending(30), far(40);
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<int>(i / 2) - 16;
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<int>(i) - 8;
  for (std::size_t i = 0; i < ascending.size(); ++i)
    ascending[i] = static_cast<int>(i) + 1;
  for (std::size_t i = 0; i < far.size(); ++i)
    far[i] = 1000 + static_cast<int>(i);

  check(
      "copy_if", empty ? negative : a, b, empty,
      [](const Vec &x, const Vec &, auto out) { return std::copy_if(par, x.begin(), x.end(), out, positive); },
      [](const Vec &x, const Vec &, auto out) { return std::copy_if(x.begin(), x.end(), out, positive); });
  check(
      "remove_copy", empty ? repeated : a, b, empty,
      [](const Vec &x, const Vec &, auto out) { return std::remove_copy(par, x.begin(), x.end(), out, 7); },
      [](const Vec &x, const Vec &, auto out) { return std::remove_copy(x.begin(), x.end(), out, 7); });
  check(
      "remove_copy_if", empty ? ascending : a, b, empty,
      [](const Vec &x, const Vec &, auto out) { return std::remove_copy_if(par, x.begin(), x.end(), out, positive); },
      [](const Vec &x, const Vec &, auto out) { return std::remove_copy_if(x.begin(), x.end(), out, positive); });
  check_raw(empty ? negative : a, empty);
  if (!empty) {
    check(
        "unique_copy", a, b, false, [](const Vec &x, const Vec &, auto out) { return std::unique_copy(par, x.begin(), x.end(), out); },
        [](const Vec &x, const Vec &, auto out) { return std::unique_copy(x.begin(), x.end(), out); });
    check(
        "set_union", a, b, false,
        [](const Vec &x, const Vec &y, auto out) { return std::set_union(par, x.begin(), x.end(), y.begin(), y.end(), out); },
        [](const Vec &x, const Vec &y, auto out) { return std::set_union(x.begin(), x.end(), y.begin(), y.end(), out); });
  }
  check(
      "set_difference", a, empty ? a : b, empty,
      [](const Vec &x, const Vec &y, auto out) { return std::set_difference(par, x.begin(), x.end(), y.begin(), y.end(), out); },
      [](const Vec &x, const Vec &y, auto out) { return std::set_difference(x.begin(), x.end(), y.begin(), y.end(), out); });
  check(
      "set_intersection", a, empty ? far : b, empty,
      [](const Vec &x, const Vec &y, auto out) { return std::set_intersection(par, x.begin(), x.end(), y.begin(), y.end(), out); },
      [](const Vec &x, const Vec &y, auto out) { return std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), out); });
  std::printf("%s", failures.empty() ? "pass" : failures.c_str());
}
