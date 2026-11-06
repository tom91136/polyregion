#pragma region case: index_division
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: polycpp {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
#pragma region stderr-count: 1 polyregion.core.transform_reduce bound to polyregion.core (spectra.implementation.
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0/257 0/1031

#pragma region case: index_division-baseline
#pragma region using: opt=O0,O3
#pragma region offload-only
#pragma region do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polycpp {spectra_stdpar} -{opt} -o {output} {input}
#pragma region stderr-count: 0 bound to polyregion.core
#pragma region stderr-count: 0 uses the inline body
#pragma region do: {output}
#pragma region requires: 0/257 0/1031

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <functional>
#include <numeric>
#include <type_traits>

template <typename N> class range {
public:
  class iterator {
    friend class range;

  public:
    using difference_type = typename std::make_signed_t<N>;
    using value_type = N;
    using pointer = const N *;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;
    using is_passed_directly = std::true_type;

    reference operator*() const { return i_; }
    iterator &operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }
    iterator &operator--() {
      --i_;
      return *this;
    }
    iterator &operator+=(N by) {
      i_ += by;
      return *this;
    }
    value_type operator[](const difference_type &i) const { return i_ + i; }
    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }
    bool operator==(const iterator &other) const { return i_ == other.i_; }
    bool operator!=(const iterator &other) const { return i_ != other.i_; }
    bool operator<(const iterator &other) const { return i_ < other.i_; }

  protected:
    explicit iterator(N start) : i_(start) {}

  private:
    N i_;
  };

  [[nodiscard]] iterator begin() const { return begin_; }
  [[nodiscard]] iterator end() const { return end_; }
  range(N begin, N end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};

int check(const size_t n) {
  const int xmin = std::atoi("0"), xmax = std::atoi("383"), ymin = std::atoi("0");
  const auto indices = range<size_t>(0, n);
  int bad = 0;
  for (size_t t = 0; t < n; ++t) {
    auto only = [=](size_t idx) {
      const size_t j = xmin + 1 + idx % (xmax - xmin + 1);
      const size_t k = ymin + 1 + idx / (xmax - xmin + 1);
      return idx == t ? static_cast<double>(j + 1 + 1000 * k) : 0.0;
    };
    const double device = std::transform_reduce(std::execution::par_unseq, indices.begin(), indices.end(), 0.0, std::plus<>(), only);
    bad += device != only(t);
  }
  return bad;
}

int main() {
  std::printf("%d/%zu %d/%zu", check(257), size_t{257}, check(1031), size_t{1031});
  return 0;
}
