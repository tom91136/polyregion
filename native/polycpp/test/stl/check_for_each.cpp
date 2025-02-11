#pragma region case: general
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: SIZE=1024 {output}
#pragma region requires: 204.800000

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

template <typename N> class ranged {
public:
  class iterator {
    friend class ranged;

  public:
    using difference_type = N;
    using value_type = N;
    using pointer = const N *;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;

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
    iterator operator--(int) {
      iterator copy(*this);
      --i_;
      return copy;
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
  ranged(N begin, N end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};

int main() {

  int size = 1024;
  if (auto sizeEnv = std::getenv("SIZE"); sizeEnv) size = std::stoi(sizeEnv);

  const double scalar = 0.4;

  std::vector<double> a(size);
  std::fill(a.begin(), a.end(), 0.1);

  std::vector<double> b(size);
  std::fill(b.begin(), b.end(), 0.2);

  std::vector<double> c(size);
  std::fill(c.begin(), c.end(), 0.0);

  // a = b + scalar * c
  ranged<int> r(0, size);
  std::for_each(                                                   //
      std::execution::par_unseq, r.begin(), r.end(),               //
      [scalar, a = a.data(), b = b.data(), c = c.data()](auto i) { //
        a[i] = b[i] + scalar * c[i];
      });

  auto checksum = std::reduce(a.begin(), a.end(), 0.0, std::plus<>());

  printf("%f", checksum);
  fflush(stdout);
  return EXIT_SUCCESS;
}
