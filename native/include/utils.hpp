#pragma once

#include <cstring>
// #include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace polyregion {

template <typename T = std::nullptr_t> //
constexpr T undefined(const std::string &file, size_t line, const std::string &message = "not implemented") {
  throw std::logic_error(file + ":" + std::to_string(line) + ": " + message);
}

[[noreturn]] static void error(const std::string &file, size_t line, const std::string &message = "not implemented") {
  throw std::logic_error(file + ":" + std::to_string(line) + ": " + message);
}

template <typename T>
std::string mk_string(const std::vector<T> &xs,
                      const std::function<std::string(const T &)> &f, //
                      const std::string &delim) {
  return std::empty(xs) ? ""
                        : std::transform_reduce(
                              ++std::begin(xs), std::end(xs), f(*std::begin(xs)),
                              [delim](auto &&a, auto &&b) -> std::string { return a + delim + b; }, f);
}

template <typename K, typename V>
std::string mk_string2(const std::unordered_map<K, V> &xs,
                       const std::function<std::string(const typename std::unordered_map<K, V>::value_type &)> &f, //
                       const std::string &delim) {
  return std::empty(xs) ? ""
                        : std::transform_reduce(
                              ++std::begin(xs), std::end(xs), f(*std::begin(xs)),
                              [delim](auto &&a, auto &&b) -> std::string { return a + delim + b; }, f);
}

template <typename T, typename U>
std::vector<U> map_vec(const std::vector<T> &xs, const std::function<U(const T &)> &f) {
  std::vector<U> ys;
  std::transform(std::begin(xs), std::end(xs), std::back_inserter(ys), f);
  return ys;
}


// See https://stackoverflow.com/a/64500326
template <typename C, typename F> auto map_vec2(C &&xs, F &&f) -> std::vector<decltype(f(std::forward<C>(xs)[0]))> {
  std::vector<decltype(f(std::forward<C>(xs)[0]))> ys;
  std::transform(std::begin(xs), std::end(xs), std::back_inserter(ys), f);
  return ys;
}

// See https://stackoverflow.com/a/64500326
template <typename O, typename F> auto map_opt(O &&o, F &&f) -> std::optional<decltype(f(*std::forward<O>(o)))> {
  if (!o.has_value()) {
    return {std::nullopt};
  }
  return {f(*std::forward<O>(o))};
}

static std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> xs;
  std::stringstream stream(str);
  std::string x;
  while (std::getline(stream, x, delim))
    xs.push_back(x);
  return xs;
}

template <typename T> std::vector<T> flatten(const std::vector<std::vector<T>> &v) {
  std::size_t total_size = 0;
  for (const auto &sub : v)
    total_size += sub.size();
  std::vector<T> result;
  result.reserve(total_size);
  for (const auto &sub : v)
    result.insert(result.end(), sub.begin(), sub.end());
  return result;
}

template <typename T> std::vector<std::vector<T>> cartesin_product(const std::vector<std::vector<T>> &v) {
  std::vector<std::vector<T>> s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<T>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = move(r);
  }
  return s;
}

template <typename S> S indent(size_t n, const S &in) {
  return mk_string<S>(
      split(in, '\n'), [n](auto &l) { return std::string(n, ' ') + l; }, "\n");
}

// template <typename T> std::vector<T> read_struct(const std::string &path) {
//   std::fstream s(path, std::ios::binary | std::ios::in);
//   if (!s.good()) {
//     throw std::invalid_argument("Cannot open binary file for reading: " + path);
//   }
//   s.ignore(std::numeric_limits<std::streamsize>::max());
//   auto len = s.gcount();
//   s.clear();
//   s.seekg(0, std::ios::beg);
//   std::vector<T> xs(len / sizeof(T));
//   s.read(reinterpret_cast<char *>(xs.data()), len);
//   s.close();
//   return xs;
// }

template <typename T, template <typename...> typename Container>
constexpr std::optional<T> get_opt(const Container<T> &xs, size_t i) {
  return i > std::size(xs) ? std::nullopt : std::make_optional(xs[i]);
}

template <typename K, typename V, template <typename...> typename Container>
constexpr std::optional<V> get_opt(const Container<K, V> &xs, const K &k) {
  if (auto v = xs.find(k); v != xs.end()) return std::make_optional(v->second);
  else
    return std::nullopt;
}

template <typename T, typename Predicate>
std::pair<std::vector<T>, std::vector<T>> take_while(const std::vector<T> &input, Predicate pred) {
  std::vector<T> xs(input.begin(), input.end());
  std::vector<T> ys;
  auto pos = std::find_if_not(xs.begin(), xs.end(), pred);
  ys.insert(ys.end(), pos, xs.end());
  xs.erase(pos, xs.end());
  return {xs, ys};
}

// https://stackoverflow.com/a/33447587/896997
template <typename N, typename = typename std::enable_if_t<std::is_arithmetic_v<N>, N>>
std::string hex(N w, size_t hex_len = sizeof(N) << 1) {
  static const char *digits = "0123456789abcdef";
  std::string rc(hex_len, '0');
  for (size_t i = 0, j = (hex_len - 1) * 4; i < hex_len; ++i, j -= 4)
    rc[i] = digits[(w >> j) & 0x0f];
  return rc;
}

static inline char *new_str(const std::string &s) {
  auto xs = new char[s.length() + 1];
  std::copy(s.c_str(), s.c_str() + s.length() + 1, xs);
  return xs;
}

static inline void free_str(const char *s) { delete[] s; }

template <typename E> constexpr auto to_underlying(E e) noexcept { return static_cast<std::underlying_type_t<E>>(e); }

constexpr uint32_t hash(const char *data, size_t const size) noexcept {
  uint32_t hash = 5381;

  for (const char *c = data; c < data + size; ++c)
    hash = ((hash << 5) + hash) + (unsigned char)*c;

  return hash;
}

constexpr uint32_t hash(const std::string_view &str) noexcept { return hash(str.data(), str.length()); }

template <typename I, typename J> static std::enable_if_t<std::is_signed_v<I> && std::is_signed_v<J>, I> int_cast(J value) {
  if (value < std::numeric_limits<I>::min() || value > std::numeric_limits<I>::max()) {
    throw std::out_of_range("out of range");
  }
  return static_cast<I>(value);
}

template <typename I, typename J> static std::enable_if_t<std::is_signed_v<I> && std::is_unsigned_v<J>, I> int_cast(J value) {
  if (value > static_cast<std::make_unsigned_t<I>>(std::numeric_limits<I>::max())) {
    throw std::out_of_range("out of range");
  }
  return static_cast<I>(value);
}

template <typename I, typename J> static std::enable_if_t<std::is_unsigned_v<I> && std::is_signed_v<J>, I> int_cast(J value) {
  if (value < 0 || static_cast<std::make_unsigned_t<J>>(value) > std::numeric_limits<I>::max()) {
    throw std::out_of_range("out of range");
  }
  return static_cast<I>(value);
}

template <typename I, typename J> static std::enable_if_t<std::is_unsigned_v<I> && std::is_unsigned_v<J>, I> int_cast(J value) {
  if (value > std::numeric_limits<I>::max()) {
    throw std::out_of_range("out of range");
  }
  return static_cast<I>(value);
}

static inline std::string &trimInplace(std::string &str) {
  str.erase(str.find_last_not_of(' ') + 1);
  str.erase(0, str.find_first_not_of(' '));
  return str;
}

static inline std::string trim(const std::string &str) {
  std::string that = str;
  return trimInplace(that);
}

static inline std::string &replaceInPlace(std::string &haystack, const std::string &needle, const std::string &replace) {
  size_t pos = 0;
  while ((pos = haystack.find(needle, pos)) != std::string::npos) {
    haystack.replace(pos, needle.length(), replace);
    pos += replace.length();
  }
  return haystack;
}

static inline std::string &replace(std::string &haystack, const std::string &needle, const std::string &replace) {
  std::string that = haystack;
  return replaceInPlace(that, needle, replace);
}

// constexpr inline unsigned int operator"" _(char const *p, size_t s) { return hash(p, s); }

} // namespace polyregion