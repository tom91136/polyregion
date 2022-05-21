#pragma once

#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace polyregion {

template <typename T = std::nullptr_t> //
constexpr T undefined(const std::string &file, size_t line, const std::string &message = "not implemented") {
  throw std::logic_error(file + ":" + std::to_string(line) + ": " + message);
}

[[noreturn]] static void error(const std::string &file, size_t line, const std::string &message = "not implemented") {
  throw std::logic_error(file + ":" + std::to_string(line) + ": " + message);
}

template <typename T, template <typename...> typename Container = std::vector>
std::string mk_string(const Container<T> &xs,
                      const std::function<std::string(const T &)> &f, //
                      const std::string &delim) {
  return std::empty(xs) ? ""
                        : std::transform_reduce(
                              ++std::begin(xs), std::end(xs), f(*std::begin(xs)),
                              [delim](auto &&a, auto &&b) -> std::string { return a + delim + b; }, f);
}

template <typename K, typename V, template <typename...> typename Container>
std::string mk_string2(const Container<K, V> &xs,
                       const std::function<std::string(const typename Container<K, V>::value_type &)> &f, //
                       const std::string &delim) {
  return std::empty(xs) ? ""
                        : std::transform_reduce(
                              ++std::begin(xs), std::end(xs), f(*std::begin(xs)),
                              [delim](auto &&a, auto &&b) -> std::string { return a + delim + b; }, f);
}

template <typename T, typename U, template <typename...> typename Container>
Container<U> map_vec(const Container<T> &xs, const std::function<U(const T &)> &f) {
  Container<U> ys(std::size(xs));
  std::transform(std::begin(xs), std::end(xs), ys.begin(), f);
  return ys;
}

template <typename T, typename F> //
std::optional<std::result_of_t<F(T)>> map_opt(const std::optional<T> &maybe, F &&f) {
  if (maybe) return std::forward<F>(f)(*maybe);
  else
    return std::nullopt;
}

template <typename T, typename F> //
std::result_of_t<F(T)> bind_opt(const std::optional<T> &maybe, F &&f) {
  if (maybe) return std::forward<F>(f)(*maybe);
  else
    return std::nullopt;
}

static std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> xs;
  std::stringstream stream(str);
  std::string x;
  while (std::getline(stream, x, delim))
    xs.push_back(x);
  return xs;
}

template <typename T> std::vector<T> read_struct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Cannot open binary file for reading: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<T> xs(len / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

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

// https://stackoverflow.com/a/33447587/896997
template <typename N, typename = typename std::enable_if_t<std::is_arithmetic_v<N>, N>>
std::string hex(N w, size_t hex_len = sizeof(N) << 1) {
  static const char *digits = "0123456789ABCDEF";
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

//constexpr inline unsigned int operator"" _(char const *p, size_t s) { return hash(p, s); }

} // namespace polyregion