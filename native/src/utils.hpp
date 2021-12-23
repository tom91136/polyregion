#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace polyregion {

template <typename T = std::nullptr_t> T undefined(const std::string &message = "not implemented") {
  throw std::logic_error(message);
}

template <typename T, template <typename...> typename Container>
std::string mk_string(const Container<T> &xs,
                      const std::function<std::string(const T &)> &f, //
                      const std::string &delim) {
  return std::empty(xs) ? ""
                        : std::transform_reduce(
                              ++std::begin(xs), std::end(xs), f(*std::begin(xs)),
                              [delim](auto &&a, auto &&b) -> std::string { return a + delim + b; }, f);
}

template <typename T, typename U, template <typename...> typename Container>
std::vector<U> map_vec(const Container<T> &xs, const std::function<U(const T &)> &f) {
  std::vector<U> ys(std::size(xs));
  std::transform(std::begin(xs), std::end(xs), ys.begin(), f);
  return ys;
}

static std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> xs;
  std::stringstream stream(str);
  std::string x;
  while (std::getline(stream, x, delim))
    xs.push_back(x);
  return xs;
}

template <typename T> std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
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

std::optional<T> get_opt(const Container<T> &xs, size_t i) {
  return i > std::size(xs) ? std::nullopt : std::make_optional(xs[i]);
}

constexpr uint32_t hash(const char *data, size_t const size) noexcept {
  uint32_t hash = 5381;

  for (const char *c = data; c < data + size; ++c)
    hash = ((hash << 5) + hash) + (unsigned char)*c;

  return hash;
}

constexpr uint32_t hash(const std::string &str) noexcept { return hash(str.data(), str.length()); }

consteval inline unsigned int operator"" _(char const *p, size_t s) { return hash(p, s); }

} // namespace polyregion