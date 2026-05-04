#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polyregion {

template <typename T> using Vector = std::vector<T>;
template <typename T> using Opt = std::optional<T>;
template <typename T> using Set = std::unordered_set<T>;
template <typename T, typename U> using Pair = std::pair<T, U>;
template <typename K, typename V> using Map = std::unordered_map<K, V>;
using String = std::string;
using StringView = std::string_view;

} // namespace polyregion
