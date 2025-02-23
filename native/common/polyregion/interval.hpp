#pragma once

#include <algorithm>
#include <map>

namespace polyregion {

template <typename T, typename N = int64_t> class IntervalMap {
public:
  struct Range {
    N offset, size;
    N end() const { return offset + size; }
    bool operator<(const Range &that) const { return this->offset < that.offset; }
  };

  std::map<Range, T> map;

  typename std::map<Range, T>::iterator find(N point) {
    auto it = map.upper_bound(Range{point, 0});
    if (it == map.begin()) return map.end();
    --it;
    if (point >= it->first.offset && point < it->first.end()) return it;
    return map.end();
  }

  template <typename Merge> typename std::map<Range, T>::iterator insert(const Range &r, const T &data, Merge merge) {
    if (r.size <= 0) return map.end();
    N newStart = r.offset;
    N newEnd = r.end();
    auto it = map.lower_bound(Range{r.offset, 0});

    T merged = std::move(data);
    if (it != map.begin()) {
      if (auto prev = std::prev(it); prev->first.end() >= r.offset) { // overlap or touching
        newStart = std::min(newStart, prev->first.offset);
        newEnd = std::max(newEnd, prev->first.end());
        merged = std::move(merge(std::move(merged), std::move(prev->second)));
        it = map.erase(prev);
      }
    }
    while (it != map.end() && it->first.offset <= newEnd) { // merge all
      newEnd = std::max(newEnd, it->first.end());
      merged = std::move(merge(std::move(merged), std::move(it->second)));
      it = map.erase(it);
    }
    return map.insert(it, {Range{newStart, newEnd - newStart}, std::move(merged)});
  }
};

} // namespace polyregion