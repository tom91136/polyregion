#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

#include "rt_protected.hpp"

namespace polyregion::rt_reflect::details {

template <typename V> class IntervalIndex {
  struct Node {
    uintptr_t base, end, maxEnd;
    const V *value;
    uint64_t priority;
    Node *left, *right;
  };

  Node *root{};
  size_t count{};
  uint64_t seed{0x9E3779B97F4A7C15ULL};

  __RT_ODR uint64_t nextPriority() {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
  }

  __RT_ODR static void refresh(Node *n) {
    n->maxEnd = n->end;
    if (n->left && n->left->maxEnd > n->maxEnd) n->maxEnd = n->left->maxEnd;
    if (n->right && n->right->maxEnd > n->maxEnd) n->maxEnd = n->right->maxEnd;
  }

  __RT_ODR static void split(Node *n, const uintptr_t key, Node *&left, Node *&right) {
    if (!n) {
      left = right = nullptr;
    } else if (n->base < key) {
      split(n->right, key, n->right, right);
      left = n;
      refresh(n);
    } else {
      split(n->left, key, left, n->left);
      right = n;
      refresh(n);
    }
  }

  __RT_ODR static Node *merge(Node *left, Node *right) {
    if (!left) return right;
    if (!right) return left;
    if (left->priority > right->priority) {
      left->right = merge(left->right, right);
      refresh(left);
      return left;
    }
    right->left = merge(left, right->left);
    refresh(right);
    return right;
  }

  __RT_ODR static const Node *find(const Node *n, const uintptr_t point, const uintptr_t minEnd, size_t &visited) {
    while (n && n->maxEnd >= minEnd) {
      ++visited;
      if (n->base > point) {
        n = n->left;
        continue;
      }
      if (const Node *hit = find(n->right, point, minEnd, visited)) return hit;
      if (n->end >= minEnd) return n;
      n = n->left;
    }
    return nullptr;
  }

  __RT_ODR static bool erase(Node *&n, const uintptr_t base) {
    if (!n) return false;
    if (n->base == base) {
      Node *gone = n;
      n = merge(gone->left, gone->right);
      __RT_ALTERNATIVE(free)(gone);
      return true;
    }
    if (!erase(base < n->base ? n->left : n->right, base)) return false;
    refresh(n);
    return true;
  }

  __RT_ODR static void destroy(Node *n) {
    if (!n) return;
    destroy(n->left);
    destroy(n->right);
    __RT_ALTERNATIVE(free)(n);
  }

public:
  __RT_ODR IntervalIndex() = default;
  __RT_ODR IntervalIndex(const IntervalIndex &) = delete;
  __RT_ODR IntervalIndex &operator=(const IntervalIndex &) = delete;
  __RT_ODR ~IntervalIndex() { destroy(root); }

  __RT_ODR void insert(const uintptr_t base, const size_t size, const V *value) {
    auto *n = static_cast<Node *>(__RT_ALTERNATIVE(malloc)(sizeof(Node)));
    new (n) Node{base, base + size, base + size, value, nextPriority(), nullptr, nullptr};
    Node **link = &root;
    while (*link && (*link)->priority >= n->priority) {
      Node *at = *link;
      if (n->end > at->maxEnd) at->maxEnd = n->end;
      link = base < at->base ? &at->left : &at->right;
    }
    split(*link, base, n->left, n->right);
    refresh(n);
    *link = n;
    ++count;
  }

  __RT_ODR bool erase(const uintptr_t base) {
    if (!erase(root, base)) return false;
    --count;
    return true;
  }

  __RT_ODR [[nodiscard]] const V *containing(const uintptr_t point, size_t &visited) const {
    const Node *n = find(root, point, point + 1, visited);
    return n ? n->value : nullptr;
  }

  __RT_ODR [[nodiscard]] const V *endingAt(const uintptr_t point, size_t &visited) const {
    const Node *n = find(root, point, point, visited);
    return n && n->end == point ? n->value : nullptr;
  }

  __RT_ODR [[nodiscard]] size_t size() const { return count; }
};

} // namespace polyregion::rt_reflect::details
