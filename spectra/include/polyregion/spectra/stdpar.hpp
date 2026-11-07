#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

#include <polyregion/spectra/host.hpp>

namespace spectra::stdpar::details {

template <class It> using value_t = typename std::iterator_traits<It>::value_type;

template <class It>
inline constexpr bool element_v =
    std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<It>::iterator_category>
    && std::is_lvalue_reference_v<typename std::iterator_traits<It>::reference> && std::is_trivially_copyable_v<value_t<It>>;

template <class A, class B> inline constexpr bool same_element_v = element_v<A> && element_v<B> && std::is_same_v<value_t<A>, value_t<B>>;

template <class, class = void> inline constexpr bool iterator_v = false;

template <class It> inline constexpr bool iterator_v<It, std::void_t<typename std::iterator_traits<It>::iterator_category>> = true;

template <class R, class Op, class... Args> struct returns : std::is_same<std::invoke_result_t<const Op &, Args...>, R> {};

template <class R, class Op, class... Args>
inline constexpr bool callable_v = std::conjunction_v<std::is_invocable<const Op &, Args...>, returns<R, Op, Args...>>;

template <class It> bool contiguous(It first, std::ptrdiff_t n) {
  if (n <= 0 || n > std::numeric_limits<std::int32_t>::max()) return false;
  if constexpr (std::is_pointer_v<It>) return true;
#if defined(__cpp_lib_concepts)
  else if constexpr (std::contiguous_iterator<It>) return true;
#endif
  else {
    const auto *base = std::addressof(*first);
    for (std::ptrdiff_t i = 1; i < n; ++i)
      if (std::addressof(first[i]) != base + i) return false;
    return true;
  }
}

template <class In, class Out> bool contiguous_disjoint(In first, Out result, std::ptrdiff_t n) {
  if (!contiguous(first, n) || !contiguous(result, n)) return false;
  const auto in = reinterpret_cast<std::uintptr_t>(std::addressof(*first));
  const auto out = reinterpret_cast<std::uintptr_t>(std::addressof(*result));
  return in + static_cast<std::uintptr_t>(n) * sizeof(value_t<In>) <= out
         || out + static_cast<std::uintptr_t>(n) * sizeof(value_t<Out>) <= in;
}

}

namespace std { // NOLINT(*-dcl58-cpp)

template <class It, class Pred> bool all_of(execution::parallel_unsequenced_policy, It first, It last, Pred pred) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n))
      return ::spectra::all_of(std::addressof(*first), static_cast<std::int32_t>(n), [pred](V x) { return pred(x); });
  }
  return std::all_of(first, last, pred);
}

template <class It, class Pred> bool any_of(execution::parallel_unsequenced_policy, It first, It last, Pred pred) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n))
      return ::spectra::any_of(std::addressof(*first), static_cast<std::int32_t>(n), [pred](V x) { return pred(x); });
  }
  return std::any_of(first, last, pred);
}

template <class It, class Pred> bool none_of(execution::parallel_unsequenced_policy, It first, It last, Pred pred) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n))
      return ::spectra::none_of(std::addressof(*first), static_cast<std::int32_t>(n), [pred](V x) { return pred(x); });
  }
  return std::none_of(first, last, pred);
}

template <class It1, class It2> bool equal(execution::parallel_unsequenced_policy, It1 first1, It1 last1, It2 first2) {
  using V = ::spectra::stdpar::details::value_t<It1>;
  if constexpr (::spectra::stdpar::details::same_element_v<It1, It2>) {
    const auto n = last1 - first1;
    if (::spectra::stdpar::details::contiguous(first1, n) && ::spectra::stdpar::details::contiguous(first2, n))
      return ::spectra::equal(std::addressof(*first1), std::addressof(*first2), static_cast<std::int32_t>(n),
                              [](V x, V y) -> bool { return x == y; });
  }
  return std::equal(first1, last1, first2);
}

template <class It1, class It2, class Pred, class = std::enable_if_t<!::spectra::stdpar::details::iterator_v<Pred>>>
bool equal(execution::parallel_unsequenced_policy, It1 first1, It1 last1, It2 first2, Pred pred) {
  using V = ::spectra::stdpar::details::value_t<It1>;
  if constexpr (::spectra::stdpar::details::same_element_v<It1, It2>
                && ::spectra::stdpar::details::callable_v<bool, Pred, const V &, const V &>) {
    const auto n = last1 - first1;
    if (::spectra::stdpar::details::contiguous(first1, n) && ::spectra::stdpar::details::contiguous(first2, n))
      return ::spectra::equal(std::addressof(*first1), std::addressof(*first2), static_cast<std::int32_t>(n),
                              [pred](V x, V y) { return pred(x, y); });
  }
  return std::equal(first1, last1, first2, pred);
}

template <class It1, class It2, class Op = std::less<>>
bool includes(execution::parallel_unsequenced_policy, It1 first1, It1 last1, It2 first2, It2 last2, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It1>;
  if constexpr (::spectra::stdpar::details::same_element_v<It1, It2>
                && ::spectra::stdpar::details::callable_v<bool, Op, const V &, const V &>) {
    const auto n1 = last1 - first1;
    const auto n2 = last2 - first2;
    if (::spectra::stdpar::details::contiguous(first1, n1) && ::spectra::stdpar::details::contiguous(first2, n2))
      return ::spectra::includes(std::addressof(*first1), static_cast<std::int32_t>(n1), std::addressof(*first2),
                                 static_cast<std::int32_t>(n2), [op](V x, V y) { return op(x, y); });
  }
  return std::includes(first1, last1, first2, last2, op);
}

template <class It, class Pred> bool is_partitioned(execution::parallel_unsequenced_policy, It first, It last, Pred pred) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n))
      return ::spectra::is_partitioned(std::addressof(*first), static_cast<std::int32_t>(n), [pred](V x) { return pred(x); });
  }
  return std::is_partitioned(first, last, pred);
}

template <class It, class Op = std::less<>> bool is_sorted(execution::parallel_unsequenced_policy, It first, It last, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n))
      return ::spectra::is_sorted(std::addressof(*first), static_cast<std::int32_t>(n), [op](V x, V y) { return op(x, y); });
  }
  return std::is_sorted(first, last, op);
}

template <class It, class OutIt, class Op = std::minus<>>
OutIt adjacent_difference(execution::parallel_unsequenced_policy, It first, It last, OutIt result, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt>
                && ::spectra::stdpar::details::callable_v<V, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::adjacent_difference(std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n),
                                     [op](V x, V y) { return op(x, y); });
      return result + n;
    }
  }
  return std::adjacent_difference(first, last, result, op);
}

template <class It, class OutIt, class Op = std::plus<>>
OutIt inclusive_scan(execution::parallel_unsequenced_policy, It first, It last, OutIt result, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt>
                && ::spectra::stdpar::details::callable_v<V, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::inclusive_scan(std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n),
                                [op](V x, V y) { return op(x, y); });
      return result + n;
    }
  }
  return std::inclusive_scan(first, last, result, op);
}

template <class It, class OutIt, class T, class Op = std::plus<>>
OutIt exclusive_scan(execution::parallel_unsequenced_policy, It first, It last, OutIt result, T init, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt> && std::is_same_v<T, V>
                && ::spectra::stdpar::details::callable_v<V, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::exclusive_scan(std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n), init,
                                [op](V x, V y) { return op(x, y); });
      return result + n;
    }
  }
  return std::exclusive_scan(first, last, result, init, op);
}

template <class It, class OutIt, class UnOp, class BinOp>
OutIt transform_inclusive_scan(execution::parallel_unsequenced_policy, It first, It last, OutIt result, BinOp binop, UnOp unop) {
  using V = ::spectra::stdpar::details::value_t<It>;
  using U = ::spectra::stdpar::details::value_t<OutIt>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::element_v<OutIt>
                && ::spectra::stdpar::details::callable_v<U, UnOp, const V &>
                && ::spectra::stdpar::details::callable_v<U, BinOp, const U &, const U &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::transform_inclusive_scan(
          std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n), [unop](V x) { return unop(x); },
          [binop](U x, U y) { return binop(x, y); });
      return result + n;
    }
  }
  return std::transform_inclusive_scan(first, last, result, binop, unop);
}

template <class It, class OutIt, class T, class UnOp, class BinOp>
OutIt transform_exclusive_scan(execution::parallel_unsequenced_policy, It first, It last, OutIt result, T init, BinOp binop, UnOp unop) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::element_v<OutIt>
                && std::is_same_v<T, ::spectra::stdpar::details::value_t<OutIt>>
                && ::spectra::stdpar::details::callable_v<T, UnOp, const V &>
                && ::spectra::stdpar::details::callable_v<T, BinOp, const T &, const T &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::transform_exclusive_scan(
          std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n), init, [unop](V x) { return unop(x); },
          [binop](T x, T y) { return binop(x, y); });
      return result + n;
    }
  }
  return std::transform_exclusive_scan(first, last, result, init, binop, unop);
}

template <class It, class Op = std::less<>> void sort(execution::parallel_unsequenced_policy, It first, It last, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::sort(std::addressof(*first), static_cast<std::int32_t>(n), [op](V x, V y) { return op(x, y); });
      return;
    }
  }
  std::sort(first, last, op);
}

template <class It, class Op = std::less<>> void stable_sort(execution::parallel_unsequenced_policy, It first, It last, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<bool, Op, const V &, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::stable_sort(std::addressof(*first), static_cast<std::int32_t>(n), [op](V x, V y) { return op(x, y); });
      return;
    }
  }
  std::stable_sort(first, last, op);
}

template <class It1, class It2, class OutIt, class Op = std::less<>>
OutIt merge(execution::parallel_unsequenced_policy, It1 first1, It1 last1, It2 first2, It2 last2, OutIt result, Op op = {}) {
  using V = ::spectra::stdpar::details::value_t<It1>;
  if constexpr (::spectra::stdpar::details::same_element_v<It1, It2> && ::spectra::stdpar::details::same_element_v<It1, OutIt>
                && ::spectra::stdpar::details::callable_v<bool, Op, const V &, const V &>) {
    const auto n1 = last1 - first1;
    const auto n2 = last2 - first2;
    if (::spectra::stdpar::details::contiguous(first1, n1) && ::spectra::stdpar::details::contiguous(first2, n2)
        && ::spectra::stdpar::details::contiguous(result, n1 + n2)) {
      ::spectra::merge(std::addressof(*first1), static_cast<std::int32_t>(n1), std::addressof(*first2), static_cast<std::int32_t>(n2),
                       std::addressof(*result), [op](V x, V y) { return op(x, y); });
      return result + (n1 + n2);
    }
  }
  return std::merge(first1, last1, first2, last2, result, op);
}

template <class It, class Gen> void generate(execution::parallel_unsequenced_policy, It first, It last, Gen gen) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<V, Gen>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::generate(std::addressof(*first), static_cast<std::int32_t>(n), [gen]() { return gen(); });
      return;
    }
  }
  std::generate(first, last, gen);
}

template <class It, class Size, class Gen> It generate_n(execution::parallel_unsequenced_policy, It first, Size count, Gen gen) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && ::spectra::stdpar::details::callable_v<V, Gen>) {
    const auto n = static_cast<std::ptrdiff_t>(count);
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::generate_n(std::addressof(*first), static_cast<std::int32_t>(n), [gen]() { return gen(); });
      return first + n;
    }
  }
  return std::generate_n(first, count, gen);
}

template <class It, class Size, class OutIt> OutIt copy_n(execution::parallel_unsequenced_policy, It first, Size count, OutIt result) {
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt>) {
    const auto n = static_cast<std::ptrdiff_t>(count);
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::copy_n(std::addressof(*first), static_cast<std::int32_t>(n), std::addressof(*result));
      return result + n;
    }
  }
  return std::copy_n(first, count, result);
}

template <class It1, class It2> It2 swap_ranges(execution::parallel_unsequenced_policy, It1 first1, It1 last1, It2 first2) {
  if constexpr (::spectra::stdpar::details::same_element_v<It1, It2>) {
    const auto n = last1 - first1;
    if (::spectra::stdpar::details::contiguous(first1, n) && ::spectra::stdpar::details::contiguous(first2, n)) {
      ::spectra::swap_ranges(std::addressof(*first1), static_cast<std::int32_t>(n), std::addressof(*first2));
      return first2 + n;
    }
  }
  return std::swap_ranges(first1, last1, first2);
}

template <class It, class T> void replace(execution::parallel_unsequenced_policy, It first, It last, T old_value, T new_value) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && std::is_same_v<T, V>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::replace(std::addressof(*first), static_cast<std::int32_t>(n), old_value, new_value,
                         [](V x, V y) -> bool { return x == y; });
      return;
    }
  }
  std::replace(first, last, old_value, new_value);
}

template <class It, class OutIt, class T>
OutIt replace_copy(execution::parallel_unsequenced_policy, It first, It last, OutIt result, T old_value, T new_value) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt> && std::is_same_v<T, V>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::replace_copy(std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n), old_value, new_value,
                              [](V x, V y) -> bool { return x == y; });
      return result + n;
    }
  }
  return std::replace_copy(first, last, result, old_value, new_value);
}

template <class It, class Pred, class T>
void replace_if(execution::parallel_unsequenced_policy, It first, It last, Pred pred, T new_value) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::element_v<It> && std::is_same_v<T, V>
                && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::replace_if(std::addressof(*first), static_cast<std::int32_t>(n), new_value, [pred](V x) { return pred(x); });
      return;
    }
  }
  std::replace_if(first, last, pred, new_value);
}

template <class It, class OutIt, class Pred, class T>
OutIt replace_copy_if(execution::parallel_unsequenced_policy, It first, It last, OutIt result, Pred pred, T new_value) {
  using V = ::spectra::stdpar::details::value_t<It>;
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt> && std::is_same_v<T, V>
                && ::spectra::stdpar::details::callable_v<bool, Pred, const V &>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::replace_copy_if(std::addressof(*first), std::addressof(*result), static_cast<std::int32_t>(n), new_value,
                                 [pred](V x) { return pred(x); });
      return result + n;
    }
  }
  return std::replace_copy_if(first, last, result, pred, new_value);
}

template <class It> void reverse(execution::parallel_unsequenced_policy, It first, It last) {
  if constexpr (::spectra::stdpar::details::element_v<It>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous(first, n)) {
      ::spectra::reverse(std::addressof(*first), static_cast<std::int32_t>(n));
      return;
    }
  }
  std::reverse(first, last);
}

template <class It, class OutIt> OutIt reverse_copy(execution::parallel_unsequenced_policy, It first, It last, OutIt result) {
  if constexpr (::spectra::stdpar::details::same_element_v<It, OutIt>) {
    const auto n = last - first;
    if (::spectra::stdpar::details::contiguous_disjoint(first, result, n)) {
      ::spectra::reverse_copy(std::addressof(*first), static_cast<std::int32_t>(n), std::addressof(*result));
      return result + n;
    }
  }
  return std::reverse_copy(first, last, result);
}

}
