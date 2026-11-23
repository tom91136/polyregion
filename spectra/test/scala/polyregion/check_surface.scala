package polyregion

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Compile-only instantiations of every generated Scala declaration; runtime package coverage is separate.
final case class SpectraSurfaceValue12(x: Int, y: Int, z: Int)

object spectraSignatures {

  def adjacent_difference[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    ???

  def all_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    ???

  def any_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    ???

  def binary_search[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Boolean =
    ???

  def copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    ???

  def copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    ???

  def copy_n[T](in: Array[T], n: Int, out: Array[T]): Unit =
    ???

  def count[T](in: Array[T], n: Int, value: T, eq: (T, T) => Boolean): Int =
    ???

  def count_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def equal[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    ???

  def exclusive_scan[T](in: Array[T], out: Array[T], n: Int, init: T, op: (T, T) => T): Unit =
    ???

  def exclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      init: V,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    ???

  def fill[T](out: Array[T], n: Int, v: T): Unit =
    ???

  def fill_n[T](out: Array[T], n: Int, v: T): Unit =
    ???

  def find[T](in: Array[T], n: Int, value: T, eq: (T, T) => Boolean): Int =
    ???

  def find_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def find_if_not[T](in: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def for_each[T](data: Array[T], n: Int, op: T => T): Unit =
    ???

  def for_each_n[T](data: Array[T], n: Int, op: T => T): Unit =
    ???

  def gather[T](map: Array[Int], n: Int, in: Array[T], in_n: Int, out: Array[T]): Unit =
    ???

  def generate[T](out: Array[T], n: Int, op: () => T): Unit =
    ???

  def generate_n[T](out: Array[T], n: Int, op: () => T): Unit =
    ???

  def includes[T](a: Array[T], na: Int, b: Array[T], nb: Int, op: (T, T) => Boolean): Boolean =
    ???

  def inclusive_scan[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    ???

  def inclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    ???

  def inner_product[T, U, V](
      a: Array[T],
      b: Array[U],
      n: Int,
      init: V,
      op_reduce: (V, V) => V,
      op_product: (T, U) => V
  ): V =
    ???

  def is_partitioned[T](data: Array[T], n: Int, op: T => Boolean): Boolean =
    ???

  def is_sorted[T](data: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    ???

  def is_sorted_until[T](data: Array[T], n: Int, op: (T, T) => Boolean): Int =
    ???

  def lower_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    ???

  def max_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): Int =
    ???

  def merge[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Unit =
    ???

  def min_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): Int =
    ???

  def minmax_element[T](in: Array[T], n: Int, min_out: Array[Int], max_out: Array[Int], op: (T, T) => Boolean): Unit =
    ???

  def mismatch[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Int =
    ???

  def none_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    ???

  def partition[T](data: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def partition_point[T](in: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def reduce[T](in: Array[T], n: Int, init: T, op: (T, T) => T): T =
    ???

  def reduce_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      kout: Array[K],
      vout: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Int =
    ???

  def reduce_unordered[T](in: Array[T], n: Int, init: T, op: (T, T) => T): T =
    ???

  def remove[T](data: Array[T], n: Int, value: T, eq: (T, T) => Boolean): Int =
    ???

  def remove_copy[T](in: Array[T], n: Int, out: Array[T], value: T, eq: (T, T) => Boolean): Int =
    ???

  def remove_copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    ???

  def remove_if[T](data: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def replace[T](io: Array[T], n: Int, oldv: T, newv: T, eq: (T, T) => Boolean): Unit =
    ???

  def replace_copy[T](in: Array[T], out: Array[T], n: Int, oldv: T, newv: T, eq: (T, T) => Boolean): Unit =
    ???

  def replace_copy_if[T](in: Array[T], out: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    ???

  def replace_if[T](data: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    ???

  def reverse[T](data: Array[T], n: Int): Unit =
    ???

  def reverse_copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    ???

  def scatter[T](in: Array[T], n: Int, map: Array[Int], out: Array[T], out_n: Int): Unit =
    ???

  def search[T](in: Array[T], n: Int, sub: Array[T], m: Int, op: (T, T) => Boolean): Int =
    ???

  def search_n[T](in: Array[T], n: Int, count: Int, value: T, op: (T, T) => Boolean): Int =
    ???

  def sequence[T](out: Array[T], n: Int, init: T, step: T, value_at: (T, T, Int) => T): Unit =
    ???

  def set_difference[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    ???

  def set_intersection[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    ???

  def set_union[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    ???

  def sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    ???

  def sort_by_key[K, V](keys: Array[K], values: Array[V], n: Int, op: (K, K) => Boolean): Unit =
    ???

  def stable_partition[T](data: Array[T], n: Int, op: T => Boolean): Int =
    ???

  def stable_sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    ???

  def swap_ranges[T](a: Array[T], n: Int, b: Array[T]): Unit =
    ???

  def tabulate[T](out: Array[T], n: Int, op: Int => T): Unit =
    ???

  def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit =
    ???

  def transform_binary[T, U, V](a: Array[T], b: Array[U], out: Array[V], n: Int, op: (T, U) => V): Unit =
    ???

  def transform_exclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, init: U, map: T => U, op: (U, U) => U): Unit =
    ???

  def transform_inclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, map: T => U, op: (U, U) => U): Unit =
    ???

  def transform_reduce[T, U](in: Array[T], n: Int, init: U, map: T => U, op: (U, U) => U): U =
    ???

  def transform_reduce_unordered[T, U](in: Array[T], n: Int, init: U, map: T => U, op: (U, U) => U): U =
    ???

  def unique[T](data: Array[T], n: Int, eq: (T, T) => Boolean): Int =
    ???

  def unique_copy[T](in: Array[T], n: Int, out: Array[T], eq: (T, T) => Boolean): Int =
    ???

  def upper_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    ???
}

object check_surface {
  def compile(): Unit = InterfaceSurfaceCheck.erase {
    val n        = 4
    val values4  = new Array[Int](n)
    val values8  = new Array[Long](n)
    val values12 = new Array[SpectraSurfaceValue12](n)
    val value4   = 0
    val value8   = 0L
    val value12  = SpectraSurfaceValue12(0, 0, 0)

    // reduce: T=w8
    spectraSignatures.reduce(values8, n, value8, (arg0: Long, arg1: Long) => 0L)

    // reduce_unordered: T=w12
    spectraSignatures.reduce_unordered(
      values12,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // transform: T=w12, U=w4
    spectraSignatures.transform(values12, values4, n, (arg0: SpectraSurfaceValue12) => 0)

    // inclusive_scan: T=w8
    spectraSignatures.inclusive_scan(values8, values8, n, (arg0: Long, arg1: Long) => 0L)

    // exclusive_scan: T=w12
    spectraSignatures.exclusive_scan(
      values12,
      values12,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // sort: T=w4
    spectraSignatures.sort(values4, n, (arg0: Int, arg1: Int) => false)

    // count_if: T=w12
    spectraSignatures.count_if(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // min_element: T=w8
    spectraSignatures.min_element(values8, n, (arg0: Long, arg1: Long) => false)

    // max_element: T=w12
    spectraSignatures.max_element(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // find_if: T=w8
    spectraSignatures.find_if(values8, n, (arg0: Long) => false)

    // is_sorted: T=w8
    spectraSignatures.is_sorted(values8, n, (arg0: Long, arg1: Long) => false)

    // merge: T=w4
    spectraSignatures.merge(values4, n, values4, n, values4, (arg0: Int, arg1: Int) => false)

    // set_difference: T=w4
    spectraSignatures.set_difference(values4, n, values4, n, values4, (arg0: Int, arg1: Int) => false)

    // set_intersection: T=w12
    spectraSignatures.set_intersection(
      values12,
      n,
      values12,
      n,
      values12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // set_union: T=w12
    spectraSignatures.set_union(
      values12,
      n,
      values12,
      n,
      values12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // includes: T=w4
    spectraSignatures.includes(values4, n, values4, n, (arg0: Int, arg1: Int) => false)

    // copy_if: T=w8
    spectraSignatures.copy_if(values8, n, values8, (arg0: Long) => false)

    // remove_copy_if: T=w4
    spectraSignatures.remove_copy_if(values4, n, values4, (arg0: Int) => false)

    // unique_copy: T=w8
    spectraSignatures.unique_copy(values8, n, values8, (arg0: Long, arg1: Long) => false)

    // remove_if: T=w8
    spectraSignatures.remove_if(values8, n, (arg0: Long) => false)

    // unique: T=w12
    spectraSignatures.unique(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // sort_by_key: K=w12, V=w4
    spectraSignatures.sort_by_key(
      values12,
      values4,
      n,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // for_each: T=w12
    spectraSignatures.for_each(values12, n, (arg0: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0))

    // gather: T=w8
    spectraSignatures.gather(values4, n, values8, n, values8)

    // scatter: T=w12
    spectraSignatures.scatter(values12, n, values4, values12, n)

    // copy: T=w12
    spectraSignatures.copy(values12, n, values12)

    // reverse: T=w12
    spectraSignatures.reverse(values12, n)

    // fill: T=w4
    spectraSignatures.fill(values4, n, value4)

    // transform_reduce: T=w4, U=w8
    spectraSignatures.transform_reduce(values4, n, value8, (arg0: Int) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_reduce_unordered: T=w8, U=w12
    spectraSignatures.transform_reduce_unordered(
      values8,
      n,
      value12,
      (arg0: Long) => SpectraSurfaceValue12(0, 0, 0),
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // transform_inclusive_scan: T=w4, U=w8
    spectraSignatures.transform_inclusive_scan(values4, values8, n, (arg0: Int) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_exclusive_scan: T=w8, U=w12
    spectraSignatures.transform_exclusive_scan(
      values8,
      values12,
      n,
      value12,
      (arg0: Long) => SpectraSurfaceValue12(0, 0, 0),
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // reduce_by_key: K=w4, V=w8
    spectraSignatures.reduce_by_key(
      values4,
      values8,
      values4,
      values8,
      n,
      (arg0: Int, arg1: Int) => false,
      (arg0: Long, arg1: Long) => 0L
    )

    // inclusive_scan_by_key: K=w12, V=w4
    spectraSignatures.inclusive_scan_by_key(
      values12,
      values4,
      values4,
      n,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false,
      (arg0: Int, arg1: Int) => 0
    )

    // exclusive_scan_by_key: K=w4, V=w8
    spectraSignatures.exclusive_scan_by_key(
      values4,
      values8,
      values8,
      n,
      value8,
      (arg0: Int, arg1: Int) => false,
      (arg0: Long, arg1: Long) => 0L
    )

    // transform_binary: T=w8, U=w12, V=w4
    spectraSignatures.transform_binary(values8, values12, values4, n, (arg0: Long, arg1: SpectraSurfaceValue12) => 0)

    // inner_product: T=w8, U=w12, V=w4
    spectraSignatures.inner_product(
      values8,
      values12,
      n,
      value4,
      (arg0: Int, arg1: Int) => 0,
      (arg0: Long, arg1: SpectraSurfaceValue12) => 0
    )

    // equal: T=w12
    spectraSignatures.equal(values12, values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // mismatch: T=w8
    spectraSignatures.mismatch(values8, values8, n, (arg0: Long, arg1: Long) => false)

    // search: T=w12
    spectraSignatures.search(
      values12,
      n,
      values12,
      n,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // search_n: T=w12
    spectraSignatures.search_n(
      values12,
      n,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // all_of: T=w12
    spectraSignatures.all_of(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // any_of: T=w12
    spectraSignatures.any_of(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // none_of: T=w4
    spectraSignatures.none_of(values4, n, (arg0: Int) => false)

    // find_if_not: T=w4
    spectraSignatures.find_if_not(values4, n, (arg0: Int) => false)

    // generate: T=w4
    spectraSignatures.generate(values4, n, () => 0)

    // generate_n: T=w8
    spectraSignatures.generate_n(values8, n, () => 0L)

    // tabulate: T=w12
    spectraSignatures.tabulate(values12, n, (arg0: Int) => SpectraSurfaceValue12(0, 0, 0))

    // lower_bound: T=w12
    spectraSignatures.lower_bound(
      values12,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // upper_bound: T=w8
    spectraSignatures.upper_bound(values8, n, value8, (arg0: Long, arg1: Long) => false)

    // binary_search: T=w12
    spectraSignatures.binary_search(
      values12,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // partition: T=w12
    spectraSignatures.partition(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // stable_partition: T=w8
    spectraSignatures.stable_partition(values8, n, (arg0: Long) => false)

    // partition_point: T=w4
    spectraSignatures.partition_point(values4, n, (arg0: Int) => false)

    // is_partitioned: T=w8
    spectraSignatures.is_partitioned(values8, n, (arg0: Long) => false)

    // is_sorted_until: T=w12
    spectraSignatures.is_sorted_until(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // stable_sort: T=w12
    spectraSignatures.stable_sort(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // for_each_n: T=w12
    spectraSignatures.for_each_n(values12, n, (arg0: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0))

    // minmax_element: T=w12
    spectraSignatures.minmax_element(
      values12,
      n,
      values4,
      values4,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // copy_n: T=w12
    spectraSignatures.copy_n(values12, n, values12)

    // reverse_copy: T=w8
    spectraSignatures.reverse_copy(values8, n, values8)

    // fill_n: T=w4
    spectraSignatures.fill_n(values4, n, value4)

    // swap_ranges: T=w8
    spectraSignatures.swap_ranges(values8, n, values8)

    // count: T=w8
    spectraSignatures.count(values8, n, value8, (arg0: Long, arg1: Long) => false)

    // find: T=w4
    spectraSignatures.find(values4, n, value4, (arg0: Int, arg1: Int) => false)

    // remove: T=w12
    spectraSignatures.remove(values12, n, value12, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // remove_copy: T=w8
    spectraSignatures.remove_copy(values8, n, values8, value8, (arg0: Long, arg1: Long) => false)

    // replace: T=w4
    spectraSignatures.replace(values4, n, value4, value4, (arg0: Int, arg1: Int) => false)

    // replace_copy: T=w12
    spectraSignatures.replace_copy(
      values12,
      values12,
      n,
      value12,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // replace_if: T=w4
    spectraSignatures.replace_if(values4, n, value4, (arg0: Int) => false)

    // replace_copy_if: T=w8
    spectraSignatures.replace_copy_if(values8, values8, n, value8, (arg0: Long) => false)

    // sequence: T=w4
    spectraSignatures.sequence(values4, n, value4, value4, (arg0: Int, arg1: Int, arg2: Int) => 0)

    // adjacent_difference: T=w4
    spectraSignatures.adjacent_difference(values4, values4, n, (arg0: Int, arg1: Int) => 0)
  }
}
