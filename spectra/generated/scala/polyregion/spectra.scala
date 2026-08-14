package polyregion

import scala.annotation.compileTimeOnly

object spectra {

  @compileTimeOnly("polyregion_interface:spectra:spectra.adjacent_difference")
  def adjacent_difference[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.adjacent_difference")

  @compileTimeOnly("polyregion_interface:spectra:spectra.all_of")
  def all_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.all_of")

  @compileTimeOnly("polyregion_interface:spectra:spectra.any_of")
  def any_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.any_of")

  @compileTimeOnly("polyregion_interface:spectra:spectra.binary_search")
  def binary_search[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.binary_search")

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy")
  def copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.copy")

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_if")
  def copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.copy_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_n")
  def copy_n[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.copy_n")

  @compileTimeOnly("polyregion_interface:spectra:spectra.count")
  def count[T](in: Array[T], n: Int, value: T): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.count")

  @compileTimeOnly("polyregion_interface:spectra:spectra.count_if")
  def count_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.count_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.equal")
  def equal[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.equal")

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan")
  def exclusive_scan[T](in: Array[T], out: Array[T], n: Int, init: T, op: (T, T) => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.exclusive_scan")

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan_by_key")
  def exclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      init: V,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.exclusive_scan_by_key")

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill")
  def fill[T](out: Array[T], n: Int, v: T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.fill")

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill_n")
  def fill_n[T](out: Array[T], n: Int, v: T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.fill_n")

  @compileTimeOnly("polyregion_interface:spectra:spectra.find")
  def find[T](in: Array[T], n: Int, value: T): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.find")

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if")
  def find_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.find_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if_not")
  def find_if_not[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.find_if_not")

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each")
  def for_each[T](data: Array[T], n: Int, op: T => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.for_each")

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each_n")
  def for_each_n[T](data: Array[T], n: Int, op: T => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.for_each_n")

  @compileTimeOnly("polyregion_interface:spectra:spectra.gather")
  def gather[T](map: Array[Int], n: Int, in: Array[T], in_n: Int, out: Array[T]): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.gather")

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate")
  def generate[T](out: Array[T], n: Int, op: () => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.generate")

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate_n")
  def generate_n[T](out: Array[T], n: Int, op: () => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.generate_n")

  @compileTimeOnly("polyregion_interface:spectra:spectra.includes")
  def includes[T](a: Array[T], na: Int, b: Array[T], nb: Int, op: (T, T) => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.includes")

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan")
  def inclusive_scan[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.inclusive_scan")

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan_by_key")
  def inclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.inclusive_scan_by_key")

  @compileTimeOnly("polyregion_interface:spectra:spectra.inner_product")
  def inner_product[T, U, V](
      a: Array[T],
      b: Array[U],
      n: Int,
      init: V,
      op_reduce: (V, V) => V,
      op_product: (T, U) => V
  ): V =
    throw UnsupportedOperationException("compiler did not replace spectra.inner_product")

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_partitioned")
  def is_partitioned[T](data: Array[T], n: Int, op: T => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.is_partitioned")

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted")
  def is_sorted[T](data: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.is_sorted")

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted_until")
  def is_sorted_until[T](data: Array[T], n: Int, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.is_sorted_until")

  @compileTimeOnly("polyregion_interface:spectra:spectra.lower_bound")
  def lower_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.lower_bound")

  @compileTimeOnly("polyregion_interface:spectra:spectra.max_element")
  def max_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): T =
    throw UnsupportedOperationException("compiler did not replace spectra.max_element")

  @compileTimeOnly("polyregion_interface:spectra:spectra.merge")
  def merge[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.merge")

  @compileTimeOnly("polyregion_interface:spectra:spectra.min_element")
  def min_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): T =
    throw UnsupportedOperationException("compiler did not replace spectra.min_element")

  @compileTimeOnly("polyregion_interface:spectra:spectra.minmax_element")
  def minmax_element[T](in: Array[T], n: Int, min_out: Array[T], max_out: Array[T], op: (T, T) => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.minmax_element")

  @compileTimeOnly("polyregion_interface:spectra:spectra.mismatch")
  def mismatch[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.mismatch")

  @compileTimeOnly("polyregion_interface:spectra:spectra.none_of")
  def none_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw UnsupportedOperationException("compiler did not replace spectra.none_of")

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition")
  def partition[T](data: Array[T], n: Int, op: T => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.partition")

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition_point")
  def partition_point[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.partition_point")

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce")
  def reduce[T](in: Array[T], n: Int, init: T, op: (T, T) => T): T =
    throw UnsupportedOperationException("compiler did not replace spectra.reduce")

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce_by_key")
  def reduce_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      kout: Array[K],
      vout: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.reduce_by_key")

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove")
  def remove[T](data: Array[T], n: Int, value: T): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.remove")

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy")
  def remove_copy[T](in: Array[T], n: Int, out: Array[T], value: T): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.remove_copy")

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy_if")
  def remove_copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.remove_copy_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_if")
  def remove_if[T](data: Array[T], n: Int, op: T => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.remove_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace")
  def replace[T](io: Array[T], n: Int, oldv: T, newv: T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.replace")

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy")
  def replace_copy[T](in: Array[T], out: Array[T], n: Int, oldv: T, newv: T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.replace_copy")

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy_if")
  def replace_copy_if[T](in: Array[T], out: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.replace_copy_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_if")
  def replace_if[T](data: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.replace_if")

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse")
  def reverse[T](data: Array[T], n: Int): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.reverse")

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse_copy")
  def reverse_copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.reverse_copy")

  @compileTimeOnly("polyregion_interface:spectra:spectra.scatter")
  def scatter[T](in: Array[T], n: Int, map: Array[Int], out: Array[T], out_n: Int): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.scatter")

  @compileTimeOnly("polyregion_interface:spectra:spectra.search")
  def search[T](in: Array[T], n: Int, sub: Array[T], m: Int, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.search")

  @compileTimeOnly("polyregion_interface:spectra:spectra.search_n")
  def search_n[T](in: Array[T], n: Int, count: Int, value: T, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.search_n")

  @compileTimeOnly("polyregion_interface:spectra:spectra.sequence")
  def sequence[T](out: Array[T], n: Int, init: T, step: T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.sequence")

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_difference")
  def set_difference[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.set_difference")

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_intersection")
  def set_intersection[T](
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      out_n: Int,
      op: (T, T) => Boolean
  ): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.set_intersection")

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_union")
  def set_union[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.set_union")

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort")
  def sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.sort")

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort_by_key")
  def sort_by_key[K, V](keys: Array[K], values: Array[V], n: Int, op: (K, K) => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.sort_by_key")

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_partition")
  def stable_partition[T](data: Array[T], n: Int, op: T => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.stable_partition")

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_sort")
  def stable_sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.stable_sort")

  @compileTimeOnly("polyregion_interface:spectra:spectra.swap_ranges")
  def swap_ranges[T](a: Array[T], n: Int, b: Array[T]): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.swap_ranges")

  @compileTimeOnly("polyregion_interface:spectra:spectra.tabulate")
  def tabulate[T](out: Array[T], n: Int, op: Int => T): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.tabulate")

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform")
  def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.transform")

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_binary")
  def transform_binary[T, U, V](a: Array[T], b: Array[U], out: Array[V], n: Int, op: (T, U) => V): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.transform_binary")

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_exclusive_scan")
  def transform_exclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, init: U, map: T => U, op: (U, U) => U): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.transform_exclusive_scan")

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_inclusive_scan")
  def transform_inclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, map: T => U, op: (U, U) => U): Unit =
    throw UnsupportedOperationException("compiler did not replace spectra.transform_inclusive_scan")

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_reduce")
  def transform_reduce[T, U](in: Array[T], n: Int, init: U, map: T => U, op: (U, U) => U): U =
    throw UnsupportedOperationException("compiler did not replace spectra.transform_reduce")

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique")
  def unique[T](data: Array[T], n: Int, eq: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.unique")

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique_copy")
  def unique_copy[T](in: Array[T], n: Int, out: Array[T], eq: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.unique_copy")

  @compileTimeOnly("polyregion_interface:spectra:spectra.upper_bound")
  def upper_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    throw UnsupportedOperationException("compiler did not replace spectra.upper_bound")
}
