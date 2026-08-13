package polyregion.spectra

import scala.annotation.StaticAnnotation

object SpectraApi {
  final class PolyregionImport(val library: String, val declaration: String) extends StaticAnnotation

  final class PolyregionImportFailure(message: String) extends RuntimeException(message)
}

trait SpectraApi {

  @SpectraApi.PolyregionImport("spectra", "spectra.adjacent_difference")
  def adjacent_difference[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.adjacent_difference")

  @SpectraApi.PolyregionImport("spectra", "spectra.all_of")
  def all_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.all_of")

  @SpectraApi.PolyregionImport("spectra", "spectra.any_of")
  def any_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.any_of")

  @SpectraApi.PolyregionImport("spectra", "spectra.binary_search")
  def binary_search[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.binary_search")

  @SpectraApi.PolyregionImport("spectra", "spectra.copy")
  def copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.copy")

  @SpectraApi.PolyregionImport("spectra", "spectra.copy_if")
  def copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.copy_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.copy_n")
  def copy_n[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.copy_n")

  @SpectraApi.PolyregionImport("spectra", "spectra.count")
  def count[T](in: Array[T], n: Int, value: T): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.count")

  @SpectraApi.PolyregionImport("spectra", "spectra.count_if")
  def count_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.count_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.equal")
  def equal[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.equal")

  @SpectraApi.PolyregionImport("spectra", "spectra.exclusive_scan")
  def exclusive_scan[T](in: Array[T], out: Array[T], n: Int, init: T, op: (T, T) => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.exclusive_scan")

  @SpectraApi.PolyregionImport("spectra", "spectra.exclusive_scan_by_key")
  def exclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      init: V,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.exclusive_scan_by_key")

  @SpectraApi.PolyregionImport("spectra", "spectra.fill")
  def fill[T](out: Array[T], n: Int, v: T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.fill")

  @SpectraApi.PolyregionImport("spectra", "spectra.fill_n")
  def fill_n[T](out: Array[T], n: Int, v: T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.fill_n")

  @SpectraApi.PolyregionImport("spectra", "spectra.find")
  def find[T](in: Array[T], n: Int, value: T): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.find")

  @SpectraApi.PolyregionImport("spectra", "spectra.find_if")
  def find_if[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.find_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.find_if_not")
  def find_if_not[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.find_if_not")

  @SpectraApi.PolyregionImport("spectra", "spectra.for_each")
  def for_each[T](data: Array[T], n: Int, op: T => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.for_each")

  @SpectraApi.PolyregionImport("spectra", "spectra.for_each_n")
  def for_each_n[T](data: Array[T], n: Int, op: T => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.for_each_n")

  @SpectraApi.PolyregionImport("spectra", "spectra.gather")
  def gather[T](map: Array[Int], n: Int, in: Array[T], in_n: Int, out: Array[T]): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.gather")

  @SpectraApi.PolyregionImport("spectra", "spectra.generate")
  def generate[T](out: Array[T], n: Int, op: () => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.generate")

  @SpectraApi.PolyregionImport("spectra", "spectra.generate_n")
  def generate_n[T](out: Array[T], n: Int, op: () => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.generate_n")

  @SpectraApi.PolyregionImport("spectra", "spectra.includes")
  def includes[T](a: Array[T], na: Int, b: Array[T], nb: Int, op: (T, T) => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.includes")

  @SpectraApi.PolyregionImport("spectra", "spectra.inclusive_scan")
  def inclusive_scan[T](in: Array[T], out: Array[T], n: Int, op: (T, T) => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.inclusive_scan")

  @SpectraApi.PolyregionImport("spectra", "spectra.inclusive_scan_by_key")
  def inclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.inclusive_scan_by_key")

  @SpectraApi.PolyregionImport("spectra", "spectra.inner_product")
  def inner_product[T, U, V](
      a: Array[T],
      b: Array[U],
      n: Int,
      init: V,
      op_reduce: (V, V) => V,
      op_product: (T, U) => V
  ): V =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.inner_product")

  @SpectraApi.PolyregionImport("spectra", "spectra.is_partitioned")
  def is_partitioned[T](data: Array[T], n: Int, op: T => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.is_partitioned")

  @SpectraApi.PolyregionImport("spectra", "spectra.is_sorted")
  def is_sorted[T](data: Array[T], n: Int, op: (T, T) => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.is_sorted")

  @SpectraApi.PolyregionImport("spectra", "spectra.is_sorted_until")
  def is_sorted_until[T](data: Array[T], n: Int, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.is_sorted_until")

  @SpectraApi.PolyregionImport("spectra", "spectra.lower_bound")
  def lower_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.lower_bound")

  @SpectraApi.PolyregionImport("spectra", "spectra.max_element")
  def max_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): T =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.max_element")

  @SpectraApi.PolyregionImport("spectra", "spectra.merge")
  def merge[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.merge")

  @SpectraApi.PolyregionImport("spectra", "spectra.min_element")
  def min_element[T](in: Array[T], n: Int, op: (T, T) => Boolean): T =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.min_element")

  @SpectraApi.PolyregionImport("spectra", "spectra.minmax_element")
  def minmax_element[T](in: Array[T], n: Int, min_out: Array[T], max_out: Array[T], op: (T, T) => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.minmax_element")

  @SpectraApi.PolyregionImport("spectra", "spectra.mismatch")
  def mismatch[T](a: Array[T], b: Array[T], n: Int, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.mismatch")

  @SpectraApi.PolyregionImport("spectra", "spectra.none_of")
  def none_of[T](in: Array[T], n: Int, op: T => Boolean): Boolean =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.none_of")

  @SpectraApi.PolyregionImport("spectra", "spectra.partition")
  def partition[T](data: Array[T], n: Int, op: T => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.partition")

  @SpectraApi.PolyregionImport("spectra", "spectra.partition_point")
  def partition_point[T](in: Array[T], n: Int, op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.partition_point")

  @SpectraApi.PolyregionImport("spectra", "spectra.reduce")
  def reduce[T](in: Array[T], n: Int, init: T, op: (T, T) => T): T =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.reduce")

  @SpectraApi.PolyregionImport("spectra", "spectra.reduce_by_key")
  def reduce_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      kout: Array[K],
      vout: Array[V],
      n: Int,
      eq: (K, K) => Boolean,
      op: (V, V) => V
  ): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.reduce_by_key")

  @SpectraApi.PolyregionImport("spectra", "spectra.remove")
  def remove[T](data: Array[T], n: Int, value: T): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.remove")

  @SpectraApi.PolyregionImport("spectra", "spectra.remove_copy")
  def remove_copy[T](in: Array[T], n: Int, out: Array[T], value: T): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.remove_copy")

  @SpectraApi.PolyregionImport("spectra", "spectra.remove_copy_if")
  def remove_copy_if[T](in: Array[T], n: Int, out: Array[T], op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.remove_copy_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.remove_if")
  def remove_if[T](data: Array[T], n: Int, op: T => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.remove_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.replace")
  def replace[T](io: Array[T], n: Int, oldv: T, newv: T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.replace")

  @SpectraApi.PolyregionImport("spectra", "spectra.replace_copy")
  def replace_copy[T](in: Array[T], out: Array[T], n: Int, oldv: T, newv: T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.replace_copy")

  @SpectraApi.PolyregionImport("spectra", "spectra.replace_copy_if")
  def replace_copy_if[T](in: Array[T], out: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.replace_copy_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.replace_if")
  def replace_if[T](data: Array[T], n: Int, new_value: T, op: T => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.replace_if")

  @SpectraApi.PolyregionImport("spectra", "spectra.reverse")
  def reverse[T](data: Array[T], n: Int): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.reverse")

  @SpectraApi.PolyregionImport("spectra", "spectra.reverse_copy")
  def reverse_copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.reverse_copy")

  @SpectraApi.PolyregionImport("spectra", "spectra.scatter")
  def scatter[T](in: Array[T], n: Int, map: Array[Int], out: Array[T], out_n: Int): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.scatter")

  @SpectraApi.PolyregionImport("spectra", "spectra.search")
  def search[T](in: Array[T], n: Int, sub: Array[T], m: Int, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.search")

  @SpectraApi.PolyregionImport("spectra", "spectra.search_n")
  def search_n[T](in: Array[T], n: Int, count: Int, value: T, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.search_n")

  @SpectraApi.PolyregionImport("spectra", "spectra.sequence")
  def sequence[T](out: Array[T], n: Int, init: T, step: T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.sequence")

  @SpectraApi.PolyregionImport("spectra", "spectra.set_difference")
  def set_difference[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.set_difference")

  @SpectraApi.PolyregionImport("spectra", "spectra.set_intersection")
  def set_intersection[T](
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      out_n: Int,
      op: (T, T) => Boolean
  ): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.set_intersection")

  @SpectraApi.PolyregionImport("spectra", "spectra.set_union")
  def set_union[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.set_union")

  @SpectraApi.PolyregionImport("spectra", "spectra.sort")
  def sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.sort")

  @SpectraApi.PolyregionImport("spectra", "spectra.sort_by_key")
  def sort_by_key[K, V](keys: Array[K], values: Array[V], n: Int, op: (K, K) => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.sort_by_key")

  @SpectraApi.PolyregionImport("spectra", "spectra.stable_partition")
  def stable_partition[T](data: Array[T], n: Int, op: T => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.stable_partition")

  @SpectraApi.PolyregionImport("spectra", "spectra.stable_sort")
  def stable_sort[T](data: Array[T], n: Int, op: (T, T) => Boolean): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.stable_sort")

  @SpectraApi.PolyregionImport("spectra", "spectra.swap_ranges")
  def swap_ranges[T](a: Array[T], n: Int, b: Array[T]): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.swap_ranges")

  @SpectraApi.PolyregionImport("spectra", "spectra.tabulate")
  def tabulate[T](out: Array[T], n: Int, op: Int => T): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.tabulate")

  @SpectraApi.PolyregionImport("spectra", "spectra.transform")
  def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.transform")

  @SpectraApi.PolyregionImport("spectra", "spectra.transform_binary")
  def transform_binary[T, U, V](a: Array[T], b: Array[U], out: Array[V], n: Int, op: (T, U) => V): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.transform_binary")

  @SpectraApi.PolyregionImport("spectra", "spectra.transform_exclusive_scan")
  def transform_exclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, init: U, map: T => U, op: (U, U) => U): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.transform_exclusive_scan")

  @SpectraApi.PolyregionImport("spectra", "spectra.transform_inclusive_scan")
  def transform_inclusive_scan[T, U](in: Array[T], out: Array[U], n: Int, map: T => U, op: (U, U) => U): Unit =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.transform_inclusive_scan")

  @SpectraApi.PolyregionImport("spectra", "spectra.transform_reduce")
  def transform_reduce[T, U](in: Array[T], n: Int, init: U, map: T => U, op: (U, U) => U): U =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.transform_reduce")

  @SpectraApi.PolyregionImport("spectra", "spectra.unique")
  def unique[T](data: Array[T], n: Int, eq: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.unique")

  @SpectraApi.PolyregionImport("spectra", "spectra.unique_copy")
  def unique_copy[T](in: Array[T], n: Int, out: Array[T], eq: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.unique_copy")

  @SpectraApi.PolyregionImport("spectra", "spectra.upper_bound")
  def upper_bound[T](in: Array[T], n: Int, value: T, op: (T, T) => Boolean): Int =
    throw SpectraApi.PolyregionImportFailure("compiler did not replace spectra.upper_bound")
}
