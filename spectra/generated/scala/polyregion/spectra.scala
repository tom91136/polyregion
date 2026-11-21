package polyregion

import scala.annotation.compileTimeOnly

object spectra {

  @compileTimeOnly("polyregion_interface:spectra:spectra.adjacent_difference")
  inline def adjacent_difference[T](in: Array[T], out: Array[T], n: Int, inline op: (T, T) => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.adjacent_difference",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.adjacent_difference")
  inline def adjacent_difference[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[T],
      n: Int,
      inline op: (T, T) => T
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.adjacent_difference", context, in, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.all_of")
  inline def all_of[T](in: Array[T], n: Int, inline op: T => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.all_of",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.all_of")
  inline def all_of[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.all_of", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.any_of")
  inline def any_of[T](in: Array[T], n: Int, inline op: T => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.any_of",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.any_of")
  inline def any_of[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.any_of", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.binary_search")
  inline def binary_search[T](in: Array[T], n: Int, value: T, inline op: (T, T) => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.binary_search",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.binary_search")
  inline def binary_search[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      value: T,
      inline op: (T, T) => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.binary_search", context, in, n, value, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy")
  inline def copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.copy", polyregion.jvm.runtime.Context.current(), in, n, out)

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy")
  inline def copy[T](context: polyregion.jvm.runtime.Context, in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.copy", context, in, n, out)

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_if")
  inline def copy_if[T](in: Array[T], n: Int, out: Array[T], inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.copy_if",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_if")
  inline def copy_if[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      out: Array[T],
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.copy_if", context, in, n, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_n")
  inline def copy_n[T](in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.copy_n",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.copy_n")
  inline def copy_n[T](context: polyregion.jvm.runtime.Context, in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.copy_n", context, in, n, out)

  @compileTimeOnly("polyregion_interface:spectra:spectra.count")
  inline def count[T](in: Array[T], n: Int, value: T, inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.count",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      value,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.count")
  inline def count[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      value: T,
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.count", context, in, n, value, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.count_if")
  inline def count_if[T](in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.count_if",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.count_if")
  inline def count_if[T](context: polyregion.jvm.runtime.Context, in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.count_if", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.equal")
  inline def equal[T](a: Array[T], b: Array[T], n: Int, inline op: (T, T) => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.equal",
      polyregion.jvm.runtime.Context.current(),
      a,
      b,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.equal")
  inline def equal[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      b: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.equal", context, a, b, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan")
  inline def exclusive_scan[T](in: Array[T], out: Array[T], n: Int, init: T, inline op: (T, T) => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.exclusive_scan",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      init,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan")
  inline def exclusive_scan[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[T],
      n: Int,
      init: T,
      inline op: (T, T) => T
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.exclusive_scan", context, in, out, n, init, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan_by_key")
  inline def exclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      init: V,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.exclusive_scan_by_key",
      polyregion.jvm.runtime.Context.current(),
      keys,
      vals,
      out,
      n,
      init,
      eq,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.exclusive_scan_by_key")
  inline def exclusive_scan_by_key[K, V](
      context: polyregion.jvm.runtime.Context,
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      init: V,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.exclusive_scan_by_key",
      context,
      keys,
      vals,
      out,
      n,
      init,
      eq,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill")
  inline def fill[T](out: Array[T], n: Int, v: T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.fill", polyregion.jvm.runtime.Context.current(), out, n, v)

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill")
  inline def fill[T](context: polyregion.jvm.runtime.Context, out: Array[T], n: Int, v: T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.fill", context, out, n, v)

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill_n")
  inline def fill_n[T](out: Array[T], n: Int, v: T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.fill_n",
      polyregion.jvm.runtime.Context.current(),
      out,
      n,
      v
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.fill_n")
  inline def fill_n[T](context: polyregion.jvm.runtime.Context, out: Array[T], n: Int, v: T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.fill_n", context, out, n, v)

  @compileTimeOnly("polyregion_interface:spectra:spectra.find")
  inline def find[T](in: Array[T], n: Int, value: T, inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.find",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      value,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.find")
  inline def find[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      value: T,
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.find", context, in, n, value, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if")
  inline def find_if[T](in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.find_if",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if")
  inline def find_if[T](context: polyregion.jvm.runtime.Context, in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.find_if", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if_not")
  inline def find_if_not[T](in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.find_if_not",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.find_if_not")
  inline def find_if_not[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.find_if_not", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each")
  inline def for_each[T](data: Array[T], n: Int, inline op: T => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.for_each",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each")
  inline def for_each[T](context: polyregion.jvm.runtime.Context, data: Array[T], n: Int, inline op: T => T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.for_each", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each_n")
  inline def for_each_n[T](data: Array[T], n: Int, inline op: T => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.for_each_n",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.for_each_n")
  inline def for_each_n[T](context: polyregion.jvm.runtime.Context, data: Array[T], n: Int, inline op: T => T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.for_each_n", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.gather")
  inline def gather[T](map: Array[Int], n: Int, in: Array[T], in_n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.gather",
      polyregion.jvm.runtime.Context.current(),
      map,
      n,
      in,
      in_n,
      out
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.gather")
  inline def gather[T](
      context: polyregion.jvm.runtime.Context,
      map: Array[Int],
      n: Int,
      in: Array[T],
      in_n: Int,
      out: Array[T]
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.gather", context, map, n, in, in_n, out)

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate")
  inline def generate[T](out: Array[T], n: Int, inline op: () => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.generate",
      polyregion.jvm.runtime.Context.current(),
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate")
  inline def generate[T](context: polyregion.jvm.runtime.Context, out: Array[T], n: Int, inline op: () => T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.generate", context, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate_n")
  inline def generate_n[T](out: Array[T], n: Int, inline op: () => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.generate_n",
      polyregion.jvm.runtime.Context.current(),
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.generate_n")
  inline def generate_n[T](context: polyregion.jvm.runtime.Context, out: Array[T], n: Int, inline op: () => T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.generate_n", context, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.includes")
  inline def includes[T](a: Array[T], na: Int, b: Array[T], nb: Int, inline op: (T, T) => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.includes",
      polyregion.jvm.runtime.Context.current(),
      a,
      na,
      b,
      nb,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.includes")
  inline def includes[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      inline op: (T, T) => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.includes", context, a, na, b, nb, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan")
  inline def inclusive_scan[T](in: Array[T], out: Array[T], n: Int, inline op: (T, T) => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.inclusive_scan",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan")
  inline def inclusive_scan[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[T],
      n: Int,
      inline op: (T, T) => T
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.inclusive_scan", context, in, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan_by_key")
  inline def inclusive_scan_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.inclusive_scan_by_key",
      polyregion.jvm.runtime.Context.current(),
      keys,
      vals,
      out,
      n,
      eq,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.inclusive_scan_by_key")
  inline def inclusive_scan_by_key[K, V](
      context: polyregion.jvm.runtime.Context,
      keys: Array[K],
      vals: Array[V],
      out: Array[V],
      n: Int,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.inclusive_scan_by_key", context, keys, vals, out, n, eq, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.inner_product")
  inline def inner_product[T, U, V](
      a: Array[T],
      b: Array[U],
      n: Int,
      init: V,
      inline op_reduce: (V, V) => V,
      inline op_product: (T, U) => V
  ): V =
    polyregion.scalalang.launch.call[V](
      "spectra:spectra.inner_product",
      polyregion.jvm.runtime.Context.current(),
      a,
      b,
      n,
      init,
      op_reduce,
      op_product
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.inner_product")
  inline def inner_product[T, U, V](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      b: Array[U],
      n: Int,
      init: V,
      inline op_reduce: (V, V) => V,
      inline op_product: (T, U) => V
  ): V =
    polyregion.scalalang.launch.call[V]("spectra:spectra.inner_product", context, a, b, n, init, op_reduce, op_product)

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_partitioned")
  inline def is_partitioned[T](data: Array[T], n: Int, inline op: T => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.is_partitioned",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_partitioned")
  inline def is_partitioned[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.is_partitioned", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted")
  inline def is_sorted[T](data: Array[T], n: Int, inline op: (T, T) => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.is_sorted",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted")
  inline def is_sorted[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.is_sorted", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted_until")
  inline def is_sorted_until[T](data: Array[T], n: Int, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.is_sorted_until",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.is_sorted_until")
  inline def is_sorted_until[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.is_sorted_until", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.lower_bound")
  inline def lower_bound[T](in: Array[T], n: Int, value: T, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.lower_bound",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.lower_bound")
  inline def lower_bound[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      value: T,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.lower_bound", context, in, n, value, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.max_element")
  inline def max_element[T](in: Array[T], n: Int, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.max_element",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.max_element")
  inline def max_element[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.max_element", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.merge")
  inline def merge[T](a: Array[T], na: Int, b: Array[T], nb: Int, out: Array[T], inline op: (T, T) => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.merge",
      polyregion.jvm.runtime.Context.current(),
      a,
      na,
      b,
      nb,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.merge")
  inline def merge[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.merge", context, a, na, b, nb, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.min_element")
  inline def min_element[T](in: Array[T], n: Int, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.min_element",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.min_element")
  inline def min_element[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.min_element", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.minmax_element")
  inline def minmax_element[T](
      in: Array[T],
      n: Int,
      min_out: Array[Int],
      max_out: Array[Int],
      inline op: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.minmax_element",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      min_out,
      max_out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.minmax_element")
  inline def minmax_element[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      min_out: Array[Int],
      max_out: Array[Int],
      inline op: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.minmax_element", context, in, n, min_out, max_out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.mismatch")
  inline def mismatch[T](a: Array[T], b: Array[T], n: Int, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.mismatch",
      polyregion.jvm.runtime.Context.current(),
      a,
      b,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.mismatch")
  inline def mismatch[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      b: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.mismatch", context, a, b, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.none_of")
  inline def none_of[T](in: Array[T], n: Int, inline op: T => Boolean): Boolean =
    polyregion.scalalang.launch.call[Boolean](
      "spectra:spectra.none_of",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.none_of")
  inline def none_of[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Boolean =
    polyregion.scalalang.launch.call[Boolean]("spectra:spectra.none_of", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition")
  inline def partition[T](data: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.partition",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition")
  inline def partition[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.partition", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition_point")
  inline def partition_point[T](in: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.partition_point",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.partition_point")
  inline def partition_point[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.partition_point", context, in, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce")
  inline def reduce[T](in: Array[T], n: Int, init: T, inline op: (T, T) => T): T =
    polyregion.scalalang.launch.call[T](
      "spectra:spectra.reduce",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      init,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce")
  inline def reduce[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      init: T,
      inline op: (T, T) => T
  ): T =
    polyregion.scalalang.launch.call[T]("spectra:spectra.reduce", context, in, n, init, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce_by_key")
  inline def reduce_by_key[K, V](
      keys: Array[K],
      vals: Array[V],
      kout: Array[K],
      vout: Array[V],
      n: Int,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.reduce_by_key",
      polyregion.jvm.runtime.Context.current(),
      keys,
      vals,
      kout,
      vout,
      n,
      eq,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.reduce_by_key")
  inline def reduce_by_key[K, V](
      context: polyregion.jvm.runtime.Context,
      keys: Array[K],
      vals: Array[V],
      kout: Array[K],
      vout: Array[V],
      n: Int,
      inline eq: (K, K) => Boolean,
      inline op: (V, V) => V
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.reduce_by_key", context, keys, vals, kout, vout, n, eq, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove")
  inline def remove[T](data: Array[T], n: Int, value: T, inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.remove",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      value,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove")
  inline def remove[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      value: T,
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.remove", context, data, n, value, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy")
  inline def remove_copy[T](in: Array[T], n: Int, out: Array[T], value: T, inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.remove_copy",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out,
      value,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy")
  inline def remove_copy[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      out: Array[T],
      value: T,
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.remove_copy", context, in, n, out, value, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy_if")
  inline def remove_copy_if[T](in: Array[T], n: Int, out: Array[T], inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.remove_copy_if",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_copy_if")
  inline def remove_copy_if[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      out: Array[T],
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.remove_copy_if", context, in, n, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_if")
  inline def remove_if[T](data: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.remove_if",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.remove_if")
  inline def remove_if[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.remove_if", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace")
  inline def replace[T](io: Array[T], n: Int, oldv: T, newv: T, inline eq: (T, T) => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.replace",
      polyregion.jvm.runtime.Context.current(),
      io,
      n,
      oldv,
      newv,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace")
  inline def replace[T](
      context: polyregion.jvm.runtime.Context,
      io: Array[T],
      n: Int,
      oldv: T,
      newv: T,
      inline eq: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.replace", context, io, n, oldv, newv, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy")
  inline def replace_copy[T](
      in: Array[T],
      out: Array[T],
      n: Int,
      oldv: T,
      newv: T,
      inline eq: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.replace_copy",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      oldv,
      newv,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy")
  inline def replace_copy[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[T],
      n: Int,
      oldv: T,
      newv: T,
      inline eq: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.replace_copy", context, in, out, n, oldv, newv, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy_if")
  inline def replace_copy_if[T](in: Array[T], out: Array[T], n: Int, new_value: T, inline op: T => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.replace_copy_if",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      new_value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_copy_if")
  inline def replace_copy_if[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[T],
      n: Int,
      new_value: T,
      inline op: T => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.replace_copy_if", context, in, out, n, new_value, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_if")
  inline def replace_if[T](data: Array[T], n: Int, new_value: T, inline op: T => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.replace_if",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      new_value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.replace_if")
  inline def replace_if[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      new_value: T,
      inline op: T => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.replace_if", context, data, n, new_value, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse")
  inline def reverse[T](data: Array[T], n: Int): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.reverse", polyregion.jvm.runtime.Context.current(), data, n)

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse")
  inline def reverse[T](context: polyregion.jvm.runtime.Context, data: Array[T], n: Int): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.reverse", context, data, n)

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse_copy")
  inline def reverse_copy[T](in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.reverse_copy",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.reverse_copy")
  inline def reverse_copy[T](context: polyregion.jvm.runtime.Context, in: Array[T], n: Int, out: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.reverse_copy", context, in, n, out)

  @compileTimeOnly("polyregion_interface:spectra:spectra.scatter")
  inline def scatter[T](in: Array[T], n: Int, map: Array[Int], out: Array[T], out_n: Int): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.scatter",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      map,
      out,
      out_n
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.scatter")
  inline def scatter[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      map: Array[Int],
      out: Array[T],
      out_n: Int
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.scatter", context, in, n, map, out, out_n)

  @compileTimeOnly("polyregion_interface:spectra:spectra.search")
  inline def search[T](in: Array[T], n: Int, sub: Array[T], m: Int, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.search",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      sub,
      m,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.search")
  inline def search[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      sub: Array[T],
      m: Int,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.search", context, in, n, sub, m, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.search_n")
  inline def search_n[T](in: Array[T], n: Int, count: Int, value: T, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.search_n",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      count,
      value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.search_n")
  inline def search_n[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      count: Int,
      value: T,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.search_n", context, in, n, count, value, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.sequence")
  inline def sequence[T](out: Array[T], n: Int, init: T, step: T, inline value_at: (T, T, Int) => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.sequence",
      polyregion.jvm.runtime.Context.current(),
      out,
      n,
      init,
      step,
      value_at
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.sequence")
  inline def sequence[T](
      context: polyregion.jvm.runtime.Context,
      out: Array[T],
      n: Int,
      init: T,
      step: T,
      inline value_at: (T, T, Int) => T
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.sequence", context, out, n, init, step, value_at)

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_difference")
  inline def set_difference[T](
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.set_difference",
      polyregion.jvm.runtime.Context.current(),
      a,
      na,
      b,
      nb,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_difference")
  inline def set_difference[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.set_difference", context, a, na, b, nb, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_intersection")
  inline def set_intersection[T](
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.set_intersection",
      polyregion.jvm.runtime.Context.current(),
      a,
      na,
      b,
      nb,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_intersection")
  inline def set_intersection[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.set_intersection", context, a, na, b, nb, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_union")
  inline def set_union[T](
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.set_union",
      polyregion.jvm.runtime.Context.current(),
      a,
      na,
      b,
      nb,
      out,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.set_union")
  inline def set_union[T](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      na: Int,
      b: Array[T],
      nb: Int,
      out: Array[T],
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.set_union", context, a, na, b, nb, out, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort")
  inline def sort[T](data: Array[T], n: Int, inline op: (T, T) => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.sort",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort")
  inline def sort[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.sort", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort_by_key")
  inline def sort_by_key[K, V](keys: Array[K], values: Array[V], n: Int, inline op: (K, K) => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.sort_by_key",
      polyregion.jvm.runtime.Context.current(),
      keys,
      values,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.sort_by_key")
  inline def sort_by_key[K, V](
      context: polyregion.jvm.runtime.Context,
      keys: Array[K],
      values: Array[V],
      n: Int,
      inline op: (K, K) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.sort_by_key", context, keys, values, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_partition")
  inline def stable_partition[T](data: Array[T], n: Int, inline op: T => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.stable_partition",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_partition")
  inline def stable_partition[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: T => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.stable_partition", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_sort")
  inline def stable_sort[T](data: Array[T], n: Int, inline op: (T, T) => Boolean): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.stable_sort",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.stable_sort")
  inline def stable_sort[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline op: (T, T) => Boolean
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.stable_sort", context, data, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.swap_ranges")
  inline def swap_ranges[T](a: Array[T], n: Int, b: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.swap_ranges",
      polyregion.jvm.runtime.Context.current(),
      a,
      n,
      b
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.swap_ranges")
  inline def swap_ranges[T](context: polyregion.jvm.runtime.Context, a: Array[T], n: Int, b: Array[T]): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.swap_ranges", context, a, n, b)

  @compileTimeOnly("polyregion_interface:spectra:spectra.tabulate")
  inline def tabulate[T](out: Array[T], n: Int, inline op: Int => T): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.tabulate",
      polyregion.jvm.runtime.Context.current(),
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.tabulate")
  inline def tabulate[T](context: polyregion.jvm.runtime.Context, out: Array[T], n: Int, inline op: Int => T): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.tabulate", context, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform")
  inline def transform[T, U](in: Array[T], out: Array[U], n: Int, inline op: T => U): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.transform",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform")
  inline def transform[T, U](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[U],
      n: Int,
      inline op: T => U
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.transform", context, in, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_binary")
  inline def transform_binary[T, U, V](a: Array[T], b: Array[U], out: Array[V], n: Int, inline op: (T, U) => V): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.transform_binary",
      polyregion.jvm.runtime.Context.current(),
      a,
      b,
      out,
      n,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_binary")
  inline def transform_binary[T, U, V](
      context: polyregion.jvm.runtime.Context,
      a: Array[T],
      b: Array[U],
      out: Array[V],
      n: Int,
      inline op: (T, U) => V
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.transform_binary", context, a, b, out, n, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_exclusive_scan")
  inline def transform_exclusive_scan[T, U](
      in: Array[T],
      out: Array[U],
      n: Int,
      init: U,
      inline map: T => U,
      inline op: (U, U) => U
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.transform_exclusive_scan",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      init,
      map,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_exclusive_scan")
  inline def transform_exclusive_scan[T, U](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[U],
      n: Int,
      init: U,
      inline map: T => U,
      inline op: (U, U) => U
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.transform_exclusive_scan",
      context,
      in,
      out,
      n,
      init,
      map,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_inclusive_scan")
  inline def transform_inclusive_scan[T, U](
      in: Array[T],
      out: Array[U],
      n: Int,
      inline map: T => U,
      inline op: (U, U) => U
  ): Unit =
    polyregion.scalalang.launch.call[Unit](
      "spectra:spectra.transform_inclusive_scan",
      polyregion.jvm.runtime.Context.current(),
      in,
      out,
      n,
      map,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_inclusive_scan")
  inline def transform_inclusive_scan[T, U](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      out: Array[U],
      n: Int,
      inline map: T => U,
      inline op: (U, U) => U
  ): Unit =
    polyregion.scalalang.launch.call[Unit]("spectra:spectra.transform_inclusive_scan", context, in, out, n, map, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_reduce")
  inline def transform_reduce[T, U](in: Array[T], n: Int, init: U, inline map: T => U, inline op: (U, U) => U): U =
    polyregion.scalalang.launch.call[U](
      "spectra:spectra.transform_reduce",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      init,
      map,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.transform_reduce")
  inline def transform_reduce[T, U](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      init: U,
      inline map: T => U,
      inline op: (U, U) => U
  ): U =
    polyregion.scalalang.launch.call[U]("spectra:spectra.transform_reduce", context, in, n, init, map, op)

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique")
  inline def unique[T](data: Array[T], n: Int, inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.unique",
      polyregion.jvm.runtime.Context.current(),
      data,
      n,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique")
  inline def unique[T](
      context: polyregion.jvm.runtime.Context,
      data: Array[T],
      n: Int,
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.unique", context, data, n, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique_copy")
  inline def unique_copy[T](in: Array[T], n: Int, out: Array[T], inline eq: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.unique_copy",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      out,
      eq
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.unique_copy")
  inline def unique_copy[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      out: Array[T],
      inline eq: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.unique_copy", context, in, n, out, eq)

  @compileTimeOnly("polyregion_interface:spectra:spectra.upper_bound")
  inline def upper_bound[T](in: Array[T], n: Int, value: T, inline op: (T, T) => Boolean): Int =
    polyregion.scalalang.launch.call[Int](
      "spectra:spectra.upper_bound",
      polyregion.jvm.runtime.Context.current(),
      in,
      n,
      value,
      op
    )

  @compileTimeOnly("polyregion_interface:spectra:spectra.upper_bound")
  inline def upper_bound[T](
      context: polyregion.jvm.runtime.Context,
      in: Array[T],
      n: Int,
      value: T,
      inline op: (T, T) => Boolean
  ): Int =
    polyregion.scalalang.launch.call[Int]("spectra:spectra.upper_bound", context, in, n, value, op)
}
