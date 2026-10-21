package polyregion

// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Compile-only instantiations of every generated Scala declaration; runtime package coverage is separate.
final case class SpectraSurfaceValue12(x: Int, y: Int, z: Int)

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
    spectra.reduce(values8, n, value8, (arg0: Long, arg1: Long) => 0L)

    // transform: T=w12, U=w4
    spectra.transform(values12, values4, n, (arg0: SpectraSurfaceValue12) => 0)

    // inclusive_scan: T=w8
    spectra.inclusive_scan(values8, values8, n, (arg0: Long, arg1: Long) => 0L)

    // exclusive_scan: T=w12
    spectra.exclusive_scan(
      values12,
      values12,
      n,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // sort: T=w4
    spectra.sort(values4, n, (arg0: Int, arg1: Int) => false)

    // count_if: T=w12
    spectra.count_if(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // min_element: T=w8
    spectra.min_element(values8, n, (arg0: Long, arg1: Long) => false)

    // max_element: T=w12
    spectra.max_element(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // find_if: T=w8
    spectra.find_if(values8, n, (arg0: Long) => false)

    // is_sorted: T=w8
    spectra.is_sorted(values8, n, (arg0: Long, arg1: Long) => false)

    // merge: T=w4
    spectra.merge(values4, n, values4, n, values4, (arg0: Int, arg1: Int) => false)

    // set_difference: T=w4
    spectra.set_difference(values4, n, values4, n, values4, (arg0: Int, arg1: Int) => false)

    // set_intersection: T=w12
    spectra.set_intersection(
      values12,
      n,
      values12,
      n,
      values12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // set_union: T=w12
    spectra.set_union(
      values12,
      n,
      values12,
      n,
      values12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // includes: T=w4
    spectra.includes(values4, n, values4, n, (arg0: Int, arg1: Int) => false)

    // copy_if: T=w8
    spectra.copy_if(values8, n, values8, (arg0: Long) => false)

    // remove_copy_if: T=w4
    spectra.remove_copy_if(values4, n, values4, (arg0: Int) => false)

    // unique_copy: T=w8
    spectra.unique_copy(values8, n, values8, (arg0: Long, arg1: Long) => false)

    // remove_if: T=w8
    spectra.remove_if(values8, n, (arg0: Long) => false)

    // unique: T=w12
    spectra.unique(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // sort_by_key: K=w12, V=w4
    spectra.sort_by_key(values12, values4, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // for_each: T=w12
    spectra.for_each(values12, n, (arg0: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0))

    // gather: T=w8
    spectra.gather(values4, n, values8, n, values8)

    // scatter: T=w12
    spectra.scatter(values12, n, values4, values12, n)

    // copy: T=w12
    spectra.copy(values12, n, values12)

    // reverse: T=w12
    spectra.reverse(values12, n)

    // fill: T=w4
    spectra.fill(values4, n, value4)

    // transform_reduce: T=w4, U=w8
    spectra.transform_reduce(values4, n, value8, (arg0: Int) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_inclusive_scan: T=w4, U=w8
    spectra.transform_inclusive_scan(values4, values8, n, (arg0: Int) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_exclusive_scan: T=w8, U=w12
    spectra.transform_exclusive_scan(
      values8,
      values12,
      n,
      value12,
      (arg0: Long) => SpectraSurfaceValue12(0, 0, 0),
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0)
    )

    // reduce_by_key: K=w4, V=w8
    spectra.reduce_by_key(
      values4,
      values8,
      values4,
      values8,
      n,
      (arg0: Int, arg1: Int) => false,
      (arg0: Long, arg1: Long) => 0L
    )

    // inclusive_scan_by_key: K=w12, V=w4
    spectra.inclusive_scan_by_key(
      values12,
      values4,
      values4,
      n,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false,
      (arg0: Int, arg1: Int) => 0
    )

    // exclusive_scan_by_key: K=w4, V=w8
    spectra.exclusive_scan_by_key(
      values4,
      values8,
      values8,
      n,
      value8,
      (arg0: Int, arg1: Int) => false,
      (arg0: Long, arg1: Long) => 0L
    )

    // transform_binary: T=w8, U=w12, V=w4
    spectra.transform_binary(values8, values12, values4, n, (arg0: Long, arg1: SpectraSurfaceValue12) => 0)

    // inner_product: T=w8, U=w12, V=w4
    spectra.inner_product(
      values8,
      values12,
      n,
      value4,
      (arg0: Int, arg1: Int) => 0,
      (arg0: Long, arg1: SpectraSurfaceValue12) => 0
    )

    // equal: T=w12
    spectra.equal(values12, values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // mismatch: T=w8
    spectra.mismatch(values8, values8, n, (arg0: Long, arg1: Long) => false)

    // search: T=w12
    spectra.search(values12, n, values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // search_n: T=w12
    spectra.search_n(values12, n, n, value12, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // all_of: T=w12
    spectra.all_of(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // any_of: T=w12
    spectra.any_of(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // none_of: T=w4
    spectra.none_of(values4, n, (arg0: Int) => false)

    // find_if_not: T=w4
    spectra.find_if_not(values4, n, (arg0: Int) => false)

    // generate: T=w4
    spectra.generate(values4, n, () => 0)

    // generate_n: T=w8
    spectra.generate_n(values8, n, () => 0L)

    // tabulate: T=w12
    spectra.tabulate(values12, n, (arg0: Int) => SpectraSurfaceValue12(0, 0, 0))

    // lower_bound: T=w12
    spectra.lower_bound(values12, n, value12, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // upper_bound: T=w8
    spectra.upper_bound(values8, n, value8, (arg0: Long, arg1: Long) => false)

    // binary_search: T=w12
    spectra.binary_search(values12, n, value12, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // partition: T=w12
    spectra.partition(values12, n, (arg0: SpectraSurfaceValue12) => false)

    // stable_partition: T=w8
    spectra.stable_partition(values8, n, (arg0: Long) => false)

    // partition_point: T=w4
    spectra.partition_point(values4, n, (arg0: Int) => false)

    // is_partitioned: T=w8
    spectra.is_partitioned(values8, n, (arg0: Long) => false)

    // is_sorted_until: T=w12
    spectra.is_sorted_until(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // stable_sort: T=w12
    spectra.stable_sort(values12, n, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // for_each_n: T=w12
    spectra.for_each_n(values12, n, (arg0: SpectraSurfaceValue12) => SpectraSurfaceValue12(0, 0, 0))

    // minmax_element: T=w12
    spectra.minmax_element(
      values12,
      n,
      values12,
      values12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // copy_n: T=w12
    spectra.copy_n(values12, n, values12)

    // reverse_copy: T=w8
    spectra.reverse_copy(values8, n, values8)

    // fill_n: T=w4
    spectra.fill_n(values4, n, value4)

    // swap_ranges: T=w8
    spectra.swap_ranges(values8, n, values8)

    // count: T=w8
    spectra.count(values8, n, value8, (arg0: Long, arg1: Long) => false)

    // find: T=w4
    spectra.find(values4, n, value4, (arg0: Int, arg1: Int) => false)

    // remove: T=w12
    spectra.remove(values12, n, value12, (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false)

    // remove_copy: T=w8
    spectra.remove_copy(values8, n, values8, value8, (arg0: Long, arg1: Long) => false)

    // replace: T=w4
    spectra.replace(values4, n, value4, value4, (arg0: Int, arg1: Int) => false)

    // replace_copy: T=w12
    spectra.replace_copy(
      values12,
      values12,
      n,
      value12,
      value12,
      (arg0: SpectraSurfaceValue12, arg1: SpectraSurfaceValue12) => false
    )

    // replace_if: T=w4
    spectra.replace_if(values4, n, value4, (arg0: Int) => false)

    // replace_copy_if: T=w8
    spectra.replace_copy_if(values8, values8, n, value8, (arg0: Long) => false)

    // sequence: T=w4
    spectra.sequence(values4, n, value4, value4, (arg0: Int, arg1: Int, arg2: Int) => 0)

    // adjacent_difference: T=w4
    spectra.adjacent_difference(values4, values4, n, (arg0: Int, arg1: Int) => 0)
  }
}
