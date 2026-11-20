package polyregion;

public final class check_surface {

  private static final class Value {}

  public static void compile() {
    int n = 4;
    int[] ints = new int[n];
    long[] longs = new long[n];
    float[] floats = new float[n];
    double[] doubles = new double[n];
    Value[] values = new Value[n];

    // adjacent_difference: T=int
    Spectra.adjacent_difference(ints, ints, n, (arg0, arg1) -> 0);

    // adjacent_difference: T=long
    Spectra.adjacent_difference(longs, longs, n, (arg0, arg1) -> 0L);

    // adjacent_difference: T=float
    Spectra.adjacent_difference(floats, floats, n, (arg0, arg1) -> 0.0f);

    // adjacent_difference: T=double
    Spectra.adjacent_difference(doubles, doubles, n, (arg0, arg1) -> 0.0);

    // adjacent_difference: T=Value
    Spectra.adjacent_difference(values, values, n, (arg0, arg1) -> null);

    // all_of: T=int
    Spectra.all_of(ints, n, (arg0) -> false);

    // all_of: T=long
    Spectra.all_of(longs, n, (arg0) -> false);

    // all_of: T=float
    Spectra.all_of(floats, n, (arg0) -> false);

    // all_of: T=double
    Spectra.all_of(doubles, n, (arg0) -> false);

    // all_of: T=Value
    Spectra.all_of(values, n, (arg0) -> false);

    // any_of: T=int
    Spectra.any_of(ints, n, (arg0) -> false);

    // any_of: T=long
    Spectra.any_of(longs, n, (arg0) -> false);

    // any_of: T=float
    Spectra.any_of(floats, n, (arg0) -> false);

    // any_of: T=double
    Spectra.any_of(doubles, n, (arg0) -> false);

    // any_of: T=Value
    Spectra.any_of(values, n, (arg0) -> false);

    // binary_search: T=int
    Spectra.binary_search(ints, n, 0, (arg0, arg1) -> false);

    // binary_search: T=long
    Spectra.binary_search(longs, n, 0L, (arg0, arg1) -> false);

    // binary_search: T=float
    Spectra.binary_search(floats, n, 0.0f, (arg0, arg1) -> false);

    // binary_search: T=double
    Spectra.binary_search(doubles, n, 0.0, (arg0, arg1) -> false);

    // binary_search: T=Value
    Spectra.binary_search(values, n, new Value(), (arg0, arg1) -> false);

    // copy: T=int
    Spectra.copy(ints, n, ints);

    // copy: T=long
    Spectra.copy(longs, n, longs);

    // copy: T=float
    Spectra.copy(floats, n, floats);

    // copy: T=double
    Spectra.copy(doubles, n, doubles);

    // copy: T=Value
    Spectra.copy(values, n, values);

    // copy_if: T=int
    Spectra.copy_if(ints, n, ints, (arg0) -> false);

    // copy_if: T=long
    Spectra.copy_if(longs, n, longs, (arg0) -> false);

    // copy_if: T=float
    Spectra.copy_if(floats, n, floats, (arg0) -> false);

    // copy_if: T=double
    Spectra.copy_if(doubles, n, doubles, (arg0) -> false);

    // copy_if: T=Value
    Spectra.copy_if(values, n, values, (arg0) -> false);

    // copy_n: T=int
    Spectra.copy_n(ints, n, ints);

    // copy_n: T=long
    Spectra.copy_n(longs, n, longs);

    // copy_n: T=float
    Spectra.copy_n(floats, n, floats);

    // copy_n: T=double
    Spectra.copy_n(doubles, n, doubles);

    // copy_n: T=Value
    Spectra.copy_n(values, n, values);

    // count: T=int
    Spectra.count(ints, n, 0, (arg0, arg1) -> false);

    // count: T=long
    Spectra.count(longs, n, 0L, (arg0, arg1) -> false);

    // count: T=float
    Spectra.count(floats, n, 0.0f, (arg0, arg1) -> false);

    // count: T=double
    Spectra.count(doubles, n, 0.0, (arg0, arg1) -> false);

    // count: T=Value
    Spectra.count(values, n, new Value(), (arg0, arg1) -> false);

    // count_if: T=int
    Spectra.count_if(ints, n, (arg0) -> false);

    // count_if: T=long
    Spectra.count_if(longs, n, (arg0) -> false);

    // count_if: T=float
    Spectra.count_if(floats, n, (arg0) -> false);

    // count_if: T=double
    Spectra.count_if(doubles, n, (arg0) -> false);

    // count_if: T=Value
    Spectra.count_if(values, n, (arg0) -> false);

    // equal: T=int
    Spectra.equal(ints, ints, n, (arg0, arg1) -> false);

    // equal: T=long
    Spectra.equal(longs, longs, n, (arg0, arg1) -> false);

    // equal: T=float
    Spectra.equal(floats, floats, n, (arg0, arg1) -> false);

    // equal: T=double
    Spectra.equal(doubles, doubles, n, (arg0, arg1) -> false);

    // equal: T=Value
    Spectra.equal(values, values, n, (arg0, arg1) -> false);

    // exclusive_scan: T=int
    Spectra.exclusive_scan(ints, ints, n, 0, (arg0, arg1) -> 0);

    // exclusive_scan: T=long
    Spectra.exclusive_scan(longs, longs, n, 0L, (arg0, arg1) -> 0L);

    // exclusive_scan: T=float
    Spectra.exclusive_scan(floats, floats, n, 0.0f, (arg0, arg1) -> 0.0f);

    // exclusive_scan: T=double
    Spectra.exclusive_scan(doubles, doubles, n, 0.0, (arg0, arg1) -> 0.0);

    // exclusive_scan: T=Value
    Spectra.exclusive_scan(values, values, n, new Value(), (arg0, arg1) -> null);

    // exclusive_scan_by_key: K=int, V=int
    Spectra.exclusive_scan_by_key(ints, ints, ints, n, 0, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // exclusive_scan_by_key: K=int, V=long
    Spectra.exclusive_scan_by_key(
        ints,
        longs,
        longs,
        n,
        0L,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // exclusive_scan_by_key: K=int, V=float
    Spectra.exclusive_scan_by_key(
        ints,
        floats,
        floats,
        n,
        0.0f,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // exclusive_scan_by_key: K=int, V=double
    Spectra.exclusive_scan_by_key(
        ints,
        doubles,
        doubles,
        n,
        0.0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // exclusive_scan_by_key: K=long, V=int
    Spectra.exclusive_scan_by_key(
        longs,
        ints,
        ints,
        n,
        0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0);

    // exclusive_scan_by_key: K=long, V=long
    Spectra.exclusive_scan_by_key(
        longs,
        longs,
        longs,
        n,
        0L,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // exclusive_scan_by_key: K=long, V=float
    Spectra.exclusive_scan_by_key(
        longs,
        floats,
        floats,
        n,
        0.0f,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // exclusive_scan_by_key: K=long, V=double
    Spectra.exclusive_scan_by_key(
        longs,
        doubles,
        doubles,
        n,
        0.0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // exclusive_scan_by_key: K=float, V=int
    Spectra.exclusive_scan_by_key(
        floats,
        ints,
        ints,
        n,
        0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0);

    // exclusive_scan_by_key: K=float, V=long
    Spectra.exclusive_scan_by_key(
        floats,
        longs,
        longs,
        n,
        0L,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // exclusive_scan_by_key: K=float, V=float
    Spectra.exclusive_scan_by_key(
        floats,
        floats,
        floats,
        n,
        0.0f,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // exclusive_scan_by_key: K=float, V=double
    Spectra.exclusive_scan_by_key(
        floats,
        doubles,
        doubles,
        n,
        0.0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // exclusive_scan_by_key: K=double, V=int
    Spectra.exclusive_scan_by_key(
        doubles,
        ints,
        ints,
        n,
        0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0);

    // exclusive_scan_by_key: K=double, V=long
    Spectra.exclusive_scan_by_key(
        doubles,
        longs,
        longs,
        n,
        0L,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // exclusive_scan_by_key: K=double, V=float
    Spectra.exclusive_scan_by_key(
        doubles,
        floats,
        floats,
        n,
        0.0f,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // exclusive_scan_by_key: K=double, V=double
    Spectra.exclusive_scan_by_key(
        doubles,
        doubles,
        doubles,
        n,
        0.0,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // exclusive_scan_by_key: K=Value, V=Value
    Spectra.exclusive_scan_by_key(
        values,
        values,
        values,
        n,
        new Value(),
        (arg0, arg1) -> false,
        (arg0, arg1) -> null);

    // fill: T=int
    Spectra.fill(ints, n, 0);

    // fill: T=long
    Spectra.fill(longs, n, 0L);

    // fill: T=float
    Spectra.fill(floats, n, 0.0f);

    // fill: T=double
    Spectra.fill(doubles, n, 0.0);

    // fill: T=Value
    Spectra.fill(values, n, new Value());

    // fill_n: T=int
    Spectra.fill_n(ints, n, 0);

    // fill_n: T=long
    Spectra.fill_n(longs, n, 0L);

    // fill_n: T=float
    Spectra.fill_n(floats, n, 0.0f);

    // fill_n: T=double
    Spectra.fill_n(doubles, n, 0.0);

    // fill_n: T=Value
    Spectra.fill_n(values, n, new Value());

    // find: T=int
    Spectra.find(ints, n, 0, (arg0, arg1) -> false);

    // find: T=long
    Spectra.find(longs, n, 0L, (arg0, arg1) -> false);

    // find: T=float
    Spectra.find(floats, n, 0.0f, (arg0, arg1) -> false);

    // find: T=double
    Spectra.find(doubles, n, 0.0, (arg0, arg1) -> false);

    // find: T=Value
    Spectra.find(values, n, new Value(), (arg0, arg1) -> false);

    // find_if: T=int
    Spectra.find_if(ints, n, (arg0) -> false);

    // find_if: T=long
    Spectra.find_if(longs, n, (arg0) -> false);

    // find_if: T=float
    Spectra.find_if(floats, n, (arg0) -> false);

    // find_if: T=double
    Spectra.find_if(doubles, n, (arg0) -> false);

    // find_if: T=Value
    Spectra.find_if(values, n, (arg0) -> false);

    // find_if_not: T=int
    Spectra.find_if_not(ints, n, (arg0) -> false);

    // find_if_not: T=long
    Spectra.find_if_not(longs, n, (arg0) -> false);

    // find_if_not: T=float
    Spectra.find_if_not(floats, n, (arg0) -> false);

    // find_if_not: T=double
    Spectra.find_if_not(doubles, n, (arg0) -> false);

    // find_if_not: T=Value
    Spectra.find_if_not(values, n, (arg0) -> false);

    // for_each: T=int
    Spectra.for_each(ints, n, (arg0) -> 0);

    // for_each: T=long
    Spectra.for_each(longs, n, (arg0) -> 0L);

    // for_each: T=float
    Spectra.for_each(floats, n, (arg0) -> 0.0f);

    // for_each: T=double
    Spectra.for_each(doubles, n, (arg0) -> 0.0);

    // for_each: T=Value
    Spectra.for_each(values, n, (arg0) -> null);

    // for_each_n: T=int
    Spectra.for_each_n(ints, n, (arg0) -> 0);

    // for_each_n: T=long
    Spectra.for_each_n(longs, n, (arg0) -> 0L);

    // for_each_n: T=float
    Spectra.for_each_n(floats, n, (arg0) -> 0.0f);

    // for_each_n: T=double
    Spectra.for_each_n(doubles, n, (arg0) -> 0.0);

    // for_each_n: T=Value
    Spectra.for_each_n(values, n, (arg0) -> null);

    // gather: T=int
    Spectra.gather(new int[n], n, ints, n, ints);

    // gather: T=long
    Spectra.gather(new int[n], n, longs, n, longs);

    // gather: T=float
    Spectra.gather(new int[n], n, floats, n, floats);

    // gather: T=double
    Spectra.gather(new int[n], n, doubles, n, doubles);

    // gather: T=Value
    Spectra.gather(new int[n], n, values, n, values);

    // generate: T=int
    Spectra.generate(ints, n, () -> 0);

    // generate: T=long
    Spectra.generate(longs, n, () -> 0L);

    // generate: T=float
    Spectra.generate(floats, n, () -> 0.0f);

    // generate: T=double
    Spectra.generate(doubles, n, () -> 0.0);

    // generate: T=Value
    Spectra.generate(values, n, () -> null);

    // generate_n: T=int
    Spectra.generate_n(ints, n, () -> 0);

    // generate_n: T=long
    Spectra.generate_n(longs, n, () -> 0L);

    // generate_n: T=float
    Spectra.generate_n(floats, n, () -> 0.0f);

    // generate_n: T=double
    Spectra.generate_n(doubles, n, () -> 0.0);

    // generate_n: T=Value
    Spectra.generate_n(values, n, () -> null);

    // includes: T=int
    Spectra.includes(ints, n, ints, n, (arg0, arg1) -> false);

    // includes: T=long
    Spectra.includes(longs, n, longs, n, (arg0, arg1) -> false);

    // includes: T=float
    Spectra.includes(floats, n, floats, n, (arg0, arg1) -> false);

    // includes: T=double
    Spectra.includes(doubles, n, doubles, n, (arg0, arg1) -> false);

    // includes: T=Value
    Spectra.includes(values, n, values, n, (arg0, arg1) -> false);

    // inclusive_scan: T=int
    Spectra.inclusive_scan(ints, ints, n, (arg0, arg1) -> 0);

    // inclusive_scan: T=long
    Spectra.inclusive_scan(longs, longs, n, (arg0, arg1) -> 0L);

    // inclusive_scan: T=float
    Spectra.inclusive_scan(floats, floats, n, (arg0, arg1) -> 0.0f);

    // inclusive_scan: T=double
    Spectra.inclusive_scan(doubles, doubles, n, (arg0, arg1) -> 0.0);

    // inclusive_scan: T=Value
    Spectra.inclusive_scan(values, values, n, (arg0, arg1) -> null);

    // inclusive_scan_by_key: K=int, V=int
    Spectra.inclusive_scan_by_key(ints, ints, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // inclusive_scan_by_key: K=int, V=long
    Spectra.inclusive_scan_by_key(ints, longs, longs, n, (arg0, arg1) -> false, (arg0, arg1) -> 0L);

    // inclusive_scan_by_key: K=int, V=float
    Spectra.inclusive_scan_by_key(
        ints,
        floats,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // inclusive_scan_by_key: K=int, V=double
    Spectra.inclusive_scan_by_key(
        ints,
        doubles,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // inclusive_scan_by_key: K=long, V=int
    Spectra.inclusive_scan_by_key(longs, ints, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // inclusive_scan_by_key: K=long, V=long
    Spectra.inclusive_scan_by_key(
        longs,
        longs,
        longs,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // inclusive_scan_by_key: K=long, V=float
    Spectra.inclusive_scan_by_key(
        longs,
        floats,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // inclusive_scan_by_key: K=long, V=double
    Spectra.inclusive_scan_by_key(
        longs,
        doubles,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // inclusive_scan_by_key: K=float, V=int
    Spectra.inclusive_scan_by_key(floats, ints, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // inclusive_scan_by_key: K=float, V=long
    Spectra.inclusive_scan_by_key(
        floats,
        longs,
        longs,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // inclusive_scan_by_key: K=float, V=float
    Spectra.inclusive_scan_by_key(
        floats,
        floats,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // inclusive_scan_by_key: K=float, V=double
    Spectra.inclusive_scan_by_key(
        floats,
        doubles,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // inclusive_scan_by_key: K=double, V=int
    Spectra.inclusive_scan_by_key(doubles, ints, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // inclusive_scan_by_key: K=double, V=long
    Spectra.inclusive_scan_by_key(
        doubles,
        longs,
        longs,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // inclusive_scan_by_key: K=double, V=float
    Spectra.inclusive_scan_by_key(
        doubles,
        floats,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // inclusive_scan_by_key: K=double, V=double
    Spectra.inclusive_scan_by_key(
        doubles,
        doubles,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // inclusive_scan_by_key: K=Value, V=Value
    Spectra.inclusive_scan_by_key(
        values,
        values,
        values,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> null);

    // inner_product: T=int, U=int, V=int
    Spectra.inner_product(ints, ints, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=int, U=int, V=long
    Spectra.inner_product(ints, ints, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=int, U=int, V=float
    Spectra.inner_product(ints, ints, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=int, U=int, V=double
    Spectra.inner_product(ints, ints, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=int, U=long, V=int
    Spectra.inner_product(ints, longs, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=int, U=long, V=long
    Spectra.inner_product(ints, longs, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=int, U=long, V=float
    Spectra.inner_product(ints, longs, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=int, U=long, V=double
    Spectra.inner_product(ints, longs, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=int, U=float, V=int
    Spectra.inner_product(ints, floats, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=int, U=float, V=long
    Spectra.inner_product(ints, floats, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=int, U=float, V=float
    Spectra.inner_product(ints, floats, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=int, U=float, V=double
    Spectra.inner_product(ints, floats, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=int, U=double, V=int
    Spectra.inner_product(ints, doubles, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=int, U=double, V=long
    Spectra.inner_product(ints, doubles, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=int, U=double, V=float
    Spectra.inner_product(ints, doubles, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=int, U=double, V=double
    Spectra.inner_product(ints, doubles, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=long, U=int, V=int
    Spectra.inner_product(longs, ints, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=long, U=int, V=long
    Spectra.inner_product(longs, ints, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=long, U=int, V=float
    Spectra.inner_product(longs, ints, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=long, U=int, V=double
    Spectra.inner_product(longs, ints, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=long, U=long, V=int
    Spectra.inner_product(longs, longs, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=long, U=long, V=long
    Spectra.inner_product(longs, longs, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=long, U=long, V=float
    Spectra.inner_product(longs, longs, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=long, U=long, V=double
    Spectra.inner_product(longs, longs, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=long, U=float, V=int
    Spectra.inner_product(longs, floats, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=long, U=float, V=long
    Spectra.inner_product(longs, floats, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=long, U=float, V=float
    Spectra.inner_product(longs, floats, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=long, U=float, V=double
    Spectra.inner_product(longs, floats, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=long, U=double, V=int
    Spectra.inner_product(longs, doubles, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=long, U=double, V=long
    Spectra.inner_product(longs, doubles, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=long, U=double, V=float
    Spectra.inner_product(longs, doubles, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=long, U=double, V=double
    Spectra.inner_product(longs, doubles, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=float, U=int, V=int
    Spectra.inner_product(floats, ints, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=float, U=int, V=long
    Spectra.inner_product(floats, ints, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=float, U=int, V=float
    Spectra.inner_product(floats, ints, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=float, U=int, V=double
    Spectra.inner_product(floats, ints, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=float, U=long, V=int
    Spectra.inner_product(floats, longs, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=float, U=long, V=long
    Spectra.inner_product(floats, longs, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=float, U=long, V=float
    Spectra.inner_product(floats, longs, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=float, U=long, V=double
    Spectra.inner_product(floats, longs, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=float, U=float, V=int
    Spectra.inner_product(floats, floats, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=float, U=float, V=long
    Spectra.inner_product(floats, floats, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=float, U=float, V=float
    Spectra.inner_product(floats, floats, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=float, U=float, V=double
    Spectra.inner_product(floats, floats, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=float, U=double, V=int
    Spectra.inner_product(floats, doubles, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=float, U=double, V=long
    Spectra.inner_product(floats, doubles, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=float, U=double, V=float
    Spectra.inner_product(floats, doubles, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=float, U=double, V=double
    Spectra.inner_product(floats, doubles, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=double, U=int, V=int
    Spectra.inner_product(doubles, ints, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=double, U=int, V=long
    Spectra.inner_product(doubles, ints, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=double, U=int, V=float
    Spectra.inner_product(doubles, ints, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=double, U=int, V=double
    Spectra.inner_product(doubles, ints, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=double, U=long, V=int
    Spectra.inner_product(doubles, longs, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=double, U=long, V=long
    Spectra.inner_product(doubles, longs, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=double, U=long, V=float
    Spectra.inner_product(doubles, longs, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=double, U=long, V=double
    Spectra.inner_product(doubles, longs, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=double, U=float, V=int
    Spectra.inner_product(doubles, floats, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=double, U=float, V=long
    Spectra.inner_product(doubles, floats, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=double, U=float, V=float
    Spectra.inner_product(doubles, floats, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=double, U=float, V=double
    Spectra.inner_product(doubles, floats, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=double, U=double, V=int
    Spectra.inner_product(doubles, doubles, n, 0, (arg0, arg1) -> 0, (arg0, arg1) -> 0);

    // inner_product: T=double, U=double, V=long
    Spectra.inner_product(doubles, doubles, n, 0L, (arg0, arg1) -> 0L, (arg0, arg1) -> 0L);

    // inner_product: T=double, U=double, V=float
    Spectra.inner_product(doubles, doubles, n, 0.0f, (arg0, arg1) -> 0.0f, (arg0, arg1) -> 0.0f);

    // inner_product: T=double, U=double, V=double
    Spectra.inner_product(doubles, doubles, n, 0.0, (arg0, arg1) -> 0.0, (arg0, arg1) -> 0.0);

    // inner_product: T=Value, U=Value, V=Value
    Spectra.inner_product(
        values,
        values,
        n,
        new Value(),
        (arg0, arg1) -> null,
        (arg0, arg1) -> null);

    // is_partitioned: T=int
    Spectra.is_partitioned(ints, n, (arg0) -> false);

    // is_partitioned: T=long
    Spectra.is_partitioned(longs, n, (arg0) -> false);

    // is_partitioned: T=float
    Spectra.is_partitioned(floats, n, (arg0) -> false);

    // is_partitioned: T=double
    Spectra.is_partitioned(doubles, n, (arg0) -> false);

    // is_partitioned: T=Value
    Spectra.is_partitioned(values, n, (arg0) -> false);

    // is_sorted: T=int
    Spectra.is_sorted(ints, n, (arg0, arg1) -> false);

    // is_sorted: T=long
    Spectra.is_sorted(longs, n, (arg0, arg1) -> false);

    // is_sorted: T=float
    Spectra.is_sorted(floats, n, (arg0, arg1) -> false);

    // is_sorted: T=double
    Spectra.is_sorted(doubles, n, (arg0, arg1) -> false);

    // is_sorted: T=Value
    Spectra.is_sorted(values, n, (arg0, arg1) -> false);

    // is_sorted_until: T=int
    Spectra.is_sorted_until(ints, n, (arg0, arg1) -> false);

    // is_sorted_until: T=long
    Spectra.is_sorted_until(longs, n, (arg0, arg1) -> false);

    // is_sorted_until: T=float
    Spectra.is_sorted_until(floats, n, (arg0, arg1) -> false);

    // is_sorted_until: T=double
    Spectra.is_sorted_until(doubles, n, (arg0, arg1) -> false);

    // is_sorted_until: T=Value
    Spectra.is_sorted_until(values, n, (arg0, arg1) -> false);

    // lower_bound: T=int
    Spectra.lower_bound(ints, n, 0, (arg0, arg1) -> false);

    // lower_bound: T=long
    Spectra.lower_bound(longs, n, 0L, (arg0, arg1) -> false);

    // lower_bound: T=float
    Spectra.lower_bound(floats, n, 0.0f, (arg0, arg1) -> false);

    // lower_bound: T=double
    Spectra.lower_bound(doubles, n, 0.0, (arg0, arg1) -> false);

    // lower_bound: T=Value
    Spectra.lower_bound(values, n, new Value(), (arg0, arg1) -> false);

    // max_element: T=int
    Spectra.max_element(ints, n, (arg0, arg1) -> false);

    // max_element: T=long
    Spectra.max_element(longs, n, (arg0, arg1) -> false);

    // max_element: T=float
    Spectra.max_element(floats, n, (arg0, arg1) -> false);

    // max_element: T=double
    Spectra.max_element(doubles, n, (arg0, arg1) -> false);

    // max_element: T=Value
    Spectra.max_element(values, n, (arg0, arg1) -> false);

    // merge: T=int
    Spectra.merge(ints, n, ints, n, ints, (arg0, arg1) -> false);

    // merge: T=long
    Spectra.merge(longs, n, longs, n, longs, (arg0, arg1) -> false);

    // merge: T=float
    Spectra.merge(floats, n, floats, n, floats, (arg0, arg1) -> false);

    // merge: T=double
    Spectra.merge(doubles, n, doubles, n, doubles, (arg0, arg1) -> false);

    // merge: T=Value
    Spectra.merge(values, n, values, n, values, (arg0, arg1) -> false);

    // min_element: T=int
    Spectra.min_element(ints, n, (arg0, arg1) -> false);

    // min_element: T=long
    Spectra.min_element(longs, n, (arg0, arg1) -> false);

    // min_element: T=float
    Spectra.min_element(floats, n, (arg0, arg1) -> false);

    // min_element: T=double
    Spectra.min_element(doubles, n, (arg0, arg1) -> false);

    // min_element: T=Value
    Spectra.min_element(values, n, (arg0, arg1) -> false);

    // minmax_element: T=int
    Spectra.minmax_element(ints, n, new int[n], new int[n], (arg0, arg1) -> false);

    // minmax_element: T=long
    Spectra.minmax_element(longs, n, new int[n], new int[n], (arg0, arg1) -> false);

    // minmax_element: T=float
    Spectra.minmax_element(floats, n, new int[n], new int[n], (arg0, arg1) -> false);

    // minmax_element: T=double
    Spectra.minmax_element(doubles, n, new int[n], new int[n], (arg0, arg1) -> false);

    // minmax_element: T=Value
    Spectra.minmax_element(values, n, new int[n], new int[n], (arg0, arg1) -> false);

    // mismatch: T=int
    Spectra.mismatch(ints, ints, n, (arg0, arg1) -> false);

    // mismatch: T=long
    Spectra.mismatch(longs, longs, n, (arg0, arg1) -> false);

    // mismatch: T=float
    Spectra.mismatch(floats, floats, n, (arg0, arg1) -> false);

    // mismatch: T=double
    Spectra.mismatch(doubles, doubles, n, (arg0, arg1) -> false);

    // mismatch: T=Value
    Spectra.mismatch(values, values, n, (arg0, arg1) -> false);

    // none_of: T=int
    Spectra.none_of(ints, n, (arg0) -> false);

    // none_of: T=long
    Spectra.none_of(longs, n, (arg0) -> false);

    // none_of: T=float
    Spectra.none_of(floats, n, (arg0) -> false);

    // none_of: T=double
    Spectra.none_of(doubles, n, (arg0) -> false);

    // none_of: T=Value
    Spectra.none_of(values, n, (arg0) -> false);

    // partition: T=int
    Spectra.partition(ints, n, (arg0) -> false);

    // partition: T=long
    Spectra.partition(longs, n, (arg0) -> false);

    // partition: T=float
    Spectra.partition(floats, n, (arg0) -> false);

    // partition: T=double
    Spectra.partition(doubles, n, (arg0) -> false);

    // partition: T=Value
    Spectra.partition(values, n, (arg0) -> false);

    // partition_point: T=int
    Spectra.partition_point(ints, n, (arg0) -> false);

    // partition_point: T=long
    Spectra.partition_point(longs, n, (arg0) -> false);

    // partition_point: T=float
    Spectra.partition_point(floats, n, (arg0) -> false);

    // partition_point: T=double
    Spectra.partition_point(doubles, n, (arg0) -> false);

    // partition_point: T=Value
    Spectra.partition_point(values, n, (arg0) -> false);

    // reduce: T=int
    Spectra.reduce(ints, n, 0, (arg0, arg1) -> 0);

    // reduce: T=long
    Spectra.reduce(longs, n, 0L, (arg0, arg1) -> 0L);

    // reduce: T=float
    Spectra.reduce(floats, n, 0.0f, (arg0, arg1) -> 0.0f);

    // reduce: T=double
    Spectra.reduce(doubles, n, 0.0, (arg0, arg1) -> 0.0);

    // reduce: T=Value
    Spectra.reduce(values, n, new Value(), (arg0, arg1) -> null);

    // reduce_by_key: K=int, V=int
    Spectra.reduce_by_key(ints, ints, ints, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // reduce_by_key: K=int, V=long
    Spectra.reduce_by_key(ints, longs, ints, longs, n, (arg0, arg1) -> false, (arg0, arg1) -> 0L);

    // reduce_by_key: K=int, V=float
    Spectra.reduce_by_key(
        ints,
        floats,
        ints,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // reduce_by_key: K=int, V=double
    Spectra.reduce_by_key(
        ints,
        doubles,
        ints,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // reduce_by_key: K=long, V=int
    Spectra.reduce_by_key(longs, ints, longs, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // reduce_by_key: K=long, V=long
    Spectra.reduce_by_key(longs, longs, longs, longs, n, (arg0, arg1) -> false, (arg0, arg1) -> 0L);

    // reduce_by_key: K=long, V=float
    Spectra.reduce_by_key(
        longs,
        floats,
        longs,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // reduce_by_key: K=long, V=double
    Spectra.reduce_by_key(
        longs,
        doubles,
        longs,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // reduce_by_key: K=float, V=int
    Spectra.reduce_by_key(floats, ints, floats, ints, n, (arg0, arg1) -> false, (arg0, arg1) -> 0);

    // reduce_by_key: K=float, V=long
    Spectra.reduce_by_key(
        floats,
        longs,
        floats,
        longs,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // reduce_by_key: K=float, V=float
    Spectra.reduce_by_key(
        floats,
        floats,
        floats,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // reduce_by_key: K=float, V=double
    Spectra.reduce_by_key(
        floats,
        doubles,
        floats,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // reduce_by_key: K=double, V=int
    Spectra.reduce_by_key(
        doubles,
        ints,
        doubles,
        ints,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0);

    // reduce_by_key: K=double, V=long
    Spectra.reduce_by_key(
        doubles,
        longs,
        doubles,
        longs,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0L);

    // reduce_by_key: K=double, V=float
    Spectra.reduce_by_key(
        doubles,
        floats,
        doubles,
        floats,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0f);

    // reduce_by_key: K=double, V=double
    Spectra.reduce_by_key(
        doubles,
        doubles,
        doubles,
        doubles,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> 0.0);

    // reduce_by_key: K=Value, V=Value
    Spectra.reduce_by_key(
        values,
        values,
        values,
        values,
        n,
        (arg0, arg1) -> false,
        (arg0, arg1) -> null);

    // remove: T=int
    Spectra.remove(ints, n, 0, (arg0, arg1) -> false);

    // remove: T=long
    Spectra.remove(longs, n, 0L, (arg0, arg1) -> false);

    // remove: T=float
    Spectra.remove(floats, n, 0.0f, (arg0, arg1) -> false);

    // remove: T=double
    Spectra.remove(doubles, n, 0.0, (arg0, arg1) -> false);

    // remove: T=Value
    Spectra.remove(values, n, new Value(), (arg0, arg1) -> false);

    // remove_copy: T=int
    Spectra.remove_copy(ints, n, ints, 0, (arg0, arg1) -> false);

    // remove_copy: T=long
    Spectra.remove_copy(longs, n, longs, 0L, (arg0, arg1) -> false);

    // remove_copy: T=float
    Spectra.remove_copy(floats, n, floats, 0.0f, (arg0, arg1) -> false);

    // remove_copy: T=double
    Spectra.remove_copy(doubles, n, doubles, 0.0, (arg0, arg1) -> false);

    // remove_copy: T=Value
    Spectra.remove_copy(values, n, values, new Value(), (arg0, arg1) -> false);

    // remove_copy_if: T=int
    Spectra.remove_copy_if(ints, n, ints, (arg0) -> false);

    // remove_copy_if: T=long
    Spectra.remove_copy_if(longs, n, longs, (arg0) -> false);

    // remove_copy_if: T=float
    Spectra.remove_copy_if(floats, n, floats, (arg0) -> false);

    // remove_copy_if: T=double
    Spectra.remove_copy_if(doubles, n, doubles, (arg0) -> false);

    // remove_copy_if: T=Value
    Spectra.remove_copy_if(values, n, values, (arg0) -> false);

    // remove_if: T=int
    Spectra.remove_if(ints, n, (arg0) -> false);

    // remove_if: T=long
    Spectra.remove_if(longs, n, (arg0) -> false);

    // remove_if: T=float
    Spectra.remove_if(floats, n, (arg0) -> false);

    // remove_if: T=double
    Spectra.remove_if(doubles, n, (arg0) -> false);

    // remove_if: T=Value
    Spectra.remove_if(values, n, (arg0) -> false);

    // replace: T=int
    Spectra.replace(ints, n, 0, 0, (arg0, arg1) -> false);

    // replace: T=long
    Spectra.replace(longs, n, 0L, 0L, (arg0, arg1) -> false);

    // replace: T=float
    Spectra.replace(floats, n, 0.0f, 0.0f, (arg0, arg1) -> false);

    // replace: T=double
    Spectra.replace(doubles, n, 0.0, 0.0, (arg0, arg1) -> false);

    // replace: T=Value
    Spectra.replace(values, n, new Value(), new Value(), (arg0, arg1) -> false);

    // replace_copy: T=int
    Spectra.replace_copy(ints, ints, n, 0, 0, (arg0, arg1) -> false);

    // replace_copy: T=long
    Spectra.replace_copy(longs, longs, n, 0L, 0L, (arg0, arg1) -> false);

    // replace_copy: T=float
    Spectra.replace_copy(floats, floats, n, 0.0f, 0.0f, (arg0, arg1) -> false);

    // replace_copy: T=double
    Spectra.replace_copy(doubles, doubles, n, 0.0, 0.0, (arg0, arg1) -> false);

    // replace_copy: T=Value
    Spectra.replace_copy(values, values, n, new Value(), new Value(), (arg0, arg1) -> false);

    // replace_copy_if: T=int
    Spectra.replace_copy_if(ints, ints, n, 0, (arg0) -> false);

    // replace_copy_if: T=long
    Spectra.replace_copy_if(longs, longs, n, 0L, (arg0) -> false);

    // replace_copy_if: T=float
    Spectra.replace_copy_if(floats, floats, n, 0.0f, (arg0) -> false);

    // replace_copy_if: T=double
    Spectra.replace_copy_if(doubles, doubles, n, 0.0, (arg0) -> false);

    // replace_copy_if: T=Value
    Spectra.replace_copy_if(values, values, n, new Value(), (arg0) -> false);

    // replace_if: T=int
    Spectra.replace_if(ints, n, 0, (arg0) -> false);

    // replace_if: T=long
    Spectra.replace_if(longs, n, 0L, (arg0) -> false);

    // replace_if: T=float
    Spectra.replace_if(floats, n, 0.0f, (arg0) -> false);

    // replace_if: T=double
    Spectra.replace_if(doubles, n, 0.0, (arg0) -> false);

    // replace_if: T=Value
    Spectra.replace_if(values, n, new Value(), (arg0) -> false);

    // reverse: T=int
    Spectra.reverse(ints, n);

    // reverse: T=long
    Spectra.reverse(longs, n);

    // reverse: T=float
    Spectra.reverse(floats, n);

    // reverse: T=double
    Spectra.reverse(doubles, n);

    // reverse: T=Value
    Spectra.reverse(values, n);

    // reverse_copy: T=int
    Spectra.reverse_copy(ints, n, ints);

    // reverse_copy: T=long
    Spectra.reverse_copy(longs, n, longs);

    // reverse_copy: T=float
    Spectra.reverse_copy(floats, n, floats);

    // reverse_copy: T=double
    Spectra.reverse_copy(doubles, n, doubles);

    // reverse_copy: T=Value
    Spectra.reverse_copy(values, n, values);

    // scatter: T=int
    Spectra.scatter(ints, n, new int[n], ints, n);

    // scatter: T=long
    Spectra.scatter(longs, n, new int[n], longs, n);

    // scatter: T=float
    Spectra.scatter(floats, n, new int[n], floats, n);

    // scatter: T=double
    Spectra.scatter(doubles, n, new int[n], doubles, n);

    // scatter: T=Value
    Spectra.scatter(values, n, new int[n], values, n);

    // search: T=int
    Spectra.search(ints, n, ints, n, (arg0, arg1) -> false);

    // search: T=long
    Spectra.search(longs, n, longs, n, (arg0, arg1) -> false);

    // search: T=float
    Spectra.search(floats, n, floats, n, (arg0, arg1) -> false);

    // search: T=double
    Spectra.search(doubles, n, doubles, n, (arg0, arg1) -> false);

    // search: T=Value
    Spectra.search(values, n, values, n, (arg0, arg1) -> false);

    // search_n: T=int
    Spectra.search_n(ints, n, n, 0, (arg0, arg1) -> false);

    // search_n: T=long
    Spectra.search_n(longs, n, n, 0L, (arg0, arg1) -> false);

    // search_n: T=float
    Spectra.search_n(floats, n, n, 0.0f, (arg0, arg1) -> false);

    // search_n: T=double
    Spectra.search_n(doubles, n, n, 0.0, (arg0, arg1) -> false);

    // search_n: T=Value
    Spectra.search_n(values, n, n, new Value(), (arg0, arg1) -> false);

    // sequence: T=int
    Spectra.sequence(ints, n, 0, 0, (arg0, arg1, arg2) -> 0);

    // sequence: T=long
    Spectra.sequence(longs, n, 0L, 0L, (arg0, arg1, arg2) -> 0L);

    // sequence: T=float
    Spectra.sequence(floats, n, 0.0f, 0.0f, (arg0, arg1, arg2) -> 0.0f);

    // sequence: T=double
    Spectra.sequence(doubles, n, 0.0, 0.0, (arg0, arg1, arg2) -> 0.0);

    // sequence: T=Value
    Spectra.sequence(values, n, new Value(), new Value(), (arg0, arg1, arg2) -> null);

    // set_difference: T=int
    Spectra.set_difference(ints, n, ints, n, ints, (arg0, arg1) -> false);

    // set_difference: T=long
    Spectra.set_difference(longs, n, longs, n, longs, (arg0, arg1) -> false);

    // set_difference: T=float
    Spectra.set_difference(floats, n, floats, n, floats, (arg0, arg1) -> false);

    // set_difference: T=double
    Spectra.set_difference(doubles, n, doubles, n, doubles, (arg0, arg1) -> false);

    // set_difference: T=Value
    Spectra.set_difference(values, n, values, n, values, (arg0, arg1) -> false);

    // set_intersection: T=int
    Spectra.set_intersection(ints, n, ints, n, ints, (arg0, arg1) -> false);

    // set_intersection: T=long
    Spectra.set_intersection(longs, n, longs, n, longs, (arg0, arg1) -> false);

    // set_intersection: T=float
    Spectra.set_intersection(floats, n, floats, n, floats, (arg0, arg1) -> false);

    // set_intersection: T=double
    Spectra.set_intersection(doubles, n, doubles, n, doubles, (arg0, arg1) -> false);

    // set_intersection: T=Value
    Spectra.set_intersection(values, n, values, n, values, (arg0, arg1) -> false);

    // set_union: T=int
    Spectra.set_union(ints, n, ints, n, ints, (arg0, arg1) -> false);

    // set_union: T=long
    Spectra.set_union(longs, n, longs, n, longs, (arg0, arg1) -> false);

    // set_union: T=float
    Spectra.set_union(floats, n, floats, n, floats, (arg0, arg1) -> false);

    // set_union: T=double
    Spectra.set_union(doubles, n, doubles, n, doubles, (arg0, arg1) -> false);

    // set_union: T=Value
    Spectra.set_union(values, n, values, n, values, (arg0, arg1) -> false);

    // sort: T=int
    Spectra.sort(ints, n, (arg0, arg1) -> false);

    // sort: T=long
    Spectra.sort(longs, n, (arg0, arg1) -> false);

    // sort: T=float
    Spectra.sort(floats, n, (arg0, arg1) -> false);

    // sort: T=double
    Spectra.sort(doubles, n, (arg0, arg1) -> false);

    // sort: T=Value
    Spectra.sort(values, n, (arg0, arg1) -> false);

    // sort_by_key: K=int, V=int
    Spectra.sort_by_key(ints, ints, n, (arg0, arg1) -> false);

    // sort_by_key: K=int, V=long
    Spectra.sort_by_key(ints, longs, n, (arg0, arg1) -> false);

    // sort_by_key: K=int, V=float
    Spectra.sort_by_key(ints, floats, n, (arg0, arg1) -> false);

    // sort_by_key: K=int, V=double
    Spectra.sort_by_key(ints, doubles, n, (arg0, arg1) -> false);

    // sort_by_key: K=long, V=int
    Spectra.sort_by_key(longs, ints, n, (arg0, arg1) -> false);

    // sort_by_key: K=long, V=long
    Spectra.sort_by_key(longs, longs, n, (arg0, arg1) -> false);

    // sort_by_key: K=long, V=float
    Spectra.sort_by_key(longs, floats, n, (arg0, arg1) -> false);

    // sort_by_key: K=long, V=double
    Spectra.sort_by_key(longs, doubles, n, (arg0, arg1) -> false);

    // sort_by_key: K=float, V=int
    Spectra.sort_by_key(floats, ints, n, (arg0, arg1) -> false);

    // sort_by_key: K=float, V=long
    Spectra.sort_by_key(floats, longs, n, (arg0, arg1) -> false);

    // sort_by_key: K=float, V=float
    Spectra.sort_by_key(floats, floats, n, (arg0, arg1) -> false);

    // sort_by_key: K=float, V=double
    Spectra.sort_by_key(floats, doubles, n, (arg0, arg1) -> false);

    // sort_by_key: K=double, V=int
    Spectra.sort_by_key(doubles, ints, n, (arg0, arg1) -> false);

    // sort_by_key: K=double, V=long
    Spectra.sort_by_key(doubles, longs, n, (arg0, arg1) -> false);

    // sort_by_key: K=double, V=float
    Spectra.sort_by_key(doubles, floats, n, (arg0, arg1) -> false);

    // sort_by_key: K=double, V=double
    Spectra.sort_by_key(doubles, doubles, n, (arg0, arg1) -> false);

    // sort_by_key: K=Value, V=Value
    Spectra.sort_by_key(values, values, n, (arg0, arg1) -> false);

    // stable_partition: T=int
    Spectra.stable_partition(ints, n, (arg0) -> false);

    // stable_partition: T=long
    Spectra.stable_partition(longs, n, (arg0) -> false);

    // stable_partition: T=float
    Spectra.stable_partition(floats, n, (arg0) -> false);

    // stable_partition: T=double
    Spectra.stable_partition(doubles, n, (arg0) -> false);

    // stable_partition: T=Value
    Spectra.stable_partition(values, n, (arg0) -> false);

    // stable_sort: T=int
    Spectra.stable_sort(ints, n, (arg0, arg1) -> false);

    // stable_sort: T=long
    Spectra.stable_sort(longs, n, (arg0, arg1) -> false);

    // stable_sort: T=float
    Spectra.stable_sort(floats, n, (arg0, arg1) -> false);

    // stable_sort: T=double
    Spectra.stable_sort(doubles, n, (arg0, arg1) -> false);

    // stable_sort: T=Value
    Spectra.stable_sort(values, n, (arg0, arg1) -> false);

    // swap_ranges: T=int
    Spectra.swap_ranges(ints, n, ints);

    // swap_ranges: T=long
    Spectra.swap_ranges(longs, n, longs);

    // swap_ranges: T=float
    Spectra.swap_ranges(floats, n, floats);

    // swap_ranges: T=double
    Spectra.swap_ranges(doubles, n, doubles);

    // swap_ranges: T=Value
    Spectra.swap_ranges(values, n, values);

    // tabulate: T=int
    Spectra.tabulate(ints, n, (arg0) -> 0);

    // tabulate: T=long
    Spectra.tabulate(longs, n, (arg0) -> 0L);

    // tabulate: T=float
    Spectra.tabulate(floats, n, (arg0) -> 0.0f);

    // tabulate: T=double
    Spectra.tabulate(doubles, n, (arg0) -> 0.0);

    // tabulate: T=Value
    Spectra.tabulate(values, n, (arg0) -> null);

    // transform: T=int, U=int
    Spectra.transform(ints, ints, n, (arg0) -> 0);

    // transform: T=int, U=long
    Spectra.transform(ints, longs, n, (arg0) -> 0L);

    // transform: T=int, U=float
    Spectra.transform(ints, floats, n, (arg0) -> 0.0f);

    // transform: T=int, U=double
    Spectra.transform(ints, doubles, n, (arg0) -> 0.0);

    // transform: T=long, U=int
    Spectra.transform(longs, ints, n, (arg0) -> 0);

    // transform: T=long, U=long
    Spectra.transform(longs, longs, n, (arg0) -> 0L);

    // transform: T=long, U=float
    Spectra.transform(longs, floats, n, (arg0) -> 0.0f);

    // transform: T=long, U=double
    Spectra.transform(longs, doubles, n, (arg0) -> 0.0);

    // transform: T=float, U=int
    Spectra.transform(floats, ints, n, (arg0) -> 0);

    // transform: T=float, U=long
    Spectra.transform(floats, longs, n, (arg0) -> 0L);

    // transform: T=float, U=float
    Spectra.transform(floats, floats, n, (arg0) -> 0.0f);

    // transform: T=float, U=double
    Spectra.transform(floats, doubles, n, (arg0) -> 0.0);

    // transform: T=double, U=int
    Spectra.transform(doubles, ints, n, (arg0) -> 0);

    // transform: T=double, U=long
    Spectra.transform(doubles, longs, n, (arg0) -> 0L);

    // transform: T=double, U=float
    Spectra.transform(doubles, floats, n, (arg0) -> 0.0f);

    // transform: T=double, U=double
    Spectra.transform(doubles, doubles, n, (arg0) -> 0.0);

    // transform: T=Value, U=Value
    Spectra.transform(values, values, n, (arg0) -> null);

    // transform_binary: T=int, U=int, V=int
    Spectra.transform_binary(ints, ints, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=int, U=int, V=long
    Spectra.transform_binary(ints, ints, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=int, U=int, V=float
    Spectra.transform_binary(ints, ints, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=int, U=int, V=double
    Spectra.transform_binary(ints, ints, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=int, U=long, V=int
    Spectra.transform_binary(ints, longs, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=int, U=long, V=long
    Spectra.transform_binary(ints, longs, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=int, U=long, V=float
    Spectra.transform_binary(ints, longs, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=int, U=long, V=double
    Spectra.transform_binary(ints, longs, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=int, U=float, V=int
    Spectra.transform_binary(ints, floats, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=int, U=float, V=long
    Spectra.transform_binary(ints, floats, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=int, U=float, V=float
    Spectra.transform_binary(ints, floats, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=int, U=float, V=double
    Spectra.transform_binary(ints, floats, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=int, U=double, V=int
    Spectra.transform_binary(ints, doubles, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=int, U=double, V=long
    Spectra.transform_binary(ints, doubles, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=int, U=double, V=float
    Spectra.transform_binary(ints, doubles, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=int, U=double, V=double
    Spectra.transform_binary(ints, doubles, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=long, U=int, V=int
    Spectra.transform_binary(longs, ints, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=long, U=int, V=long
    Spectra.transform_binary(longs, ints, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=long, U=int, V=float
    Spectra.transform_binary(longs, ints, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=long, U=int, V=double
    Spectra.transform_binary(longs, ints, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=long, U=long, V=int
    Spectra.transform_binary(longs, longs, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=long, U=long, V=long
    Spectra.transform_binary(longs, longs, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=long, U=long, V=float
    Spectra.transform_binary(longs, longs, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=long, U=long, V=double
    Spectra.transform_binary(longs, longs, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=long, U=float, V=int
    Spectra.transform_binary(longs, floats, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=long, U=float, V=long
    Spectra.transform_binary(longs, floats, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=long, U=float, V=float
    Spectra.transform_binary(longs, floats, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=long, U=float, V=double
    Spectra.transform_binary(longs, floats, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=long, U=double, V=int
    Spectra.transform_binary(longs, doubles, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=long, U=double, V=long
    Spectra.transform_binary(longs, doubles, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=long, U=double, V=float
    Spectra.transform_binary(longs, doubles, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=long, U=double, V=double
    Spectra.transform_binary(longs, doubles, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=float, U=int, V=int
    Spectra.transform_binary(floats, ints, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=float, U=int, V=long
    Spectra.transform_binary(floats, ints, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=float, U=int, V=float
    Spectra.transform_binary(floats, ints, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=float, U=int, V=double
    Spectra.transform_binary(floats, ints, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=float, U=long, V=int
    Spectra.transform_binary(floats, longs, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=float, U=long, V=long
    Spectra.transform_binary(floats, longs, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=float, U=long, V=float
    Spectra.transform_binary(floats, longs, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=float, U=long, V=double
    Spectra.transform_binary(floats, longs, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=float, U=float, V=int
    Spectra.transform_binary(floats, floats, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=float, U=float, V=long
    Spectra.transform_binary(floats, floats, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=float, U=float, V=float
    Spectra.transform_binary(floats, floats, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=float, U=float, V=double
    Spectra.transform_binary(floats, floats, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=float, U=double, V=int
    Spectra.transform_binary(floats, doubles, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=float, U=double, V=long
    Spectra.transform_binary(floats, doubles, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=float, U=double, V=float
    Spectra.transform_binary(floats, doubles, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=float, U=double, V=double
    Spectra.transform_binary(floats, doubles, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=double, U=int, V=int
    Spectra.transform_binary(doubles, ints, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=double, U=int, V=long
    Spectra.transform_binary(doubles, ints, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=double, U=int, V=float
    Spectra.transform_binary(doubles, ints, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=double, U=int, V=double
    Spectra.transform_binary(doubles, ints, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=double, U=long, V=int
    Spectra.transform_binary(doubles, longs, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=double, U=long, V=long
    Spectra.transform_binary(doubles, longs, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=double, U=long, V=float
    Spectra.transform_binary(doubles, longs, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=double, U=long, V=double
    Spectra.transform_binary(doubles, longs, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=double, U=float, V=int
    Spectra.transform_binary(doubles, floats, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=double, U=float, V=long
    Spectra.transform_binary(doubles, floats, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=double, U=float, V=float
    Spectra.transform_binary(doubles, floats, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=double, U=float, V=double
    Spectra.transform_binary(doubles, floats, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=double, U=double, V=int
    Spectra.transform_binary(doubles, doubles, ints, n, (arg0, arg1) -> 0);

    // transform_binary: T=double, U=double, V=long
    Spectra.transform_binary(doubles, doubles, longs, n, (arg0, arg1) -> 0L);

    // transform_binary: T=double, U=double, V=float
    Spectra.transform_binary(doubles, doubles, floats, n, (arg0, arg1) -> 0.0f);

    // transform_binary: T=double, U=double, V=double
    Spectra.transform_binary(doubles, doubles, doubles, n, (arg0, arg1) -> 0.0);

    // transform_binary: T=Value, U=Value, V=Value
    Spectra.transform_binary(values, values, values, n, (arg0, arg1) -> null);

    // transform_exclusive_scan: T=int, U=int
    Spectra.transform_exclusive_scan(ints, ints, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_exclusive_scan: T=int, U=long
    Spectra.transform_exclusive_scan(ints, longs, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_exclusive_scan: T=int, U=float
    Spectra.transform_exclusive_scan(ints, floats, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_exclusive_scan: T=int, U=double
    Spectra.transform_exclusive_scan(ints, doubles, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_exclusive_scan: T=long, U=int
    Spectra.transform_exclusive_scan(longs, ints, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_exclusive_scan: T=long, U=long
    Spectra.transform_exclusive_scan(longs, longs, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_exclusive_scan: T=long, U=float
    Spectra.transform_exclusive_scan(longs, floats, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_exclusive_scan: T=long, U=double
    Spectra.transform_exclusive_scan(longs, doubles, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_exclusive_scan: T=float, U=int
    Spectra.transform_exclusive_scan(floats, ints, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_exclusive_scan: T=float, U=long
    Spectra.transform_exclusive_scan(floats, longs, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_exclusive_scan: T=float, U=float
    Spectra.transform_exclusive_scan(floats, floats, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_exclusive_scan: T=float, U=double
    Spectra.transform_exclusive_scan(floats, doubles, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_exclusive_scan: T=double, U=int
    Spectra.transform_exclusive_scan(doubles, ints, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_exclusive_scan: T=double, U=long
    Spectra.transform_exclusive_scan(doubles, longs, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_exclusive_scan: T=double, U=float
    Spectra.transform_exclusive_scan(
        doubles,
        floats,
        n,
        0.0f,
        (arg0) -> 0.0f,
        (arg0, arg1) -> 0.0f);

    // transform_exclusive_scan: T=double, U=double
    Spectra.transform_exclusive_scan(doubles, doubles, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_exclusive_scan: T=Value, U=Value
    Spectra.transform_exclusive_scan(
        values,
        values,
        n,
        new Value(),
        (arg0) -> null,
        (arg0, arg1) -> null);

    // transform_inclusive_scan: T=int, U=int
    Spectra.transform_inclusive_scan(ints, ints, n, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_inclusive_scan: T=int, U=long
    Spectra.transform_inclusive_scan(ints, longs, n, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_inclusive_scan: T=int, U=float
    Spectra.transform_inclusive_scan(ints, floats, n, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_inclusive_scan: T=int, U=double
    Spectra.transform_inclusive_scan(ints, doubles, n, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_inclusive_scan: T=long, U=int
    Spectra.transform_inclusive_scan(longs, ints, n, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_inclusive_scan: T=long, U=long
    Spectra.transform_inclusive_scan(longs, longs, n, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_inclusive_scan: T=long, U=float
    Spectra.transform_inclusive_scan(longs, floats, n, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_inclusive_scan: T=long, U=double
    Spectra.transform_inclusive_scan(longs, doubles, n, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_inclusive_scan: T=float, U=int
    Spectra.transform_inclusive_scan(floats, ints, n, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_inclusive_scan: T=float, U=long
    Spectra.transform_inclusive_scan(floats, longs, n, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_inclusive_scan: T=float, U=float
    Spectra.transform_inclusive_scan(floats, floats, n, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_inclusive_scan: T=float, U=double
    Spectra.transform_inclusive_scan(floats, doubles, n, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_inclusive_scan: T=double, U=int
    Spectra.transform_inclusive_scan(doubles, ints, n, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_inclusive_scan: T=double, U=long
    Spectra.transform_inclusive_scan(doubles, longs, n, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_inclusive_scan: T=double, U=float
    Spectra.transform_inclusive_scan(doubles, floats, n, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_inclusive_scan: T=double, U=double
    Spectra.transform_inclusive_scan(doubles, doubles, n, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_inclusive_scan: T=Value, U=Value
    Spectra.transform_inclusive_scan(values, values, n, (arg0) -> null, (arg0, arg1) -> null);

    // transform_reduce: T=int, U=int
    Spectra.transform_reduce(ints, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=int, U=long
    Spectra.transform_reduce(ints, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=int, U=float
    Spectra.transform_reduce(ints, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=int, U=double
    Spectra.transform_reduce(ints, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_reduce: T=long, U=int
    Spectra.transform_reduce(longs, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=long, U=long
    Spectra.transform_reduce(longs, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=long, U=float
    Spectra.transform_reduce(longs, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=long, U=double
    Spectra.transform_reduce(longs, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_reduce: T=float, U=int
    Spectra.transform_reduce(floats, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=float, U=long
    Spectra.transform_reduce(floats, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=float, U=float
    Spectra.transform_reduce(floats, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=float, U=double
    Spectra.transform_reduce(floats, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_reduce: T=double, U=int
    Spectra.transform_reduce(doubles, n, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=double, U=long
    Spectra.transform_reduce(doubles, n, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=double, U=float
    Spectra.transform_reduce(doubles, n, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=double, U=double
    Spectra.transform_reduce(doubles, n, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_reduce: T=Value, U=Value
    Spectra.transform_reduce(values, n, new Value(), (arg0) -> null, (arg0, arg1) -> null);

    // unique: T=int
    Spectra.unique(ints, n, (arg0, arg1) -> false);

    // unique: T=long
    Spectra.unique(longs, n, (arg0, arg1) -> false);

    // unique: T=float
    Spectra.unique(floats, n, (arg0, arg1) -> false);

    // unique: T=double
    Spectra.unique(doubles, n, (arg0, arg1) -> false);

    // unique: T=Value
    Spectra.unique(values, n, (arg0, arg1) -> false);

    // unique_copy: T=int
    Spectra.unique_copy(ints, n, ints, (arg0, arg1) -> false);

    // unique_copy: T=long
    Spectra.unique_copy(longs, n, longs, (arg0, arg1) -> false);

    // unique_copy: T=float
    Spectra.unique_copy(floats, n, floats, (arg0, arg1) -> false);

    // unique_copy: T=double
    Spectra.unique_copy(doubles, n, doubles, (arg0, arg1) -> false);

    // unique_copy: T=Value
    Spectra.unique_copy(values, n, values, (arg0, arg1) -> false);

    // upper_bound: T=int
    Spectra.upper_bound(ints, n, 0, (arg0, arg1) -> false);

    // upper_bound: T=long
    Spectra.upper_bound(longs, n, 0L, (arg0, arg1) -> false);

    // upper_bound: T=float
    Spectra.upper_bound(floats, n, 0.0f, (arg0, arg1) -> false);

    // upper_bound: T=double
    Spectra.upper_bound(doubles, n, 0.0, (arg0, arg1) -> false);

    // upper_bound: T=Value
    Spectra.upper_bound(values, n, new Value(), (arg0, arg1) -> false);
  }
}
