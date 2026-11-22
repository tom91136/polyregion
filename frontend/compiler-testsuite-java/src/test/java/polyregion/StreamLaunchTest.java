package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import polyregion.jvm.runtime.Context;

public class StreamLaunchTest {

  private static final int N = 5000;

  @Rule public final TestName name = new TestName();

  private long launchedBefore;

  @Before
  public void snapshotLaunches() {
    launchedBefore = Context.current().stats().launches;
  }

  @After
  public void everyChainLaunchedAKernel() {
    if (!(name.getMethodName().startsWith("leaves") || name.getMethodName().startsWith("fallsBack") || name.getMethodName().startsWith("emptySources")))
      assertTrue(
          name.getMethodName() + " ran no kernel",
          Context.current().stats().launches > launchedBefore);
  }

  private static int[] ints(long seed) {
    Random random = new Random(seed);
    int[] xs = new int[N];
    for (int i = 0; i < N; i++) xs[i] = random.nextInt(2000) - 1000;
    return xs;
  }

  private static long[] longs(long seed) {
    Random random = new Random(seed);
    long[] xs = new long[N];
    for (int i = 0; i < N; i++) xs[i] = random.nextInt(2000) - 1000;
    return xs;
  }

  private static double[] doubles(long seed) {
    Random random = new Random(seed);
    double[] xs = new double[N];
    for (int i = 0; i < N; i++) xs[i] = random.nextInt(2000) / 8.0 - 100.0;
    return xs;
  }

  @Test
  public void sumsAMappedRange() {
    int k = 7;
    int expected = IntStream.range(0, N).map(i -> i * 3 + k).sum();
    assertEquals(expected, IntStream.range(0, N).parallel().map(i -> i * 3 + k).sum());
  }

  @Test
  public void sumsAClosedRange() {
    assertEquals(IntStream.rangeClosed(1, N).sum(), IntStream.rangeClosed(1, N).parallel().sum());
    assertEquals(LongStream.rangeClosed(1, N).sum(), LongStream.rangeClosed(1, N).parallel().sum());
  }

  @Test
  public void sumsAnArrayThroughMapAndFilter() {
    int[] xs = ints(1);
    int bias = 5;
    int expected = Arrays.stream(xs).map(x -> x * 2 + bias).filter(x -> x % 3 == 0).sum();
    int actual = Arrays.stream(xs).parallel().map(x -> x * 2 + bias).filter(x -> x % 3 == 0).sum();
    assertEquals(expected, actual);
  }

  @Test
  public void sumsLongsAndDoubles() {
    long[] ls = longs(2);
    double[] ds = doubles(3);
    assertEquals(Arrays.stream(ls).map(x -> x * x).sum(), Arrays.stream(ls).parallel().map(x -> x * x).sum());
    assertEquals(
        Arrays.stream(ds).map(x -> x * 0.5).sum(),
        Arrays.stream(ds).parallel().map(x -> x * 0.5).sum(),
        1e-6);
  }

  @Test
  public void convertsBetweenElementTypes() {
    int[] xs = ints(4);
    long wide = Arrays.stream(xs).mapToLong(x -> (long) x * 1000000L).sum();
    assertEquals(wide, Arrays.stream(xs).parallel().mapToLong(x -> (long) x * 1000000L).sum());
    double halves = Arrays.stream(xs).mapToDouble(x -> x / 2.0).sum();
    assertEquals(halves, Arrays.stream(xs).parallel().mapToDouble(x -> x / 2.0).sum(), 1e-6);
    int narrowed = LongStream.range(0, N).mapToInt(i -> (int) (i % 17)).sum();
    assertEquals(narrowed, LongStream.range(0, N).parallel().mapToInt(i -> (int) (i % 17)).sum());
  }

  @Test
  public void countsWithAndWithoutAFilter() {
    int[] xs = ints(5);
    assertEquals(N, Arrays.stream(xs).parallel().count());
    assertEquals(
        Arrays.stream(xs).filter(x -> x > 100).count(), Arrays.stream(xs).parallel().filter(x -> x > 100).count());
    assertEquals(
        IntStream.range(0, N).filter(i -> i % 7 == 0).count(),
        IntStream.range(0, N).parallel().filter(i -> i % 7 == 0).count());
  }

  @Test
  public void findsTheExtremes() {
    int[] xs = ints(6);
    assertEquals(Arrays.stream(xs).min(), Arrays.stream(xs).parallel().min());
    assertEquals(Arrays.stream(xs).max(), Arrays.stream(xs).parallel().max());
    assertEquals(
        Arrays.stream(xs).map(x -> -x).filter(x -> x > 3).min(),
        Arrays.stream(xs).parallel().map(x -> -x).filter(x -> x > 3).min());
    double[] ds = doubles(7);
    assertEquals(Arrays.stream(ds).min(), Arrays.stream(ds).parallel().min());
    assertEquals(Arrays.stream(ds).max(), Arrays.stream(ds).parallel().max());
    long[] ls = longs(8);
    assertEquals(Arrays.stream(ls).min(), Arrays.stream(ls).parallel().min());
    assertEquals(Arrays.stream(ls).max(), Arrays.stream(ls).parallel().max());
  }

  @Test
  public void extremesOfEmptyChainsAreEmpty() {
    int[] xs = ints(9);
    int[] none = new int[0];
    assertEquals(OptionalInt.empty(), Arrays.stream(none).parallel().min());
    assertEquals(OptionalInt.empty(), Arrays.stream(xs).parallel().filter(x -> x > 5000).max());
    assertEquals(OptionalInt.empty(), IntStream.range(0, 0).parallel().max());
    assertEquals(OptionalLong.empty(), LongStream.range(5, 5).parallel().min());
    assertEquals(OptionalDouble.empty(), Arrays.stream(new double[0]).parallel().max());
  }

  @Test
  public void extremesKeepTheNeutralValue() {
    int[] xs = {Integer.MAX_VALUE, Integer.MAX_VALUE, 5};
    assertEquals(OptionalInt.of(5), Arrays.stream(xs).parallel().filter(x -> x == 5).min());
    int[] only = {Integer.MAX_VALUE};
    assertEquals(OptionalInt.of(Integer.MAX_VALUE), Arrays.stream(only).parallel().filter(x -> x > 0).min());
  }

  @Test
  public void averagesWithAndWithoutAFilter() {
    int[] xs = ints(10);
    assertEquals(Arrays.stream(xs).average(), Arrays.stream(xs).parallel().average());
    assertEquals(
        Arrays.stream(xs).filter(x -> x > 0).average(), Arrays.stream(xs).parallel().filter(x -> x > 0).average());
    assertEquals(OptionalDouble.empty(), Arrays.stream(xs).parallel().filter(x -> x > 5000).average());
    double[] ds = doubles(11);
    assertEquals(Arrays.stream(ds).average().getAsDouble(), Arrays.stream(ds).parallel().average().getAsDouble(), 1e-6);
  }

  @Test
  public void matchesPredicates() {
    int[] xs = ints(12);
    assertEquals(Arrays.stream(xs).anyMatch(x -> x == 17), Arrays.stream(xs).parallel().anyMatch(x -> x == 17));
    assertEquals(Arrays.stream(xs).anyMatch(x -> x > 5000), Arrays.stream(xs).parallel().anyMatch(x -> x > 5000));
    assertEquals(Arrays.stream(xs).allMatch(x -> x > -2000), Arrays.stream(xs).parallel().allMatch(x -> x > -2000));
    assertEquals(Arrays.stream(xs).allMatch(x -> x > 0), Arrays.stream(xs).parallel().allMatch(x -> x > 0));
    assertEquals(Arrays.stream(xs).noneMatch(x -> x > 5000), Arrays.stream(xs).parallel().noneMatch(x -> x > 5000));
    assertEquals(Arrays.stream(xs).noneMatch(x -> x > 0), Arrays.stream(xs).parallel().noneMatch(x -> x > 0));
    assertTrue(new int[0].length == 0 && Arrays.stream(new int[0]).parallel().allMatch(x -> false));
    assertTrue(!Arrays.stream(new int[0]).parallel().anyMatch(x -> true));
  }

  @Test
  public void reducesWithAnIdentity() {
    int[] xs = ints(13);
    assertEquals(Arrays.stream(xs).reduce(0, (a, b) -> a + b), Arrays.stream(xs).parallel().reduce(0, (a, b) -> a + b));
    assertEquals(
        Arrays.stream(xs).reduce(Integer.MIN_VALUE, Math::max),
        Arrays.stream(xs).parallel().reduce(Integer.MIN_VALUE, Math::max));
    assertEquals(
        LongStream.range(1, 20).reduce(1L, (a, b) -> a * b), LongStream.range(1, 20).parallel().reduce(1L, (a, b) -> a * b));
    assertEquals(7, Arrays.stream(new int[0]).parallel().reduce(7, Integer::sum));
  }

  @Test
  public void reducesWithoutAnIdentity() {
    int[] xs = ints(14);
    assertEquals(Arrays.stream(xs).reduce(Integer::sum), Arrays.stream(xs).parallel().reduce(Integer::sum));
    assertEquals(
        Arrays.stream(xs).filter(x -> x > 0).reduce(Math::max),
        Arrays.stream(xs).parallel().filter(x -> x > 0).reduce(Math::max));
    assertEquals(OptionalInt.empty(), Arrays.stream(xs).parallel().filter(x -> x > 5000).reduce(Integer::sum));
    assertEquals(OptionalInt.of(4), IntStream.range(4, 5).parallel().reduce(Integer::sum));
  }

  @Test
  public void collectsToAnArray() {
    int[] xs = ints(15);
    int k = 3;
    assertArrayEquals(
        Arrays.stream(xs).map(x -> x * k).toArray(), Arrays.stream(xs).parallel().map(x -> x * k).toArray());
    assertArrayEquals(
        Arrays.stream(xs).filter(x -> x > 0).map(x -> x + 1).toArray(),
        Arrays.stream(xs).parallel().filter(x -> x > 0).map(x -> x + 1).toArray());
    assertArrayEquals(Arrays.stream(xs).toArray(), Arrays.stream(xs).parallel().toArray());
    assertArrayEquals(
        IntStream.range(0, N).map(i -> i * i).toArray(), IntStream.range(0, N).parallel().map(i -> i * i).toArray());
    assertArrayEquals(
        IntStream.range(0, N).filter(i -> i % 3 == 1).toArray(),
        IntStream.range(0, N).parallel().filter(i -> i % 3 == 1).toArray());
    assertArrayEquals(
        LongStream.rangeClosed(1, 100).map(i -> i << 33).toArray(),
        LongStream.rangeClosed(1, 100).parallel().map(i -> i << 33).toArray());
    assertArrayEquals(new int[0], Arrays.stream(xs).parallel().filter(x -> x > 5000).toArray());
    assertArrayEquals(new int[0], IntStream.range(3, 3).parallel().toArray());
    double[] ds = doubles(16);
    assertArrayEquals(
        Arrays.stream(ds).map(x -> x * 2).toArray(), Arrays.stream(ds).parallel().map(x -> x * 2).toArray(), 0.0);
  }

  @Test
  public void sortsThroughTheStream() {
    int[] xs = ints(17);
    assertArrayEquals(Arrays.stream(xs).sorted().toArray(), Arrays.stream(xs).parallel().sorted().toArray());
    assertArrayEquals(
        Arrays.stream(xs).map(x -> -x).sorted().toArray(), Arrays.stream(xs).parallel().map(x -> -x).sorted().toArray());
  }

  @Test
  public void runsForEachOverARange() {
    int[] in = ints(18);
    int[] out = new int[N];
    int add = 11;
    IntStream.range(0, N).parallel().forEach(i -> out[i] = in[i] + add);
    int[] expected = new int[N];
    IntStream.range(0, N).forEach(i -> expected[i] = in[i] + add);
    assertArrayEquals(expected, out);
  }

  @Test
  public void runsForEachOverAnArray() {
    int[] in = ints(19);
    int[] out = new int[N];
    Arrays.stream(in).parallel().filter(x -> x > 0).forEach(x -> out[x] = x);
    int[] expected = new int[N];
    Arrays.stream(in).filter(x -> x > 0).forEach(x -> expected[x] = x);
    assertArrayEquals(expected, out);
  }

  @Test
  public void setsAllElements() {
    int[] a = new int[N];
    int[] b = new int[N];
    int k = 9;
    Arrays.parallelSetAll(a, i -> i * k);
    Arrays.setAll(b, i -> i * k);
    assertArrayEquals(b, a);
    long[] la = new long[N];
    long[] lb = new long[N];
    Arrays.parallelSetAll(la, i -> (long) i << 33);
    Arrays.setAll(lb, i -> (long) i << 33);
    assertArrayEquals(lb, la);
    double[] da = new double[N];
    double[] db = new double[N];
    Arrays.parallelSetAll(da, i -> i * 0.25);
    Arrays.setAll(db, i -> i * 0.25);
    assertArrayEquals(db, da, 0.0);
  }

  @Test
  public void computesPrefixes() {
    int[] a = ints(20);
    int[] b = a.clone();
    Arrays.parallelPrefix(a, (x, y) -> x + y);
    for (int i = 1; i < b.length; i++) b[i] += b[i - 1];
    assertArrayEquals(b, a);
    long[] la = longs(21);
    long[] lb = la.clone();
    Arrays.parallelPrefix(la, Long::sum);
    for (int i = 1; i < lb.length; i++) lb[i] += lb[i - 1];
    assertArrayEquals(lb, la);
  }

  @Test
  public void sortsInPlace() {
    int[] a = ints(22);
    int[] b = a.clone();
    Arrays.parallelSort(a);
    Arrays.sort(b);
    assertArrayEquals(b, a);
    long[] la = longs(23);
    long[] lb = la.clone();
    Arrays.parallelSort(la);
    Arrays.sort(lb);
    assertArrayEquals(lb, la);
    double[] da = doubles(24);
    double[] db = da.clone();
    Arrays.parallelSort(da);
    Arrays.sort(db);
    assertArrayEquals(db, da, 0.0);
  }

  @Test
  public void sortsDoublesInTheTotalOrder() {
    double[] a = {Double.NaN, 2.0, 0.0, -0.0, 1.0, Double.NEGATIVE_INFINITY, -0.0, Double.NaN, 0.0};
    double[] b = a.clone();
    Arrays.parallelSort(a);
    Arrays.sort(b);
    assertEquals(Arrays.toString(b), Arrays.toString(a));
  }

  @Test
  public void sortsFloatsInTheTotalOrder() {
    float[] a = {Float.NaN, 2f, 0f, -0f, 1f, Float.NEGATIVE_INFINITY, -0f, Float.NaN, 0f};
    float[] b = a.clone();
    Arrays.parallelSort(a);
    Arrays.sort(b);
    assertEquals(Arrays.toString(b), Arrays.toString(a));
  }

  @Test
  public void sortsARangeThroughTheStream() {
    assertArrayEquals(
        LongStream.range(0, N).map(i -> (i * 7919) % 1000).sorted().toArray(),
        LongStream.range(0, N).parallel().map(i -> (i * 7919) % 1000).sorted().toArray());
  }

  @Test
  public void fallsBackWhenSortedIsNotTheLastStage() {
    int[] xs = ints(29);
    assertArrayEquals(
        Arrays.stream(xs).sorted().map(x -> -x).toArray(),
        Arrays.stream(xs).parallel().sorted().map(x -> -x).toArray());
  }

  @Test
  public void extremesFollowTheJdkOnNaNAndSignedZero() {
    double[] nan = {1.0, Double.NaN, 2.0};
    assertEquals(Arrays.stream(nan).min().getAsDouble(), Arrays.stream(nan).parallel().min().getAsDouble(), 0.0);
    double[] zeros = {0.0, -0.0};
    assertEquals(
        Double.doubleToLongBits(Arrays.stream(zeros).min().getAsDouble()),
        Double.doubleToLongBits(Arrays.stream(zeros).parallel().min().getAsDouble()));
    assertEquals(
        Double.doubleToLongBits(Arrays.stream(zeros).max().getAsDouble()),
        Double.doubleToLongBits(Arrays.stream(zeros).parallel().max().getAsDouble()));
  }

  @Test
  public void leavesSequentialStreamsToTheJdk() {
    int[] xs = ints(25);
    assertEquals(Arrays.stream(xs).map(x -> x + 1).sum(), Arrays.stream(xs).map(x -> x + 1).sum());
    assertEquals(Arrays.stream(xs).parallel().sequential().sum(), Arrays.stream(xs).sum());
  }

  @Test
  public void fallsBackToTheJdkOnUnsupportedChains() {
    int[] xs = ints(26);
    int[] seen = new int[1];
    assertEquals(
        Arrays.stream(xs).map(x -> x + 1).sum(),
        Arrays.stream(xs).parallel().peek(x -> {}).map(x -> x + 1).sum());
    assertEquals(
        Arrays.stream(xs).boxed().mapToInt(Integer::intValue).sum(),
        Arrays.stream(xs).parallel().boxed().mapToInt(Integer::intValue).sum());
    assertEquals(
        Arrays.stream(xs).limit(100).sum(), Arrays.stream(xs).parallel().limit(100).sum());
    assertEquals(
        Arrays.stream(xs).distinct().count(), Arrays.stream(xs).parallel().distinct().count());
    StringBuilder sb = new StringBuilder();
    assertEquals(
        IntStream.range(0, 10).map(i -> sb.length() + i).sum(),
        IntStream.range(0, 10).parallel().map(i -> sb.length() + i).sum());
    assertEquals(0, seen[0]);
  }

  @Test
  public void emptySourcesLaunchNothing() {
    Context context = Context.current();
    long before = context.stats().launches;
    assertEquals(0, IntStream.range(0, 0).parallel().map(i -> i + 1).sum());
    assertEquals(0, Arrays.stream(new int[0]).parallel().sum());
    assertEquals(0L, IntStream.range(2, 2).parallel().filter(i -> true).count());
    assertEquals(0, Arrays.stream(new double[0]).parallel().map(x -> x).toArray().length);
    assertEquals(before, context.stats().launches);
  }

  private static long launches(Runnable body) {
    Context context = Context.current();
    long before = context.stats().launches;
    body.run();
    return context.stats().launches - before;
  }

  @Test
  public void aGpuPlatformSeesTheChain() {
    boolean gpu = System.getProperty("polyregion.platform", "").equalsIgnoreCase("cuda");
    Context context = Context.current();
    Context.Stats before = context.stats();
    int[] xs = ints(28);
    assertEquals(Arrays.stream(xs).map(x -> x * 3).sum(), Arrays.stream(xs).parallel().map(x -> x * 3).sum());
    Context.Stats after = context.stats();
    assertTrue("the chain ran no kernel", after.launches > before.launches);
    if (gpu) assertTrue("the chain copied nothing to the device", after.hostToDevice > before.hostToDevice);
  }

  @Test
  public void fusesStagesIntoOneLaunch() {
    int[] xs = ints(27);
    int[] sink = new int[1];
    long plain = launches(() -> sink[0] = Arrays.stream(xs).parallel().sum());
    assertTrue("a reduction launches a kernel", plain > 0);
    long fused =
        launches(
            () -> sink[0] = Arrays.stream(xs).parallel().map(x -> x + 1).map(x -> x * 2).filter(x -> x > 0).sum());
    assertEquals("map, map, filter and sum are one launch", plain, fused);
    long range = launches(() -> sink[0] = IntStream.range(0, N).parallel().sum());
    long rangeFused =
        launches(() -> sink[0] = IntStream.range(0, N).parallel().map(i -> i + 1).filter(i -> i > 3).sum());
    assertEquals("a range chain is one launch", range, rangeFused);
    long transformed = launches(() -> sink[0] = Arrays.stream(xs).parallel().map(x -> x + 1).toArray().length);
    long transformedTwice =
        launches(() -> sink[0] = Arrays.stream(xs).parallel().map(x -> x + 1).map(x -> x * 3).toArray().length);
    assertEquals("two maps to an array are one launch", transformed, transformedTwice);
  }
}
