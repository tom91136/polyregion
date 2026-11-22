package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.javalang.Par;
import polyregion.jvm.runtime.Context;

public class ParLaunchTest {

  private static final int N = 5000;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  private static double bound(double[] terms) {
    double magnitude = 0;
    for (double term : terms) magnitude += Math.abs(term);
    return terms.length * Math.ulp(magnitude);
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
    assertEquals(IntStream.range(0, N).map(i -> i * 3 + k).sum(), Par.range(0, N).map(i -> i * 3 + k).sum());
    assertEquals(IntStream.rangeClosed(1, N).sum(), Par.rangeClosed(1, N).sum());
    assertEquals(LongStream.rangeClosed(1, N).sum(), Par.rangeClosed(1L, N).sum());
  }

  @Test
  public void sumsAnArrayThroughMapAndFilter() {
    int[] xs = ints(1);
    int bias = 5;
    assertEquals(
        Arrays.stream(xs).map(x -> x * 2 + bias).filter(x -> x % 3 == 0).sum(),
        Par.ints(xs).map(x -> x * 2 + bias).filter(x -> x % 3 == 0).sum());
    long[] ls = longs(2);
    assertEquals(Arrays.stream(ls).map(x -> x * x).sum(), Par.longs(ls).map(x -> x * x).sum());
    double[] ds = doubles(3);
    double[] halved = Arrays.stream(ds).map(x -> x * 0.5).toArray();
    assertEquals(Arrays.stream(halved).sum(), Par.doubles(ds).map(x -> x * 0.5).sum(), bound(halved));
  }

  @Test
  public void convertsBetweenElementTypes() {
    int[] xs = ints(4);
    assertEquals(
        Arrays.stream(xs).mapToLong(x -> (long) x * 1000000L).sum(),
        Par.ints(xs).mapToLong(x -> (long) x * 1000000L).sum());
    double[] halves = Arrays.stream(xs).mapToDouble(x -> x / 2.0).toArray();
    assertEquals(Arrays.stream(halves).sum(), Par.ints(xs).mapToDouble(x -> x / 2.0).sum(), bound(halves));
    assertEquals(
        LongStream.range(0, N).mapToInt(i -> (int) (i % 17)).sum(),
        Par.range(0L, N).mapToInt(i -> (int) (i % 17)).sum());
  }

  @Test
  public void countsAndExtremes() {
    int[] xs = ints(5);
    assertEquals(N, Par.ints(xs).count());
    assertEquals(Arrays.stream(xs).filter(x -> x > 100).count(), Par.ints(xs).filter(x -> x > 100).count());
    assertEquals(Arrays.stream(xs).min(), Par.ints(xs).min());
    assertEquals(Arrays.stream(xs).max(), Par.ints(xs).max());
    assertEquals(OptionalInt.empty(), Par.ints(xs).filter(x -> x > 5000).max());
    assertEquals(OptionalLong.empty(), Par.range(5L, 5L).min());
    assertEquals(OptionalDouble.empty(), Par.doubles(new double[0]).max());
    assertEquals(Arrays.stream(xs).average(), Par.ints(xs).average());
    assertEquals(OptionalDouble.empty(), Par.ints(xs).filter(x -> x > 5000).average());
  }

  @Test
  public void matchesPredicates() {
    int[] xs = ints(6);
    assertEquals(Arrays.stream(xs).anyMatch(x -> x == 17), Par.ints(xs).anyMatch(x -> x == 17));
    assertEquals(Arrays.stream(xs).allMatch(x -> x > -2000), Par.ints(xs).allMatch(x -> x > -2000));
    assertEquals(Arrays.stream(xs).noneMatch(x -> x > 0), Par.ints(xs).noneMatch(x -> x > 0));
  }

  @Test
  public void reduces() {
    int[] xs = ints(7);
    assertEquals(
        Arrays.stream(xs).reduce(Integer.MIN_VALUE, Math::max), Par.ints(xs).reduce(Integer.MIN_VALUE, Math::max));
    assertEquals(Arrays.stream(xs).reduce(Integer::sum), Par.ints(xs).reduce(Integer::sum));
    assertEquals(OptionalInt.empty(), Par.ints(xs).filter(x -> x > 5000).reduce(Integer::sum));
  }

  @Test
  public void collectsToAnArray() {
    int[] xs = ints(8);
    int k = 3;
    assertArrayEquals(Arrays.stream(xs).map(x -> x * k).toArray(), Par.ints(xs).map(x -> x * k).toArray());
    assertArrayEquals(
        Arrays.stream(xs).filter(x -> x > 0).map(x -> x + 1).toArray(),
        Par.ints(xs).filter(x -> x > 0).map(x -> x + 1).toArray());
    assertArrayEquals(IntStream.range(0, N).map(i -> i * i).toArray(), Par.range(0, N).map(i -> i * i).toArray());
    assertArrayEquals(
        IntStream.range(0, N).filter(i -> i % 3 == 1).toArray(), Par.range(0, N).filter(i -> i % 3 == 1).toArray());
    assertArrayEquals(Arrays.stream(xs).sorted().toArray(), Par.ints(xs).sorted().toArray());
    assertArrayEquals(new int[0], Par.ints(xs).filter(x -> x > 5000).toArray());
  }

  @Test
  public void runsForEach() {
    int[] in = ints(9);
    int[] out = new int[N];
    int add = 11;
    Par.range(0, N).forEach(i -> out[i] = in[i] + add);
    int[] expected = new int[N];
    IntStream.range(0, N).forEach(i -> expected[i] = in[i] + add);
    assertArrayEquals(expected, out);
  }

  @Test
  public void setsPrefixesAndSorts() {
    int[] a = new int[N];
    int[] b = new int[N];
    int k = 9;
    Par.setAll(a, i -> i * k);
    Arrays.setAll(b, i -> i * k);
    assertArrayEquals(b, a);
    int[] p = ints(10);
    int[] q = p.clone();
    Par.prefix(p, (x, y) -> x + y);
    for (int i = 1; i < q.length; i++) q[i] += q[i - 1];
    assertArrayEquals(q, p);
    double[] d = doubles(11);
    double[] e = d.clone();
    Par.sort(d);
    Arrays.sort(e);
    assertArrayEquals(e, d, 0.0);
    float[] f = {Float.NaN, 2f, -0f, 0f, 1f};
    float[] g = f.clone();
    Par.sort(f);
    Arrays.sort(g);
    assertEquals(Arrays.toString(g), Arrays.toString(f));
  }

  @Test
  public void selectsAnExplicitContext() {
    Context context = Context.current();
    int[] xs = ints(12);
    assertEquals(Arrays.stream(xs).map(x -> x + 1).sum(), Par.ints(xs).on(context).map(x -> x + 1).sum());
    int[] a = new int[N];
    Par.setAll(context, a, i -> i);
    assertArrayEquals(IntStream.range(0, N).toArray(), a);
    long before = context.stats().launches;
    assertEquals(Arrays.stream(xs).sum(), Par.ints(xs).on(context).sum());
    assertTrue(context.stats().launches > before);
  }

  private static long launches(Runnable body) {
    Context context = Context.current();
    long before = context.stats().launches;
    body.run();
    return context.stats().launches - before;
  }

  @Test
  public void fusesStagesIntoOneLaunch() {
    int[] xs = ints(13);
    int[] sink = new int[1];
    long plain = launches(() -> sink[0] = Par.ints(xs).sum());
    assertTrue("a reduction launches a kernel", plain > 0);
    long fused = launches(() -> sink[0] = Par.ints(xs).map(x -> x + 1).map(x -> x * 2).filter(x -> x > 0).sum());
    assertEquals("map, map, filter and sum are one launch", plain, fused);
    long range = launches(() -> sink[0] = Par.range(0, N).sum());
    long rangeFused = launches(() -> sink[0] = Par.range(0, N).map(i -> i + 1).filter(i -> i > 3).sum());
    assertEquals("a range chain is one launch", range, rangeFused);
  }
}
