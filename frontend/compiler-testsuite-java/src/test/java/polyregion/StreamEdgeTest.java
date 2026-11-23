package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.javalang.Par;
import polyregion.jvm.runtime.Context;

public class StreamEdgeTest {

  private static final int N = 5000;

  private static final long SPAN = Long.getLong("polyregion.stream.span", 4_294_967_306L);

  @Rule
  public final TestRule kernels =
      LaunchRule.requireLaunch(
          method -> method.startsWith("fallsBack") || method.startsWith("refuses"));

  private static long launches(Runnable body) {
    Context context = Context.current();
    long before = context.stats().launches;
    body.run();
    return context.stats().launches - before;
  }

  private static void assertEveryHit(int[] hits) {
    for (int i = 0; i < hits.length; i++) assertEquals("element " + i, 1, hits[i]);
  }

  private static <T> T record(List<String> log, String what, T value) {
    log.add(what);
    return value;
  }

  private static int[] permutation(long seed) {
    int[] xs = IntStream.range(0, N).toArray();
    Random random = new Random(seed);
    for (int i = N - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int t = xs[i];
      xs[i] = xs[j];
      xs[j] = t;
    }
    return xs;
  }

  private static double bound(double[] terms) {
    double magnitude = 0;
    for (double term : terms) magnitude += Math.abs(term);
    return terms.length * Math.ulp(magnitude);
  }

  @Test
  public void fallsBackOnForEachOrdered() {
    int[] last = {-1};
    long launched =
        launches(() -> IntStream.range(0, N).parallel().forEachOrdered(i -> last[0] = i));
    assertEquals(N - 1, last[0]);
    assertEquals("forEachOrdered stays on the JDK", 0, launched);
  }

  @Test
  public void reducesRangesBeyondIntLimitsOnCuda() {
    assumeTrue(
        "2^32-element ranges exceed the host time budget",
        System.getProperty("polyregion.platform", "").split("[@!]", 2)[0].equalsIgnoreCase("cuda"));
    long cycles = SPAN / 3;
    long rest = SPAN % 3;
    assertEquals(cycles * 3 + rest * (rest - 1) / 2, LongStream.range(0, SPAN).parallel().map(x -> x % 3).sum());
    assertEquals((SPAN + 6) / 7, LongStream.range(0, SPAN).parallel().filter(x -> x % 7 == 0).count());
    assertEquals(SPAN, LongStream.range(0, SPAN).parallel().count());
    assertEquals(OptionalLong.of(SPAN - 1), LongStream.range(0, SPAN).parallel().max());
    assertEquals(OptionalLong.of(0), LongStream.range(0, SPAN).parallel().min());
    assertEquals(
        OptionalLong.of(SPAN - 1), LongStream.range(0, SPAN).parallel().filter(x -> x > SPAN - 3).max());
    assertTrue(LongStream.range(0, SPAN).parallel().anyMatch(x -> x == SPAN - 1));
    assertFalse(LongStream.range(0, SPAN).parallel().allMatch(x -> x < SPAN - 1));
    assertTrue(LongStream.range(0, SPAN).parallel().noneMatch(x -> x >= SPAN));
    assertEquals(cycles * 3 + rest * (rest - 1) / 2, LongStream.range(0, SPAN).parallel().map(x -> x % 3).reduce(0L, Long::sum));
    long[] seen = {0};
    LongStream.range(0, SPAN).parallel().forEach(x -> {
      if (x == SPAN - 1) seen[0] = 1;
    });
    assertEquals("the last element ran", 1, seen[0]);
    assertEquals(
        Integer.MAX_VALUE - 6L,
        IntStream.range(Integer.MIN_VALUE, Integer.MAX_VALUE).parallel().filter(i -> i > 5).count());
    assertEquals(
        Integer.MAX_VALUE - 5L,
        IntStream.rangeClosed(Integer.MIN_VALUE, Integer.MAX_VALUE).parallel().filter(i -> i > 5).count());
  }

  @Test
  public void refusesArraysBeyondTheJdkLimit() {
    assertThrows(IllegalArgumentException.class, () -> LongStream.range(0, SPAN).parallel().toArray());
    assertThrows(IllegalArgumentException.class, () -> LongStream.range(0, SPAN).parallel().map(x -> x + 1).toArray());
    assertThrows(IllegalArgumentException.class, () -> IntStream.range(0, Integer.MAX_VALUE).parallel().toArray());
    assertThrows(
        IllegalArgumentException.class, () -> LongStream.range(0, Integer.MAX_VALUE - 8L).parallel().toArray());
    assertThrows(
        IllegalArgumentException.class,
        () -> LongStream.range(0, SPAN).parallel().sorted().toArray());
  }

  @Test
  public void coversTheEdgesOfTheIntegerTypes() {
    assertArrayEquals(
        LongStream.rangeClosed(Long.MAX_VALUE - 4, Long.MAX_VALUE).toArray(),
        LongStream.rangeClosed(Long.MAX_VALUE - 4, Long.MAX_VALUE).parallel().toArray());
    assertArrayEquals(
        LongStream.range(Long.MIN_VALUE, Long.MIN_VALUE + 5).toArray(),
        LongStream.range(Long.MIN_VALUE, Long.MIN_VALUE + 5).parallel().toArray());
    assertArrayEquals(
        IntStream.rangeClosed(Integer.MAX_VALUE - 4, Integer.MAX_VALUE).map(i -> i - 1).toArray(),
        IntStream.rangeClosed(Integer.MAX_VALUE - 4, Integer.MAX_VALUE).parallel().map(i -> i - 1).toArray());
    assertArrayEquals(
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + 5).toArray(),
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + 5).parallel().toArray());
    assertEquals(
        LongStream.rangeClosed(Long.MAX_VALUE - 4, Long.MAX_VALUE).count(),
        LongStream.rangeClosed(Long.MAX_VALUE - 4, Long.MAX_VALUE).parallel().filter(x -> true).count());
  }

  @Test
  public void refusesSpansBeyondALong() {
    assertThrows(
        ArithmeticException.class,
        () -> LongStream.rangeClosed(Long.MIN_VALUE, Long.MAX_VALUE).parallel().filter(x -> x == 0).count());
    assertThrows(
        ArithmeticException.class,
        () -> LongStream.range(Long.MIN_VALUE, Long.MAX_VALUE).parallel().filter(x -> x == 0).count());
  }

  @Test
  public void sumsDoublesWithinTheNaiveSummationBound() {
    double[] tenths = new double[10];
    Arrays.fill(tenths, 0.1);
    assertEquals(
        DoubleStream.of(tenths).sum(), DoubleStream.of(tenths).parallel().sum(), bound(tenths));
    assertEquals(
        DoubleStream.of(tenths).average().getAsDouble(),
        DoubleStream.of(tenths).parallel().average().getAsDouble(),
        bound(tenths) / tenths.length);
    double[] mixed = new double[N];
    Random random = new Random(41);
    for (int i = 0; i < N; i++) mixed[i] = (random.nextInt(2000) - 1000) / 7.0;
    assertEquals(Arrays.stream(mixed).sum(), Arrays.stream(mixed).parallel().sum(), bound(mixed));
  }

  @Test
  public void mapsOncePerElementAcrossAFilter() {
    int[] seqMap = new int[N];
    int[] seqKeep = new int[N];
    int[] seqLate = new int[N];
    int[] expected =
        IntStream.range(0, N)
            .map(i -> { seqMap[i]++; return i * 2; })
            .filter(v -> { seqKeep[v / 2]++; return v % 4 == 0; })
            .map(v -> { seqLate[v / 2]++; return v + 1; })
            .filter(v -> v % 3 != 0)
            .toArray();
    int[] parMap = new int[N];
    int[] parKeep = new int[N];
    int[] parLate = new int[N];
    int[] actual =
        IntStream.range(0, N)
            .parallel()
            .map(i -> { parMap[i]++; return i * 2; })
            .filter(v -> { parKeep[v / 2]++; return v % 4 == 0; })
            .map(v -> { parLate[v / 2]++; return v + 1; })
            .filter(v -> v % 3 != 0)
            .toArray();
    assertArrayEquals(expected, actual);
    assertArrayEquals(seqMap, parMap);
    assertArrayEquals(seqKeep, parKeep);
    assertArrayEquals(seqLate, parLate);
    assertEveryHit(parMap);
  }

  @Test
  public void mapsOncePerElementOfAnArrayAcrossAFilter() {
    int[] xs = permutation(51);
    int[] seqMap = new int[N];
    int[] parMap = new int[N];
    int[] expected =
        Arrays.stream(xs).map(x -> { seqMap[x]++; return x * 3; }).filter(v -> v % 2 == 0).toArray();
    int[] actual =
        Arrays.stream(xs).parallel().map(x -> { parMap[x]++; return x * 3; }).filter(v -> v % 2 == 0).toArray();
    assertArrayEquals(expected, actual);
    assertEveryHit(parMap);
  }

  @Test
  public void mapsOncePerElementInFilteredReductions() {
    int[] avg = new int[N];
    int[] min = new int[N];
    int[] max = new int[N];
    int[] reduce = new int[N];
    int[] mapped = new int[N];
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).average(),
        IntStream.range(0, N).parallel().map(i -> { avg[i]++; return i * 2; }).filter(v -> v % 3 == 0).average());
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).min(),
        IntStream.range(0, N).parallel().map(i -> { min[i]++; return i * 2; }).filter(v -> v % 3 == 0).min());
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).max(),
        IntStream.range(0, N).parallel().map(i -> { max[i]++; return i * 2; }).filter(v -> v % 3 == 0).max());
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).reduce(Math::max),
        IntStream.range(0, N).parallel().map(i -> { reduce[i]++; return i * 2; }).filter(v -> v % 3 == 0).reduce(Math::max));
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).sum(),
        IntStream.range(0, N).parallel().map(i -> { mapped[i]++; return i * 2; }).filter(v -> v % 3 == 0).sum());
    assertEveryHit(avg);
    assertEveryHit(min);
    assertEveryHit(max);
    assertEveryHit(reduce);
    assertEveryHit(mapped);
  }

  @Test
  public void filtersOnceWhenTheExtremeIsTheNeutralValue() {
    int[] hits = new int[3];
    assertEquals(
        OptionalInt.of(Integer.MAX_VALUE),
        IntStream.range(0, 3).parallel().map(i -> { hits[i]++; return Integer.MAX_VALUE; }).filter(v -> true).min());
    assertEveryHit(hits);
  }

  @Test
  public void mapsOncePerElementOnTheParRoute() {
    int[] hits = new int[N];
    int[] expected = IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 4 == 0).toArray();
    assertArrayEquals(
        expected,
        Par.range(0, N).map(i -> { hits[i]++; return i * 2; }).filter(v -> v % 4 == 0).toArray());
    assertEveryHit(hits);
    int[] avgHits = new int[N];
    assertEquals(
        IntStream.range(0, N).map(i -> i * 2).filter(v -> v % 3 == 0).average(),
        Par.range(0, N).map(i -> { avgHits[i]++; return i * 2; }).filter(v -> v % 3 == 0).average());
    assertEveryHit(avgHits);
  }

  @Test
  public void evaluatesTheSourceBeforeTheContext() {
    List<String> log = new ArrayList<>();
    int[] xs = {1, 2, 3};
    Context context = Context.current();
    int total = Par.ints(record(log, "source", xs)).on(record(log, "context", context)).sum();
    assertEquals(6, total);
    assertEquals(Arrays.asList("source", "context"), log);
    log.clear();
    long reduced =
        Par.range(record(log, "lo", 0), record(log, "hi", 4))
            .mapToLong(i -> i)
            .on(record(log, "context", context))
            .reduce(record(log, "identity", 0L), Long::sum);
    assertEquals(6, reduced);
    assertEquals(Arrays.asList("lo", "hi", "context", "identity"), log);
    log.clear();
    int[] a = {3, 1, 2};
    Par.sort(record(log, "context", context), record(log, "array", a));
    assertArrayEquals(new int[] {1, 2, 3}, a);
    assertEquals(Arrays.asList("context", "array"), log);
  }

  @Test
  public void evaluatesJdkChainOperandsInOrder() {
    List<String> log = new ArrayList<>();
    long total =
        LongStream.range(record(log, "lo", 0L), record(log, "hi", 4L))
            .parallel()
            .reduce(record(log, "identity", 0L), Long::sum);
    assertEquals(6, total);
    assertEquals(Arrays.asList("lo", "hi", "identity"), log);
  }
}
