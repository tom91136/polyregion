package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static polyregion.javalang.Runtime.offload;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntSupplier;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.jvm.runtime.Context;
import polyregion.javalang.OffloadFunction;

public class SpectraLaunchTest {

  private static final int N = 4096;

  @Rule public final TestRule kernels = LaunchRule.skipOnGap();

  private static int sum(int[] xs) {
    int total = 0;
    for (int x : xs) total += x;
    return total;
  }

  private static int[] sorted(int[] xs) {
    int[] copy = xs.clone();
    Arrays.sort(copy);
    return copy;
  }

  private static int[] ints(long seed) {
    Random random = new Random(seed);
    int[] xs = new int[N];
    for (int i = 0; i < N; i++) xs[i] = random.nextInt(1 << 20) - (1 << 19);
    return xs;
  }

  private static int[] shuffled(long seed) {
    int[] xs = new int[N];
    for (int i = 0; i < N; i++) xs[i] = i;
    Random random = new Random(seed);
    for (int i = N - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int t = xs[i];
      xs[i] = xs[j];
      xs[j] = t;
    }
    return xs;
  }

  @Test
  public void sortsIntsWithAComparator() {
    int[] xs = ints(1);
    int[] expected = xs.clone();
    Arrays.sort(expected);
    Spectra.sort(xs, N, (a, b) -> a < b);
    assertArrayEquals(expected, xs);
  }

  @Test
  public void sortsDescending() {
    int[] xs = ints(2);
    int[] expected = xs.clone();
    Arrays.sort(expected);
    for (int i = 0; i < N / 2; i++) {
      int t = expected[i];
      expected[i] = expected[N - 1 - i];
      expected[N - 1 - i] = t;
    }
    Spectra.sort(xs, N, (a, b) -> a > b);
    assertArrayEquals(expected, xs);
  }

  @Test
  public void sortsFloatsAndDoubles() {
    Random random = new Random(3);
    float[] fs = new float[N];
    double[] ds = new double[N];
    for (int i = 0; i < N; i++) {
      fs[i] = random.nextFloat() * 100f - 50f;
      ds[i] = random.nextDouble() * 100.0 - 50.0;
    }
    float[] expectedF = fs.clone();
    double[] expectedD = ds.clone();
    Arrays.sort(expectedF);
    Arrays.sort(expectedD);
    Spectra.sort(fs, N, (a, b) -> a < b);
    Spectra.sort(ds, N, (a, b) -> a < b);
    assertArrayEquals(expectedF, fs, 0f);
    assertArrayEquals(expectedD, ds, 0.0);
  }

  @Test
  public void sortsByAKeyArrayTheComparatorCaptures() {
      int[] keys = shuffled(4);
      int[] order = new int[N];
      for (int i = 0; i < N; i++) order[i] = i;
      int[] expected = new int[N];
      for (int i = 0; i < N; i++) expected[keys[i]] = i;
      Spectra.sort(order, N, (a, b) -> keys[a] < keys[b]);
      assertArrayEquals(expected, order);
  }

  @Test
  public void transformsWithACapturedScalar() {
    int[] in = ints(5);
    int[] out = new int[N];
    int k = 7;
    long offset = 3;
    Spectra.transform(in, out, N, x -> x * k + (int) offset);
    int[] expected = new int[N];
    for (int i = 0; i < N; i++) expected[i] = in[i] * k + (int) offset;
    assertArrayEquals(expected, out);
  }

  @Test
  public void reducesWithACombiner() {
    int[] xs = ints(6);
    int expected = 0;
    for (int x : xs) expected += x;
    assertEquals(expected, Spectra.reduce(xs, N, 0, (a, b) -> a + b));
  }

  @Test
  public void transformReduceComputesADotProduct() {
      int[] xs = ints(7);
      int[] ys = ints(8);
      long expected = 0;
      for (int i = 0; i < N; i++) expected += (long) xs[i] * ys[i];
      long actual = Offload.transform_reduce(N, 0L, i -> (long) xs[(int) i] * ys[(int) i], (a, b) -> a + b);
      assertEquals(expected, actual);
  }

  @Test
  public void transformReduceMapsAndCombinesDoubles() {
      int[] xs = ints(9);
      double scale = 0.5;
      double expected = 0;
      for (int i = 0; i < N; i++) expected = Math.max(expected, Math.abs(xs[i] * scale));
      double actual =
          Offload.transform_reduce(
              N, 0.0, i -> Math.abs(xs[(int) i] * scale), (a, b) -> a > b ? a : b);
      assertEquals(expected, actual, 0.0);
  }

  @Test
  public void forEachWritesCapturedArrays() {
      int[] in = ints(10);
      int[] out = new int[N];
      Offload.for_each(N, i -> out[(int) i] = in[(int) i] * 2 + 1);
      int[] expected = new int[N];
      for (int i = 0; i < N; i++) expected[i] = in[i] * 2 + 1;
      assertArrayEquals(expected, out);
  }

  @Test
  public void aTaskIsAForEachOfOne() {
      int[] cell = new int[1];
      int v = 41;
      Offload.for_each(1, i -> cell[0] = v + 1);
      assertEquals(42, cell[0]);
  }

  @Test
  public void anExplicitContextSelectsTheDevice() {
      int[] in = ints(11);
      int[] out = new int[N];
      Offload.for_each(Context.current(), N, i -> out[(int) i] = in[(int) i] - 1);
      int[] expected = new int[N];
      for (int i = 0; i < N; i++) expected[i] = in[i] - 1;
      assertArrayEquals(expected, out);
  }

  @Test
  public void aGpuPlatformSeesTheLaunch() {
    Context context = Context.current();
    long before = context.stats().launches;
    int[] xs = ints(20);
    int[] expected = xs.clone();
    Arrays.sort(expected);
    Spectra.sort(xs, N, (a, b) -> a < b);
    assertArrayEquals(expected, xs);
    long launched = context.stats().launches - before;
    if (onGpu()) assertTrue("the sort ran no kernel", launched > 0);
  }

  @Test
  public void repeatedLaunchesReuseTheSite() {
    for (int round = 0; round < 3; round++) {
      int[] xs = ints(12 + round);
      int[] expected = xs.clone();
      Arrays.sort(expected);
      Spectra.sort(xs, N, (a, b) -> a < b);
      assertArrayEquals(expected, xs);
    }
  }

  static final class FieldInitialisers {
    static final int[] DATA = ints(30);
    static final int TOTAL = Spectra.reduce(DATA, N, 0, (a, b) -> a + b);
    static final int[] SORTED;
    static {
      SORTED = DATA.clone();
      Spectra.sort(SORTED, N, (a, b) -> a < b);
    }
  }

  @Test
  public void staticInitialisersLaunch() {
    assertEquals(sum(FieldInitialisers.DATA), FieldInitialisers.TOTAL);
    assertArrayEquals(sorted(FieldInitialisers.DATA), FieldInitialisers.SORTED);
  }

  static final class InstanceInitialisers {
    final int[] data = ints(31);
    final int total = Spectra.reduce(data, N, 0, (a, b) -> a + b);
    final int[] sorted;
    {
      sorted = data.clone();
      Spectra.sort(sorted, N, (a, b) -> a < b);
    }
  }

  @Test
  public void instanceInitialisersLaunch() {
    InstanceInitialisers holder = new InstanceInitialisers();
    assertEquals(sum(holder.data), holder.total);
    assertArrayEquals(sorted(holder.data), holder.sorted);
  }

  static final class MemberClass {
    static final int[] DATA = ints(32);
    static final int TOTAL = Spectra.reduce(DATA, N, 0, (a, b) -> a + b);

    static int total(int[] xs) {
      return Spectra.reduce(xs, N, 0, (a, b) -> a + b);
    }
  }

  final class InnerClass {
    final int[] data = ints(33);
    final int total = Spectra.reduce(data, N, 0, (a, b) -> a + b);

    int totalOf(int[] xs) {
      return Spectra.reduce(xs, N, 0, (a, b) -> a + b);
    }
  }

  @Test
  public void memberClassesLaunch() {
    assertEquals(sum(MemberClass.DATA), MemberClass.TOTAL);
    assertEquals(sum(ints(34)), MemberClass.total(ints(34)));
  }

  @Test
  public void innerClassesLaunch() {
    InnerClass inner = new InnerClass();
    assertEquals(sum(inner.data), inner.total);
    assertEquals(sum(ints(35)), inner.totalOf(ints(35)));
  }

  @Test
  public void anonymousClassesLaunch() {
    int[] xs = ints(36);
    IntSupplier supplier =
        new IntSupplier() {
          @Override
          public int getAsInt() {
            return Spectra.reduce(xs, N, 0, (a, b) -> a + b);
          }
        };
    assertEquals(sum(xs), supplier.getAsInt());
  }

  @Test
  public void localClassesLaunch() {
    int[] xs = ints(37);
    class Local {
      int total() {
        return Spectra.reduce(xs, N, 0, (a, b) -> a + b);
      }
    }
    assertEquals(sum(xs), new Local().total());
  }

  @Test
  public void liftedLambdasLaunch() {
    int[] xs = ints(38);
    IntSupplier supplier = () -> Spectra.reduce(xs, N, 0, (a, b) -> a + b);
    assertEquals(sum(xs), supplier.getAsInt());
  }

  private static boolean onGpu() {
    String backend = System.getProperty("polyregion.platform", "").split("[@!]", 2)[0];
    return !backend.isEmpty() && !backend.equalsIgnoreCase("host");
  }

  @Test
  public void copyNeedsNoCallable() {
    int[] in = ints(39);
    int[] out = new int[N];
    long before = Context.current().stats().launches;
    Spectra.copy(in, N, out);
    assertArrayEquals(in, out);
    if (onGpu())
      assertTrue("the copy ran no kernel", Context.current().stats().launches - before > 0);
  }

  @Test
  public void fillNeedsNoCallable() {
    int[] out = new int[N];
    long before = Context.current().stats().launches;
    Spectra.fill(out, N, 7);
    int[] expected = new int[N];
    Arrays.fill(expected, 7);
    assertArrayEquals(expected, out);
    if (onGpu())
      assertTrue("the fill ran no kernel", Context.current().stats().launches - before > 0);
  }

  @Test
  public void aStaticMethodReferenceIsACallable() {
    int[] xs = ints(40);
    assertEquals(sum(xs), Spectra.reduce(xs, N, 0, Integer::sum));
  }

  private static int plus(int a, int b) {
    return a + b;
  }

  @Test
  public void aStaticMethodReferenceToASourceMethodIsACallable() {
    int[] xs = ints(41);
    assertEquals(sum(xs), Spectra.reduce(xs, N, 0, SpectraLaunchTest::plus));
  }

  private static void assertRejectedInRegion(String call, OffloadFunction<Integer> region) {
    try {
      offload(region);
    } catch (UnsupportedOperationException e) {
      String message = String.valueOf(e.getMessage());
      assertTrue("the diagnostic does not name `" + call + "`: " + message, message.contains(call));
      assertTrue("the diagnostic does not name the region: " + message, message.contains("offload region"));
      return;
    }
    fail("a launch site inside an offload region must be refused");
  }

  @Test
  public void anInterfaceCallInsideARegionIsRefused() {
    int[] xs = {3, 1, 2};
    assertRejectedInRegion(
        "polyregion.Spectra.sort",
        () -> {
            Spectra.sort(xs, 3, (a, b) -> a < b);
            return xs[0];
        });
  }

  @Test
  public void aRegionApiCallInsideARegionIsRefused() {
    int[] xs = new int[4];
    assertRejectedInRegion(
        "polyregion.Offload.for_each",
        () -> {
            Offload.for_each(4, i -> xs[(int) i] = 1);
            return xs[0];
        });
  }
}
