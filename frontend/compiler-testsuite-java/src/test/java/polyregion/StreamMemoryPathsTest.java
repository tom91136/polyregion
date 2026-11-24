package polyregion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.OptionalDouble;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.jvm.runtime.Context;

public class StreamMemoryPathsTest {

  private static final long SPAN = Long.getLong("polyregion.stream.span", 4_294_967_306L);
  private static final String PATH = System.getProperty("polyregion.stream.path", "all");
  private static final long STEP = 1_000_003;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  @Test
  public void materialisingPathsStayBounded() {
    long mirrorsBefore = Context.current().stats().mirrorAllocations;
    switch (PATH) {
      case "average":
        average();
        break;
      case "toArray":
        toArray();
        break;
      case "sorted":
        sorted();
        break;
      case "reduce":
        reduce();
        break;
      case "reduceId":
        reduceId();
        break;
      case "output":
        output();
        break;
      default:
        average();
        toArray();
        sorted();
        reduce();
        reduceId();
        output();
    }
    long mirrorsAfter = Context.current().stats().mirrorAllocations;
    assertTrue(
        PATH + " left " + (mirrorsAfter - mirrorsBefore) + " device mirrors",
        mirrorsAfter <= mirrorsBefore + 16);
  }

  private static long cycles() {
    return SPAN / STEP + 1;
  }

  private static long last() {
    return (cycles() - 1) * STEP;
  }

  private static void average() {
    assertEquals(
        OptionalDouble.of((double) last() / 2),
        LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).average());
    int stepInt = 1_000_003;
    long intCycles = (Integer.MAX_VALUE - 1L) / stepInt + 1;
    long intLast = (intCycles - 1) * stepInt;
    assertEquals(
        OptionalDouble.of((double) intLast / 2),
        IntStream.range(0, Integer.MAX_VALUE).parallel().filter(x -> x % stepInt == 0).average());
  }

  private static long[] filtered() {
    long[] xs = LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).toArray();
    assertEquals(cycles(), xs.length);
    assertEquals(last(), xs[xs.length - 1]);
    return xs;
  }

  private static void toArray() {
    filtered();
  }

  private static void sorted() {
    long[] xs =
        LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).sorted().toArray();
    assertEquals(cycles(), xs.length);
    assertEquals(last(), xs[xs.length - 1]);
  }

  private static long sum() {
    return SPAN % 2 == 0 ? (SPAN / 2) * (SPAN - 1) : SPAN * ((SPAN - 1) / 2);
  }

  private static void reduce() {
    assertEquals(sum(), LongStream.range(0, SPAN).parallel().reduce((a, b) -> a + b).getAsLong());
  }

  private static void reduceId() {
    assertEquals(sum(), LongStream.range(0, SPAN).parallel().reduce(0L, Long::sum));
  }

  private static void output() {
    int[] xs = IntStream.range(0, 1 << 20).parallel().toArray();
    assertEquals(1 << 20, xs.length);
    assertEquals((1 << 20) - 1, xs[xs.length - 1]);
  }
}
