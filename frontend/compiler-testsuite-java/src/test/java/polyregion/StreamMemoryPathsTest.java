package polyregion;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;

public class StreamMemoryPathsTest {

  private static final long SPAN = Long.getLong("polyregion.stream.span", 1L << 20);
  private static final String PATH = System.getProperty("polyregion.stream.path", "all");
  private static final long STEP = 1_000_003;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  @Test
  public void materialisingPathsStayBounded() {
    switch (PATH) {
      case "averageLong":
        averageLong();
        break;
      case "averageInt":
        averageInt();
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
        averageLong();
        averageInt();
        toArray();
        sorted();
        reduce();
        reduceId();
        output();
    }  }

  private static long cycles() {
    return (SPAN - 1) / STEP + 1;
  }

  private static long last() {
    return (cycles() - 1) * STEP;
  }

  private static void averageLong() {
    assertEquals(
        OptionalDouble.of((double) last() / 2),
        LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).average());
  }

  private static void averageInt() {
    assertEquals(3, Arrays.stream(new int[] {1, 2}).parallel().sum());
    int step = (int) Math.min(STEP, Math.max(1, SPAN / 4));
    long limit = Math.min(SPAN, Integer.MAX_VALUE - 1L);
    long count = (limit - 1) / step + 1;
    double expected = ((count - 1) * step) / 2.0;
    assertEquals(
        OptionalDouble.of((double) expected),
        IntStream.range(0, (int) limit).parallel().filter(x -> x % step == 0).average());
  }

  private static long[] filtered() {
    long[] xs = LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).toArray();
    assertEquals(cycles(), xs.length);
    for (int i = 0; i < xs.length; i++) assertEquals(i * STEP, xs[i]);
    return xs;
  }

  private static void toArray() {
    filtered();
  }

  private static void sorted() {
    long[] xs = LongStream.range(0, SPAN).parallel().filter(x -> x % STEP == 0).sorted().toArray();
    assertEquals(cycles(), xs.length);
    for (int i = 0; i < xs.length; i++) assertEquals(i * STEP, xs[i]);
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
    for (int i = 0; i < xs.length; i++) assertEquals(i, xs[i]);
  }
}
