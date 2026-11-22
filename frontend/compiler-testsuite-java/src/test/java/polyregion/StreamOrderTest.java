package polyregion;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.javalang.Par;

public class StreamOrderTest {

  private static final int N = 5000;

  @Rule public final TestRule kernels = LaunchRule.skipOnGap();

  @Test
  public void keepsEncounterOrderInReduce() {
    assertEquals(OptionalInt.of(99_999), IntStream.range(0, 100_000).parallel().reduce((a, b) -> b));
    assertEquals(OptionalInt.of(0), IntStream.range(0, 100_000).parallel().reduce((a, b) -> a));
    assertEquals(OptionalLong.of(99_999), LongStream.range(0, 100_000).parallel().reduce((a, b) -> b));
    assertEquals(OptionalLong.of(0), LongStream.range(0, 100_000).parallel().reduce((a, b) -> a));
    assertEquals(
        OptionalInt.of(49_998),
        IntStream.range(0, 100_000).parallel().map(i -> i).filter(i -> i % 2 == 0).filter(i -> i < 50_000).reduce((a, b) -> b));
    double[] ds = DoubleStream.iterate(0.5, d -> d + 1).limit(N).toArray();
    assertEquals(OptionalDouble.of(ds[N - 1]), Arrays.stream(ds).parallel().reduce((a, b) -> b));
    assertEquals(OptionalDouble.of(ds[0]), Arrays.stream(ds).parallel().reduce((a, b) -> a));
  }

  @Test
  public void keepsEncounterOrderInReduceOnThePar() {
    int[] xs = IntStream.range(0, 100_000).toArray();
    assertEquals(OptionalInt.of(99_999), Par.ints(xs).reduce((a, b) -> b));
    assertEquals(OptionalInt.of(0), Par.ints(xs).reduce((a, b) -> a));
  }
}
