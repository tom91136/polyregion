package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.stream.IntStream;
import org.junit.Test;
import polyregion.jvm.runtime.Context;

public class StreamReceiverTest {

  private static final int N = 1000;

  private final int bias = 3;

  private final int[] data = new int[N];

  @Test
  public void fallsBackOnAMapThatReadsAField() {
    Context context = Context.current();
    long before = context.stats().launches;
    int actual = IntStream.range(0, N).parallel().map(i -> i + bias).sum();
    assertEquals(IntStream.range(0, N).map(i -> i + bias).sum(), actual);
    assertEquals("a lambda that reads a field stays on the JDK", before, context.stats().launches);
  }

  @Test
  public void fallsBackOnAPredicateAndAForEachThatReadAField() {
    Context context = Context.current();
    long before = context.stats().launches;
    long kept = IntStream.range(0, N).parallel().filter(i -> i % bias == 0).count();
    assertEquals(IntStream.range(0, N).filter(i -> i % bias == 0).count(), kept);
    IntStream.range(0, N).parallel().forEach(i -> data[i] = i * bias);
    assertArrayEquals(IntStream.range(0, N).map(i -> i * bias).toArray(), data);
    assertEquals(before, context.stats().launches);
    Arrays.fill(data, 0);
  }
}
