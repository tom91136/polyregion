package polyregion.javalang;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.OptionalLong;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.LaunchRule;

public class StreamChunkTest {

  private static final int N = 5000;

  private static final long CHUNK = 1L << 10;

  private static final long BASE = (1L << 32) + 7;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  private long savedIndex;
  private long savedArray;

  @Before
  public void shrinkChunks() {
    savedIndex = Streams.indexChunk;
    savedArray = Streams.arrayChunk;
    Streams.indexChunk = CHUNK;
    Streams.arrayChunk = CHUNK;
  }

  @After
  public void restoreChunks() {
    Streams.indexChunk = savedIndex;
    Streams.arrayChunk = savedArray;
  }

  private static void assertEveryHit(int[] hits) {
    for (int i = 0; i < hits.length; i++) assertEquals("element " + i, 1, hits[i]);
  }

  @Test
  public void sumsLongRangesBeyondIntAcrossChunks() {
    assertEquals(
        LongStream.range(BASE, BASE + N).sum(), LongStream.range(BASE, BASE + N).parallel().sum());
  }

  @Test
  public void countsEveryLongRangeElementAcrossChunks() {
    assertEquals(N, LongStream.range(BASE, BASE + N).parallel().filter(x -> true).count());
  }

  @Test
  public void visitsEveryLongRangeElementOnceAcrossChunks() {
    int[] hits = new int[N];
    LongStream.range(BASE, BASE + N).parallel().map(x -> x - BASE).forEach(x -> hits[(int) x]++);
    assertEveryHit(hits);
  }

  @Test
  public void filtersLongRangesAcrossChunks() {
    assertArrayEquals(
        LongStream.range(BASE, BASE + N).filter(x -> x % 3 != 0).toArray(),
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x % 3 != 0).toArray());
  }

  @Test
  public void filtersLongRangesAtTheEdgesAcrossChunks() {
    assertArrayEquals(
        new long[] {BASE, BASE + 1, BASE + 2},
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x < BASE + 3).toArray());
    assertArrayEquals(
        new long[] {BASE + N - 2, BASE + N - 1},
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x >= BASE + N - 2).toArray());
    assertEquals(
        OptionalLong.of(BASE + N - 1),
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x > BASE + N - 3).reduce((a, b) -> b));
    assertArrayEquals(
        LongStream.range(BASE, BASE + N)
            .filter(x -> x <= BASE + 2 || (x >= BASE + CHUNK - 1 && x <= BASE + CHUNK + 1))
            .map(x -> x - BASE)
            .filter(x -> x % 2 == 0)
            .toArray(),
        LongStream.range(BASE, BASE + N)
            .parallel()
            .filter(x -> x <= BASE + 2 || (x >= BASE + CHUNK - 1 && x <= BASE + CHUNK + 1))
            .map(x -> x - BASE)
            .filter(x -> x % 2 == 0)
            .toArray());
  }

  @Test
  public void reducesLongRangesAcrossChunks() {
    assertEquals(
        LongStream.range(BASE, BASE + N).reduce((a, b) -> a + b),
        LongStream.range(BASE, BASE + N).parallel().reduce((a, b) -> a + b));
  }

  @Test
  public void reducesFilteredLongRangesAcrossChunks() {
    assertEquals(
        LongStream.range(BASE, BASE + N).filter(x -> x <= BASE + CHUNK).max(),
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x <= BASE + CHUNK).max());
    assertEquals(
        LongStream.range(BASE, BASE + N).filter(x -> x >= BASE + CHUNK).min(),
        LongStream.range(BASE, BASE + N).parallel().filter(x -> x >= BASE + CHUNK).min());
  }

  @Test
  public void matchesLongRangesAcrossChunks() {
    assertTrue(LongStream.range(BASE, BASE + N).parallel().anyMatch(x -> x == BASE + CHUNK));
    assertTrue(LongStream.range(BASE, BASE + N).parallel().filter(x -> true).anyMatch(x -> x == BASE + N - 1));
    assertTrue(LongStream.range(BASE, BASE + N).parallel().filter(x -> true).anyMatch(x -> x >= BASE));
    assertFalse(LongStream.range(BASE, BASE + N).parallel().noneMatch(x -> x == BASE + CHUNK));
    assertTrue(LongStream.range(BASE, BASE + N).parallel().noneMatch(x -> x >= BASE + N));
    assertTrue(LongStream.range(BASE, BASE + N).parallel().allMatch(x -> x >= BASE));
    assertFalse(LongStream.range(BASE, BASE + N).parallel().allMatch(x -> x != BASE + CHUNK));
  }

  @Test
  public void countsIntRangesAtTheLowerEdgeAcrossChunks() {
    assertEquals(
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + N).filter(i -> i > Integer.MIN_VALUE + 5).count(),
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + N)
            .parallel()
            .filter(i -> i > Integer.MIN_VALUE + 5)
            .count());
  }

  @Test
  public void countsIntRangesAtTheUpperEdgeAcrossChunks() {
    assertEquals(
        N,
        IntStream.rangeClosed(Integer.MAX_VALUE - N + 1, Integer.MAX_VALUE).parallel().filter(i -> true).count());
  }

  @Test
  public void coversTheEdgesOfTheLongTypesAcrossChunks() {
    assertArrayEquals(
        LongStream.rangeClosed(Long.MAX_VALUE - N + 1, Long.MAX_VALUE).toArray(),
        LongStream.rangeClosed(Long.MAX_VALUE - N + 1, Long.MAX_VALUE).parallel().filter(x -> true).toArray());
    assertEquals(
        N, LongStream.rangeClosed(Long.MAX_VALUE - N + 1, Long.MAX_VALUE).parallel().filter(x -> true).count());
    assertArrayEquals(
        LongStream.range(Long.MIN_VALUE, Long.MIN_VALUE + N).toArray(),
        LongStream.range(Long.MIN_VALUE, Long.MIN_VALUE + N).parallel().filter(x -> true).toArray());
  }

  @Test
  public void coversTheEdgesOfTheIntTypesAcrossChunks() {
    assertArrayEquals(
        IntStream.rangeClosed(Integer.MAX_VALUE - N + 1, Integer.MAX_VALUE).map(i -> i - 1).toArray(),
        IntStream.rangeClosed(Integer.MAX_VALUE - N + 1, Integer.MAX_VALUE)
            .parallel()
            .map(i -> i - 1)
            .filter(i -> true)
            .toArray());
    assertArrayEquals(
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + N).toArray(),
        IntStream.range(Integer.MIN_VALUE, Integer.MIN_VALUE + N).parallel().filter(i -> true).toArray());
  }
}
