package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import org.junit.Test;
import polyregion.javalang.Par;
import polyregion.jvm.runtime.Context;

public class StreamPairsTest {

  private static final int N = 4096;

  @Test
  public void sumOfSquaresOfTheEvens() {
    int[] xs = new Random(1).ints(N, -100, 100).toArray();
    int expected = Arrays.stream(xs).filter(x -> x % 2 == 0).map(x -> x * x).sum();
    assertEquals(
        expected, Arrays.stream(xs).parallel().filter(x -> x % 2 == 0).map(x -> x * x).sum());
    assertEquals(expected, Par.ints(xs).filter(x -> x % 2 == 0).map(x -> x * x).sum());
  }

  @Test
  public void integralOverARange() {
    double step = 1.0 / N;
    double expected = IntStream.range(0, N).mapToDouble(i -> 4.0 / (1.0 + (i + 0.5) * step * (i + 0.5) * step)).sum() * step;
    double jdk =
        IntStream.range(0, N)
                .parallel()
                .mapToDouble(i -> 4.0 / (1.0 + (i + 0.5) * step * (i + 0.5) * step))
                .sum()
            * step;
    double par = Par.range(0, N).mapToDouble(i -> 4.0 / (1.0 + (i + 0.5) * step * (i + 0.5) * step)).sum() * step;
    assertEquals(expected, jdk, 1e-9);
    assertEquals(expected, par, 1e-9);
  }

  @Test
  public void saxpyThroughForEach() {
    int[] xi = new Random(2).ints(N, 0, 64).toArray();
    int[] yi = new Random(4).ints(N, 0, 64).toArray();
    int a = 3;
    int[] expected = new int[N];
    IntStream.range(0, N).forEach(i -> expected[i] = a * xi[i] + yi[i]);
    int[] jdk = new int[N];
    IntStream.range(0, N).parallel().forEach(i -> jdk[i] = a * xi[i] + yi[i]);
    int[] par = new int[N];
    Par.range(0, N).forEach(i -> par[i] = a * xi[i] + yi[i]);
    assertArrayEquals(expected, jdk);
    assertArrayEquals(expected, par);
  }

  @Test
  public void setAllPrefixAndSortInPlace() {
    int[] seq = new int[N];
    Arrays.setAll(seq, i -> (i * 7919) % 1009);
    int[] jdk = new int[N];
    Arrays.parallelSetAll(jdk, i -> (i * 7919) % 1009);
    int[] par = new int[N];
    Par.setAll(par, i -> (i * 7919) % 1009);
    assertArrayEquals(seq, jdk);
    assertArrayEquals(seq, par);

    for (int i = 1; i < N; i++) seq[i] += seq[i - 1];
    Arrays.parallelPrefix(jdk, Integer::sum);
    Par.prefix(par, Integer::sum);
    assertArrayEquals(seq, jdk);
    assertArrayEquals(seq, par);

    Arrays.sort(seq);
    Arrays.parallelSort(jdk);
    Par.sort(par);
    assertArrayEquals(seq, jdk);
    assertArrayEquals(seq, par);
  }

  @Test
  public void explicitDeviceAndTheChainTheJdkKeeps() {
    int[] xs = new Random(3).ints(N, 0, 1000).toArray();
    Context context = Context.current();
    int expected = Arrays.stream(xs).map(x -> x + 1).sum();
    assertEquals(expected, Par.ints(xs).on(context).map(x -> x + 1).sum());
    long distinct = Arrays.stream(xs).parallel().distinct().count();
    assertEquals(Arrays.stream(xs).distinct().count(), distinct);
  }
}
