package polyregion;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.javalang.Par;

public class StreamSumTest {

  private static final double[] CANCELLING = {1e16, 1.0, 1.0, -1e16};
  private static final double[] TENTHS = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  private static final double[] LARGE_THEN_TENTHS = {1e16, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1e16};

  @Rule public final TestRule kernels = LaunchRule.skipOnGap();

  private static double magnitude(double[] terms) {
    double total = 0;
    for (double term : terms) total += Math.abs(term);
    return total;
  }

  private static double[] scattered(int n) {
    Random random = new Random(7);
    double[] terms = new double[n];
    for (int i = 0; i < n; i++) terms[i] = (random.nextDouble() - 0.5) * Math.pow(10, random.nextInt(12));
    return terms;
  }

  @Test
  public void oracleIsCompensated() {
    assertEquals(2.0, DoubleStream.of(CANCELLING).sum(), 0.0);
    assertEquals(0.0, Arrays.stream(CANCELLING).reduce(0.0, Double::sum), 0.0);
    assertEquals(1.0, DoubleStream.of(TENTHS).sum(), 0.0);
  }

  @Test
  public void sumsCancellingTermsLikeTheJdk() {
    double[] terms = CANCELLING.clone();
    double expected = DoubleStream.of(terms).sum();
    assertEquals(expected, Arrays.stream(CANCELLING).parallel().sum(), 0.0);
    assertEquals(expected, Par.doubles(CANCELLING).sum(), 0.0);
    assertEquals(expected, Arrays.stream(CANCELLING).parallel().map(x -> x).sum(), 0.0);
    assertEquals(expected, Par.doubles(CANCELLING).map(x -> x).sum(), 0.0);
    assertEquals(expected, IntStream.range(0, 4).parallel().mapToDouble(i -> terms[i]).sum(), 0.0);
    assertEquals(expected, Par.range(0, 4).mapToDouble(i -> terms[i]).sum(), 0.0);
  }

  @Test
  public void sumsTenthsLikeTheJdk() {
    double expected = DoubleStream.of(TENTHS).sum();
    assertEquals(expected, Arrays.stream(TENTHS).parallel().sum(), 0.0);
    assertEquals(expected, Par.doubles(TENTHS).sum(), 0.0);
    double bracketed = DoubleStream.of(LARGE_THEN_TENTHS).sum();
    assertEquals(bracketed, Arrays.stream(LARGE_THEN_TENTHS).parallel().sum(), 0.0);
    assertEquals(bracketed, Par.doubles(LARGE_THEN_TENTHS).sum(), 0.0);
  }

  @Test
  public void sumsFilteredTermsLikeTheJdk() {
    double expected = DoubleStream.of(CANCELLING).filter(x -> x == x).sum();
    assertEquals(expected, Arrays.stream(CANCELLING).parallel().filter(x -> x == x).sum(), 0.0);
    assertEquals(expected, Par.doubles(CANCELLING).filter(x -> x == x).sum(), 0.0);
    double dropped = DoubleStream.of(CANCELLING).filter(x -> x != 1.0).sum();
    assertEquals(dropped, Arrays.stream(CANCELLING).parallel().filter(x -> x != 1.0).sum(), 0.0);
    assertEquals(dropped, Par.doubles(CANCELLING).filter(x -> x != 1.0).sum(), 0.0);
  }

  @Test
  public void averagesLikeTheJdk() {
    OptionalDouble expected = DoubleStream.of(CANCELLING).average();
    assertEquals(expected, Arrays.stream(CANCELLING).parallel().average());
    assertEquals(expected, Par.doubles(CANCELLING).average());
    OptionalDouble filtered = DoubleStream.of(CANCELLING).filter(x -> x == x).average();
    assertEquals(filtered, Arrays.stream(CANCELLING).parallel().filter(x -> x == x).average());
    assertEquals(filtered, Par.doubles(CANCELLING).filter(x -> x == x).average());
    OptionalDouble tenths = DoubleStream.of(TENTHS).average();
    assertEquals(tenths, Arrays.stream(TENTHS).parallel().average());
    assertEquals(tenths, Par.doubles(TENTHS).average());
  }

  @Test
  public void sumsNonFiniteTermsLikeTheJdk() {
    double[] infinite = {1.0, Double.POSITIVE_INFINITY, 2.0};
    assertEquals(DoubleStream.of(infinite).sum(), Arrays.stream(infinite).parallel().sum(), 0.0);
    double[] undefined = {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 1.0};
    assertEquals(Double.NaN, Arrays.stream(undefined).parallel().sum(), 0.0);
    assertEquals(Double.NaN, DoubleStream.of(undefined).sum(), 0.0);
    double[] nan = {1.0, Double.NaN};
    assertEquals(Double.NaN, Par.doubles(nan).sum(), 0.0);
  }

  @Test
  public void staysWithinOneUlpOfTheJdkAcrossChunks() {
    double[] terms = scattered(100_000);
    double expected = DoubleStream.of(terms).sum();
    double bound = Math.ulp(magnitude(terms));
    assertEquals(expected, Arrays.stream(terms).parallel().sum(), bound);
    assertEquals(expected, Par.doubles(terms).sum(), bound);
    assertEquals(expected, IntStream.range(0, terms.length).parallel().mapToDouble(i -> terms[i]).sum(), bound);
    double average = DoubleStream.of(terms).average().getAsDouble();
    assertEquals(average, Arrays.stream(terms).parallel().average().getAsDouble(), bound / terms.length);
    assertEquals(average, Par.doubles(terms).average().getAsDouble(), bound / terms.length);
  }

  @Test
  public void leavesIntegralSumsExact() {
    int[] xs = new Random(3).ints(100_000, -1000, 1000).toArray();
    assertEquals(Arrays.stream(xs).asLongStream().sum(), Arrays.stream(xs).parallel().asLongStream().sum());
    assertEquals(Arrays.stream(xs).average(), Arrays.stream(xs).parallel().average());
  }
}
