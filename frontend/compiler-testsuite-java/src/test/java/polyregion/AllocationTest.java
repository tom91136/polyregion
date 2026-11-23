package polyregion;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class AllocationTest extends BaseSuite {

  @Test
  public void allocatesFixedSize() {
    assertMatchesJdk(
        () -> {
          int[] scratch = new int[4];
          for (int i = 0; i < 4; i++) scratch[i] = i * i;
          int acc = 0;
          for (int i = 0; i < 4; i++) acc += scratch[i];
          return acc;
        });
  }

  @Test
  public void allocatesFromCapturedLength() {
    int n = 6;
    assertMatchesJdk(
        () -> {
          int[] scratch = new int[n];
          for (int i = 0; i < n; i++) scratch[i] = i;
          int acc = 0;
          for (int i = 0; i < n; i++) acc += scratch[i];
          return acc;
        });
  }

  @Test
  public void allocatesWithInitialiser() {
    assertMatchesJdk(
        () -> {
          int[] xs = {5, 10, 15};
          return xs[0] + xs[1] + xs[2];
        });
  }

  @Test
  public void allocatesDouble() {
    assertMatchesJdk(
        () -> {
          double[] xs = new double[3];
          xs[0] = 1.5;
          xs[1] = 2.5;
          xs[2] = xs[0] + xs[1];
          return xs[2];
        });
  }

  @Test
  public void localArrayLengthIsKnown() {
    assertMatchesJdk(
        () -> {
          int[] xs = new int[7];
          return xs.length;
        });
  }

  @Test
  public void freshArrayIsZeroed() {
    assertMatchesJdk(
        () -> {
          int[] xs = new int[4];
          return xs[0] + xs[1] + xs[2] + xs[3];
        });
  }

  @Test
  public void freshDoubleArrayIsZeroed() {
    assertMatchesJdk(
        () -> {
          double[] xs = new double[3];
          return xs[0] + xs[2];
        });
  }

  @Test
  public void zeroedFromCapturedLength() {
    int n = 5;
    assertMatchesJdk(
        () -> {
          int[] xs = new int[n];
          int acc = 0;
          for (int i = 0; i < n; i++) acc += xs[i];
          return acc;
        });
  }

  @Test
  public void allocatesZeroLengthFromCapturedLength() {
    int n = 0;
    assertMatchesJdk(
        () -> {
          int[] scratch = new int[n];
          int acc = scratch.length;
          for (int i = 0; i < n; i++) acc += scratch[i];
          return acc;
        });
  }

  @Test
  public void copiesIntoCapturedArray() {
    int[] out = new int[3];
    OffloadRunnable r =
        () -> {
          int[] scratch = {7, 8, 9};
          for (int i = 0; i < 3; i++) out[i] = scratch[i];
        };
    polyregion.javalang.Runtime.offload(r);
    assertArrayEquals(new int[] {7, 8, 9}, out);
  }
}
