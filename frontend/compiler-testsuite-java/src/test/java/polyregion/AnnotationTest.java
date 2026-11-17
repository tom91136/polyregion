package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;

public class AnnotationTest {

  static int twice(int x) {
    return x * 2;
  }

  static int sumTo(int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) acc += i;
    return acc;
  }

  @Test
  public void capturedArithmetic() {
    int a = 3;
    int b = 4;
    int r = offload(() -> a * b + 1);
    assertEquals(13, r);
  }

  @Test
  public void mixedWidthCaptures() {
    long a = 7L;
    double b = 0.5;
    double r = offload(() -> a * b);
    assertEquals(3.5, r, 0.0);
  }

  @Test
  public void callsUserMethod() {
    int a = 21;
    int r = offload(() -> twice(a));
    assertEquals(42, r);
  }

  @Test
  public void transitiveCallWithLoop() {
    int n = 10;
    int r = offload(() -> sumTo(n));
    assertEquals(45, r);
  }

  @Test
  public void branches() {
    int a = 5;
    int r = offload(() -> a > 3 ? a - 3 : a + 3);
    assertEquals(2, r);
  }

  @Test
  public void writesCapturedArray() {
    int[] xs = new int[4];
    OffloadRunnable fill =
        () -> {
          for (int i = 0; i < 4; i++) xs[i] = i * 2;
        };
    offload(fill);
    assertArrayEquals(new int[] {0, 2, 4, 6}, xs);
  }

  @Test
  public void readsCapturedArray() {
    int[] xs = {1, 2, 3, 4};
    OffloadFunction<Integer> total =
        () -> {
          int acc = 0;
          for (int i = 0; i < 4; i++) acc += xs[i];
          return acc;
        };
    assertEquals(10, (int) offload(total));
  }
}
