package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class CaptureTest {

  int field = 7;

  @Test
  public void capturesInstanceFieldThroughLocal() {
    int f = field;
    assertEquals(14, (int) offload(() -> f * 2));
  }

  @Test
  public void reoffloadSeesHostMutation() {
    int[] xs = {1, 2, 3};
    OffloadRunnable bump =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = xs[i] + 1;
        };
    offload(bump);
    assertArrayEquals(new int[] {2, 3, 4}, xs);
    xs[0] = 100;
    offload(bump);
    assertArrayEquals(new int[] {101, 4, 5}, xs);
  }

  @Test
  public void repeatedOffloadAccumulates() {
    int[] acc = new int[1];
    OffloadRunnable inc = () -> acc[0] = acc[0] + 1;
    for (int i = 0; i < 5; i++) offload(inc);
    assertArrayEquals(new int[] {5}, acc);
  }

  @Test
  public void sameArrayCapturedTwice() {
    int[] xs = {1, 2, 3};
    int[] out = new int[3];
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) out[i] = xs[i] + xs[i];
        };
    offload(r);
    assertArrayEquals(new int[] {2, 4, 6}, out);
  }

  @Test
  public void distinctArraysOfEqualContent() {
    int[] a = {1, 2, 3};
    int[] b = {1, 2, 3};
    int[] out = new int[3];
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) {
            b[i] = 9;
            out[i] = a[i];
          }
        };
    offload(r);
    assertArrayEquals(new int[] {1, 2, 3}, out);
    assertArrayEquals(new int[] {9, 9, 9}, b);
  }

  @Test
  public void captureOrderIsPreserved() {
    int a = 1;
    long b = 2L;
    double c = 3.0;
    int[] d = {4};
    float e = 5.0f;
    assertEquals(15.0, (double) offload(() -> a + b + c + d[0] + e), 0.0);
  }

  @Test
  public void twoRegionsShareOneArray() {
    int[] xs = {1, 2, 3};
    offload((OffloadRunnable) () -> xs[0] = 10);
    offload((OffloadRunnable) () -> xs[1] = xs[0] * 2);
    assertArrayEquals(new int[] {10, 20, 3}, xs);
  }
}
