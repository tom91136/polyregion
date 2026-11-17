package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class PrimitiveTypesTest {

  @Test
  public void byteArithmeticPromotesToInt() {
    byte a = 100;
    byte b = 100;
    assertEquals(200, (int) offload(() -> a + b));
  }

  @Test
  public void shortArithmetic() {
    short a = 300;
    short b = 12;
    assertEquals(312, (int) offload(() -> a + b));
  }

  @Test
  public void charArithmetic() {
    char c = 'A';
    assertEquals('C', (int) offload(() -> c + 2));
  }

  @Test
  public void charComparison() {
    char c = 'z';
    assertEquals(1, (int) offload(() -> c >= 'a' && c <= 'z' ? 1 : 0));
  }

  @Test
  public void longArithmetic() {
    long a = 1L << 40;
    assertEquals((1L << 40) + 7, (long) offload(() -> a + 7));
  }

  @Test
  public void floatArithmetic() {
    float a = 1.5f;
    float b = 2.25f;
    assertEquals(3.75f, (float) offload(() -> a + b), 0.0f);
  }

  @Test
  public void doubleArithmetic() {
    double a = 1.0;
    double b = 3.0;
    assertEquals(1.0 / 3.0, (double) offload(() -> a / b), 1e-15);
  }

  @Test
  public void intDivisionTruncatesTowardZero() {
    int a = -7;
    int b = 2;
    assertEquals(-3, (int) offload(() -> a / b));
  }

  @Test
  public void remainderKeepsSignOfDividend() {
    int a = -7;
    int b = 2;
    assertEquals(-1, (int) offload(() -> a % b));
  }

  @Test
  public void intOverflowWraps() {
    int a = Integer.MAX_VALUE;
    assertEquals(Integer.MIN_VALUE, (int) offload(() -> a + 1));
  }

  @Test
  public void wideningCast() {
    int a = 7;
    assertEquals(7.0, (double) offload(() -> (double) a), 0.0);
  }

  @Test
  public void narrowingCastTruncates() {
    double a = 3.99;
    assertEquals(3, (int) offload(() -> (int) a));
  }

  @Test
  public void narrowingToByteWraps() {
    int a = 300;
    assertEquals((byte) 300, (int) (byte) offload(() -> (byte) a));
  }

  @Test
  public void longToIntNarrowing() {
    long a = (1L << 32) + 5;
    assertEquals(5, (int) offload(() -> (int) a));
  }

  @Test
  public void mixedIntLongPromotion() {
    int a = 3;
    long b = 4L;
    assertEquals(7L, (long) offload(() -> a + b));
  }

  @Test
  public void mixedFloatDoublePromotion() {
    float a = 0.5f;
    double b = 0.25;
    assertEquals(0.75, (double) offload(() -> a + b), 0.0);
  }

  @Test
  public void bitwiseOps() {
    int a = 0b1100;
    int b = 0b1010;
    assertEquals(0b1000, (int) offload(() -> a & b));
    assertEquals(0b1110, (int) offload(() -> a | b));
    assertEquals(0b0110, (int) offload(() -> a ^ b));
    assertEquals(~0b1100, (int) offload(() -> ~a));
  }

  @Test
  public void shifts() {
    int a = -8;
    assertEquals(-16, (int) offload(() -> a << 1));
    assertEquals(-4, (int) offload(() -> a >> 1));
    assertEquals(-8 >>> 1, (int) offload(() -> a >>> 1));
  }

  @Test
  public void longShiftByInt() {
    long a = 1L;
    int s = 40;
    assertEquals(1L << 40, (long) offload(() -> a << s));
  }

  @Test
  public void booleanLogic() {
    boolean t = true;
    boolean f = false;
    assertEquals(Boolean.FALSE, offload(() -> t && f));
    assertEquals(Boolean.TRUE, offload(() -> t || f));
    assertEquals(Boolean.TRUE, offload(() -> t ^ f));
    assertEquals(Boolean.FALSE, offload(() -> !t));
  }

  @Test
  public void unaryMinusOnNarrowType() {
    byte a = 5;
    assertEquals(-5, (int) offload(() -> -a));
  }

  @Test
  public void incrementAndDecrement() {
    assertEquals(
        4,
        (int)
            offload(
                () -> {
                  int i = 5;
                  i--;
                  return i;
                }));
    assertEquals(
        6,
        (int)
            offload(
                () -> {
                  int i = 5;
                  ++i;
                  return i;
                }));
  }

  @Test
  public void compoundAssignmentNarrows() {
    assertEquals(
        100,
        (int)
            offload(
                () -> {
                  byte b = 50;
                  b += 50;
                  return b;
                }));
  }

  @Test
  public void doubleArrayRoundTrip() {
    double[] xs = {1.5, 2.5, 3.5};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i] = xs[i] * 2.0;
        };
    offload(r);
    assertArrayEquals(new double[] {3.0, 5.0, 7.0}, xs, 0.0);
  }

  @Test
  public void longArrayRoundTrip() {
    long[] xs = {1L, 2L, 3L};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i] = xs[i] << 32;
        };
    offload(r);
    assertArrayEquals(new long[] {1L << 32, 2L << 32, 3L << 32}, xs);
  }

  @Test
  public void byteArrayRoundTrip() {
    byte[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i] = (byte) (xs[i] + 1);
        };
    offload(r);
    assertArrayEquals(new byte[] {2, 3, 4}, xs);
  }

  @Test
  public void floatArrayRoundTrip() {
    float[] xs = {1.0f, 2.0f};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 2; i++) xs[i] = xs[i] / 4.0f;
        };
    offload(r);
    assertArrayEquals(new float[] {0.25f, 0.5f}, xs, 0.0f);
  }

  @Test
  public void booleanArrayRoundTrip() {
    boolean[] xs = {true, false, true};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i] = !xs[i];
        };
    offload(r);
    assertArrayEquals(new boolean[] {false, true, false}, new boolean[] {xs[0], xs[1], xs[2]});
  }
}
