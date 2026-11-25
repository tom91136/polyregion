package polyregion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;
import polyregion.javalang.Runtime;

public class ExceptionClassTest extends BaseSuite {

  static class Boom extends IllegalStateException {
    Boom(String message) {
      super(message);
    }
  }

  private static <E extends Throwable> E trap(Class<E> expected, OffloadRunnable f) {
    try {
      Runtime.offload(f);
    } catch (Throwable t) {
      if (expected.isInstance(t)) return expected.cast(t);
      fail("Expected " + expected.getName() + ", got " + t);
    }
    fail("Expected the region to raise " + expected.getName());
    return null;
  }

  @Test
  public void illegalStateThrowKeepsClassAndMessage() {
    IllegalStateException e = trap(IllegalStateException.class, () -> { throw new IllegalStateException("x"); });
    assertEquals("x", e.getMessage());
  }

  @Test
  public void userSubclassThrowKeepsClassAndMessage() {
    Boom e = trap(Boom.class, () -> { throw new Boom("boom"); });
    assertEquals("boom", e.getMessage());
  }

  @Test
  public void arrayIndexThrowKeepsClassAndMessage() {
    ArrayIndexOutOfBoundsException e =
        trap(ArrayIndexOutOfBoundsException.class, () -> { throw new ArrayIndexOutOfBoundsException("idx"); });
    assertEquals("idx", e.getMessage());
  }

  @Test
  public void outOfRangeArrayAccessRaisesArrayIndexOutOfBoundsException() {
    ArrayIndexOutOfBoundsException e = trap(ArrayIndexOutOfBoundsException.class, () -> {
      int[] xs = new int[2];
      int ignored = xs[2];
      if (ignored == 0) ignored++;
    });
    assertEquals("array index out of bounds", e.getMessage());
  }

  @Test
  public void divideByZeroRaisesArithmeticException() {
    int zero = 0;
    int one = 1;
    ArithmeticException e = trap(ArithmeticException.class, () -> {
      int ignored = one / zero;
      if (ignored == 0) ignored++;
    });
    assertEquals("/ by zero", e.getMessage());
  }

  @Test
  public void bmpMessageSurvivesTheJvmBoundary() {
    String expected = "caf\u00e9 \u2603";
    IllegalStateException e =
        trap(IllegalStateException.class, () -> { throw new IllegalStateException("caf\u00e9 \u2603"); });
    assertEquals(expected, e.getMessage());
  }

  @Test
  public void nonBmpMessageSurvivesTheJvmBoundary() {
    String expected = "smile \ud83d\ude00";
    IllegalStateException e =
        trap(IllegalStateException.class, () -> { throw new IllegalStateException("smile \ud83d\ude00"); });
    assertEquals(expected, e.getMessage());
  }

  @Test
  public void nullThrowMessageIsRefused() {
    int a = 1;
    assertRejected(
        "`throw`",
        () -> {
          if (a < 0) throw new IllegalStateException((String) null);
          return a;
        });
  }
}
