package polyregion;

import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;

public class MathUlpTest {

  private static void assertWithinUlps(String what, int maxUlps, OffloadFunction<Double> f) {
    double expected = f.run();
    double actual = offload(f);
    if (Double.compare(expected, actual) == 0) return;
    long diff = Math.abs(Double.doubleToLongBits(expected) - Double.doubleToLongBits(actual));
    assertEquals(
        what + " differed by " + diff + " ulp (max " + maxUlps + ")", true, diff <= maxUlps);
  }

  @Test
  public void hypot() {
    double a = 3.0;
    double b = 4.0;
    assertWithinUlps("hypot", 2, () -> Math.hypot(a, b));
  }

  @Test
  public void cbrt() {
    double a = 27.5;
    assertWithinUlps("cbrt", 2, () -> Math.cbrt(a));
  }

  @Test
  public void expm1() {
    double a = 1e-8;
    assertWithinUlps("expm1", 2, () -> Math.expm1(a));
  }

  @Test
  public void log1p() {
    double a = 1e-8;
    assertWithinUlps("log1p", 2, () -> Math.log1p(a));
  }

  @Test
  public void atan2() {
    double a = 1.0;
    double b = 2.0;
    assertWithinUlps("atan2", 2, () -> Math.atan2(a, b));
  }

  @Test
  public void tan() {
    double a = 1.23;
    assertWithinUlps("tan", 2, () -> Math.tan(a));
  }
}
