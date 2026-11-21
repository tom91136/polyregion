package polyregion;

import org.junit.Test;

public class JvmIntrinsicTest extends BaseSuite {

  @Test
  public void roundIsHalfUp() {
    double a = 2.5;
    double b = -2.5;
    double c = 0.49999999999999994;
    float d = -0.5f;
    assertMatchesJdk("up", () -> Math.round(a));
    assertMatchesJdk("down", () -> Math.round(b));
    assertMatchesJdk("below-half", () -> Math.round(c));
    assertMatchesJdk("float", () -> Math.round(d));
    assertMatchesJdk("nan", () -> Math.round(Double.NaN));
    assertMatchesJdk("saturates", () -> Math.round(1e30));
  }

  @Test
  public void minMaxPropagateNan() {
    double nan = Double.NaN;
    double one = 1.0;
    float fnan = Float.NaN;
    assertMatchesJdk("min-nan-left", () -> Math.min(nan, one));
    assertMatchesJdk("min-nan-right", () -> Math.min(one, nan));
    assertMatchesJdk("max-nan-left", () -> Math.max(nan, one));
    assertMatchesJdk("max-float", () -> Math.max(fnan, 2.0f));
  }

  @Test
  public void minMaxOrderSignedZero() {
    double pos = 0.0;
    double neg = -0.0;
    assertMatchesJdk("min-zero", () -> Math.min(pos, neg));
    assertMatchesJdk("max-zero", () -> Math.max(neg, pos));
  }

  @Test
  public void integerAndLongBits() {
    int x = 0x0f0f0f0f;
    long y = 1L << 40;
    assertMatchesJdk("bitCount", () -> Integer.bitCount(x));
    assertMatchesJdk("nlz", () -> Long.numberOfLeadingZeros(y));
    assertMatchesJdk("ntz", () -> Long.numberOfTrailingZeros(y));
    assertMatchesJdk("rotate", () -> Integer.rotateLeft(x, 5));
    assertMatchesJdk("lowest", () -> Integer.lowestOneBit(x));
  }

  @Test
  public void floatingPredicates() {
    double nan = Double.NaN;
    double inf = Double.POSITIVE_INFINITY;
    assertMatchesJdk("isNaN", () -> Double.isNaN(nan));
    assertMatchesJdk("isInfinite", () -> Double.isInfinite(inf));
    assertMatchesJdk("bits", () -> Double.doubleToRawLongBits(1.5));
  }

  @Test
  public void floorDivAndModAcrossSigns() {
    int a = -7;
    int b = 3;
    assertMatchesJdk("div", () -> Math.floorDiv(a, b));
    assertMatchesJdk("mod", () -> Math.floorMod(a, b));
    assertMatchesJdk("mod-neg", () -> Math.floorMod(7, -b));
  }

  @Test
  public void mathWithoutAnEntryLowersFromBytecode() {
    double x = 1536.5;
    double y = -0.15625;
    assertMatchesJdk("getExponent", () -> Math.getExponent(x));
    assertMatchesJdk("getExponent-neg", () -> Math.getExponent(y));
    assertMatchesJdk("nextUp", () -> Math.nextUp(x));
    assertMatchesJdk("nextUp-neg", () -> Math.nextUp(y));
  }

  @Test
  public void missingIntrinsicsAreRefused() {
    double a = 1.5;
    assertRejected("unsupported JVM intrinsic java/lang/StrictMath.IEEEremainder(DD)D", () -> Math.IEEEremainder(a, a));
    assertRejected("native", () -> Math.IEEEremainder(a, a));
  }
}
