package polyregion;

import org.junit.Test;

public class MathIntrinsicTest extends BaseSuite {

  @Test
  public void trigonometric() {
    double a = 1.23;
    assertMatchesJdk("sin", () -> Math.sin(a));
    assertMatchesJdk("cos", () -> Math.cos(a));
    assertMatchesJdk("asin", () -> Math.asin(0.5));
    assertMatchesJdk("acos", () -> Math.acos(0.5));
    assertMatchesJdk("sinh", () -> Math.sinh(a));
    assertMatchesJdk("cosh", () -> Math.cosh(a));
    assertMatchesJdk("tanh", () -> Math.tanh(a));
  }

  @Test
  public void rounding() {
    double a = 2.5;
    double b = -2.5;
    assertMatchesJdk("ceil", () -> Math.ceil(a));
    assertMatchesJdk("floor", () -> Math.floor(a));
    assertMatchesJdk("rint", () -> Math.rint(a));
    assertMatchesJdk("ceil-neg", () -> Math.ceil(b));
    assertMatchesJdk("floor-neg", () -> Math.floor(b));
  }

  @Test
  public void exponentialAndLog() {
    double a = 1.23;
    assertMatchesJdk("sqrt", () -> Math.sqrt(a));
    assertMatchesJdk("exp", () -> Math.exp(a));
    assertMatchesJdk("log", () -> Math.log(a));
    assertMatchesJdk("log10", () -> Math.log10(a));
    assertMatchesJdk("pow", () -> Math.pow(a, 3.0));
  }

  @Test
  public void absAcrossWidths() {
    int i = Integer.MIN_VALUE;
    long l = -42L;
    float f = -1.5f;
    double d = -2.5;
    assertMatchesJdk("abs-int-min", () -> Math.abs(i));
    assertMatchesJdk("abs-int", () -> Math.abs(-7));
    assertMatchesJdk("abs-long", () -> Math.abs(l));
    assertMatchesJdk("abs-float", () -> Math.abs(f));
    assertMatchesJdk("abs-double", () -> Math.abs(d));
  }

  @Test
  public void minMaxAcrossWidths() {
    int a = 3;
    int b = 9;
    long la = 3L;
    long lb = 9L;
    double da = 3.0;
    double db = 9.0;
    assertMatchesJdk("min-int", () -> Math.min(a, b));
    assertMatchesJdk("max-int", () -> Math.max(a, b));
    assertMatchesJdk("min-long", () -> Math.min(la, lb));
    assertMatchesJdk("max-long", () -> Math.max(la, lb));
    assertMatchesJdk("min-double", () -> Math.min(da, db));
    assertMatchesJdk("max-double", () -> Math.max(da, db));
  }

  @Test
  public void constantsFoldToLiterals() {
    assertMatchesJdk("pi", () -> Math.PI);
    assertMatchesJdk("e", () -> Math.E);
    assertMatchesJdk("pi-expr", () -> Math.PI * 2.0);
  }

  @Test
  public void widensIntArgumentToDouble() {
    int n = 9;
    assertMatchesJdk("sqrt-int", () -> Math.sqrt(n));
  }
}
