package polyregion;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.jvm.twin.TwinException;

public class ObjectCaptureTest extends BaseSuite {

  enum Colour {
    RED,
    GREEN,
    BLUE
  }

  record Vec(int x, int y) {
    int dot(Vec o) {
      return x * o.x + y * o.y;
    }
  }

  interface Op {
    int apply(int x);
  }

  static final class AddOne implements Op {
    @Override
    public int apply(int x) {
      return x + 1;
    }
  }

  static final class TimesTwo implements Op {
    @Override
    public int apply(int x) {
      return x * 2;
    }
  }

  static final class Negate implements Op {
    @Override
    public int apply(int x) {
      return -x;
    }
  }

  interface Scaled {
    int of(int x);
  }

  static final class ByField implements Scaled {
    final int k;

    ByField(int k) {
      this.k = k;
    }

    @Override
    public int of(int x) {
      return x * k;
    }
  }

  static final class Offset implements Scaled {
    final int lo;
    final int hi;

    Offset(int lo, int hi) {
      this.lo = lo;
      this.hi = hi;
    }

    @Override
    public int of(int x) {
      return x + lo * hi;
    }
  }

  abstract static class Shape {
    abstract int area(int n);
  }

  static final class Square extends Shape {
    @Override
    int area(int n) {
      return n * n;
    }
  }

  static final class Strip extends Shape {
    @Override
    int area(int n) {
      return n * 2;
    }
  }

  static int dispatch(Op op, int x) {
    return op.apply(x);
  }

  static int scale(Scaled s, int x) {
    return s.of(x);
  }

  static int area(Shape s, int n) {
    return s.area(n);
  }

  static <T> int identity(int x) {
    return x;
  }

  @Test
  public void boxedIntegerCapture() {
    Integer a = 7;
    assertMatchesJdk(() -> a + 1);
  }

  @Test
  public void boxedDoubleCapture() {
    Double a = 2.5;
    assertMatchesJdk(() -> a * 2);
  }

  @Test
  public void enumCapture() {
    Colour c = Colour.GREEN;
    assertMatchesJdk(() -> c == Colour.GREEN ? 1 : 0);
  }

  @Test
  public void enumOrdinal() {
    Colour c = Colour.BLUE;
    assertMatchesJdk(() -> c.ordinal());
  }

  @Test
  public void everyEnumConstantKeepsItsOrdinal() {
    assertMatchesJdk(
        "constants are numbered in declaration order",
        () -> Colour.RED.ordinal() * 100 + Colour.GREEN.ordinal() * 10 + Colour.BLUE.ordinal());
  }

  @Test
  public void enumCaptureComparedAgainstEveryConstant() {
    Colour c = Colour.GREEN;
    assertMatchesJdk(
        () ->
            (c == Colour.RED ? 1 : 0)
                + (c == Colour.GREEN ? 20 : 0)
                + (c == Colour.BLUE ? 300 : 0));
  }

  @Test
  public void enumInequality() {
    Colour c = Colour.RED;
    assertMatchesJdk(() -> c != Colour.BLUE ? 7 : 9);
  }

  @Test
  public void enumOrdinalDrivesALoop() {
    Colour c = Colour.BLUE;
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i <= c.ordinal(); i++) acc += i;
          return acc;
        });
  }

  @Test
  public void enumPassedToHelper() {
    Colour c = Colour.GREEN;
    assertMatchesJdk(() -> weigh(c));
  }

  static int weigh(Colour c) {
    return c == Colour.BLUE ? 5 : c.ordinal() + 1;
  }

  enum Weighted {
    LIGHT(1),
    HEAVY(9);

    final int kg;

    Weighted(int kg) {
      this.kg = kg;
    }
  }

  @Test
  public void rejectsEnumWithInstanceState() {
    Weighted w = Weighted.HEAVY;
    OffloadFunction<Integer> region = () -> w.ordinal();
    try {
      polyregion.javalang.Runtime.offload(region);
      org.junit.Assert.fail("expected a stateful enum to be rejected");
    } catch (UnsupportedOperationException e) {
      org.junit.Assert.assertTrue(e.getMessage(), e.getMessage().contains("carry state"));
    }
  }

  @Test
  public void recordCapture() {
    Vec v = new Vec(3, 4);
    assertMatchesJdk(() -> v.x() * v.y());
  }

  @Test
  public void recordMethodCall() {
    Vec a = new Vec(1, 2);
    Vec b = new Vec(3, 4);
    assertMatchesJdk(() -> a.dot(b));
  }

  @Test
  public void virtualDispatchThroughInterface() {
    Op op = new TimesTwo();
    assertMatchesJdk(() -> dispatch(op, 5));
  }

  @Test
  public void twoImplementationsOfOneInterface() {
    Op a = new AddOne();
    Op b = new TimesTwo();
    assertMatchesJdk(() -> dispatch(a, 3) + dispatch(b, 3));
  }

  @Test
  public void threeImplementationsOfOneInterface() {
    Op a = new AddOne();
    Op b = new TimesTwo();
    Op c = new Negate();
    assertMatchesJdk(() -> dispatch(a, 4) + dispatch(b, 4) + dispatch(c, 4));
  }

  @Test
  public void dispatchesOverImplementationsWithFields() {
    Scaled a = new ByField(3);
    Scaled b = new Offset(2, 5);
    assertMatchesJdk(() -> scale(a, 7) + scale(b, 7));
  }

  @Test
  public void dispatchThroughAbstractClass() {
    Shape a = new Square();
    Shape b = new Strip();
    assertMatchesJdk(() -> area(a, 6) - area(b, 6));
  }

  @Test
  public void virtualCallDirectlyInRegion() {
    Op op = new Negate();
    assertMatchesJdk(() -> op.apply(11));
  }

  @Test
  public void dispatchInsideLoop() {
    Op op = new TimesTwo();
    assertMatchesJdk(
        () -> {
          int total = 0;
          for (int i = 0; i < 4; i++) total += dispatch(op, i);
          return total;
        });
  }

  @Test
  public void rejectsImplementationTheRegionWasNotCompiledAgainst() {
    Op hidden =
        new Op() {
          @Override
          public int apply(int x) {
            return x - 1;
          }
        };
    OffloadFunction<Integer> region = () -> dispatch(hidden, 5);
    try {
      polyregion.javalang.Runtime.offload(region);
      org.junit.Assert.fail("expected the launch to reject an unknown implementation");
    } catch (TwinException e) {
      org.junit.Assert.assertTrue(
          e.getMessage(), e.getMessage().contains(hidden.getClass().getName()));
    }
  }

  @Test
  public void nullReferenceCapture() {
    int[] xs = null;
    assertMatchesJdk(() -> xs == null ? 1 : 0);
  }

  @Test
  public void nonNullReferenceCompare() {
    int[] xs = {1, 2, 3};
    assertMatchesJdk(() -> xs == null ? 1 : 0);
  }

  @Test
  public void notNullReferenceCompare() {
    int[] xs = {9};
    int[] ys = null;
    assertMatchesJdk(
        "a null and a non-null array must not compare alike",
        () -> (xs != null ? 2 : 0) + (ys != null ? 4 : 0));
  }

  @Test
  public void nullOnTheLeftOfCompare() {
    double[] xs = null;
    assertMatchesJdk(() -> null == xs ? 1 : 0);
  }

  @Test
  public void nullArrayStillReportsZeroLength() {
    int[] xs = null;
    assertMatchesJdk(() -> xs == null ? -1 : xs.length);
  }

  @Test
  public void emptyArrayReportsZeroLength() {
    int[] xs = new int[0];
    assertMatchesJdk(() -> xs.length);
  }

  @Test
  public void emptyArrayIsNotNull() {
    int[] xs = new int[0];
    assertMatchesJdk("an empty array is not a null one", () -> xs != null ? 1 : 0);
  }

  @Test
  public void genericHelperCall() {
    int a = 5;
    assertMatchesJdk(() -> ObjectCaptureTest.<String>identity(a));
  }
}
