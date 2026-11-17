package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class StructTest extends BaseSuite {

  static final class Point {
    int x;
    int y;

    Point(int x, int y) {
      this.x = x;
      this.y = y;
    }
  }

  static final class Nested {
    Point a;
    Point b;

    Nested(Point a, Point b) {
      this.a = a;
      this.b = b;
    }
  }

  static final class Bag {
    Point[] ps;

    Bag(Point[] ps) {
      this.ps = ps;
    }
  }

  static final class Row {
    int[] xs;

    Row(int[] xs) {
      this.xs = xs;
    }
  }

  static final class Mixture {
    byte b;
    char c;
    short s;
    int i;
    long l;
    float f;
    double d;
    boolean z;
  }

  @Test
  public void readsFieldOfCapturedObject() {
    Point p = new Point(3, 4);
    assertMatchesJdk(() -> p.x + p.y);
  }

  @Test
  public void writesFieldOfCapturedObject() {
    Point p = new Point(1, 1);
    OffloadRunnable r = () -> p.x = 42;
    polyregion.javalang.Runtime.offload(r);
    assertEquals(42, p.x);
  }

  @Test
  public void readsNestedField() {
    Nested n = new Nested(new Point(1, 2), new Point(3, 4));
    assertMatchesJdk(() -> n.a.x + n.b.y);
  }

  @Test
  public void writesNestedField() {
    Point a = new Point(1, 2);
    Nested n = new Nested(a, new Point(3, 4));
    OffloadRunnable r = () -> n.a.x = n.b.y * 10;
    polyregion.javalang.Runtime.offload(r);
    assertEquals(40, n.a.x);
    assertEquals(40, a.x);
  }

  @Test
  public void readsEveryFieldWidth() {
    Mixture m = new Mixture();
    m.b = 1;
    m.c = 2;
    m.s = 3;
    m.i = 4;
    m.l = 5L;
    m.f = 6.0f;
    m.d = 7.0;
    m.z = true;
    assertMatchesJdk(() -> m.b + m.c + m.s + m.i + m.l + m.f + m.d + (m.z ? 1 : 0));
  }

  static final class Computed {
    int a;
    int b;

    Computed(int n) {
      a = n * 2;
      b = a + 1;
    }
  }

  static final class PartlyInitialised {
    int set;
    int left;

    PartlyInitialised(int n) {
      set = n;
    }
  }

  @Test
  public void allocatesObjectInRegion() {
    assertMatchesJdk(
        () -> {
          Point p = new Point(2, 5);
          return p.x * p.y;
        });
  }

  @Test
  public void allocatesTwoObjectsThatDoNotAlias() {
    assertMatchesJdk(
        () -> {
          Point a = new Point(1, 2);
          Point b = new Point(30, 40);
          a.x = 7;
          return a.x * 1000 + a.y * 100 + b.x + b.y;
        });
  }

  @Test
  public void allocatesInALoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i < 4; i++) {
            Point p = new Point(i, i + 1);
            acc += p.x * p.y;
          }
          return acc;
        });
  }

  @Test
  public void constructorBodyRuns() {
    assertMatchesJdk(
        () -> {
          Computed c = new Computed(5);
          return c.a * 100 + c.b;
        });
  }

  @Test
  public void unassignedFieldReadsBackAsZero() {
    assertMatchesJdk(
        "a field the constructor skips is cleared, not left as whatever was there",
        () -> {
          PartlyInitialised p = new PartlyInitialised(9);
          return p.set * 10 + p.left;
        });
  }

  @Test
  public void allocatedObjectPassedToHelper() {
    assertMatchesJdk(() -> manhattan(new Point(3, 4)));
  }

  @Test
  public void capturesArrayOfObjects() {
    Point[] ps = {new Point(1, 2), new Point(3, 4)};
    assertMatchesJdk(() -> ps[0].x + ps[1].y);
  }

  @Test
  public void mutatesSeveralFields() {
    Mixture m = new Mixture();
    OffloadRunnable r =
        () -> {
          m.i = 11;
          m.l = 22L;
          m.d = 3.5;
          m.z = true;
        };
    polyregion.javalang.Runtime.offload(r);
    assertEquals(11, m.i);
    assertEquals(22L, m.l);
    assertEquals(3.5, m.d, 0.0);
    assertEquals(true, m.z);
  }

  @Test
  public void capturesTwoObjects() {
    Point a = new Point(1, 2);
    Point b = new Point(10, 20);
    assertMatchesJdk(() -> a.x + b.y);
  }

  @Test
  public void objectPassedToStaticHelper() {
    Point p = new Point(3, 4);
    assertMatchesJdk(() -> manhattan(p));
  }

  static int manhattan(Point p) {
    return p.x + p.y;
  }

  @Test
  public void fieldDrivesALoop() {
    Point p = new Point(0, 5);
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i < p.y; i++) acc += i;
          return acc;
        });
  }

  @Test
  public void writesBackThroughHelper() {
    Point p = new Point(1, 1);
    OffloadRunnable r = () -> bump(p);
    polyregion.javalang.Runtime.offload(r);
    assertEquals(2, p.x);
    assertEquals(3, p.y);
  }

  static void bump(Point p) {
    p.x = p.x + 1;
    p.y = p.y + 2;
  }

  @Test
  public void writesElementsOfObjectArray() {
    Point[] ps = {new Point(1, 1), new Point(2, 2)};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 2; i++) ps[i].x = ps[i].x * 10;
        };
    polyregion.javalang.Runtime.offload(r);
    assertEquals(10, ps[0].x);
    assertEquals(20, ps[1].x);
  }

  @Test
  public void readsObjectArrayHeldInAField() {
    Bag bag = new Bag(new Point[] {new Point(1, 2), new Point(3, 4)});
    assertMatchesJdk(() -> bag.ps[0].x + bag.ps[1].y);
  }

  @Test
  public void writesElementsOfAnObjectArrayHeldInAField() {
    Bag bag = new Bag(new Point[] {new Point(1, 1), new Point(2, 2)});
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 2; i++) bag.ps[i].x = bag.ps[i].x * 10;
        };
    polyregion.javalang.Runtime.offload(r);
    assertEquals(10, bag.ps[0].x);
    assertEquals(20, bag.ps[1].x);
  }

  @Test
  public void readsPrimitiveArrayHeldInAField() {
    Row row = new Row(new int[] {5, 6, 7});
    assertMatchesJdk(() -> row.xs[0] + row.xs[2]);
  }

  @Test
  public void writesElementsOfAPrimitiveArrayHeldInAField() {
    Row row = new Row(new int[] {1, 2, 3});
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) row.xs[i] = row.xs[i] * 10;
        };
    polyregion.javalang.Runtime.offload(r);
    assertArrayEquals(new int[] {10, 20, 30}, row.xs);
  }

  @Test
  public void emptyObjectArrayReportsZeroLength() {
    Point[] ps = new Point[0];
    assertMatchesJdk(() -> ps.length);
  }

  @Test
  public void emptyObjectArrayIsNotNull() {
    Point[] ps = new Point[0];
    assertMatchesJdk("an empty array is not a null one", () -> ps != null ? 1 : 0);
  }

  @Test
  public void rejectsNullElementOfAnObjectArray() {
    Point[] ps = new Point[2];
    try {
      polyregion.javalang.Runtime.offload(() -> ps.length);
      org.junit.Assert.fail("expected a null element to be rejected");
    } catch (IllegalArgumentException e) {
      org.junit.Assert.assertTrue(
          String.valueOf(e.getMessage()), String.valueOf(e.getMessage()).contains("null"));
    }
  }

  @Test
  public void emptyObjectArrayGuardsIndexing() {
    Point[] ps = new Point[0];
    assertMatchesJdk(() -> ps.length == 0 ? 7 : ps[0].x);
  }

  @Test
  public void callsInstanceMethodOnCapture() {
    Counter c = new Counter(5);
    assertMatchesJdk(() -> c.doubled());
  }

  static final class Counter {
    int n;

    Counter(int n) {
      this.n = n;
    }

    int doubled() {
      return n * 2;
    }
  }
}
