package polyregion;

import org.junit.Test;

public class DispatchPaddingTest extends BaseSuite {

  abstract static class Shape {
    abstract double area();
  }

  static final class Circle extends Shape {
    final double r;

    Circle(double r) {
      this.r = r;
    }

    double area() {
      return 3.0 * r * r;
    }
  }

  static final class Square extends Shape {
    final double side;

    Square(double side) {
      this.side = side;
    }

    double area() {
      return side * side;
    }
  }

  static final class Tiny extends Shape {
    final int n;

    Tiny(int n) {
      this.n = n;
    }

    double area() {
      return n;
    }
  }

  static double through(Shape s) {
    return s.area();
  }

  @Test
  public void aSubtypeChosenByATernaryKeepsItsStateThroughAReference() {
    boolean circle = true;
    boolean square = false;
    assertMatchesJdk("circle", () -> {
      Shape s = circle ? new Circle(2) : new Square(3);
      return through(s);
    });
    assertMatchesJdk("square", () -> {
      Shape s = square ? new Circle(2) : new Square(3);
      return through(s);
    });
  }

  @Test
  public void anOwnerNamedOnlyByALocalIsLaidOut() {
    boolean circle = true;
    assertMatchesJdk("local", () -> {
      Shape s = circle ? new Circle(2) : new Square(3);
      return s.area();
    });
  }

  @Test
  public void aSubtypeSmallerThanTheOwnerCannotBeCopiedIntoIt() {
    boolean tiny = true;
    assertRejected("would read past the source allocation", () -> {
      Shape s = tiny ? new Tiny(1) : new Circle(2);
      return through(s);
    });
  }
}
