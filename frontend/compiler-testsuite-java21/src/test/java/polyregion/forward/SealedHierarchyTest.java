package polyregion.forward;

import org.junit.Test;
import polyregion.BaseSuite;

public class SealedHierarchyTest extends BaseSuite {

  sealed interface Shape permits Circle, Square {}

  record Circle(double radius) implements Shape {}

  record Square(double side) implements Shape {}

  @Test
  public void exhaustiveSwitchNeedsNoDefault() {
    Shape shape = new Circle(2.0);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (shape) {
              case Circle c -> c.radius() * 2.0;
              case Square s -> s.side() * 4.0;
            });
  }

  @Test
  public void exhaustiveSwitchWithDeconstruction() {
    Shape shape = new Square(3.0);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (shape) {
              case Circle(double r) -> r * 2.0;
              case Square(double s) -> s * 4.0;
            });
  }

  @Test
  public void instanceofAgainstAPermittedSubtype() {
    Shape shape = new Circle(2.0);
    assertRejected(Rejections.INSTANCEOF, () -> shape instanceof Circle c ? c.radius() : 0.0);
  }

  @Test
  public void virtualCallThroughTheSealedInterface() {
    Shape shape = new Circle(2.0);
    assertRejected("Circle.toString", () -> shape.toString().length());
  }
}
