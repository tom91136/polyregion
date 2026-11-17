package polyregion.forward;

import org.junit.Test;
import polyregion.BaseSuite;

public class RecordPatternTest extends BaseSuite {

  record Point(int x, int y) {}

  record Line(Point from, Point to) {}

  @Test
  public void deconstructionInInstanceof() {
    Object o = new Point(1, 2);
    assertRejected(Rejections.INSTANCEOF, () -> o instanceof Point(int x, int y) ? x + y : 0);
  }

  @Test
  public void deconstructionInSwitch() {
    Object o = new Point(1, 2);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (o) {
              case Point(int x, int y) -> x + y;
              default -> 0;
            });
  }

  @Test
  public void nestedDeconstruction() {
    Object o = new Line(new Point(1, 2), new Point(3, 4));
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (o) {
              case Line(Point(int ax, int ay), Point(int bx, int by)) -> (bx - ax) + (by - ay);
              default -> 0;
            });
  }

  @Test
  public void deconstructionWithVar() {
    Object o = new Point(1, 2);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (o) {
              case Point(var x, var y) -> x * y;
              default -> 0;
            });
  }
}
