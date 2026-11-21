package polyregion.helpers;

public final class Point {

  int x;
  int y;
  double w;

  public Point(int x, int y) {
    this.x = x;
    this.y = y;
  }

  public Point(int x) {
    this(x, 0);
  }
}
