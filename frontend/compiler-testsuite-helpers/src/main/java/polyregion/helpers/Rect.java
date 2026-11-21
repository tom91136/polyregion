package polyregion.helpers;

public final class Rect extends Shape {

  final int w;
  final int h;

  public Rect(int w, int h) {
    this.w = w;
    this.h = h;
  }

  @Override
  public int area() {
    return w * h;
  }
}
