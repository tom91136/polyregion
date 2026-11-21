package polyregion.helpers;

public final class Square extends Shape {

  final int side;

  public Square(int side) {
    this.side = side;
  }

  @Override
  public int area() {
    return side * side;
  }
}
