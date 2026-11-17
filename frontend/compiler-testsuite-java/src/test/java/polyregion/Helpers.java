package polyregion;

public final class Helpers {

  public static final int CONSTANT = 10;

  public static int counter = 0;

  public static int triple(int x) {
    return x * 3;
  }

  public static int sumRange(int[] xs, int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) acc += xs[i];
    return acc;
  }

  public static void scale(int[] xs, int n, int by) {
    for (int i = 0; i < n; i++) xs[i] = xs[i] * by;
  }

  public static int viaAnotherHelper(int x) {
    return triple(x) + CONSTANT;
  }
}
