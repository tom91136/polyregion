package polyregion;

import org.junit.Test;

public class RecursionTest extends BaseSuite {

  static int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
  }

  static int isEven(int n) {
    return n == 0 ? 1 : isOdd(n - 1);
  }

  static int isOdd(int n) {
    return n == 0 ? 0 : isEven(n - 1);
  }

  static int countdown(int n) {
    if (n <= 0) return 0;
    return 1 + countdown(n - 1);
  }

  static long factorial(long n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
  }

  static int ackermann(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
  }

  static int sumArray(int[] xs, int i, int n) {
    return i >= n ? 0 : xs[i] + sumArray(xs, i + 1, n);
  }

  @Test
  public void binaryRecursion() {
    int n = 15;
    assertMatchesJdk(() -> fib(n));
  }

  @Test
  public void mutualRecursion() {
    int n = 7;
    assertMatchesJdk("odd", () -> isEven(n));
    assertMatchesJdk("even", () -> isEven(8));
  }

  @Test
  public void tailRecursion() {
    int n = 100;
    assertMatchesJdk(() -> countdown(n));
  }

  @Test
  public void recursionOnLong() {
    long n = 20L;
    assertMatchesJdk(() -> factorial(n));
  }

  @Test
  public void nestedRecursion() {
    assertMatchesJdk(() -> ackermann(2, 3));
  }

  @Test
  public void recursionOverCapturedArray() {
    int[] xs = {1, 2, 3, 4, 5};
    assertMatchesJdk(() -> sumArray(xs, 0, 5));
  }
}
