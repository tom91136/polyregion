package polyregion.helpers;

public final class Kernels {

  private Kernels() {}

  public static int clamp(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  }

  public static double lerp(double a, double b, double t) {
    return a + (b - a) * t;
  }

  public static long mix(long x) {
    x ^= x >>> 33;
    x *= 0xff51afd7ed558ccdL;
    x ^= x >>> 33;
    return x;
  }

  public static int sumTo(int n) {
    int acc = 0;
    for (int i = 1; i <= n; i++) acc += i;
    return acc;
  }

  public static int collatz(int n) {
    int steps = 0;
    while (n != 1) {
      n = (n & 1) == 0 ? n >> 1 : 3 * n + 1;
      steps++;
    }
    return steps;
  }

  public static int dot(int[] a, int[] b, int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
  }

  public static void scale(double[] xs, int n, double k) {
    for (int i = 0; i < n; i++) xs[i] *= k;
  }

  public static int pick(int x) {
    return x + 1;
  }

  public static int pick(long x) {
    return (int) (x * 2);
  }

  public static int throwing(int x) {
    if (x < 0) throw new IllegalArgumentException("negative");
    return x;
  }

  public static int measured(int[] xs) {
    return xs.length;
  }

  public static int clampTwice(int x, int lo, int hi) {
    return clamp(clamp(x, lo, hi), lo, hi - 1);
  }

  public static int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
  }

  public static double hypotenuse(double a, double b) {
    return Math.sqrt(a * a + b * b);
  }

  public static int overloadsTogether(int x) {
    return pick(x) + pick((long) x);
  }

  public static int signedMagnitude(int x) {
    return Integer.signum(x) * Math.abs(x);
  }

  public static void bump(int[] xs, int n) {
    for (int i = 0; i < n; i++) xs[i] = clamp(xs[i], 0, 3);
  }
}
