package polyregion.javalang;

public final class Ops {

  private Ops() {
    throw new AssertionError();
  }

  public static final class Sum {
    final double high;
    final double compensation;
    final double simple;

    Sum(double high, double compensation, double simple) {
      this.high = high;
      this.compensation = compensation;
      this.simple = simple;
    }
  }

  public static Sum sumMerge(Sum a, Sum b) {
    double high = a.high;
    double tmp = b.high - a.compensation;
    double next = high + tmp;
    double compensation = (next - high) - tmp;
    high = next;
    tmp = -b.compensation - compensation;
    next = high + tmp;
    compensation = (next - high) - tmp;
    return new Sum(next, compensation, a.simple + b.simple);
  }

  public static final int BLOCK = 1024;

  public static long blockFrom(long block) {
    return block * BLOCK;
  }

  public static long blockTo(long n, long block) {
    return Math.min(n, (block + 1) * BLOCK);
  }

  public static int slot(long block, int stride, int field) {
    return (int) (stride * block + field);
  }

  public static void putD(double[] a, int i, double v) {
    a[i] = v;
  }

  public static void putI(int[] a, int i, int v) {
    a[i] = v;
  }

  public static void putJ(long[] a, int i, long v) {
    a[i] = v;
  }

  public static double compensated(double high, double error, double x) {
    return high + (x - error);
  }

  public static double compensation(double high, double error, double x) {
    double tmp = x - error;
    return ((high + tmp) - high) - tmp;
  }

  public static double sumValue(Sum a) {
    double value = a.high - a.compensation;
    return Double.isNaN(value) && Double.isInfinite(a.simple) ? a.simple : value;
  }

  public static int atIJ(int origin, long index) {
    return (int) (origin + index);
  }

  public static long atJJ(long origin, long index) {
    return origin + index;
  }

  public static int atII(int origin, int index) {
    return origin + index;
  }

  public static long atJI(long origin, int index) {
    return origin + index;
  }

  public static int getI(int[] xs, long index) {
    return xs[(int) index];
  }

  public static long getJ(long[] xs, long index) {
    return xs[(int) index];
  }

  public static double getD(double[] xs, long index) {
    return xs[(int) index];
  }

  public static int sumI(int a, int b) {
    return a + b;
  }

  public static long sumJ(long a, long b) {
    return a + b;
  }

  public static double sumD(double a, double b) {
    return a + b;
  }

  public static int minI(int a, int b) {
    return Math.min(a, b);
  }

  public static long minJ(long a, long b) {
    return Math.min(a, b);
  }

  public static double minD(double a, double b) {
    return Math.min(a, b);
  }

  public static int maxI(int a, int b) {
    return Math.max(a, b);
  }

  public static long maxJ(long a, long b) {
    return Math.max(a, b);
  }

  public static double maxD(double a, double b) {
    return Math.max(a, b);
  }

  public static long widenI(int a) {
    return a;
  }

  public static long bit(boolean a) {
    return a ? 1L : 0L;
  }

  public static long notBit(boolean a) {
    return a ? 0L : 1L;
  }

  public static boolean ltI(int a, int b) {
    return a < b;
  }

  public static boolean ltJ(long a, long b) {
    return a < b;
  }

  public static boolean ltF(float a, float b) {
    if (a < b) return true;
    if (a > b) return false;
    if (b != b) return a == a;
    if (a != a) return false;
    return 1 / a < 0 && !(1 / b < 0);
  }

  public static boolean ltD(double a, double b) {
    if (a < b) return true;
    if (a > b) return false;
    if (b != b) return a == a;
    if (a != a) return false;
    return 1 / a < 0 && !(1 / b < 0);
  }
}
