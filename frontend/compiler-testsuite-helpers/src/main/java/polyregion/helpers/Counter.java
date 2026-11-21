package polyregion.helpers;

public final class Counter {

  int n;

  public Counter(int n) {
    this.n = n;
  }

  public int next() {
    return n++;
  }

  public int add(int k) {
    n += k;
    return n;
  }

  public int get() {
    return n;
  }

  private int twice() {
    return n * 2;
  }

  public int viaPrivate() {
    return twice() + 1;
  }
}
