package polyregion;

public final class NdHelper {

  private NdHelper() {}

  public static void fill(long n0, long n1, int[] out) {
    Offload.for_each(n0, n1, (i, j) -> out[(int) (i * n1 + j)] = (int) (100 * i + j));
  }
}
