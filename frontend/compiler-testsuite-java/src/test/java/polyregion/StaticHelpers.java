package polyregion;

public final class StaticHelpers {

  public static int total = 0;

  public static void accumulate(int x) {
    total = total + x;
  }
}
