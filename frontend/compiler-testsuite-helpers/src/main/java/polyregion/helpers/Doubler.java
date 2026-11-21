package polyregion.helpers;

public final class Doubler implements Scorer {

  @Override
  public int score(int x) {
    return 2 * x;
  }
}
