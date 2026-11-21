package polyregion.helpers;

public final class Negator implements Scorer {

  @Override
  public int score(int x) {
    return -x;
  }
}
