package polyregion;

import org.junit.Test;

public class DispatchArgsTest extends BaseSuite {

  abstract static class Reducer {
    abstract int reduce(int[] xs);
  }

  static final class Measuring extends Reducer {
    int reduce(int[] xs) {
      return xs.length;
    }
  }

  static final class First extends Reducer {
    int reduce(int[] xs) {
      return xs[0];
    }
  }

  static final class Last extends Reducer {
    int reduce(int[] xs) {
      return xs[xs.length - 1];
    }
  }

  @Test
  public void implementationsTakingDifferentExtraArguments() {
    int[] xs = {4, 8, 15};
    Reducer a = new Measuring();
    Reducer b = new First();
    Reducer c = new Last();
    assertMatchesJdk("measuring", () -> a.reduce(xs));
    assertMatchesJdk("first", () -> b.reduce(xs));
    assertMatchesJdk("last", () -> c.reduce(xs));
    assertMatchesJdk("together", () -> a.reduce(xs) * 10000 + b.reduce(xs) * 100 + c.reduce(xs));
  }
}
