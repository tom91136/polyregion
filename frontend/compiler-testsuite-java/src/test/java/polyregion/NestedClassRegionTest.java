package polyregion;

import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import java.util.function.IntSupplier;
import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;

public class NestedClassRegionTest extends BaseSuite {

  int outerField = 100;

  static final class StaticNested {
    static int sum(int a, int b) {
      return offload(() -> a + b);
    }

    static int scaled(int[] xs, int k) {
      OffloadRunnable r =
          () -> {
            for (int i = 0; i < xs.length; i++) xs[i] *= k;
          };
      offload(r);
      return xs[0] + xs[1];
    }
  }

  final class Inner {
    int innerField = 7;

    int combined(int x) {
      return offload(() -> x + innerField + outerField);
    }

    void bump() {
      OffloadRunnable r = () -> innerField += outerField;
      offload(r);
    }
  }

  static final class Holder {
    static int viaLambda(int a) {
      IntSupplier s = () -> offload(() -> a * 3);
      return s.getAsInt();
    }

    int instanceViaLambda(int a) {
      IntSupplier s = () -> offload(() -> a + base);
      return s.getAsInt();
    }

    final int base = 1000;
  }

  static final class Labelled {
    String label = "outer";

    final class Inner {
      int x = 5;

      int plusOne() {
        return offload(() -> x + 1);
      }
    }
  }

  @Test
  public void innerClassIgnoringAnUnloweredOuter() {
    assertEquals(6, new Labelled().new Inner().plusOne());
  }

  @Test
  public void staticNestedClass() {
    assertEquals(5, StaticNested.sum(2, 3));
    assertEquals(9, StaticNested.scaled(new int[] {1, 2}, 3));
  }

  @Test
  public void innerClass() {
    Inner inner = new Inner();
    assertEquals(3 + 7 + 100, inner.combined(3));
    inner.bump();
    assertEquals(107, inner.innerField);
  }

  @Test
  public void lambdaInNestedClass() {
    assertEquals(12, Holder.viaLambda(4));
    assertEquals(1004, new Holder().instanceViaLambda(4));
  }
}
