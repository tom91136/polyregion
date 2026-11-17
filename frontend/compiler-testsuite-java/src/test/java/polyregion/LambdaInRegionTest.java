package polyregion;

import java.util.function.IntBinaryOperator;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import org.junit.Test;
import polyregion.javalang.OffloadFunction;

public class LambdaInRegionTest extends BaseSuite {

  interface IntOp {
    int apply(int x);
  }

  static int applyTwice(IntUnaryOperator f, int x) {
    return f.applyAsInt(f.applyAsInt(x));
  }

  static int triple(int x) {
    return x * 3;
  }

  static int combine(IntUnaryOperator f, IntUnaryOperator g, int x) {
    return f.applyAsInt(x) * 10 + g.applyAsInt(x);
  }

  static IntUnaryOperator makeDoubler() {
    return x -> x * 2;
  }

  @Test
  public void lambdaDeclaredAndCalledInRegion() {
    int a = 5;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator twice = x -> x * 2;
          return twice.applyAsInt(a);
        });
  }

  @Test
  public void customFunctionalInterfaceInRegion() {
    int a = 5;
    assertMatchesJdk(
        () -> {
          IntOp op = x -> x + 1;
          return op.apply(a);
        });
  }

  @Test
  public void lambdaCapturedFromHost() {
    IntUnaryOperator twice = x -> x * 2;
    int a = 5;
    assertMatchesJdk(() -> twice.applyAsInt(a));
  }

  @Test
  public void lambdaPassedToHelper() {
    IntUnaryOperator inc = x -> x + 1;
    int a = 5;
    assertMatchesJdk(() -> applyTwice(inc, a));
  }

  @Test
  public void twoLambdasCapturedFromHost() {
    IntUnaryOperator twice = x -> x * 2;
    IntUnaryOperator inc = x -> x + 1;
    int a = 5;
    assertMatchesJdk(
        "each captured lambda keeps its own body",
        () -> twice.applyAsInt(a) * 100 + inc.applyAsInt(a));
  }

  @Test
  public void capturedLambdaSitsAmongOtherCaptures() {
    IntUnaryOperator twice = x -> x * 2;
    int a = 3;
    int[] xs = {10, 20, 30};
    double d = 0.5;
    assertMatchesJdk(
        "every other capture keeps its slot",
        () -> twice.applyAsInt(a) + xs[1] + xs.length + (int) (d * 8));
  }

  @Test
  public void twoRegionsShareOneHostLambda() {
    IntUnaryOperator twice = x -> x * 2;
    int a = 5;
    assertMatchesJdk(() -> twice.applyAsInt(a));
    assertMatchesJdk(() -> twice.applyAsInt(a) + 1);
  }

  @Test
  public void methodReferenceCapturedFromHost() {
    IntUnaryOperator abs = Math::abs;
    IntUnaryOperator tri = LambdaInRegionTest::triple;
    int a = -5;
    assertMatchesJdk(
        "each reference keeps its own target", () -> abs.applyAsInt(a) * 100 + tri.applyAsInt(a));
  }

  @Test
  public void twoLambdasPassedToOneHelper() {
    IntUnaryOperator inc = x -> x + 1;
    IntUnaryOperator twice = x -> x * 2;
    int a = 5;
    assertMatchesJdk("each parameter keeps its own operator", () -> combine(inc, twice, a));
  }

  @Test
  public void lambdasPassedToTheSameHelperTwice() {
    IntUnaryOperator inc = x -> x + 1;
    IntUnaryOperator twice = x -> x * 2;
    int a = 5;
    assertMatchesJdk(
        "each call gets its own operator", () -> applyTwice(inc, a) * 100 + applyTwice(twice, a));
  }

  @Test
  public void capturingLambdaFromHostIsRefused() {
    int k = 3;
    IntUnaryOperator scale = x -> x * k;
    int a = 5;
    assertRejected("devirtualise", () -> scale.applyAsInt(a));
  }

  @Test
  public void lambdaFromAHostCallIsRefused() {
    IntUnaryOperator op = makeDoubler();
    int a = 5;
    assertRejected("devirtualise", () -> op.applyAsInt(a));
  }

  @Test
  public void methodReferenceInRegion() {
    int a = -5;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator abs = Math::abs;
          return abs.applyAsInt(a);
        });
  }

  @Test
  public void lambdaCapturingRegionLocal() {
    assertMatchesJdk(
        () -> {
          int base = 10;
          IntSupplier s = () -> base * 2;
          return s.getAsInt();
        });
  }

  @Test
  public void binaryOperatorInRegion() {
    int a = 3;
    int b = 4;
    assertMatchesJdk(
        () -> {
          IntBinaryOperator add = (x, y) -> x + y;
          return add.applyAsInt(a, b);
        });
  }

  @Test
  public void nestedOffloadRegion() {
    int a = 2;
    assertMatchesJdk(
        () -> {
          OffloadFunction<Integer> inner = () -> a * 3;
          return inner.run();
        });
  }

  @Test
  public void lambdaReturningLambda() {
    int a = 2;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator outer = x -> x + a;
          IntUnaryOperator composed = x -> outer.applyAsInt(x) * 2;
          return composed.applyAsInt(1);
        });
  }

  @Test
  public void lambdaAppliedTwiceInRegion() {
    int a = 5;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator inc = x -> x + 1;
          return inc.applyAsInt(inc.applyAsInt(a));
        });
  }

  @Test
  public void lambdaWithBlockBody() {
    int a = 4;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator f =
              x -> {
                int y = x * 3;
                return y - 1;
              };
          return f.applyAsInt(a);
        });
  }

  @Test
  public void lambdaWithConditional() {
    int a = -9;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator f = x -> x > 0 ? x : -x;
          return f.applyAsInt(a);
        });
  }

  @Test
  public void methodReferenceToUserMethod() {
    int a = 7;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator f = LambdaInRegionTest::triple;
          return f.applyAsInt(a);
        });
  }

  @Test
  public void lambdaCallingHelperMethod() {
    int a = 6;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator f = x -> triple(x) + 1;
          return f.applyAsInt(a);
        });
  }

  @Test
  public void twoLambdasInOneRegion() {
    int a = 3;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator inc = x -> x + 1;
          IntBinaryOperator mul = (x, y) -> x * y;
          return mul.applyAsInt(inc.applyAsInt(a), a);
        });
  }

  @Test
  public void lambdaOverDoubles() {
    double a = 2.5;
    assertMatchesJdk(
        () -> {
          java.util.function.DoubleUnaryOperator f = x -> x * x + 1.0;
          return f.applyAsDouble(a);
        });
  }

  @Test
  public void lambdaInLoop() {
    int n = 5;
    assertMatchesJdk(
        () -> {
          IntUnaryOperator sq = x -> x * x;
          int total = 0;
          for (int i = 0; i < n; i++) total += sq.applyAsInt(i);
          return total;
        });
  }
}
