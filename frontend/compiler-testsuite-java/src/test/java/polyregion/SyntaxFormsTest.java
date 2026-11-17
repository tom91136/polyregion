package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;

public class SyntaxFormsTest extends BaseSuite {

  static int square(int x) {
    return x * x;
  }

  static int addAll(int[] xs, int n) {
    int acc = 0;
    for (int i = 0; i < n; i++) acc += xs[i];
    return acc;
  }

  static int depth3(int x) {
    return depth2(x) + 1;
  }

  static int depth2(int x) {
    return square(x) + 1;
  }

  static int constant() {
    return 99;
  }

  int instanceField = 11;

  @Test
  public void varDeclaration() {
    int n = 4;
    assertEquals(
        6,
        (int)
            offload(
                () -> {
                  var acc = 0;
                  for (var i = 0; i < n; i++) acc += i;
                  return acc;
                }));
  }

  @Test
  public void varWithArrayType() {
    int[] xs = {2, 4, 6};
    assertEquals(
        12,
        (int)
            offload(
                () -> {
                  var acc = 0;
                  for (var i = 0; i < 3; i++) {
                    var x = xs[i];
                    acc += x;
                  }
                  return acc;
                }));
  }

  @Test
  public void expressionLambda() {
    int a = 6;
    assertEquals(36, (int) offload(() -> square(a)));
  }

  @Test
  public void blockLambda() {
    int a = 6;
    assertEquals(
        37,
        (int)
            offload(
                () -> {
                  int r = square(a);
                  return r + 1;
                }));
  }

  @Test
  public void lambdaStoredInLocal() {
    int a = 3;
    OffloadFunction<Integer> f = () -> a * 3;
    assertEquals(9, (int) offload(f));
  }

  @Test
  public void runnableStoredInLocal() {
    int[] out = new int[1];
    OffloadRunnable r = () -> out[0] = 42;
    offload(r);
    assertArrayEquals(new int[] {42}, out);
  }

  @Test
  public void parenthesisedLambda() {
    int a = 2;
    assertEquals(4, (int) offload((() -> a * 2)));
  }

  @Test
  public void transitiveCallChain() {
    int a = 3;
    assertEquals(11, (int) offload(() -> depth3(a)));
  }

  @Test
  public void callWithArrayArgument() {
    int[] xs = {1, 2, 3, 4};
    assertEquals(10, (int) offload(() -> addAll(xs, 4)));
  }

  @Test
  public void capturesMultipleArrays() {
    int[] a = {1, 2, 3};
    int[] b = {10, 20, 30};
    int[] out = new int[3];
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) out[i] = a[i] + b[i];
        };
    offload(r);
    assertArrayEquals(new int[] {11, 22, 33}, out);
  }

  @Test
  public void capturesManyScalars() {
    int a = 1;
    int b = 2;
    long c = 3L;
    double d = 4.0;
    float e = 5.0f;
    assertEquals(15.0, (double) offload(() -> a + b + c + d + e), 0.0);
  }

  @Test
  public void noCaptures() {
    assertEquals(7, (int) offload(() -> 3 + 4));
  }

  @Test
  public void shadowedLocal() {
    int x = 1;
    assertEquals(
        5,
        (int)
            offload(
                () -> {
                  int acc = x;
                  {
                    int y = 4;
                    acc += y;
                  }
                  return acc;
                }));
  }

  @Test
  public void nestedBlocks() {
    assertEquals(
        3,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  {
                    acc += 1;
                    {
                      acc += 2;
                    }
                  }
                  return acc;
                }));
  }

  @Test
  public void loopLocalShadowsOuter() {
    assertEquals(
        3,
        (int)
            offload(
                () -> {
                  int i = 100;
                  int acc = 0;
                  for (int j = 0; j < 3; j++) acc++;
                  return acc;
                }));
  }

  @Test
  public void capturesInstanceFieldViaLocal() {
    int f = instanceField;
    assertEquals(22, (int) offload(() -> f * 2));
  }

  @Test
  public void arrayElementCompoundAssignment() {
    int[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i] *= 10;
        };
    offload(r);
    assertArrayEquals(new int[] {10, 20, 30}, xs);
  }

  @Test
  public void arrayElementIncrement() {
    int[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 3; i++) xs[i]++;
        };
    offload(r);
    assertArrayEquals(new int[] {2, 3, 4}, xs);
  }

  @Test
  public void writeThenReadSameArray() {
    int[] xs = new int[4];
    assertEquals(
        6,
        (int)
            offload(
                () -> {
                  for (int i = 0; i < 4; i++) xs[i] = i;
                  int acc = 0;
                  for (int i = 0; i < 4; i++) acc += xs[i];
                  return acc;
                }));
  }

  @Test
  public void staticMethodReference() {
    assertEquals(99, (int) offload(SyntaxFormsTest::constant));
  }

  @Test
  public void charArrayRoundTrip() {
    char[] xs = {'a', 'b', 'c'};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = (char) (xs[i] - 32);
        };
    offload(r);
    assertEquals("ABC", new String(xs));
  }

  @Test
  public void shortArrayRoundTrip() {
    short[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = (short) (xs[i] * 100);
        };
    offload(r);
    assertArrayEquals(new short[] {100, 200, 300}, xs);
  }

  @Test
  public void arrayLengthIsVisible() {
    int[] xs = new int[7];
    assertEquals(7, (int) offload(() -> xs.length));
  }

  @Test
  public void lengthOfSecondArray() {
    int[] a = new int[2];
    int[] b = new int[5];
    assertEquals(7, (int) offload(() -> a.length + b.length));
  }

  @Test
  public void voidRegionWithNoReturnStatement() {
    int[] out = new int[1];
    OffloadRunnable r =
        () -> {
          out[0] = 5;
        };
    offload(r);
    assertArrayEquals(new int[] {5}, out);
  }

  @Test
  public void voidRegionWithBareReturn() {
    int[] out = new int[1];
    OffloadRunnable r =
        () -> {
          out[0] = 5;
          return;
        };
    offload(r);
    assertArrayEquals(new int[] {5}, out);
  }

  static int sum(int... xs) {
    int acc = 0;
    for (int x : xs) acc += x;
    return acc;
  }

  static int weighted(int... xs) {
    int acc = 0;
    for (int i = 0; i < xs.length; i++) acc += xs[i] * (i + 1);
    return acc;
  }

  static int measured(int[] xs) {
    return xs.length;
  }

  static int forwarded(int[] xs) {
    return measured(xs);
  }

  @Test
  public void varargsCall() {
    assertMatchesJdk(() -> sum(1, 2, 3));
  }

  @Test
  public void varargsCallsOfDifferentArity() {
    assertMatchesJdk("each call carries its own length", () -> sum(1, 2, 3) * 100 + sum(4, 5));
  }

  @Test
  public void varargsCallWithNoArguments() {
    assertMatchesJdk(() -> sum());
  }

  @Test
  public void varargsElementsKeepTheirOrder() {
    assertMatchesJdk("each element keeps its position", () -> weighted(10, 20, 30, 40));
  }

  @Test
  public void helperMeasuresACapturedArray() {
    int[] xs = new int[6];
    assertMatchesJdk(() -> measured(xs));
  }

  @Test
  public void helperForwardsAnArrayToOneThatMeasuresIt() {
    int[] xs = new int[6];
    int[] ys = new int[3];
    assertMatchesJdk("each array keeps its own length", () -> forwarded(xs) * 100 + forwarded(ys));
  }

  @Test
  public void helperMeasuresAnArrayAllocatedInTheRegion() {
    assertMatchesJdk(() -> measured(new int[9]));
  }
}
