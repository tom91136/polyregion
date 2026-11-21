package polyregion;

import org.junit.Test;
import polyregion.helpers.Kernels;
import polyregion.helpers.Plain;
import polyregion.helpers.Shape;

public class BytecodeTierTest extends BaseSuite {

  @Test
  public void scalarStatics() {
    int x = 9;
    double a = 1.5;
    long l = 0x12345678L;
    assertMatchesJdk("clamp-hi", () -> Kernels.clamp(x, 0, 4));
    assertMatchesJdk("clamp-lo", () -> Kernels.clamp(-x, 0, 4));
    assertMatchesJdk("lerp", () -> Kernels.lerp(a, 4.0, 0.25));
    assertMatchesJdk("mix", () -> Kernels.mix(l));
  }

  @Test
  public void loops() {
    int n = 40;
    assertMatchesJdk("sum", () -> Kernels.sumTo(n));
    assertMatchesJdk("collatz", () -> Kernels.collatz(27));
  }

  @Test
  public void primitiveArrays() {
    int[] a = {1, 2, 3, 4};
    int[] b = {5, 6, 7, 8};
    assertMatchesJdk("dot", () -> Kernels.dot(a, b, 4));
    assertMatchesJdk(
        "scale",
        () -> {
          double[] xs = {1.0, 2.0, 3.0};
          Kernels.scale(xs, 3, 2.5);
          return xs[0] + xs[1] + xs[2];
        });
  }

  @Test
  public void overloadsStaySeparate() {
    int x = 5;
    long y = 5L;
    assertMatchesJdk("int", () -> Kernels.pick(x));
    assertMatchesJdk("long", () -> Kernels.pick(y));
  }

  @Test
  public void jdkStatics() {
    int x = -9;
    long y = 77L;
    assertMatchesJdk("signum", () -> Integer.signum(x));
    assertMatchesJdk("signum-long", () -> Long.signum(y));
    assertMatchesJdk("compare", () -> Integer.compare(x, 3));
  }

  @Test
  public void unsupportedBytecodeIsNamed() {
    int x = 3;
    int[] xs = {1, 2, 3};
    assertRejected("Kernels.throwing", () -> Kernels.throwing(x));
    assertRejected("throw", () -> Kernels.throwing(x));
    assertRejected("Kernels.measured", () -> Kernels.measured(xs));
    assertRejected("length", () -> Kernels.measured(xs));
  }

  @Test
  public void staticCallsInsideBytecode() {
    int x = 9;
    assertMatchesJdk("nested", () -> Kernels.clampTwice(x, 0, 4));
    assertMatchesJdk("recursive", () -> Kernels.gcd(84, 36));
    assertMatchesJdk("overloads", () -> Kernels.overloadsTogether(x));
    assertMatchesJdk("intrinsic", () -> Kernels.hypotenuse(3.0, 4.0));
    assertMatchesJdk("jdk-and-intrinsic", () -> Kernels.signedMagnitude(-x));
  }

  @Test
  public void staticCallsOverArrays() {
    assertMatchesJdk(
        "bump",
        () -> {
          int[] xs = {-2, 1, 5, 3};
          Kernels.bump(xs, 4);
          return xs[0] * 1000 + xs[1] * 100 + xs[2] * 10 + xs[3];
        });
  }

  @Test
  public void jdkStaticsCallingStatics() {
    int x = -7;
    int y = 3;
    assertMatchesJdk("floorMod", () -> Math.floorMod(x, y));
    assertMatchesJdk("floorDiv", () -> Math.floorDiv(x, y));
  }

  @Test
  public void objectsLiveInsideBytecode() {
    int x = 3;
    int y = 4;
    assertMatchesJdk("fields", () -> Kernels.viaPoint(x, y));
    assertMatchesJdk("delegating constructor", () -> Kernels.chained(x));
  }

  @Test
  public void scratchArrays() {
    int n = 6;
    assertMatchesJdk("loop", () -> Kernels.scratch(n));
    assertMatchesJdk("zero filled", () -> Kernels.zeroed(n));
    assertMatchesJdk("initialiser", () -> Kernels.literal());
  }

  @Test
  public void escapingAllocationsAreRefused() {
    int n = 4;
    assertRejected("escapes", () -> Kernels.leakArray(n));
    assertRejected("cannot be returned", () -> Kernels.lengthSquared(Kernels.leakPoint(n)));
  }

  @Test
  public void lambdasCalledWhereTheyAreMade() {
    int a = 5;
    int b = 3;
    assertMatchesJdk("plain", () -> Kernels.lambdaLocal(a));
    assertMatchesJdk("capturing", () -> Kernels.lambdaCapturing(a, b));
    assertMatchesJdk("method reference", () -> Kernels.lambdaReference(-a));
  }

  @Test
  public void unsupportedDynamicCallSitesAreNamed() {
    int a = 5;
    assertRejected("functional value", () -> Kernels.lambdaEscapes(a));
    assertRejected("string concatenation", () -> Kernels.concatenated(a));
  }

  @Test
  public void receiversOfFinalClasses() {
    int a = 4;
    assertMatchesJdk("counter", () -> Kernels.counting(a));
  }

  static final class Triangle extends Shape {
    final int base;
    final int height;

    Triangle(int base, int height) {
      this.base = base;
      this.height = height;
    }

    @Override
    public int area() {
      return base * height / 2;
    }
  }

  static final class Derived extends Plain {
    @Override
    public int m() {
      return 2;
    }
  }

  static class Seed {
    int x;

    Seed() {
      x = 7;
    }
  }

  static class Middle extends Seed {}

  static final class Bottom extends Middle {
    int get() {
      return x;
    }
  }

  @Test
  public void aConcreteOwnerIsItsOwnCandidate() {
    assertMatchesJdk("built in bytecode", () -> Kernels.builtPlain());
    assertMatchesJdk("source subclass", () -> Kernels.callM(new Derived()));
  }

  @Test
  public void aClassFileOwnerBuiltInSourceIsRefused() {
    assertRejected("No constructor body available", () -> Kernels.callM(new Plain()));
  }

  @Test
  public void anonymousSubclassesAreCandidates() {
    int a = 4;
    assertMatchesJdk(
        "anonymous",
        () ->
            Kernels.areaOf(
                new Shape() {
                  @Override
                  public int area() {
                    return a * 11;
                  }
                }));
  }

  @Test
  public void stateInheritedThroughAnIntermediateClassIsRefused() {
    assertRejected("carries instance state", () -> new Bottom().get());
  }

  @Test
  public void openReceiversOfStatefulOwnersAreRefused() {
    int a = 4;
    assertRejected("carries instance state", () -> Kernels.openReceiver(a));
  }

  @Test
  public void openReceiversDispatchOverClassesTheRegionBuilds() {
    int a = 4;
    int b = 7;
    assertMatchesJdk("square", () -> Kernels.areaChosen(b, a));
    assertMatchesJdk("rect", () -> Kernels.areaChosen(a, b));
    assertMatchesJdk("interface, doubled", () -> Kernels.scored(a, true));
    assertMatchesJdk("interface, negated", () -> Kernels.scored(a, false));
  }

  @Test
  public void openReceiversDispatchOverSourceClasses() {
    int a = 4;
    assertMatchesJdk("source subclass", () -> Kernels.areaOf(new Triangle(a, 3)));
    assertMatchesJdk("both kinds", () -> Kernels.areaBoth(new Triangle(a, 3), 5));
  }

  @Test
  public void inheritedImplementationsAreRefused() {
    assertRejected("inherits its implementation", () -> Kernels.inheritedId());
  }
}
