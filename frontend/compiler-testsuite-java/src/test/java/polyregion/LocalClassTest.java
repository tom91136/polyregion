package polyregion;

import org.junit.Test;

public class LocalClassTest extends BaseSuite {

  interface Shape {
    int area();
  }

  abstract static class Base {
    abstract int value();

    int doubled() {
      return value() * 2;
    }
  }

  abstract static class Mid extends Base {
    int quadrupled() {
      return doubled() * 2;
    }
  }

  static class Tripler extends Base {
    @Override
    int value() {
      return 5;
    }

    @Override
    int doubled() {
      return value() * 3;
    }
  }

  abstract static class Scaler {
    abstract int base();

    int scaled(int k) {
      return base() * k;
    }
  }

  abstract static class Accum {
    abstract void bump(int x);

    abstract int total();

    void bumpTwice(int x) {
      bump(x);
      bump(x);
    }
  }

  @Test
  public void localClassInRegion() {
    int a = 3;
    assertMatchesJdk(
        () -> {
          class Box {
            int v;

            Box(int v) {
              this.v = v;
            }

            int twice() {
              return v * 2;
            }
          }
          return new Box(a).twice();
        });
  }

  @Test
  public void anonymousClassInRegion() {
    int a = 4;
    assertMatchesJdk(
        () -> {
          Shape s =
              new Shape() {
                @Override
                public int area() {
                  return a * a;
                }
              };
          return s.area();
        });
  }

  @Test
  public void twoAnonymousImplementationsInOneRegion() {
    int a = 4;
    assertMatchesJdk(
        "each anonymous class keeps its own body",
        () -> {
          Shape lo =
              new Shape() {
                @Override
                public int area() {
                  return a;
                }
              };
          Shape hi =
              new Shape() {
                @Override
                public int area() {
                  return a * 100;
                }
              };
          return lo.area() + hi.area();
        });
  }

  @Test
  public void anonymousClassCapturesTwoLocals() {
    int a = 3;
    int b = 7;
    assertMatchesJdk(
        "each captured local keeps its own field",
        () -> {
          Shape s =
              new Shape() {
                @Override
                public int area() {
                  return a * 10 + b;
                }
              };
          return s.area();
        });
  }

  @Test
  public void anonymousSubclassCallsInheritedMethod() {
    int a = 6;
    assertMatchesJdk(
        () -> {
          Base b =
              new Base() {
                @Override
                int value() {
                  return a;
                }
              };
          return b.doubled();
        });
  }

  @Test
  public void twoSubclassesShareAnInheritedMethod() {
    int a = 6;
    assertMatchesJdk(
        "each subclass keeps its own override",
        () -> {
          Base lo =
              new Base() {
                @Override
                int value() {
                  return a;
                }
              };
          Base hi =
              new Base() {
                @Override
                int value() {
                  return a * 100;
                }
              };
          return lo.doubled() + hi.doubled();
        });
  }

  @Test
  public void subclassOverrideWinsOverTheInheritedBody() {
    assertMatchesJdk(
        "the override wins",
        () -> {
          Base b = new Tripler();
          return b.doubled();
        });
  }

  @Test
  public void inheritedMethodThroughTwoLevels() {
    int a = 3;
    assertMatchesJdk(
        () -> {
          Mid m =
              new Mid() {
                @Override
                int value() {
                  return a;
                }
              };
          return m.quadrupled();
        });
  }

  @Test
  public void inheritedMethodTakesAnArgument() {
    int a = 4;
    int k = 7;
    assertMatchesJdk(
        "the caller's own `k` survives the call",
        () -> {
          Scaler s =
              new Scaler() {
                @Override
                int base() {
                  return a;
                }
              };
          return s.scaled(3) * 100 + k;
        });
  }

  @Test
  public void inheritedMethodMutatesTheReceiver() {
    assertMatchesJdk(
        "the mutation is visible to the caller",
        () -> {
          class Acc extends Accum {
            int n;

            @Override
            void bump(int x) {
              n += x;
            }

            @Override
            int total() {
              return n;
            }
          }
          Acc acc = new Acc();
          acc.bumpTwice(5);
          return acc.total();
        });
  }

  @Test
  public void localRecordInRegion() {
    int a = 3;
    int b = 4;
    assertMatchesJdk(
        () -> {
          record Pair(int x, int y) {}
          Pair p = new Pair(a, b);
          return p.x() + p.y();
        });
  }

  @Test
  public void localClassCapturesSeveralLocals() {
    int a = 3;
    int b = 5;
    assertMatchesJdk(
        "each captured local keeps its own field",
        () -> {
          class Weighted {
            int mix() {
              return a * 10 + b;
            }
          }
          return new Weighted().mix();
        });
  }

  @Test
  public void localClassAllocatedInALoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          for (int i = 0; i < 4; i++) {
            class Cell {
              int v;

              Cell(int v) {
                this.v = v;
              }
            }
            acc += new Cell(i).v * i;
          }
          return acc;
        });
  }

  @Test
  public void localClassWithStaticLikeCounter() {
    assertMatchesJdk(
        () -> {
          class Acc {
            int total;

            void add(int x) {
              total += x;
            }
          }
          Acc acc = new Acc();
          for (int i = 0; i < 4; i++) acc.add(i);
          return acc.total;
        });
  }
}
