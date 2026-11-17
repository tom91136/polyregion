package polyregion;

import org.junit.Test;

public class UnsupportedStatementTest extends BaseSuite {

  static class Boom extends RuntimeException {}

  @Test
  public void tryCatch() {
    int a = 1;
    assertRejected(
        "`try`/`catch`",
        () -> {
          int r;
          try {
            r = a;
          } catch (RuntimeException e) {
            r = -1;
          }
          return r;
        });
  }

  @Test
  public void tryFinally() {
    int[] out = new int[1];
    assertRejected(
        "`finally`",
        () -> {
          try {
            out[0] = 1;
          } finally {
            out[0] = 2;
          }
          return out[0];
        });
  }

  @Test
  public void throwStatement() {
    int a = 1;
    assertRejected(
        "`throw`",
        () -> {
          if (a < 0) throw new Boom();
          return a;
        });
  }

  @Test
  public void synchronizedBlock() {
    Object lock = new Object();
    int a = 1;
    assertRejected(
        "`synchronized`",
        () -> {
          synchronized (lock) {
            return a;
          }
        });
  }

  @Test
  public void assertStatement() {
    int a = 1;
    assertMatchesJdk(
        () -> {
          assert a > 0;
          return a;
        });
  }

  @Test
  public void failingAssertFollowsTheHostsAssertionStatus() {
    int a = -1;
    if (UnsupportedStatementTest.class.desiredAssertionStatus())
      assertTraps(
          "a must be positive",
          () -> {
            assert a > 0 : "a must be positive";
            return a;
          });
    else
      assertMatchesJdk(
          "with assertions off the guard is dead",
          () -> {
            assert a > 0 : "a must be positive";
            return a;
          });
  }

  @Test
  public void explicitAssertionErrorTrapsOnTheDevice() {
    int a = -1;
    assertTraps(
        "negative",
        () -> {
          if (a < 0) throw new AssertionError("negative");
          return a;
        });
  }

  @Test
  public void unreachableAssertionErrorDoesNotTrap() {
    int a = 1;
    assertMatchesJdk(
        () -> {
          if (a < 0) throw new AssertionError("negative");
          return a;
        });
  }

  @Test
  public void instanceofPattern() {
    Object o = Integer.valueOf(3);
    assertRejected("`instanceof`", () -> o instanceof Integer i ? i + 1 : 0);
  }

  @Test
  public void stringLiteralAndLength() {
    assertRejected("`String` literal", () -> "hello".length());
  }

  @Test
  public void stringConcatenation() {
    int a = 1;
    assertRejected("string concatenation", () -> ("v" + a).length());
  }
}
