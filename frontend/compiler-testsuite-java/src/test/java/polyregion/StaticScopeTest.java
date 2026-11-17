package polyregion;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class StaticScopeTest extends BaseSuite {

  static int counter = 0;
  static long ticks = 0;

  @Before
  public void reset() {
    counter = 0;
    ticks = 0;
    StaticHelpers.total = 0;
  }

  static int readCounter() {
    return counter;
  }

  static void bumpCounter() {
    counter = counter + 1;
  }

  static int mix() {
    return counter * 10 + StaticHelpers.total;
  }

  static int outer() {
    return inner() + 1;
  }

  static int inner() {
    return counter * 2;
  }

  static void addTicks(long n) {
    ticks = ticks + n;
  }

  @Test
  public void helperReadsStatic() {
    counter = 5;
    assertMatchesJdk(() -> readCounter());
  }

  @Test
  public void helperWritesStatic() {
    OffloadRunnable r = () -> bumpCounter();
    polyregion.javalang.Runtime.offload(r);
    assertEquals(1, counter);
  }

  @Test
  public void helperInAnotherClassWritesStatic() {
    OffloadRunnable r = () -> StaticHelpers.accumulate(4);
    polyregion.javalang.Runtime.offload(r);
    assertEquals(4, StaticHelpers.total);
  }

  @Test
  public void helperReadsStaticsOfTwoClasses() {
    counter = 3;
    StaticHelpers.total = 7;
    assertMatchesJdk("each static keeps its own buffer", () -> mix());
  }

  @Test
  public void staticReachedThroughTwoHelpers() {
    counter = 4;
    assertMatchesJdk(() -> outer());
  }

  @Test
  public void entryAndHelperShareOneBuffer() {
    counter = 2;
    OffloadRunnable r =
        () -> {
          counter = counter + 3;
          bumpCounter();
        };
    polyregion.javalang.Runtime.offload(r);
    assertEquals(6, counter);
  }

  @Test
  public void helperWritesStaticTwice() {
    OffloadRunnable r =
        () -> {
          StaticHelpers.accumulate(4);
          StaticHelpers.accumulate(5);
        };
    polyregion.javalang.Runtime.offload(r);
    assertEquals(9, StaticHelpers.total);
  }

  @Test
  public void helperWritesWideStatic() {
    ticks = 1L;
    OffloadRunnable r = () -> addTicks(1L << 40);
    polyregion.javalang.Runtime.offload(r);
    assertEquals((1L << 40) + 1L, ticks);
  }
}
