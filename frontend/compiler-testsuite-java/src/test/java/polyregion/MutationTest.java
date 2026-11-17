package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Before;
import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class MutationTest {

  static int staticCounter = 0;
  static int[] staticArray = {1, 2, 3};

  int instanceField = 4;

  @Before
  public void reset() {
    staticCounter = 0;
    staticArray = new int[] {1, 2, 3};
    Helpers.counter = 0;
  }

  @Test
  public void mutatesCapturedArrayInPlace() {
    int[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = xs[i] * 5;
        };
    offload(r);
    assertArrayEquals(new int[] {5, 10, 15}, xs);
  }

  @Test
  public void mutatesOnlyPartOfArray() {
    int[] xs = {1, 1, 1, 1};
    OffloadRunnable r =
        () -> {
          xs[1] = 9;
          xs[3] = 9;
        };
    offload(r);
    assertArrayEquals(new int[] {1, 9, 1, 9}, xs);
  }

  @Test
  public void mutatesArrayThroughStaticHelper() {
    int[] xs = {1, 2, 3};
    OffloadRunnable r = () -> Helpers.scale(xs, 3, 4);
    offload(r);
    assertArrayEquals(new int[] {4, 8, 12}, xs);
  }

  @Test
  public void scalarCaptureIsByValue() {
    int a = 5;
    int[] out = new int[1];
    OffloadRunnable r =
        () -> {
          int local = a;
          local = local + 100;
          out[0] = local;
        };
    offload(r);
    assertArrayEquals(new int[] {105}, out);
    assertEquals(5, a);
  }

  @Test
  public void callsStaticHelperInAnotherClass() {
    int a = 4;
    assertEquals(12, (int) offload(() -> Helpers.triple(a)));
  }

  @Test
  public void callsStaticHelperTakingArray() {
    int[] xs = {1, 2, 3, 4};
    assertEquals(10, (int) offload(() -> Helpers.sumRange(xs, 4)));
  }

  @Test
  public void staticHelperCallsAnotherStaticHelper() {
    int a = 2;
    assertEquals(16, (int) offload(() -> Helpers.viaAnotherHelper(a)));
  }

  @Test
  public void readsStaticFinalConstant() {
    assertEquals(10, (int) offload(() -> Helpers.CONSTANT));
  }

  @Test
  public void readsStaticField() {
    staticCounter = 12;
    assertEquals(12, (int) offload(() -> staticCounter));
  }

  @Test
  public void writesStaticField() {
    OffloadRunnable r = () -> staticCounter = 42;
    offload(r);
    assertEquals(42, staticCounter);
  }

  @Test
  public void incrementsStaticField() {
    staticCounter = 1;
    OffloadRunnable r = () -> staticCounter = staticCounter + 1;
    offload(r);
    assertEquals(2, staticCounter);
  }

  @Test
  public void writesStaticFieldInAnotherClass() {
    OffloadRunnable r = () -> Helpers.counter = 7;
    offload(r);
    assertEquals(7, Helpers.counter);
  }

  @Test
  public void mutatesStaticArray() {
    int[] xs = staticArray;
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = xs[i] + 10;
        };
    offload(r);
    assertArrayEquals(new int[] {11, 12, 13}, staticArray);
  }

  static long staticLong = 0L;
  static double staticDouble = 0.0;
  static boolean staticFlag = false;

  @Test
  public void writesStaticsOfEachWidth() {
    OffloadRunnable r =
        () -> {
          staticLong = 1L << 40;
          staticDouble = 2.5;
          staticFlag = true;
        };
    offload(r);
    assertEquals(1L << 40, staticLong);
    assertEquals(2.5, staticDouble, 0.0);
    assertEquals(true, staticFlag);
  }

  @Test
  public void accumulatesIntoStaticInLoop() {
    staticCounter = 0;
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < 5; i++) staticCounter = staticCounter + i;
        };
    offload(r);
    assertEquals(10, staticCounter);
  }

  @Test
  public void mixesStaticsCapturesAndArrays() {
    staticCounter = 100;
    int a = 5;
    int[] xs = {1, 2, 3};
    OffloadRunnable r =
        () -> {
          for (int i = 0; i < xs.length; i++) xs[i] = xs[i] + a;
          staticCounter = staticCounter + xs.length;
        };
    offload(r);
    assertArrayEquals(new int[] {6, 7, 8}, xs);
    assertEquals(103, staticCounter);
  }

  @Test
  public void staticCompoundAssignment() {
    staticCounter = 4;
    OffloadRunnable r = () -> staticCounter *= 3;
    offload(r);
    assertEquals(12, staticCounter);
  }

  @Test
  public void staticIncrement() {
    staticCounter = 7;
    OffloadRunnable r = () -> staticCounter++;
    offload(r);
    assertEquals(8, staticCounter);
  }

  @Test
  public void readsStaticFromTwoClasses() {
    staticCounter = 3;
    Helpers.counter = 4;
    assertEquals(7, (int) offload(() -> staticCounter + Helpers.counter));
  }

  @Test
  public void instanceFieldRoundTripsThroughLocalAndArray() {
    int f = instanceField;
    int[] out = new int[1];
    OffloadRunnable r = () -> out[0] = f * 3;
    offload(r);
    instanceField = out[0];
    assertEquals(12, instanceField);
  }
}
