package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static polyregion.javalang.Runtime.offload;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;

public class ControlFlowTest {

  @Test
  public void whileLoop() {
    int n = 5;
    assertEquals(
        10,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  int i = 0;
                  while (i < n) {
                    acc += i;
                    i++;
                  }
                  return acc;
                }));
  }

  @Test
  public void doWhileRunsOnce() {
    int n = 0;
    assertEquals(
        1,
        (int)
            offload(
                () -> {
                  int count = 0;
                  do {
                    count++;
                  } while (count < n);
                  return count;
                }));
  }

  @Test
  public void forWithMultipleUpdates() {
    assertEquals(
        8,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0, j = 4; i < j; i++, j--) acc += i + j;
                  return acc;
                }));
  }

  @Test
  public void nestedLoops() {
    assertEquals(
        9,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) acc++;
                  return acc;
                }));
  }

  @Test
  public void breakExitsLoop() {
    assertEquals(
        6,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 100; i++) {
                    if (i > 3) break;
                    acc += i;
                  }
                  return acc;
                }));
  }

  @Test
  public void continueSkipsIteration() {
    assertEquals(
        20,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 10; i++) {
                    if (i % 2 != 0) continue;
                    acc += i;
                  }
                  return acc;
                }));
  }

  @Test
  public void enhancedForOverArray() {
    int[] xs = {1, 2, 3, 4};
    assertEquals(
        10,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int x : xs) acc += x;
                  return acc;
                }));
  }

  @Test
  public void enhancedForWrites() {
    int[] xs = {1, 2, 3};
    int[] out = new int[3];
    OffloadRunnable r =
        () -> {
          int i = 0;
          for (int x : xs) {
            out[i] = x * x;
            i++;
          }
        };
    offload(r);
    assertArrayEquals(new int[] {1, 4, 9}, out);
  }

  @Test
  public void switchOnInt() {
    int k = 2;
    assertEquals(
        20,
        (int)
            offload(
                () -> {
                  int r;
                  switch (k) {
                    case 1:
                      r = 10;
                      break;
                    case 2:
                      r = 20;
                      break;
                    default:
                      r = -1;
                  }
                  return r;
                }));
  }

  @Test
  public void switchFallsThrough() {
    int k = 1;
    assertEquals(
        3,
        (int)
            offload(
                () -> {
                  int r = 0;
                  switch (k) {
                    case 1:
                      r += 1;
                    case 2:
                      r += 2;
                      break;
                    case 3:
                      r += 100;
                      break;
                  }
                  return r;
                }));
  }

  @Test
  public void switchDefaultOnly() {
    int k = 99;
    assertEquals(
        -1,
        (int)
            offload(
                () -> {
                  int r;
                  switch (k) {
                    case 1:
                      r = 1;
                      break;
                    default:
                      r = -1;
                  }
                  return r;
                }));
  }

  @Test
  public void earlyReturnFromLoop() {
    int[] xs = {5, 7, 9, 11};
    assertEquals(
        2,
        (int)
            offload(
                () -> {
                  for (int i = 0; i < 4; i++) if (xs[i] == 9) return i;
                  return -1;
                }));
  }

  @Test
  public void chainedTernary() {
    int a = 7;
    assertEquals(2, (int) offload(() -> a < 5 ? 1 : a < 10 ? 2 : 3));
  }

  @Test
  public void shortCircuitAvoidsDivideByZero() {
    int d = 0;
    OffloadFunction<Integer> f = () -> (d != 0 && 10 / d > 1) ? 1 : 0;
    assertEquals(0, (int) offload(f));
  }

  @Test
  public void continueInWhileReevaluatesCondition() {
    int n = 6;
    assertEquals(
        9,
        (int)
            offload(
                () -> {
                  int i = 0;
                  int acc = 0;
                  while (i < n) {
                    i++;
                    if (i % 2 == 0) continue;
                    acc += i;
                  }
                  return acc;
                }));
  }

  @Test
  public void continueInDoWhile() {
    assertEquals(
        9,
        (int)
            offload(
                () -> {
                  int i = 0;
                  int acc = 0;
                  do {
                    i++;
                    if (i % 2 == 0) continue;
                    acc += i;
                  } while (i < 6);
                  return acc;
                }));
  }

  @Test
  public void switchInsideLoop() {
    int[] xs = {1, 2, 3, 1};
    assertEquals(
        23,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 4; i++) {
                    switch (xs[i]) {
                      case 1:
                        acc += 10;
                        break;
                      case 2:
                        acc += 2;
                        break;
                      default:
                        acc += 1;
                    }
                  }
                  return acc;
                }));
  }

  @Test
  public void breakInsideSwitchDoesNotExitLoop() {
    assertEquals(
        3,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 3; i++) {
                    switch (i) {
                      default:
                        acc += 1;
                        break;
                    }
                  }
                  return acc;
                }));
  }

  @Test
  public void nestedLoopBreakOnlyExitsInner() {
    assertEquals(
        3,
        (int)
            offload(
                () -> {
                  int acc = 0;
                  for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 10; j++) {
                      if (j > 0) break;
                      acc++;
                    }
                  return acc;
                }));
  }

  @Test
  public void emptyBodyLoop() {
    assertEquals(
        5,
        (int)
            offload(
                () -> {
                  int i = 0;
                  while (i < 5) i++;
                  return i;
                }));
  }
}
