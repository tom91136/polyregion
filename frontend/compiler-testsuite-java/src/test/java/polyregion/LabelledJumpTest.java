package polyregion;

import org.junit.Test;

public class LabelledJumpTest extends BaseSuite {

  @Test
  public void labelledBreakExitsOuterLoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) {
              if (j == 2) continue outer;
              if (i == 3) break outer;
              acc++;
            }
          return acc;
        });
  }

  @Test
  public void labelledContinueSkipsOuterIteration() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) if (j > i) continue outer;
            acc += i;
          }
          return acc;
        });
  }

  @Test
  public void plainBreakInsideLabelledLoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 5; i++) {
            if (i == 3) break;
            acc += i;
          }
          return acc;
        });
  }

  @Test
  public void plainContinueInsideLabelledLoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 6; i++) {
            if (i % 2 == 0) continue;
            acc += i;
          }
          return acc;
        });
  }

  @Test
  public void labelledWhileLoop() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          int i = 0;
          outer:
          while (i < 10) {
            i++;
            if (i > 4) break outer;
            acc += i;
          }
          return acc;
        });
  }

  @Test
  public void breakOuterFromSwitch() {
    int[] xs = {1, 2, 9, 3};
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 4; i++) {
            switch (xs[i]) {
              case 9:
                break outer;
              default:
                acc += xs[i];
            }
          }
          return acc;
        });
  }

  @Test
  public void continueOuterFromDoWhile() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 5; i++) {
            int j = 0;
            do {
              j++;
              if (j == 2) continue outer;
            } while (j < 5);
            acc += 100;
          }
          return acc;
        });
  }

  @Test
  public void nestedLabelledLoops() {
    assertMatchesJdk(
        () -> {
          int acc = 0;
          outer:
          for (int i = 0; i < 3; i++) {
            inner:
            for (int j = 0; j < 3; j++) {
              if (j == 1) continue outer;
              if (i == 2) break outer;
              acc++;
            }
          }
          return acc;
        });
  }

  @Test
  public void labelledBlockBreak() {
    assertMatchesJdk(
        () -> {
          int acc = 1;
          done:
          {
            if (acc == 1) break done;
            acc = 99;
          }
          return acc;
        });
  }
}
