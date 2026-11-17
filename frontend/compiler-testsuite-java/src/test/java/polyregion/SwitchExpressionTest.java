package polyregion;

import org.junit.Test;

public class SwitchExpressionTest extends BaseSuite {

  @Test
  public void arrowSwitchStatement() {
    int k = 2;
    assertMatchesJdk(
        () -> {
          int r = 0;
          switch (k) {
            case 1 -> r = 10;
            case 2 -> r = 20;
            default -> r = -1;
          }
          return r;
        });
  }

  @Test
  public void switchExpression() {
    int k = 3;
    assertMatchesJdk(
        () ->
            switch (k) {
              case 1 -> 10;
              case 2 -> 20;
              default -> 30;
            });
  }

  @Test
  public void switchExpressionWithYield() {
    int k = 1;
    assertMatchesJdk(
        () ->
            switch (k) {
              case 1 -> {
                int t = 5;
                yield t * 2;
              }
              default -> 0;
            });
  }

  @Test
  public void multipleLabelsPerCase() {
    int k = 3;
    assertMatchesJdk(
        () ->
            switch (k) {
              case 1, 2, 3 -> 100;
              default -> 0;
            });
  }

  @Test
  public void switchOnChar() {
    char c = 'b';
    assertMatchesJdk(
        () ->
            switch (c) {
              case 'a' -> 1;
              case 'b' -> 2;
              default -> 0;
            });
  }
}
