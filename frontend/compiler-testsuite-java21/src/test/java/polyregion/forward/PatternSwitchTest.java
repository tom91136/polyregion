package polyregion.forward;

import org.junit.Test;
import polyregion.BaseSuite;

public class PatternSwitchTest extends BaseSuite {

  @Test
  public void typePatternArm() {
    Object o = Integer.valueOf(3);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (o) {
              case Integer i -> i + 1;
              default -> 0;
            });
  }

  @Test
  public void guardedPatternArm() {
    Object o = Integer.valueOf(3);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (o) {
              case Integer i when i > 2 -> 1;
              case Integer i -> 2;
              default -> 0;
            });
  }

  @Test
  public void nullCaseLabel() {
    Integer boxed = Integer.valueOf(3);
    assertRejected(
        Rejections.PATTERN,
        () ->
            switch (boxed) {
              case null -> -1;
              case Integer i -> i;
            });
  }

  @Test
  public void patternArmInStatementSwitch() {
    Object o = Integer.valueOf(3);
    int[] out = new int[1];
    assertRejected(
        Rejections.STATEMENT_SWITCH_ARM,
        () -> {
          switch (o) {
            case Integer i -> out[0] = i;
            default -> out[0] = 0;
          }
          return out[0];
        });
  }
}
