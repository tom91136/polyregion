package polyregion;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import polyregion.javalang.OffloadRunnable;

public class ThisCaptureTest extends BaseSuite {

  int field = 7;
  int[] data = {1, 2, 3};

  int helper(int x) {
    return x + field;
  }

  @Test
  public void readsInstanceField() {
    assertMatchesJdk(() -> field * 2);
  }

  @Test
  public void writesInstanceField() {
    OffloadRunnable r = () -> field = 99;
    polyregion.javalang.Runtime.offload(r);
    assertEquals(99, field);
  }

  @Test
  public void readsInstanceArrayField() {
    assertMatchesJdk(() -> data[0] + data[2]);
  }

  @Test
  public void callsInstanceMethod() {
    assertMatchesJdk(() -> helper(3));
  }

  @Test
  public void explicitThis() {
    assertMatchesJdk(() -> this.field + 1);
  }
}
