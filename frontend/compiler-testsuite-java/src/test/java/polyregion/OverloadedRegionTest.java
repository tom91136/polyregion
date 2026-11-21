package polyregion;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import polyregion.javalang.OffloadFunction;

public class OverloadedRegionTest {

  static final class Left {
    int v = 3;
  }

  static final class Right {
    int w = 5;
  }

  static int run(Left c) {
    OffloadFunction<Integer> f = () -> c.v + 1;
    return polyregion.javalang.Runtime.offload(f);
  }

  static int run(Right o) {
    OffloadFunction<Integer> f = () -> o.w + 2;
    return polyregion.javalang.Runtime.offload(f);
  }

  @Test
  public void overloadsCapturingDifferentClasses() {
    assertEquals(4, run(new Left()));
    assertEquals(7, run(new Right()));
  }
}
