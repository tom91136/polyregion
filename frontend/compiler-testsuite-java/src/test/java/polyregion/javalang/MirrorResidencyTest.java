package polyregion.javalang;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.LaunchRule;
import polyregion.jvm.runtime.Context;

public class MirrorResidencyTest {

  private static final int ELEMENTS = 1 << 20;
  private static final int ITERATIONS = 16;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  @Test
  public void temporaryCapturesLeaveNoDeviceMirrors() {
    int[] seed = new int[ELEMENTS];
    Arrays.fill(seed, 7);
    assertEquals(8 * ELEMENTS, Arrays.stream(seed).parallel().map(x -> x + 1).sum());

    long before = Context.current().stats().mirrorAllocations;
    for (int i = 0; i < ITERATIONS; i++) {
      int[] xs = new int[ELEMENTS];
      Arrays.fill(xs, i);
      assertEquals((i + 1) * ELEMENTS, Arrays.stream(xs).parallel().map(x -> x + 1).sum());
    }
    long after = Context.current().stats().mirrorAllocations;
    assertTrue(
        "temporary mirrors grew from " + before + " to " + after + " over " + ITERATIONS + " launches",
        after <= before + 8);
  }
}
