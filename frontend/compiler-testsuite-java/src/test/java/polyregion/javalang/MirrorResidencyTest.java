package polyregion.javalang;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.LaunchRule;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.twin.Marshal;

public class MirrorResidencyTest {

  private static final int ELEMENTS = 1 << 20;
  private static final int ITERATIONS = 16;

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(method -> false);

  @Test
  public void temporaryCapturesLeaveNoDeviceMirrors() {
    Context context = Context.current();
    int[] seed = new int[ELEMENTS];
    Arrays.fill(seed, 7);
    assertEquals(8 * ELEMENTS, Arrays.stream(seed).parallel().map(x -> x + 1).sum());

    long before = context.stats().mirrorAllocations;
    long waitsBefore = context.stats().hostReleaseWaits;
    for (int i = 0; i < ITERATIONS; i++) {
      int[] xs = new int[ELEMENTS];
      Arrays.fill(xs, i);
      assertEquals((i + 1) * ELEMENTS, Arrays.stream(xs).parallel().map(x -> x + 1).sum());
    }
    assertEquals("a launch mirror leaked", before, context.stats().mirrorAllocations);
    assertTrue(
        "closing a shared-mode marshal did not wait for the device queue",
        context.stats().hostReleaseWaits > waitsBefore);
  }

  @Test
  public void sharedHostBlocksAreReleasedByTheLastMarshal() {
    Context context = Context.current();
    assertEquals(3, Arrays.stream(new int[] {1, 2}).parallel().sum());
    ByteBuffer buffer = ByteBuffer.allocateDirect(1 << 20).order(ByteOrder.nativeOrder());
    long forgets = context.stats().hostForgets;
    long blocks = context.stats().hostBlocks;
    long records = context.stats().hostRecords;

    Marshal first = Marshal.inPlace(null, null);
    first.stage(buffer);
    Marshal second = Marshal.inPlace(null, null);
    second.stage(buffer);
    assertEquals(blocks + 1, context.stats().hostBlocks);
    assertEquals(records + 2, context.stats().hostRecords);

    first.close();
    assertEquals(blocks + 1, context.stats().hostBlocks);
    assertEquals(records + 1, context.stats().hostRecords);
    assertEquals("the first close released the shared mirror", forgets, context.stats().hostForgets);

    second.close();
    assertEquals(blocks, context.stats().hostBlocks);
    assertEquals(records, context.stats().hostRecords);
    assertEquals("the last close did not release the mirror", forgets + 1, context.stats().hostForgets);
  }
}
