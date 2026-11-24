package polyregion.javalang;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.LaunchRule;
import polyregion.javalang.Par;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.PlatformInfo;
import polyregion.jvm.twin.Marshal;

public class MirrorResidencyTest {

  private static final int ELEMENTS = 1 << 20;
  private static final int ITERATIONS = 16;

  @Rule
  public final TestRule kernels =
      LaunchRule.requireLaunch(method -> method.equals("aCreatedContextWaitsBeforeItsMirrorsAreFreed"));

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

    Marshal first = Marshal.inPlace(context, null, null);
    first.stage(buffer);
    Marshal second = Marshal.inPlace(context, null, null);
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

  @Test
  public void aCreatedContextWaitsBeforeItsMirrorsAreFreed() {
    assumeTrue(
        "created CUDA contexts only", System.getProperty("polyregion.platform", "").equalsIgnoreCase("cuda"));
    Context created = null;
    for (PlatformInfo platform : Context.platforms())
      if (platform.name.toLowerCase().contains("cuda") && platform.devices.length > 0) {
        created = Context.create(platform.index, platform.devices[0].index);
        break;
      }
    assertNotNull("a CUDA device", created);
    try {
      int[] xs = new int[ELEMENTS];
      Arrays.fill(xs, 1);
      Context defaultContext = Context.current();
      long defaultWaits = defaultContext.stats().hostReleaseWaits;
      long waits = created.stats().hostReleaseWaits;
      assertEquals(ELEMENTS, Par.ints(xs).on(created).sum());
      assertTrue("the created context did not wait", created.stats().hostReleaseWaits > waits);
      assertEquals("the default context waited for another context", defaultWaits, defaultContext.stats().hostReleaseWaits);
    } finally {
      created.close();
    }
  }
}
