package polyregion.javalang;

import static org.junit.Assert.assertEquals;

import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import polyregion.jvm.runtime.Device;
import polyregion.jvm.runtime.Platform;
import polyregion.jvm.runtime.Platforms;

public class QueueReleaseTest {

  private static final Platform PLATFORM = Platforms.create().Relocatable();

  private static Device.Queue queue() {
    Device[] devices = PLATFORM.devices();
    return devices[0].createQueue(0);
  }

  private static void register(Device.Queue queue, int[] xs, AtomicInteger writes) {
    queue.registerAndInvalidateIfAbsent(
        xs,
        a -> a.length * Integer.BYTES,
        (a, b) -> {
          writes.incrementAndGet();
          b.asIntBuffer().put(a);
        },
        (b, a) -> b.asIntBuffer().get(a),
        null);
  }

  @Test
  public void releaseAllDropsEveryLiveRegistration() {
    Device.Queue queue = queue();
    AtomicInteger writes = new AtomicInteger();
    int[] a = {1, 2, 3};
    int[] b = {4, 5, 6};
    int[] c = {7, 8, 9};
    register(queue, a, writes);
    register(queue, b, writes);
    register(queue, c, writes);
    assertEquals(3, queue.references.get().size());
    queue.releaseAll();
    assertEquals(0, queue.references.get().size());
    queue.close();
  }

  @Test
  public void closeDropsEveryLiveRegistration() {
    Device.Queue queue = queue();
    AtomicInteger writes = new AtomicInteger();
    int[] a = {1, 2, 3};
    int[] b = {4, 5, 6};
    register(queue, a, writes);
    register(queue, b, writes);
    queue.close();
    assertEquals(0, queue.references.get().size());
  }

  @Test
  public void reRegisteringAfterReleaseAllBuildsAFreshProxy() {
    Device.Queue queue = queue();
    AtomicInteger writes = new AtomicInteger();
    int[] xs = {1, 2, 3};
    register(queue, xs, writes);
    assertEquals(1, writes.get());
    register(queue, xs, writes);
    assertEquals("a live registration is reused", 1, writes.get());
    queue.releaseAll();
    register(queue, xs, writes);
    assertEquals("the released registration is rebuilt", 2, writes.get());
    queue.close();
  }

  @Test
  public void releaseAllWithExplicitObjectsLeavesTheRest() {
    Device.Queue queue = queue();
    AtomicInteger writes = new AtomicInteger();
    int[] a = {1, 2, 3};
    int[] b = {4, 5, 6};
    register(queue, a, writes);
    register(queue, b, writes);
    queue.releaseAll(a);
    assertEquals(1, queue.references.get().size());
    queue.sync(b, null);
    queue.close();
  }
}
