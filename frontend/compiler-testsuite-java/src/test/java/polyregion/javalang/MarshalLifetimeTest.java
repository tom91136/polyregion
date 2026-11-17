package polyregion.javalang;

import static org.junit.Assert.assertEquals;

import org.junit.After;
import org.junit.Test;
import polyregion.jvm.runtime.Device;

public class MarshalLifetimeTest {

  @After
  public void clearSeam() {
    BinaryOffloadExecutable.afterPack = null;
  }

  private static void collect(Device.Queue queue) {
    for (int i = 0; i < 8; i++) {
      System.gc();
      queue.references.get().size();
      try {
        Thread.sleep(5);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException("Interrupted while forcing a collection", e);
      }
    }
  }

  @Test
  public void pointerRowSurvivesACollectionBeforeLaunch() {
    int[][] grid = {{1, 2}, {3, 4}};
    BinaryOffloadExecutable.afterPack = MarshalLifetimeTest::collect;
    OffloadFunction<Integer> f = () -> grid[0][0] + grid[1][1];
    assertEquals(Integer.valueOf(5), Runtime.offload(f));
  }

  @Test
  public void nestedPointerRowsSurviveACollectionBeforeLaunch() {
    int[][][] cube = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    BinaryOffloadExecutable.afterPack = MarshalLifetimeTest::collect;
    OffloadFunction<Integer> f = () -> cube[0][1][0] + cube[1][0][1];
    assertEquals(Integer.valueOf(9), Runtime.offload(f));
  }
}
