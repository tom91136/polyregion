package polyregion.javalang;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.Queue;

public class LaunchResidencyTest {

  private static Queue.Stats stats() {
    return Context.current().queue().stats();
  }

  @Test
  public void everyStagedBufferIsReleasedAfterTheLaunch() {
    int[] xs = {1, 2, 3};
    double[] ys = {0.5, 1.5};
    Queue.Stats before = stats();
    OffloadFunction<Double> f = () -> xs[0] + xs[2] + ys[1];
    assertEquals(Double.valueOf(5.5), Runtime.offload(f));
    Queue.Stats after = stats();
    long invalidated = after.invalidations - before.invalidations;
    assertEquals(0, invalidated);
    assertEquals(invalidated, after.releases - before.releases);
  }

  @Test
  public void aFailingLaunchStillReleases() {
    int[] xs = {1, 2, 3};
    Queue.Stats before = stats();
    OffloadFunction<Integer> f =
        () -> {
          assert xs[0] == 0 : "boom";
          return xs[1];
        };
    try {
      Runtime.offload(f);
    } catch (AssertionError | RuntimeException expected) {
    }
    Queue.Stats after = stats();
    assertEquals(
        after.invalidations - before.invalidations, after.releases - before.releases);
  }

  @Test
  public void concurrentLaunchesDoNotReleaseEachOthersBuffers() throws Exception {
    int threads = 8;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Future<Integer>> results = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int seed = t;
        results.add(
            pool.submit(
                (Callable<Integer>)
                    () -> {
                      int total = 0;
                      for (int i = 0; i < 25; i++) {
                        int[] xs = {seed, i, 3};
                        int[][] grid = {{seed, 1}, {i, 2}};
                        OffloadFunction<Integer> f = () -> xs[0] + xs[1] + xs[2] + grid[1][0];
                        total += Runtime.offload(f) - (seed + 2 * i + 3);
                      }
                      return total;
                    }));
      }
      for (Future<Integer> r : results) assertEquals(Integer.valueOf(0), r.get());
    } finally {
      pool.shutdownNow();
    }
  }
}
