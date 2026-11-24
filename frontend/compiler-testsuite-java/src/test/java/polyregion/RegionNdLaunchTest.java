package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestRule;
import polyregion.jvm.runtime.Context;

public class RegionNdLaunchTest {

  @Rule public final TestRule kernels = LaunchRule.requireLaunch(name -> false);

  @Test
  public void forEachWritesRankTwoInRowMajorOrder() {
    int[] out = new int[3 * 4];
    Offload.for_each(3L, 4L, (i, j) -> out[(int) (i * 4 + j)] = (int) (i * 10 + j));
    int[] expected = new int[out.length];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) expected[i * 4 + j] = i * 10 + j;
    assertArrayEquals(expected, out);
  }

  @Test
  public void forEachWritesRankThreeInRowMajorOrder() {
    int[] out = new int[2 * 3 * 4];
    Offload.for_each(2L, 3L, 4L, (i, j, k) -> out[(int) ((i * 3 + j) * 4 + k)] = (int) (100 * i + 10 * j + k));
    int[] expected = new int[out.length];
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 4; k++) expected[(i * 3 + j) * 4 + k] = 100 * i + 10 * j + k;
    assertArrayEquals(expected, out);
  }

  @Test
  public void transformReducesRankTwo() {
    long total = Offload.transform_reduce(3L, 4L, 0L, (i, j) -> i * 4 + j, (l, r) -> l + r);
    assertEquals(66L, total);
  }

  @Test
  public void transformReducesRankThree() {
    double total = Offload.transform_reduce(2L, 3L, 4L, 0.0, (i, j, k) -> i + j + k, (l, r) -> l + r);
    assertEquals(84.0, total, 0.0);
  }

  @Test
  public void transformReducesRankTwoUnordered() {
    int total = Offload.transform_reduce_unordered(3L, 4L, 0, (i, j) -> (int) (i + j), (l, r) -> l + r);
    assertEquals(36, total);
  }

  @Test
  public void taskRunsTheBody() {
    int[] cell = new int[1];
    Offload.task(() -> cell[0] = 17);
    assertEquals(17, cell[0]);
  }

  @Test
  public void contextOverloadsLaunch() {
    Context ctx = Context.current();
    int[] out = new int[6];
    Offload.for_each(ctx, 2L, 3L, (i, j) -> out[(int) (i * 3 + j)] = (int) (i * 3 + j));
    long total = Offload.transform_reduce(ctx, 2L, 3L, 0L, (i, j) -> i * 3 + j, (l, r) -> l + r);
    int[] cell = new int[1];
    Offload.task(ctx, () -> cell[0] = 5);
    assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5}, out);
    assertEquals(15L, total);
    assertEquals(5, cell[0]);
    assertTrue(ctx.stats().launches > 0);
  }

  private static void fill(long n0, long n1, int[] out) {
    Offload.for_each(n0, n1, (i, j) -> out[(int) (i * n1 + j)] = (int) (100 * i + j));
  }

  @Test
  public void extentsAreCapturesSoOneKernelServesTwoShapes() {
    int[] first = new int[6];
    int[] second = new int[6];
    fill(2L, 3L, first);
    fill(3L, 2L, second);
    assertArrayEquals(new int[] {0, 1, 2, 100, 101, 102}, first);
    assertArrayEquals(new int[] {0, 1, 100, 101, 200, 201}, second);
  }
}
