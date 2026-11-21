package polyregion.javalang;

import java.util.concurrent.Callable;

public final class ChildLoaderHarness implements Callable<Integer> {
  @Override
  public Integer call() {
    ChildLoaderCell c = new ChildLoaderCell();
    OffloadRunnable r = () -> c.v += 4;
    Runtime.offload(r);
    return c.v;
  }
}
