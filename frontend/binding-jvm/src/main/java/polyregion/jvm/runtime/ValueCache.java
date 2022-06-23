package polyregion.jvm.runtime;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

public class ValueCache<V> {

  private final AtomicReference<V> reference = new AtomicReference<>();
  private final Supplier<V> supplier;

  public ValueCache(Supplier<V> makeV) {
    this.supplier = makeV;
  }

  public V getCached() {
    V result = reference.get();
    if (result == null) {
      result = supplier.get();
      if (!reference.compareAndSet(null, result)) {
        return reference.get();
      }
    }
    return result;
  }
}
