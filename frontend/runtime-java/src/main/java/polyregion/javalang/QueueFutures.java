package polyregion.javalang;

import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import polyregion.jvm.runtime.Completion;
import polyregion.jvm.runtime.Queue;

public final class QueueFutures {

  private QueueFutures() {}

  private static CompletableFuture<Void> adapt(Consumer<Completion> start) {
    CompletableFuture<Void> future = new CompletableFuture<>();
    try {
      start.accept(
          failure -> {
            if (failure == null) future.complete(null);
            else future.completeExceptionally(failure);
          });
    } catch (Throwable t) {
      future.completeExceptionally(t);
    }
    return future;
  }

  public static CompletableFuture<Void> invalidate(Queue queue, ByteBuffer... buffers) {
    return adapt(cb -> queue.invalidateAll(cb, buffers));
  }

  public static CompletableFuture<Void> sync(Queue queue, ByteBuffer... buffers) {
    return adapt(cb -> queue.syncAll(cb, buffers));
  }
}
