package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

/**
 * Explicit residency for JVM-owned host buffers on one {@link Context}.
 *
 * <p>A resident is a direct {@link ByteBuffer} (identified by object identity) with a device copy. {@link
 * #invalidate} uploads the host bytes, {@link #sync} writes the device bytes back to the host, {@link #release} frees
 * the device copy. On a shared address space device the device copy is the host memory itself and no bytes move.
 *
 * <p>Operations are in order with kernel launches on the context. A completion runs once, on a thread owned by the
 * queue and never on the calling thread or a driver thread, in the order the operations settled. A completion must
 * not wait for another queue completion ({@link #requireNotCompletionThread} fails fast in the adapters that block;
 * {@code CompletableFuture#join} and the like are not detected). An exception thrown by a completion goes to the
 * thread's uncaught exception handler. {@link #release} and {@link #close} block until pending copies settle. A
 * resident stays resident until {@link #release}, {@link #close} or the end of its scope; a buffer can be resident
 * on one context at a time, registering it on a second one fails. Failures
 * of the runtime (a closed context, a failed allocation or copy) reach the completion; a misused argument (null, a
 * non-direct or unregistered buffer) throws before anything starts.
 *
 * <p>The host owns the bytes between {@link #invalidate} and the launch, and again after {@link #sync}. The queue does
 * not detect host writes: a write after {@link #invalidate} reaches the device only through another {@link
 * #invalidate}, and a device write reaches the host only through {@link #sync}.
 *
 * <p>{@link #resident} layers a scope on top: it uploads on entry, hands out device pointers, and writes back and
 * releases on exit, also when the body throws.
 */
public final class Queue implements AutoCloseable {

  public static final class Stats {
    public final long invalidations, syncs, releases, stagedIn, stagedOut;

    Stats(long invalidations, long syncs, long releases, long stagedIn, long stagedOut) {
      this.invalidations = invalidations;
      this.syncs = syncs;
      this.releases = releases;
      this.stagedIn = stagedIn;
      this.stagedOut = stagedOut;
    }

    @Override
    public String toString() {
      return "Stats{invalidations="
          + invalidations
          + ", syncs="
          + syncs
          + ", releases="
          + releases
          + ", stagedIn="
          + stagedIn
          + ", stagedOut="
          + stagedOut
          + '}';
    }
  }

  /**
   * The residents of one {@link #resident} call. The scope uploads a snapshot of each array on entry. Host writes to a
   * resident array inside the scope are not tracked: the device never sees them unless {@link #invalidate} uploads
   * them, and the write-back on exit overwrites them. For a direct buffer on a shared address space device the host
   * and the kernel alias the same bytes.
   */
  public static final class Scope {
    private final Queue queue;

    private Scope(Queue queue) {
      this.queue = queue;
    }

    public long pointer(Object resident) {
      long pointer = queue.devicePointer(resident);
      if (pointer == 0) throw new IllegalArgumentException("Not resident: " + resident);
      return pointer;
    }

    public void invalidate(Object... residents) {
      for (Object resident : residents) {
        ByteBuffer buffer = queue.stagingOf(resident);
        if (resident != buffer) {
          queue.encode(resident, buffer);
          queue.stagedIn.incrementAndGet();
        }
        queue.await(done -> queue.start(buffer, true, done, false));
      }
    }

    public void sync(Object... residents) {
      for (Object resident : residents) {
        ByteBuffer buffer = queue.stagingOf(resident);
        queue.await(done -> queue.start(buffer, false, done, false));
        if (resident != buffer) {
          queue.decode(buffer, resident);
          queue.stagedOut.incrementAndGet();
        }
      }
    }
  }

  private static final class Entry {
    final ByteBuffer region;
    final long bytes;
    final long pointer;
    final boolean shared;
    int pending;

    Entry(ByteBuffer region, long bytes, long pointer, boolean shared) {
      this.region = region;
      this.bytes = bytes;
      this.pointer = pointer;
      this.shared = shared;
    }
  }

  private static final class Owned {
    final Object resident;
    final ByteBuffer buffer;
    boolean uploaded;

    Owned(Object resident, ByteBuffer buffer) {
      this.resident = resident;
      this.buffer = buffer;
    }
  }

  interface Hook {
    void at(String point);
  }

  static volatile Hook hook = point -> {};

  private static final IdentityHashMap<ByteBuffer, Queue> OWNERS = new IdentityHashMap<>();

  private static final class CompletionThread extends Thread {
    CompletionThread(Runnable task) {
      super(task, "polyregion-queue");
      setDaemon(true);
    }
  }

  private final Context context;
  private final Object lock = new Object();
  private final IdentityHashMap<ByteBuffer, Entry> entries = new IdentityHashMap<>();
  private final IdentityHashMap<Object, ByteBuffer> staging = new IdentityHashMap<>();
  private final ThreadPoolExecutor completions;
  private final AtomicLong invalidations = new AtomicLong();
  private final AtomicLong syncs = new AtomicLong();
  private final AtomicLong releases = new AtomicLong();
  private final AtomicLong stagedIn = new AtomicLong();
  private final AtomicLong stagedOut = new AtomicLong();

  Queue(Context context) {
    this.context = context;
    this.completions =
        new ThreadPoolExecutor(
            1,
            1,
            5,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(),
            CompletionThread::new);
    this.completions.allowCoreThreadTimeOut(true);
  }

  private static void requireDirect(ByteBuffer buffer) {
    Objects.requireNonNull(buffer, "buffer");
    if (!buffer.isDirect()) throw new IllegalArgumentException("Expected a direct buffer");
    if (!buffer.hasRemaining()) throw new IllegalArgumentException("Expected a non-empty buffer");
  }

  private Entry entryOf(ByteBuffer buffer) {
    synchronized (lock) {
      Entry entry = entries.get(buffer);
      if (entry == null) throw new IllegalArgumentException("Not resident: " + buffer);
      return entry;
    }
  }

  private ByteBuffer stagingOf(Object resident) {
    synchronized (lock) {
      ByteBuffer buffer = resident instanceof ByteBuffer ? (ByteBuffer) resident : staging.get(resident);
      if (buffer == null || !entries.containsKey(buffer))
        throw new IllegalArgumentException("Not resident: " + resident);
      return buffer;
    }
  }

  private Entry acquire(ByteBuffer host, boolean upload) {
    synchronized (lock) {
      Entry found = entries.get(host);
      if (found != null) {
        hook.at("acquire.found");
        found.pending++;
        return found;
      }
    }
    if (!upload) throw new IllegalArgumentException("Not resident: " + host);
    requireUnowned(host);
    context.requireOpen();
    hook.at("allocate");
    ByteBuffer region = host.slice();
    long bytes = region.capacity();
    boolean shared = context.sharedAddressSpace();
    Entry fresh = new Entry(region, bytes, shared ? Context.addressOf(region) : context.malloc(bytes), shared);
    Entry winner = null;
    synchronized (OWNERS) {
      Queue owner = OWNERS.get(host);
      if (owner == null || owner == this)
        synchronized (lock) {
          winner = entries.get(host);
          if (winner == null) {
            OWNERS.put(host, this);
            entries.put(host, fresh);
            winner = fresh;
          }
          winner.pending++;
        }
    }
    if (winner != fresh && !shared) context.freeRaw(fresh.pointer);
    if (winner == null) throw new IllegalArgumentException("Already resident on another context: " + host);
    return winner;
  }

  private void requireUnowned(ByteBuffer host) {
    Queue owner = ownerOf(host);
    if (owner != null && owner != this)
      throw new IllegalArgumentException("Already resident on another context: " + host);
  }

  private Entry register(ByteBuffer host) {
    Entry entry = acquire(host, true);
    settle(entry);
    return entry;
  }

  private void settle(Entry entry) {
    synchronized (lock) {
      entry.pending--;
      lock.notifyAll();
    }
  }

  static void report(Throwable failure) {
    Thread thread = Thread.currentThread();
    thread.getUncaughtExceptionHandler().uncaughtException(thread, failure);
  }

  void dispatch(Completion cb, Throwable failure) {
    completions.execute(
        () -> {
          try {
            cb.complete(failure);
          } catch (Throwable t) {
            report(t);
          }
        });
  }

  public static void requireNotCompletionThread() {
    if (Thread.currentThread() instanceof CompletionThread)
      throw new IllegalStateException(
          "Cannot wait on the queue from one of its completions: the wait would block the thread that runs it");
  }

  private long start(ByteBuffer host, boolean upload, Completion cb, boolean dispatched) {
    Completion settled = dispatched ? failure -> dispatch(cb, failure) : cb;
    Entry entry;
    try {
      entry = acquire(host, upload);
    } catch (PolyregionRuntimeException e) {
      settled.complete(e);
      return 0;
    }
    (upload ? invalidations : syncs).incrementAndGet();
    Completion done =
        failure -> {
          settle(entry);
          hook.at("settled");
          settled.complete(failure);
        };
    hook.at("copy.begin");
    if (entry.shared) done.complete(null);
    else
      try {
        if (upload) context.copyToDeviceDirect(entry.region, entry.pointer, entry.bytes, done);
        else context.copyFromDeviceDirect(entry.pointer, entry.region, entry.bytes, done);
      } catch (RuntimeException e) {
        done.complete(e);
      }
    hook.at("copy.end");
    return entry.pointer;
  }

  private void await(java.util.function.Consumer<Completion> start) {
    CountDownLatch latch = new CountDownLatch(1);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    start.accept(
        f -> {
          failure.set(f);
          latch.countDown();
        });
    try {
      latch.await();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new PolyregionRuntimeException("Interrupted while waiting for the device");
    }
    rethrow(failure.get());
  }

  private static void rethrow(Throwable failure) {
    if (failure == null) return;
    if (failure instanceof RuntimeException) throw (RuntimeException) failure;
    if (failure instanceof Error) throw (Error) failure;
    throw new PolyregionRuntimeException(failure.toString());
  }

  public long invalidate(ByteBuffer host, Completion cb) {
    Objects.requireNonNull(cb, "cb");
    requireDirect(host);
    return start(host, true, cb, true);
  }

  public void sync(ByteBuffer host, Completion cb) {
    Objects.requireNonNull(cb, "cb");
    requireDirect(host);
    entryOf(host);
    start(host, false, cb, true);
  }

  public void release(ByteBuffer host) {
    requireDirect(host);
    Entry entry;
    synchronized (lock) {
      entry = entries.get(host);
      if (entry == null) throw new IllegalArgumentException("Not resident: " + host);
      while (entry.pending > 0) {
        try {
          lock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new PolyregionRuntimeException("Interrupted while waiting for the device");
        }
      }
      if (entries.get(host) != entry) throw new IllegalArgumentException("Not resident: " + host);
      entries.remove(host);
      staging.values().removeIf(b -> b == host);
    }
    synchronized (OWNERS) {
      OWNERS.remove(host);
    }
    releases.incrementAndGet();
    if (!entry.shared) context.freeRaw(entry.pointer);
  }

  private ByteBuffer[] residentsOf(ByteBuffer[] hosts) {
    synchronized (lock) {
      return hosts.length == 0 ? entries.keySet().toArray(new ByteBuffer[0]) : hosts;
    }
  }

  private void all(boolean upload, Completion cb, ByteBuffer[] hosts) {
    Objects.requireNonNull(cb, "cb");
    ByteBuffer[] targets = residentsOf(hosts);
    for (ByteBuffer host : targets) {
      requireDirect(host);
      if (!upload) entryOf(host);
    }
    if (targets.length == 0) {
      dispatch(cb, null);
      return;
    }
    AtomicInteger remaining = new AtomicInteger(targets.length);
    AtomicReference<Throwable> first = new AtomicReference<>();
    Completion each =
        failure -> {
          if (failure != null) first.compareAndSet(null, failure);
          if (remaining.decrementAndGet() == 0) cb.complete(first.get());
        };
    for (ByteBuffer host : targets) start(host, upload, each, true);
  }

  public void invalidateAll(Completion cb, ByteBuffer... hosts) {
    all(true, cb, hosts);
  }

  public void syncAll(Completion cb, ByteBuffer... hosts) {
    all(false, cb, hosts);
  }

  public void releaseAll(ByteBuffer... hosts) {
    for (ByteBuffer host : residentsOf(hosts)) release(host);
  }

  /** The queue the buffer is resident on, or null. */
  public static Queue ownerOf(ByteBuffer buffer) {
    synchronized (OWNERS) {
      return OWNERS.get(buffer);
    }
  }

  public Context context() {
    return context;
  }

  public long devicePointer(Object resident) {
    synchronized (lock) {
      ByteBuffer buffer = resident instanceof ByteBuffer ? (ByteBuffer) resident : staging.get(resident);
      Entry entry = buffer == null ? null : entries.get(buffer);
      return entry == null ? 0 : entry.pointer;
    }
  }

  public Stats stats() {
    return new Stats(
        invalidations.get(), syncs.get(), releases.get(), stagedIn.get(), stagedOut.get());
  }

  private static int elementSize(Object array) {
    if (array instanceof byte[]) return 1;
    if (array instanceof short[]) return Short.BYTES;
    if (array instanceof char[]) return Character.BYTES;
    if (array instanceof int[]) return Integer.BYTES;
    if (array instanceof long[]) return Long.BYTES;
    if (array instanceof float[]) return Float.BYTES;
    if (array instanceof double[]) return Double.BYTES;
    throw new IllegalArgumentException(
        "Expected a direct buffer or an array of a primitive type other than boolean, got " + array);
  }

  private static int length(Object array) {
    return java.lang.reflect.Array.getLength(array);
  }

  private void encode(Object array, ByteBuffer target) {
    ByteBuffer view = target.duplicate().order(ByteOrder.nativeOrder());
    if (array instanceof byte[]) view.put((byte[]) array);
    else if (array instanceof short[]) view.asShortBuffer().put((short[]) array);
    else if (array instanceof char[]) view.asCharBuffer().put((char[]) array);
    else if (array instanceof int[]) view.asIntBuffer().put((int[]) array);
    else if (array instanceof long[]) view.asLongBuffer().put((long[]) array);
    else if (array instanceof float[]) view.asFloatBuffer().put((float[]) array);
    else view.asDoubleBuffer().put((double[]) array);
  }

  private void decode(ByteBuffer source, Object array) {
    ByteBuffer view = source.duplicate().order(ByteOrder.nativeOrder());
    if (array instanceof byte[]) view.get((byte[]) array);
    else if (array instanceof short[]) view.asShortBuffer().get((short[]) array);
    else if (array instanceof char[]) view.asCharBuffer().get((char[]) array);
    else if (array instanceof int[]) view.asIntBuffer().get((int[]) array);
    else if (array instanceof long[]) view.asLongBuffer().get((long[]) array);
    else if (array instanceof float[]) view.asFloatBuffer().get((float[]) array);
    else view.asDoubleBuffer().get((double[]) array);
  }

  private void enter(Object[] residents, List<Owned> owned) {
    for (Object resident : residents) {
      Objects.requireNonNull(resident, "resident");
      boolean known;
      synchronized (lock) {
        known = resident instanceof ByteBuffer ? entries.containsKey(resident) : staging.containsKey(resident);
      }
      if (known) continue;
      ByteBuffer buffer;
      if (resident instanceof ByteBuffer) {
        buffer = (ByteBuffer) resident;
        requireDirect(buffer);
      } else {
        int bytes = Math.multiplyExact(elementSize(resident), length(resident));
        if (bytes == 0) throw new IllegalArgumentException("Expected a non-empty array");
        buffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder());
        encode(resident, buffer);
        stagedIn.incrementAndGet();
      }
      register(buffer);
      Owned entry = new Owned(resident, buffer);
      owned.add(entry);
      if (resident != buffer)
        synchronized (lock) {
          staging.put(resident, buffer);
        }
      await(done -> start(buffer, true, done, false));
      entry.uploaded = true;
    }
  }

  private Throwable leave(List<Owned> owned) {
    Throwable failure = null;
    for (Owned entry : owned) {
      try {
        if (entry.uploaded) {
          await(done -> start(entry.buffer, false, done, false));
          if (entry.resident != entry.buffer) {
            decode(entry.buffer, entry.resident);
            stagedOut.incrementAndGet();
          }
        }
      } catch (Throwable t) {
        if (failure == null) failure = t;
        else failure.addSuppressed(t);
      }
      try {
        release(entry.buffer);
      } catch (Throwable t) {
        if (failure == null) failure = t;
        else failure.addSuppressed(t);
      }
    }
    return failure;
  }

  /**
   * Uploads {@code residents} (direct buffers or arrays of a primitive type other than boolean), runs {@code body},
   * then writes them back and releases them. Residents that are already resident, from an enclosing scope or from
   * {@link #invalidate}, are used as they are and left to their owner.
   */
  public <R> R resident(Object[] residents, Function<Scope, R> body) {
    Objects.requireNonNull(body, "body");
    List<Owned> owned = new ArrayList<>();
    R result = null;
    Throwable failure = null;
    try {
      enter(residents, owned);
      result = body.apply(new Scope(this));
    } catch (Throwable t) {
      failure = t;
    }
    Throwable exit = leave(owned);
    if (failure != null) {
      if (exit != null) failure.addSuppressed(exit);
      rethrow(failure);
    }
    rethrow(exit);
    return result;
  }

  @Override
  public void close() {
    releaseAll();
  }
}
