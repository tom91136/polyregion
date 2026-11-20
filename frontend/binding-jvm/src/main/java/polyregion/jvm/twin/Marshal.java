package polyregion.jvm.twin;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import polyregion.jvm.runtime.Completion;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.PolyregionRuntimeException;
import polyregion.jvm.runtime.Queue;

/**
 * The device image of one launch: every buffer staged through a {@link Queue}, so the pointers
 * written into structs are device pointers, and the identity maps that keep an object reachable
 * twice staged once.
 *
 * <p>Typical use: write the captures through {@link #instance} and {@link #array}, launch, then
 * {@link #settle} to copy the device state back into the host objects and {@link #close} to release
 * the buffers. Generated code that picks its own layout calls {@link #stage}, {@link #sync} and
 * {@link #host} directly and keeps its objects in {@link #bind}.
 */
public final class Marshal implements AutoCloseable {

  /** Views memory the marshal did not stage, e.g. an object the kernel allocated on the host. */
  public interface HostView {
    ByteBuffer view(long pointer, long size);
  }

  private static final class Staged {
    final ByteBuffer buffer;
    final long pointer;

    Staged(ByteBuffer buffer, long pointer) {
      this.buffer = buffer;
      this.pointer = pointer;
    }
  }

  private interface ReadBack {
    void run();
  }

  private final Queue queue;
  private Boolean shared;
  private final Twins twins;
  private final HostView foreign;
  private final List<Staged> staged = new ArrayList<>();
  private final TreeMap<Long, Staged> byPointer = new TreeMap<>();
  private final IdentityHashMap<Object, Long> pointers = new IdentityHashMap<>();
  private final Map<Long, Object> objects = new HashMap<>();
  private final IdentityHashMap<Object, ReadBack> readBacks = new IdentityHashMap<>();

  public Marshal(Queue queue) {
    this(queue, null, null);
  }

  public Marshal(Queue queue, Twins twins) {
    this(queue, twins, null);
  }

  public Marshal(Queue queue, Twins twins, HostView foreign) {
    this.queue = queue;
    this.twins = twins;
    this.foreign = foreign;
  }

  public Queue queue() {
    return queue;
  }

  private boolean shared() {
    if (shared == null) shared = queue.context().sharedAddressSpace();
    return shared;
  }

  private static final class Settled implements Completion {
    private final CountDownLatch latch = new CountDownLatch(1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();

    @Override
    public void complete(Throwable failure) {
      this.failure.set(failure);
      latch.countDown();
    }

    void await() {
      try {
        latch.await();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new PolyregionRuntimeException("Interrupted while waiting for the device");
      }
      Throwable t = failure.get();
      if (t instanceof RuntimeException) throw (RuntimeException) t;
      if (t instanceof Error) throw (Error) t;
      if (t != null) throw new PolyregionRuntimeException(t.toString());
    }
  }

  /** A direct buffer the queue accepts: native order and never empty. */
  public static ByteBuffer allocate(int bytes) {
    return ByteBuffer.allocateDirect(Math.max(bytes, 1)).order(ByteOrder.nativeOrder());
  }

  /**
   * Uploads a finished buffer and returns the device pointer of its first byte. A device that
   * shares the host address space reads the buffer in place, so nothing is registered with the
   * queue.
   */
  public long stage(ByteBuffer host) {
    long pointer;
    if (shared()) pointer = Context.addressOf(host);
    else {
      Settled settled = new Settled();
      pointer = queue.invalidate(host, settled);
      settled.await();
    }
    Staged s = new Staged(host, pointer);
    staged.add(s);
    byPointer.put(pointer, s);
    return pointer;
  }

  /** The number of buffers staged and not yet released. */
  public int stagedBuffers() {
    return staged.size();
  }

  /** Copies every staged buffer back to the host. */
  public void sync() {
    if (shared()) return;
    for (Staged s : staged) {
      Settled settled = new Settled();
      queue.sync(s.buffer, settled);
      settled.await();
    }
  }

  /**
   * The host image of {@code size} bytes at a device pointer: a view into the staged buffer that
   * contains it, or the {@link HostView} for memory outside any.
   */
  public ByteBuffer host(long pointer, long size) {
    Map.Entry<Long, Staged> e = byPointer.floorEntry(pointer);
    if (e != null) {
      Staged s = e.getValue();
      long offset = pointer - s.pointer;
      if (offset + size <= s.buffer.capacity()) {
        ByteBuffer view = s.buffer.duplicate().order(ByteOrder.nativeOrder());
        view.position((int) offset).limit((int) (offset + size));
        return view.slice().order(ByteOrder.nativeOrder());
      }
    }
    if (foreign == null)
      throw new TwinException(
          "Device pointer 0x" + Long.toHexString(pointer) + " is not inside any staged buffer");
    return foreign.view(pointer, size);
  }

  /** The device pointer {@code root} was written at, or {@code absent}. */
  public long pointerOf(Object root, long absent) {
    Long p = pointers.get(root);
    return p == null ? absent : p;
  }

  public void bind(Object root, long pointer) {
    pointers.put(root, pointer);
  }

  public Object objectAt(long pointer) {
    return objects.get(pointer);
  }

  public void bindObject(long pointer, Object root) {
    objects.put(pointer, root);
  }

  private Twins twins() {
    if (twins == null) throw new TwinException("This marshal was created without twins");
    return twins;
  }

  /** Stages the image of a captured object as a buffer of its own. */
  public long instance(Object root) {
    Long known = pointers.get(root);
    if (known != null) return known;
    final Twin twin = twins().of(root.getClass());
    final ByteBuffer buffer = allocate(twin.layout().size);
    twin.write(this, root, buffer, 0);
    long pointer = stage(buffer);
    pointers.put(root, pointer);
    final Object target = root;
    readBacks.put(
        root,
        new ReadBack() {
          public void run() {
            twin.read(Marshal.this, target, buffer, 0);
          }
        });
    return pointer;
  }

  /**
   * Stages a captured array and returns its device pointer: primitives packed, objects as structs
   * at the layout's stride, arrays of arrays as a table of row pointers.
   */
  public long array(Object array) {
    Long known = pointers.get(array);
    if (known != null) return known;
    Class<?> component = array.getClass().getComponentType();
    long pointer;
    if (component.isPrimitive()) pointer = primitiveArray(array, Prim.of(component));
    else if (component.isArray()) pointer = rowTable((Object[]) array);
    else pointer = structArray((Object[]) array);
    pointers.put(array, pointer);
    return pointer;
  }

  private long primitiveArray(final Object array, final Prim prim) {
    final ByteBuffer buffer = allocate(java.lang.reflect.Array.getLength(array) * prim.size);
    prim.writeArray(array, Prim.native0(buffer));
    long pointer = stage(buffer);
    readBacks.put(
        array,
        new ReadBack() {
          public void run() {
            prim.readArray(Prim.native0(buffer), array);
          }
        });
    return pointer;
  }

  private long rowTable(Object[] rows) {
    long[] ptrs = new long[rows.length];
    for (int i = 0; i < rows.length; i++) {
      if (rows[i] == null)
        throw new TwinException("Row " + i + " of a captured array is null");
      ptrs[i] = array(rows[i]);
    }
    ByteBuffer buffer = allocate(ptrs.length * Long.BYTES);
    Prim.native0(buffer).asLongBuffer().put(ptrs);
    return stage(buffer);
  }

  private long structArray(final Object[] rows) {
    Twin twin = null;
    for (Object row : rows) {
      if (row == null)
        throw new TwinException("Element of a captured object array is null");
      Twin t = twins().of(row.getClass());
      if (twin != null && t != twin)
        throw new TwinException("A captured object array must hold one concrete type");
      twin = t;
    }
    final Twin element = twin;
    final int stride = twin == null ? 0 : twin.layout().size;
    final ByteBuffer buffer = allocate(stride * rows.length);
    for (int i = 0; i < rows.length; i++) element.write(this, rows[i], buffer, i * stride);
    long pointer = stage(buffer);
    readBacks.put(
        rows,
        new ReadBack() {
          public void run() {
            for (int i = 0; i < rows.length; i++) element.read(Marshal.this, rows[i], buffer, i * stride);
          }
        });
    return pointer;
  }

  /** Copies the synced device state of one staged object or array back into it. */
  public void readBack(Object target) {
    ReadBack r = readBacks.get(target);
    if (r == null) throw new TwinException("Nothing was staged for " + target.getClass().getName());
    r.run();
  }

  /**
   * Syncs every staged buffer, then copies the device state into the objects staged through {@link
   * #instance} and {@link #array}.
   */
  public void settle() {
    sync();
    for (ReadBack r : new ArrayList<>(readBacks.values())) r.run();
  }

  /** Releases every staged buffer, also after a failed launch. */
  @Override
  public void close() {
    RuntimeException failure = null;
    if (!staged.isEmpty() && !shared())
      for (Staged s : staged)
        try {
          queue.release(s.buffer);
        } catch (RuntimeException e) {
          if (failure == null) failure = e;
          else failure.addSuppressed(e);
        }
    staged.clear();
    byPointer.clear();
    if (failure != null) throw failure;
  }
}
