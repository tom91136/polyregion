package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.ToIntFunction;

import polyregion.jvm.Natives;

@SuppressWarnings("unused")
public final class Device implements AutoCloseable {

  public static class Queue implements AutoCloseable {

    final long nativePeer;
    public final Device device;

    private final class MemoryProxy<T> {
      private final ToIntFunction<T> sizeInBytes;
      private final BiConsumer<T, ByteBuffer> encode;
      private final BiConsumer<ByteBuffer, T> decode;

      private long devicePtr;
      private ByteBuffer buffer;

      MemoryProxy(
          ToIntFunction<T> sizeInBytes,
          BiConsumer<T, ByteBuffer> encode,
          BiConsumer<ByteBuffer, T> decode) {
        this.sizeInBytes = Objects.requireNonNull(sizeInBytes);
        this.encode = Objects.requireNonNull(encode);
        this.decode = decode; // null decode for write-only
      }

      MemoryProxy<T> attachBuffer(ByteBuffer buffer) {
        this.buffer = buffer.rewind();
        return this;
      }

      MemoryProxy<T> invalidate(T object, Runnable cb) {
        int byteSize = sizeInBytes.applyAsInt(object);
        if (buffer != null && buffer.capacity() != byteSize) {
          // Size is different, and we have a previous allocation, deallocate and recreate.
          release();
        }
        if (buffer == null)
          buffer = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder());

        encode.accept(Objects.requireNonNull(object), buffer.rewind());

        if (device.sharedAddressSpace) {
          devicePtr = Natives.pointerOfDirectBuffer(buffer);
          if (cb != null) cb.run();
        } else {
          if (devicePtr == 0)
            devicePtr = device.malloc(byteSize, decode == null ? Access.RO : Access.RW);
          enqueueHostToDeviceAsync(buffer, devicePtr, buffer.capacity(), cb);
        }

        return this;
      }

      void sync(T object, Runnable cb) {
        if (decode == null) return;
        if (buffer == null) {
          throw new IllegalStateException(
              "Buffer was already released or not previously invalidated.");
        }
        if (device.sharedAddressSpace) {
          decode.accept(buffer.rewind(), Objects.requireNonNull(object));
          if (cb != null) cb.run();
          return;
        }
        enqueueDeviceToHostAsync(
            devicePtr,
            buffer,
            buffer.capacity(),
            () -> {
              decode.accept(buffer.rewind(), Objects.requireNonNull(object));
              if (cb != null) cb.run();
            });
      }

      void release() {
        if (devicePtr != 0) device.free(devicePtr);
        buffer = null;
        devicePtr = 0;
      }
    }

    public final Map<Object, MemoryProxy<Object>> references =
        Collections.synchronizedMap(new WeakHashMap<>());

    Queue(long nativePeer, Device device) {
      this.nativePeer = nativePeer;
      this.device = Objects.requireNonNull(device);
    }

    @SuppressWarnings("unchecked")
    public <T> long registerAndInvalidateIfAbsent(
        T object,
        ToIntFunction<T> sizeInBytes,
        BiConsumer<T, ByteBuffer> write,
        BiConsumer<ByteBuffer, T> read,
        Runnable cb) {
      return references.computeIfAbsent(
              Objects.requireNonNull(object),
              key ->
                  ((MemoryProxy<Object>) new MemoryProxy<>(sizeInBytes, write, read))
                      .invalidate(key, cb))
          .devicePtr;
    }

    public long registerAndInvalidateIfAbsent(Object object, ByteBuffer buffer, Runnable cb) {
      return references.computeIfAbsent(
              Objects.requireNonNull(object),
              key ->
                  new MemoryProxy<>(ignored -> buffer.capacity(), (s, d) -> {}, (d, s) -> {})
                      .attachBuffer(buffer)
                      .invalidate(key, cb))
          .devicePtr;
    }

    public void invalidate(Object o, Runnable cb) {
      MemoryProxy<Object> proxy = references.get(Objects.requireNonNull(o));
      if (proxy != null) proxy.invalidate(o, cb);
      else
        throw new IllegalArgumentException(
            "Object " + o + " is not currently registered for invalidation.");
    }

    public void sync(Object o, Runnable cb) {
      MemoryProxy<Object> proxy = references.get(Objects.requireNonNull(o));
      if (proxy != null) proxy.sync(o, cb);
      else
        throw new IllegalArgumentException(
            "Object " + o + " is not currently registered for sync.");
    }

    public void release(Object o) {
      MemoryProxy<?> orphan = references.remove(Objects.requireNonNull(o));
      if (orphan != null) orphan.release();
      else
        throw new IllegalArgumentException(
            "Object " + o + " is not currently registered for release.");
    }

    private void runAll(
        String action,
        BiConsumer<Entry<Object, MemoryProxy<Object>>, Runnable> f,
        Runnable cb,
        Object... objects) {
      final Set<Entry<Object, MemoryProxy<Object>>> xs;
      if (objects.length == 0) xs = references.entrySet();
      else {
        xs = new HashSet<>(objects.length);
        for (Object o : objects) {
          MemoryProxy<Object> proxy = references.get(o);
          if (proxy == null)
            throw new IllegalArgumentException(
                "Object " + o + " is not currently registered for " + action + ".");
          xs.add(new SimpleImmutableEntry<>(o, proxy));
        }
      }
      if (xs.isEmpty()) {
        if (cb != null) cb.run();
        return;
      }
      AtomicInteger pending = new AtomicInteger(xs.size());
      Runnable latched =
          cb == null
              ? null
              : () -> {
                if (pending.decrementAndGet() == 0) cb.run();
              };
      for (Entry<Object, MemoryProxy<Object>> e : xs) f.accept(e, latched);
    }

    public void invalidateAll(Runnable cb, Object... objects) {
      runAll(
          "invalidate", (e, latched) -> e.getValue().invalidate(e.getKey(), latched), cb, objects);
    }

    public void syncAll(Runnable cb, Object... objects) {
      runAll("sync", (e, latched) -> e.getValue().sync(e.getKey(), latched), cb, objects);
    }

    public void releaseAll(Object... objects) {
      if (objects.length != 0) for (Object o : objects) release(o);
      else for (MemoryProxy<?> e : references.values()) release(e);
    }

    public void enqueueHostToDeviceAsync(ByteBuffer src, long dst, int size, Runnable cb) {
      Platform.enqueueHostToDeviceAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueDeviceToHostAsync(long src, ByteBuffer dst, int size, Runnable cb) {
      Platform.enqueueDeviceToHostAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        List<Arg<?>> args,
        Arg<?> rtn,
        Policy policy,
        Runnable cb) {

      int bytes = 0;
      byte[] argTys = new byte[args.size() + 1];
      for (int i = 0; i < args.size(); ++i) {
        bytes += args.get(i).type.sizeInBytes;
        argTys[i] = args.get(i).type.value;
      }
      bytes += rtn.type.sizeInBytes;
      argTys[args.size()] = rtn.type.value;

      ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder());
      for (Arg<?> arg : args) arg.drainTo(buffer);
      rtn.drainTo(buffer);

      Platform.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, buffer.array(), policy, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        byte[] argTypes,
        byte[] argData,
        Policy policy,
        Runnable cb) {
      Platform.enqueueInvokeAsync0(nativePeer, moduleName, symbol, argTypes, argData, policy, cb);
    }

    @Override
    public void close() {
      releaseAll();
      Platform.deleteQueuePeer0(nativePeer);
    }
  }

  final long nativePeer;
  public final long id;
  public final String name;
  public final boolean sharedAddressSpace;
  private final ValueCache<String[]> cachedFeatures;

  Device(long nativePeer, long id, String name, boolean sharedAddressSpace) {
    this.nativePeer = nativePeer;
    this.id = id;
    this.name = Objects.requireNonNull(name);
    this.sharedAddressSpace = sharedAddressSpace;
    this.cachedFeatures = new ValueCache<>(() -> Platform.deviceFeatures0(nativePeer));
  }

  public Property[] properties() {
    return Platform.deviceProperties0(nativePeer);
  }

  public String[] features() {
    return cachedFeatures.getCached();
  }

  public Queue createQueue() {
    Queue q = Platform.createQueue0(nativePeer, this);
    if (q.device != this)
      throw new AssertionError("Invalid device associated with Queue, check JNI implementation.");
    return q;
  }

  public void loadModule(String name, byte[] image) {
    Platform.loadModule0(nativePeer, name, image);
  }

  public boolean moduleLoaded(String name) {
    return Platform.moduleLoaded0(nativePeer, name);
  }

  public long malloc(long size, Access access) {
    return Platform.malloc0(nativePeer, size, access.value);
  }

  public void free(long data) {
    Platform.free0(nativePeer, data);
  }

  @Override
  public void close() {
    Platform.deleteDevicePeer0(nativePeer);
  }

  @Override
  public String toString() {
    return "Device{" + "nativePeer=" + nativePeer + ", id=" + id + ", name='" + name + '\'' + '}';
  }
}
