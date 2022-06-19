package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.util.Objects;

import polyregion.jvm.runtime.Device.Queue;

@SuppressWarnings("unused")
public final class Platform implements AutoCloseable {

  final long nativePeer;
  public final String name;

  Platform(long nativePeer, String name) {
    this.nativePeer = nativePeer;
    this.name = Objects.requireNonNull(name);
  }

  public Property[] properties() {
    return Platform.runtimeProperties0(nativePeer);
  }

  public Device[] devices() {
    return Platform.devices0(nativePeer);
  }

  @Override
  public void close() {
    Platform.deletePlatformPeer0(nativePeer);
  }

  @Override
  public String toString() {
    return "Runtime{" + "nativePeer=" + nativePeer + ", name='" + name + '\'' + '}';
  }

  static native Property[] runtimeProperties0(long nativePeer);

  static native Device[] devices0(long nativePeer);

  static native Property[] deviceProperties0(long nativePeer);

  static native String[] deviceFeatures0(long nativePeer);

  static native void loadModule0(long nativePeer, String name, byte[] image);

  static native boolean moduleLoaded0(long nativePeer, String name);

  static native long malloc0(long nativePeer, long size, byte access);

  static native void free0(long nativePeer, long handle);

  static native Queue createQueue0(long nativePeer, Device owner);

  static native void enqueueHostToDeviceAsync0(
      long nativePeer, ByteBuffer src, long dst, int size, Runnable cb);

  static native void enqueueDeviceToHostAsync0(
      long nativePeer, long src, ByteBuffer dst, int size, Runnable cb);

  static native void enqueueInvokeAsync0(
      long nativePeer,
      String moduleName,
      String symbol,
      byte[] argTypes,
      byte[] argData,
      Policy policy,
      Runnable cb);

  static native void deleteDevicePeer0(long nativePeer);

  static native void deleteQueuePeer0(long nativePeer);

  static native void deletePlatformPeer0(long nativePeer);

  //  private static final AtomicLong PENDING_CALLBACKS = new AtomicLong(0);
  //
  //  static {
  //    java.lang.Runtime.getRuntime()
  //        .addShutdownHook(
  //            new Thread(
  //                () -> {
  //                  while (PENDING_CALLBACKS.get() > 0) {
  //                    System.out.println("Waiting on " + PENDING_CALLBACKS.get());
  //                    synchronized (PENDING_CALLBACKS) {
  //                      try {
  //                        PENDING_CALLBACKS.wait();
  //                      } catch (InterruptedException ignored) {
  //                      }
  //                    }
  //                  }
  //                }));
  //  }
  //
  //
  //  static Runnable nonDaemon(Runnable r) {
  //    PENDING_CALLBACKS.incrementAndGet();
  //    return () -> {
  //      r.run();
  //      PENDING_CALLBACKS.decrementAndGet();
  //      synchronized (PENDING_CALLBACKS) {
  //        PENDING_CALLBACKS.notifyAll();
  //      }
  //    };
  //  }

}
