package polyregion.jvm.runtime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import polyregion.jvm.Loader;
import polyregion.jvm.runtime.Device.Queue;

public final class Runtime implements AutoCloseable {

  final long nativePeer;
  public final String name;

  Runtime(long nativePeer, String name) {
    this.nativePeer = nativePeer;
    this.name = Objects.requireNonNull(name);
  }

  public Property[] properties() {
    return Runtime.runtimeProperties0(nativePeer);
  }

  public Device[] devices() {
    return Runtime.devices0(nativePeer);
  }

  @Override
  public void close() {
    Runtime.deleteRuntimePeer0(nativePeer);
  }

  @Override
  public String toString() {
    return "Runtime{" + "nativePeer=" + nativePeer + ", name='" + name + '\'' + '}';
  }

  public static Runtime CUDA() {
    return CUDA0();
  }

  public static Runtime HIP() {
    return HIP0();
  }

  public static Runtime OpenCL() {
    return OpenCL0();
  }

  public static Runtime Relocatable() {
    return Relocatable0();
  }

  public static Runtime Dynamic() {
    return Dynamic0();
  }

  public static long[] directBufferPointers(Buffer[] buffers) {
    return pointers(buffers);
  }

  static final byte //
      TYPE_VOID = 1,
      TYPE_BOOL = 2,
      TYPE_BYTE = 3,
      TYPE_CHAR = 4,
      TYPE_SHORT = 5,
      TYPE_INT = 6,
      TYPE_LONG = 7,
      TYPE_FLOAT = 8,
      TYPE_DOUBLE = 9,
      TYPE_PTR = 10;
  static final byte ACCESS_RW = 1, ACCESS_R0 = 2, ACCESS_WO = 3;

  static native long[] pointers(Buffer[] buffers);

  static native Runtime CUDA0();

  static native Runtime HIP0();

  static native Runtime OpenCL0();

  static native Runtime Relocatable0();

  static native Runtime Dynamic0();

  static native Property[] runtimeProperties0(long nativePeer);

  static native Device[] devices0(long nativePeer);

  static native Property[] deviceProperties0(long nativePeer);

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

  static native void deleteAllPeer0();

  static native void deleteDevicePeer0(long nativePeer);

  static native void deleteQueuePeer0(long nativePeer);

  static native void deleteRuntimePeer0(long nativePeer);

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

  private static final Path RESOURCE_DIR = Loader.HOME_DIR.resolve(".polyregion");
  private static final AtomicBoolean loaded = new AtomicBoolean();

  static {
    if (!Boolean.getBoolean("polyregion.runtime.noautoload")) {
      load();
    }
  }

  public static void load() {

    if (!loaded.getAndSet(true)) {
      Loader.loadDirect(
          Paths.get(
              //
              "/home/tom/polyregion/native/cmake-build-release-clang/bindings/jvm/libpolyregion-runtime-jvm.so"
              // "/home/tom/polyregion/native/cmake-build-release-clang/bindings/libjava-runtime.so"
              ),
          RESOURCE_DIR);
      java.lang.Runtime.getRuntime().addShutdownHook(new Thread(Runtime::deleteAllPeer0));
    }
  }
}
