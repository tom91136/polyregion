package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import polyregion.jvm.Loader;
import polyregion.jvm.runtime.Device.Queue;

@SuppressWarnings("unused")
public final class Runtimes {

  public static native Runtime CUDA();

  public static native Runtime HIP();

  public static native Runtime OpenCL();

  public static native Runtime Relocatable();

  public static native Runtime Dynamic();

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

  static native void deleteAllPeer();

  static native void deleteDevicePeer(long nativePeer);

  static native void deleteQueuePeer(long nativePeer);

  static native void deleteRuntimePeer(long nativePeer);

  static native Property[] runtimeProperties(long nativePeer);

  static native Device[] devices0(long nativePeer);

  static native Property[] deviceProperties(long nativePeer);

  static native void loadModule0(long nativePeer, String name, byte[] image);

  static native long malloc0(long nativePeer, long size, byte access);

  static native void free0(long nativePeer, long handle);

  static native Queue createQueue0(long nativePeer);

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
      java.lang.Runtime.getRuntime().addShutdownHook(new Thread(Runtimes::deleteAllPeer));
    }
  }
}
