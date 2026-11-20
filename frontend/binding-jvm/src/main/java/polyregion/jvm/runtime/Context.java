package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import polyregion.jvm.Loader;
import polyregion.jvm.NativeLibrary;

@SuppressWarnings("unused")
public final class Context implements AutoCloseable {

  static native int platformCount0();

  static native long[] platformInfo0(int platform);

  static native String platformName0(int platform);

  static native long[] deviceInfo0(int platform, int device);

  static native String deviceName0(int platform, int device);

  static native long create0(int platform, int device);

  static native void destroy0(long context);

  static native long current0();

  static native long malloc0(long context, long bytes);

  static native void free0(long context, long pointer);

  static native void copyToDevice0(long context, ByteBuffer src, long dst, long bytes);

  static native void copyFromDevice0(long context, long src, ByteBuffer dst, long bytes);

  static native int select0(
      long context,
      String[] imageNames,
      byte[] imageTargets,
      String[][] imageFeatures,
      byte[][] images);

  static native void launch0(
      long context,
      String[] imageNames,
      byte[] imageTargets,
      String[][] imageFeatures,
      byte[][] images,
      String moduleName,
      String kernel,
      byte[] argTypes,
      byte[] argData,
      long[] dims);

  static native void loadHostProgram0(String name, byte[] image);

  static native boolean hostProgramLoaded0(String name);

  static native void callHost0(
      long context,
      String[] imageNames,
      byte[] imageTargets,
      String[][] imageFeatures,
      byte[][] images,
      String programName,
      String entry,
      byte[] argTypes,
      byte[] argData);

  private static final class Library {
    static final NativeLibrary INSTANCE = load();

    private static NativeLibrary load() {
      Loader.touch();
      String name = System.mapLibraryName("polyrt-JNI");
      return NativeLibrary.load(
          Loader.searchAndCopyResourceIfNeeded(name, Paths.get("."))
              .orElseThrow(() -> new RuntimeException("Cannot find library: " + name))
              .toAbsolutePath()
              .toString());
    }
  }

  private static void ensureLoaded() {
    if (Library.INSTANCE == null) throw new AssertionError();
  }

  private final long handle;
  private final boolean owned;
  private final AtomicBoolean closed = new AtomicBoolean();

  private Context(long handle, boolean owned) {
    this.handle = handle;
    this.owned = owned;
  }

  public static PlatformInfo[] platforms() {
    ensureLoaded();
    PlatformInfo[] platforms = new PlatformInfo[platformCount0()];
    for (int p = 0; p < platforms.length; ++p) {
      long[] info = platformInfo0(p);
      DeviceInfo[] devices = new DeviceInfo[(int) info[2]];
      for (int d = 0; d < devices.length; ++d) {
        long[] device = deviceInfo0(p, d);
        devices[d] =
            new DeviceInfo(
                p,
                d,
                (int) device[0],
                (int) device[1],
                (int) device[2],
                (int) device[3],
                (int) device[4],
                (int) device[5],
                device[6],
                device[7],
                deviceName0(p, d));
      }
      platforms[p] = new PlatformInfo(p, (int) info[0], (int) info[1], platformName0(p), devices);
    }
    return platforms;
  }

  public static Context create(int platform, int device) {
    ensureLoaded();
    return new Context(create0(platform, device), true);
  }

  public static Context create(DeviceInfo device) {
    return create(device.platform, device.index);
  }

  public static Context current() {
    ensureLoaded();
    long handle = current0();
    if (handle == 0) throw new PolyregionRuntimeException("The default context has no device");
    return new Context(handle, false);
  }

  private void requireOpen() {
    if (closed.get()) throw new PolyregionRuntimeException("Context " + handle + " is closed");
  }

  public long malloc(long bytes) {
    requireOpen();
    return malloc0(handle, bytes);
  }

  public void free(long pointer) {
    requireOpen();
    free0(handle, pointer);
  }

  public void copyToDevice(ByteBuffer src, long dst, long bytes) {
    requireOpen();
    copyToDevice0(handle, src.slice(), dst, bytes);
  }

  public void copyFromDevice(long src, ByteBuffer dst, long bytes) {
    requireOpen();
    copyFromDevice0(handle, src, dst.slice(), bytes);
  }

  public int select(ModuleImage[] images) {
    requireOpen();
    return select0(
        handle,
        ModuleImage.names(images),
        ModuleImage.targets(images),
        ModuleImage.features(images),
        ModuleImage.bytes(images));
  }

  public void launch(
      ModuleImage[] images,
      String moduleName,
      String kernel,
      byte[] argTypes,
      byte[] argData,
      Policy policy) {
    requireOpen();
    Objects.requireNonNull(policy);
    Dim3 local = policy.local;
    long[] dims = {
      policy.global.x,
      policy.global.y,
      policy.global.z,
      local == null ? 0 : local.x,
      local == null ? 0 : local.y,
      local == null ? 0 : local.z,
      policy.localMemoryBytes
    };
    launch0(
        handle,
        ModuleImage.names(images),
        ModuleImage.targets(images),
        ModuleImage.features(images),
        ModuleImage.bytes(images),
        moduleName,
        kernel,
        argTypes,
        argData,
        dims);
  }

  public static void loadHostProgram(String name, byte[] image) {
    ensureLoaded();
    loadHostProgram0(name, image);
  }

  public static boolean hostProgramLoaded(String name) {
    ensureLoaded();
    return hostProgramLoaded0(name);
  }

  public void callHost(
      ModuleImage[] images, String programName, String entry, byte[] argTypes, byte[] argData) {
    requireOpen();
    callHost0(
        handle,
        ModuleImage.names(images),
        ModuleImage.targets(images),
        ModuleImage.features(images),
        ModuleImage.bytes(images),
        programName,
        entry,
        argTypes,
        argData);
  }

  @Override
  public void close() {
    if (owned && closed.compareAndSet(false, true)) destroy0(handle);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    return handle == ((Context) o).handle;
  }

  @Override
  public int hashCode() {
    return Long.hashCode(handle);
  }

  @Override
  public String toString() {
    return "Context{handle=" + handle + ", owned=" + owned + '}';
  }
}
