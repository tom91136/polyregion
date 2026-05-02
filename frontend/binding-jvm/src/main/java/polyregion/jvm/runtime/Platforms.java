package polyregion.jvm.runtime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Paths;

import polyregion.jvm.Loader;
import polyregion.jvm.NativeLibrary;

public final class Platforms implements AutoCloseable {

  // These ordinals must match the native enum `polyregion::runtime::Type` in common/polyregion/types.h.
  static final byte //
      TYPE_VOID = 1,
      TYPE_BOOL = 2,
      TYPE_BYTE = 7,    // signed Java byte -> native IntS8
      TYPE_CHAR = 4,    // unsigned Java char -> native IntU16
      TYPE_SHORT = 8,   // signed Java short -> native IntS16
      TYPE_INT = 9,     // signed Java int -> native IntS32
      TYPE_LONG = 10,   // signed Java long -> native IntS64
      TYPE_FLOAT = 12,  // native Float32
      TYPE_DOUBLE = 13, // native Float64
      TYPE_PTR = 14;
  static final byte //
      ACCESS_RW = 1,
      ACCESS_RO = 2,
      ACCESS_WO = 3;

  static native Platform CUDA0();

  static native Platform HIP0();

  static native Platform HSA0();

  static native Platform Metal0();

  static native Platform Vulkan0();

  static native Platform OpenCL0();

  static native Platform Relocatable0();

  static native Platform Dynamic0();

  static native void deleteAllPeers0();

  static native long[] pointerOfDirectBuffers0(Buffer[] buffers);

  static native long pointerOfDirectBuffer0(Buffer buffer);

  static native ByteBuffer directBufferFromPointer0(long ptr, long size);

  private final NativeLibrary library;

  private Platforms(NativeLibrary library) {
    this.library = library;
  }

  public static Platforms create() {
    Loader.touch();
    String name = "libpolyinvoke-JNI.so";
    return new Platforms(
        NativeLibrary.load(
            Loader.searchAndCopyResourceIfNeeded(name, Paths.get("."))
                .orElseThrow(() -> new RuntimeException("Cannot find library: " + name))
                .toAbsolutePath()
                .toString()));
  }

  public Platform CUDA() {
    return CUDA0();
  }

  public Platform HIP() {
    return HIP0();
  }

  public Platform HSA() {
    return HSA0();
  }

  public Platform OpenCL() {
    return OpenCL0();
  }

  public Platform Relocatable() {
    return Relocatable0();
  }

  public Platform Dynamic() {
    return Dynamic0();
  }

  public long[] pointerOfDirectBuffers(Buffer[] buffers) {
    return pointerOfDirectBuffers0(buffers);
  }

  public long pointerOfDirectBuffer(Buffer buffers) {
    return pointerOfDirectBuffer0(buffers);
  }

  public ByteBuffer directBufferFromPointer(long ptr, long size) {
    return directBufferFromPointer0(ptr, size).order(ByteOrder.nativeOrder());
  }

  @Override
  public void close() {
    Platforms.deleteAllPeers0();
    library.close();
  }
}
