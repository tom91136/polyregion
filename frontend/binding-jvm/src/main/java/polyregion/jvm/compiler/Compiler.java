package polyregion.jvm.compiler;

import java.nio.file.Paths;
import java.util.Objects;
import java.util.Optional;

import polyregion.jvm.Loader;
import polyregion.jvm.NativeLibrary;

@SuppressWarnings("unused")
public final class Compiler implements AutoCloseable {

  static final byte //
      Opt_O0 = 10,
      Opt_O1 = 11,
      Opt_O2 = 12,
      Opt_O3 = 13,
      Opt_Ofast = 14;

  static final byte //
      Target_UNSUPPORTED = 1;

  static final byte //
      Target_Object_LLVM_HOST = 10,
      Target_Object_LLVM_x86_64 = 11,
      Target_Object_LLVM_AArch64 = 12,
      Target_Object_LLVM_ARM = 13;
  static final byte //
      Target_Object_LLVM_NVPTX64 = 20,
      Target_Object_LLVM_AMDGCN = 21,
      Target_Object_LLVM_SPIRV32 = 22,
      Target_Object_LLVM_SPIRV64 = 23;
  static final byte //
      Target_Source_C_C11 = 30,
      Target_Source_C_OpenCL1_1 = 31,
      Target_Source_C_Metal1_0 = 32;

  private static native String hostTriplet0();

  private static native byte hostTarget0();

  private static native Compilation compile0(
      byte[] function, boolean emitAssembly, Options options, byte opt);

  private static native Layout[] layoutOf0(byte[] structDef, Options options);

  private final NativeLibrary library;

  private Compiler(NativeLibrary library) {
    this.library = Objects.requireNonNull(library);
  }

  public static Compiler create() {
    Loader.touch();
    String name = "libpolyc-JNI.so";
    return new Compiler(
        NativeLibrary.load(
            Loader.searchAndCopyResourceIfNeeded(name, Paths.get("."))
                .orElseThrow(() -> new RuntimeException("Cannot find library: " + name))
                .toAbsolutePath()
                .toString()));
  }

  private static byte cachedTarget = 0;

  public Optional<Target> hostTarget() {
    if (cachedTarget == 0) cachedTarget = hostTarget0();
    if (cachedTarget == Target_UNSUPPORTED) return Optional.empty();
    for (Target t : Target.VALUES) if (t.value == cachedTarget) return Optional.of(t);
    throw new AssertionError("Target enum not implemented in Java:" + cachedTarget);
  }

  public String hostTriplet() {
    return hostTriplet0();
  }

  public Compilation compile(byte[] function, boolean emitAssembly, Options options, byte opt) {
    return compile0(function, emitAssembly, options, opt);
  }

  public Layout[] layoutOf(byte[] structDef, Options options) {
    return layoutOf0(structDef, options);
  }

  @Override
  public void close() {
    library.close();
  }
}
