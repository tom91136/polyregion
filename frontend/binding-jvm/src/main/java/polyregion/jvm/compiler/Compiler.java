package polyregion.jvm.compiler;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import polyregion.jvm.Loader;

public final class Compiler {

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
      Target_Object_LLVM_SPIRV64 = 22;
  static final byte //
      Target_Source_C_C11 = 30,
      Target_Source_C_OpenCL1_1 = 31;

  public static native String hostTriplet();

  static native byte hostTarget0();

  private static byte cachedTarget = 0;

  public Optional<Target> hostTarget() {
    if (cachedTarget == 0) cachedTarget = hostTarget0();
    if (cachedTarget == Target_UNSUPPORTED) return Optional.empty();
    for (Target t : Target.VALUES) if (t.value == cachedTarget) return Optional.of(t);
    throw new AssertionError("Target enum not implemented in Java:" + cachedTarget);
  }

  public static native Compilation compile(
      byte[] function, boolean emitAssembly, Options options, byte opt);

  public static native Layout layoutOf(byte[] structDef, Options options);

  private static AtomicBoolean loaded = new AtomicBoolean();

  private static final Path RESOURCE_DIR = Loader.HOME_DIR.resolve(".polyregion");

  static {
    if (!Boolean.getBoolean("polyregion.compiler.noautoload")) {
      load();
    }
  }

  public static void load() {
    if (!loaded.getAndSet(true)) {
      //			System.setProperty("ASAN_OPTIONS", "verify_asan_link_order=0,verbosity=2");
      //
      //
      //	Loader.loadDirect(Paths.get("/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so"), RESOURCE_DIR);
      Loader.loadDirect(
          Paths.get(
              "/home/tom/polyregion/native/cmake-build-debug-clang/bindings/jvm/libpolyregion-compiler-jvm.so"
              // "/home/tom/polyregion/native/cmake-build-release-clang/bindings/libjava-compiler.so"
              ),
          RESOURCE_DIR);
    }
  }
}
