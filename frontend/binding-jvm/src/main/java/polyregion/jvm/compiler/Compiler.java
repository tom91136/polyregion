package polyregion.jvm.compiler;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;

import polyregion.jvm.Loader;

public final class Compiler {

  public static final byte TargetObjectLLVM_x86_64 = 1;
  public static final byte TargetObjectLLVM_AArch64 = 2;
  public static final byte TargetObjectLLVM_ARM = 3;
  public static final byte TargetObjectLLVM_NVPTX64 = 4;
  public static final byte TargetObjectLLVM_AMDGCN = 5;
  public static final byte TargetSourceC_OpenCL1_1 = 6;
  public static final byte TargetSourceC_C11 = 7;

  public static native Compilation compile(byte[] function, boolean emitAssembly, Options options);

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
              "/home/tom/polyregion/native/cmake-build-debug-clang/bindings/libjava-compiler.so"
              // "/home/tom/polyregion/native/cmake-build-release-clang/bindings/libjava-compiler.so"
              ),
          RESOURCE_DIR);
    }
  }
}
