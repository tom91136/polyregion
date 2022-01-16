package polyregion;

import polyregion.loader.Loader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;

public class PolyregionCompiler {

  public static final short BACKEND_LLVM = 0;

  public static native Compilation compile(byte[] ast, boolean emitAssembly, short backend);

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
              "/home/tom/polyregion/native/cmake-build-debug-clang/bindings/libjava-compiler.so"),
          RESOURCE_DIR);
    }
  }
}
