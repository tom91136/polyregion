package polyregion;

import polyregion.loader.Loader;

import java.nio.Buffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;

public class PolyregionRuntime {

  public static final byte TYPE_BOOL = 1;
  public static final byte TYPE_BYTE = 2;
  public static final byte TYPE_CHAR = 3;
  public static final byte TYPE_SHORT = 4;
  public static final byte TYPE_INT = 5;
  public static final byte TYPE_LONG = 6;
  public static final byte TYPE_FLOAT = 7;
  public static final byte TYPE_DOUBLE = 8;
  public static final byte TYPE_PTR = 9;
  public static final byte TYPE_VOID = 10;

  public static native void invoke(
      byte[] object,
      String symbol, //
      byte returnType,
      Buffer returnPtr, //
      byte[] paramTypes,
      Buffer[] paramPtrs //
      );

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
              "/home/tom/polyregion/native/cmake-build-debug-clang/bindings/libjava-runtime.so"),
          RESOURCE_DIR);
    }
  }
}
