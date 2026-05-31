package polyregion.jvm.compiler;

import java.nio.file.Paths;
import java.util.Objects;
import java.util.Optional;

import polyregion.jvm.Loader;
import polyregion.jvm.NativeLibrary;

@SuppressWarnings("unused")
public final class Compiler implements AutoCloseable {

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
    String name = System.mapLibraryName("polyc-JNI");
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
    for (Target t : Target.VALUES) if (t.value == cachedTarget) return Optional.of(t);
    return Optional.empty();
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
