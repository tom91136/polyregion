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

  private static native byte[] loadPackage0(String name, String roots, String capabilities);

  private static native byte[][] compileInterface0(
      String name,
      String roots,
      String declaration,
      String entry,
      byte[] request,
      String capabilities,
      String targets,
      byte opt,
      int stackDepth);

  private static native String capabilitiesFor0(String capabilities, String target);

  private static native byte targetByName0(String name);

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

  public byte[] loadPackage(String name, String roots, String capabilities) {
    return loadPackage0(name, roots, capabilities);
  }

  public byte[][] compileInterface(
      String name,
      String roots,
      String declaration,
      String entry,
      byte[] request,
      String capabilities,
      String targets,
      Opt opt,
      int stackDepth) {
    return compileInterface0(
        name, roots, declaration, entry, request, capabilities, targets, opt.value, stackDepth);
  }

  public String capabilitiesFor(String capabilities, String target) {
    return capabilitiesFor0(capabilities, target);
  }

  public Target targetByName(String name) {
    return target(targetByName0(name));
  }

  private static Target target(byte value) {
    for (Target t : Target.VALUES) if (t.value == value) return t;
    throw new IllegalStateException("No target has value " + value);
  }

  @Override
  public void close() {
    library.close();
  }
}
