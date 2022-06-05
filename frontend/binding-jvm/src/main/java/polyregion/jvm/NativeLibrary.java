package polyregion.jvm;

import java.util.Objects;

@SuppressWarnings("unused")
public final class NativeLibrary implements AutoCloseable {

  static {
    Loader.touch();
  }

  public final long handle;
  public final String name;

  private NativeLibrary(long handle, String name) {
    if (handle == 0) throw new AssertionError("Library handle is NULL(0)");
    this.handle = handle;
    this.name = Objects.requireNonNull(name);
  }

  public static NativeLibrary load(String name) {
    System.out.println("Loading " + name);
    long handle = Natives.dynamicLibraryLoad0(Objects.requireNonNull(name));
    System.out.println("Loaded " + name + " handle=" + handle);
    return new NativeLibrary(handle, name);
  }

  @Override
  public void close() {
    Natives.dynamicLibraryRelease0(handle);
  }

  @Override
  public String toString() {
    return "NativeLibrary{" + "handle=" + handle + ", name='" + name + '\'' + '}';
  }
}
