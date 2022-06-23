package polyregion.jvm;

import java.io.File;
import java.nio.Buffer;

@SuppressWarnings("unused")
public final class Natives {

  private Natives() {
    throw new AssertionError();
  }

  static {
    Loader.touch();
  }

  static native void registerFilesToDropOnUnload0(File file);

  static native long dynamicLibraryLoad0(String name);

  static native void dynamicLibraryRelease0(long handle);
}
