package polyregion;

import java.nio.ByteBuffer;

final class SignalNatives {

  private SignalNatives() {}

  static native long map(long bytes);

  static native ByteBuffer buffer(long base, long bytes);

  static native void guard(long base, long bytes);

  static native void guardThenForeign(long base, long bytes);

  static native void foreignThenGuard(long base, long bytes);

  static native void foreignThenInstall();

  static native void foreign();

  static native void rearm(long base, long bytes);

  static native long resolutions();
}
