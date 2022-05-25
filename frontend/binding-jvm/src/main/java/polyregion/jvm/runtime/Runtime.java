package polyregion.jvm.runtime;

import java.util.Arrays;
import java.util.Objects;

public final class Runtime {

  final long nativePeer;
  public final String name;
  public final Property[] properties;

  Runtime(long nativePeer, String name, Property[] properties) {
    this.nativePeer = nativePeer;
    this.name = Objects.requireNonNull(name);
    this.properties = Objects.requireNonNull(properties);
  }

  public Device[] devices() {
    return Runtimes.devices0(nativePeer);
  }

  @Override
  public String toString() {
    return "Runtime{"
        + "nativePeer="
        + nativePeer
        + ", name='"
        + name
        + '\''
        + ", properties="
        + Arrays.toString(properties)
        + '}';
  }
}
