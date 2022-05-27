package polyregion.jvm.runtime;

import java.util.Arrays;
import java.util.Objects;

public final class Runtime implements AutoCloseable {

  final long nativePeer;
  public final String name;

  Runtime(long nativePeer, String name) {
    this.nativePeer = nativePeer;
    this.name = Objects.requireNonNull(name);
  }

  public Property[] properties() {
    return Runtimes.runtimeProperties(nativePeer);
  }

  public Device[] devices() {
    return Runtimes.devices0(nativePeer);
  }

  @Override
  public void close() {
    Runtimes.deleteRuntimePeer(nativePeer);
  }

  @Override
  public String toString() {
    return "Runtime{" + "nativePeer=" + nativePeer + ", name='" + name + '\'' + '}';
  }
}
