package polyregion.jvm.runtime;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

@SuppressWarnings("unused")
public final class Device implements AutoCloseable {

  // Simple value class that is unique for each instance
  private static final class NullObjectToken {
    private static final AtomicLong counter = new AtomicLong();
    private final long identity;

    NullObjectToken() {
      this.identity = counter.getAndIncrement();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      NullObjectToken that = (NullObjectToken) o;
      return identity == that.identity;
    }

    @Override
    public int hashCode() {
      return Objects.hash(identity);
    }
  }

  final long nativePeer;
  public final long id;
  public final String name;
  public final boolean sharedAddressSpace;
  private final ValueCache<String[]> cachedFeatures;

  Device(long nativePeer, long id, String name, boolean sharedAddressSpace) {
    this.nativePeer = nativePeer;
    this.id = id;
    this.name = Objects.requireNonNull(name);
    this.sharedAddressSpace = sharedAddressSpace;
    this.cachedFeatures = new ValueCache<>(() -> Platform.deviceFeatures0(nativePeer));
  }

  public Property[] properties() {
    return Platform.deviceProperties0(nativePeer);
  }

  public String[] features() {
    return cachedFeatures.getCached();
  }

  public void loadModule(String name, byte[] image) {
    Platform.loadModule0(nativePeer, name, image);
  }

  public boolean moduleLoaded(String name) {
    return Platform.moduleLoaded0(nativePeer, name);
  }

  public long malloc(long size, Access access) {
    return Platform.malloc0(nativePeer, size, access.value);
  }

  public void free(long data) {
    Platform.free0(nativePeer, data);
  }

  @Override
  public void close() {
    Platform.deleteDevicePeer0(nativePeer);
  }

  @Override
  public String toString() {
    return "Device{" + "nativePeer=" + nativePeer + ", id=" + id + ", name='" + name + '\'' + '}';
  }
}
