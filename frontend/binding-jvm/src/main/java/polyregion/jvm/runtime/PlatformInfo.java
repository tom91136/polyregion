package polyregion.jvm.runtime;

import java.util.Arrays;
import java.util.Objects;

@SuppressWarnings("unused")
public final class PlatformInfo {
  public final int index;
  public final int backend;
  public final int kind;
  public final String name;
  public final DeviceInfo[] devices;

  PlatformInfo(int index, int backend, int kind, String name, DeviceInfo[] devices) {
    this.index = index;
    this.backend = backend;
    this.kind = kind;
    this.name = Objects.requireNonNull(name);
    this.devices = Objects.requireNonNull(devices);
  }

  @Override
  public String toString() {
    return "PlatformInfo{index="
        + index
        + ", name='"
        + name
        + "', devices="
        + Arrays.toString(devices)
        + '}';
  }
}
