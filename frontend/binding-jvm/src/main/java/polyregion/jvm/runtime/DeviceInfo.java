package polyregion.jvm.runtime;

import java.util.Objects;

@SuppressWarnings("unused")
public final class DeviceInfo {
  public final int platform;
  public final int index;
  public final int backend;
  public final int platformKind;
  public final int moduleFormat;
  public final int deviceKind;
  public final int cudaArchitectureMajor;
  public final int cudaArchitectureMinor;
  public final long globalMemoryBytes;
  public final long computeUnits;
  public final String name;

  DeviceInfo(
      int platform,
      int index,
      int backend,
      int platformKind,
      int moduleFormat,
      int deviceKind,
      int cudaArchitectureMajor,
      int cudaArchitectureMinor,
      long globalMemoryBytes,
      long computeUnits,
      String name) {
    this.platform = platform;
    this.index = index;
    this.backend = backend;
    this.platformKind = platformKind;
    this.moduleFormat = moduleFormat;
    this.deviceKind = deviceKind;
    this.cudaArchitectureMajor = cudaArchitectureMajor;
    this.cudaArchitectureMinor = cudaArchitectureMinor;
    this.globalMemoryBytes = globalMemoryBytes;
    this.computeUnits = computeUnits;
    this.name = Objects.requireNonNull(name);
  }

  public boolean isGpu() {
    return deviceKind == 1;
  }

  @Override
  public String toString() {
    return "DeviceInfo{platform=" + platform + ", index=" + index + ", name='" + name + "'}";
  }
}
