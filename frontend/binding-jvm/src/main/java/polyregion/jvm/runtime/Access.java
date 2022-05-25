package polyregion.jvm.runtime;

public enum Access implements ByteEnum {
  RW(Runtimes.ACCESS_RW),
  RO(Runtimes.ACCESS_R0),
  WO(Runtimes.ACCESS_WO);

  final byte value;

  Access(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
