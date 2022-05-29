package polyregion.jvm.runtime;

public enum Access implements ByteEnum {
  RW(Runtime.ACCESS_RW),
  RO(Runtime.ACCESS_R0),
  WO(Runtime.ACCESS_WO);

  final byte value;

  Access(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
