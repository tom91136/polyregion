package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Access implements ByteEnum {
  RW(Platforms.ACCESS_RW),
  RO(Platforms.ACCESS_RO),
  WO(Platforms.ACCESS_WO);

  public static final Access[] VALUES = values();

  final byte value;

  Access(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
