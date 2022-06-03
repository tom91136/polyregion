package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;
import polyregion.jvm.compiler.Target;

public enum Access implements ByteEnum {
  RW(Platform.ACCESS_RW),
  RO(Platform.ACCESS_RO),
  WO(Platform.ACCESS_WO);

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
