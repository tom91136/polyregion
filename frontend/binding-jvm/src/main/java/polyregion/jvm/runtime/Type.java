package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;
import polyregion.jvm.compiler.Target;

public enum Type implements ByteEnum {
  BOOL(Platform.TYPE_BOOL, Byte.BYTES),
  BYTE(Platform.TYPE_BYTE, Byte.BYTES),
  CHAR(Platform.TYPE_CHAR, Character.BYTES),
  SHORT(Platform.TYPE_SHORT, Short.BYTES),
  INT(Platform.TYPE_INT, Integer.BYTES),
  LONG(Platform.TYPE_LONG, Long.BYTES),
  FLOAT(Platform.TYPE_FLOAT, Float.BYTES),
  DOUBLE(Platform.TYPE_DOUBLE, Double.BYTES),
  PTR(Platform.TYPE_PTR, Long.BYTES),
  VOID(Platform.TYPE_VOID, 0);

  public static final Type[] VALUES = values();

  final byte value;
  final int sizeInBytes;

  Type(byte value, int sizeInBytes) {
    this.value = value;
    this.sizeInBytes = sizeInBytes;
  }

  @Override
  public byte value() {
    return value;
  }

  public int sizeInBytes() {
    return sizeInBytes;
  }
}
