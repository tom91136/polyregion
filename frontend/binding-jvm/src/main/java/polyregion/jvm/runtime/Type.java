package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Type implements ByteEnum {
  BOOL(Platforms.TYPE_BOOL, Byte.BYTES),
  BYTE(Platforms.TYPE_BYTE, Byte.BYTES),
  CHAR(Platforms.TYPE_CHAR, Character.BYTES),
  SHORT(Platforms.TYPE_SHORT, Short.BYTES),
  INT(Platforms.TYPE_INT, Integer.BYTES),
  LONG(Platforms.TYPE_LONG, Long.BYTES),
  FLOAT(Platforms.TYPE_FLOAT, Float.BYTES),
  DOUBLE(Platforms.TYPE_DOUBLE, Double.BYTES),
  PTR(Platforms.TYPE_PTR, Long.BYTES),
  VOID(Platforms.TYPE_VOID, 0);

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
