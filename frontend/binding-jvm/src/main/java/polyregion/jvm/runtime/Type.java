package polyregion.jvm.runtime;

public enum Type implements ByteEnum {
  BOOL(Runtime.TYPE_BOOL, Byte.BYTES),
  BYTE(Runtime.TYPE_BYTE, Byte.BYTES),
  CHAR(Runtime.TYPE_CHAR, Character.BYTES),
  SHORT(Runtime.TYPE_SHORT, Short.BYTES),
  INT(Runtime.TYPE_INT, Integer.BYTES),
  LONG(Runtime.TYPE_LONG, Long.BYTES),
  FLOAT(Runtime.TYPE_FLOAT, Float.BYTES),
  DOUBLE(Runtime.TYPE_DOUBLE, Double.BYTES),
  PTR(Runtime.TYPE_PTR, Long.BYTES),
  VOID(Runtime.TYPE_VOID, 0);

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
