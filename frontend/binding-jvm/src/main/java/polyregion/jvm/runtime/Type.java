package polyregion.jvm.runtime;

public enum Type implements ByteEnum {
  BOOL(Runtime.TYPE_BOOL, Byte.SIZE),
  BYTE(Runtime.TYPE_BYTE, Byte.SIZE),
  CHAR(Runtime.TYPE_CHAR, Character.SIZE),
  SHORT(Runtime.TYPE_SHORT, Short.SIZE),
  INT(Runtime.TYPE_INT, Integer.SIZE),
  LONG(Runtime.TYPE_LONG, Long.SIZE),
  FLOAT(Runtime.TYPE_FLOAT, Float.SIZE),
  DOUBLE(Runtime.TYPE_DOUBLE, Double.SIZE),
  PTR(Runtime.TYPE_PTR, Long.SIZE),
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
}
