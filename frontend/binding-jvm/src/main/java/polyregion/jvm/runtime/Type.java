package polyregion.jvm.runtime;

public enum Type implements ByteEnum {
  BOOL(Runtimes.TYPE_BOOL, Byte.SIZE),
  BYTE(Runtimes.TYPE_BYTE, Byte.SIZE),
  CHAR(Runtimes.TYPE_CHAR, Character.SIZE),
  SHORT(Runtimes.TYPE_SHORT, Short.SIZE),
  INT(Runtimes.TYPE_INT, Integer.SIZE),
  LONG(Runtimes.TYPE_LONG, Long.SIZE),
  FLOAT(Runtimes.TYPE_FLOAT, Float.SIZE),
  DOUBLE(Runtimes.TYPE_DOUBLE, Double.SIZE),
  PTR(Runtimes.TYPE_PTR, Long.SIZE),
  VOID(Runtimes.TYPE_VOID, 0);

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
