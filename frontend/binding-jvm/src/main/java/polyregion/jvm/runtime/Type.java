package polyregion.jvm.runtime;

public enum Type implements ByteEnum {
  BOOL(Runtimes.TYPE_BOOL),
  BYTE(Runtimes.TYPE_BYTE),
  CHAR(Runtimes.TYPE_CHAR),
  SHORT(Runtimes.TYPE_SHORT),
  INT(Runtimes.TYPE_INT),
  LONG(Runtimes.TYPE_LONG),
  FLOAT(Runtimes.TYPE_FLOAT),
  DOUBLE(Runtimes.TYPE_DOUBLE),
  PTR(Runtimes.TYPE_PTR),
  VOID(Runtimes.TYPE_VOID);

  final byte value;

  Type(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
