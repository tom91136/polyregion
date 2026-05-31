// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Type implements ByteEnum {
  VOID((byte) 1, 0),
  BOOL((byte) 2, Byte.BYTES),
  CHAR((byte) 4, Character.BYTES),
  BYTE((byte) 7, Byte.BYTES),
  SHORT((byte) 8, Short.BYTES),
  INT((byte) 9, Integer.BYTES),
  LONG((byte) 10, Long.BYTES),
  FLOAT((byte) 12, Float.BYTES),
  DOUBLE((byte) 13, Double.BYTES),
  PTR((byte) 14, Long.BYTES);

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
