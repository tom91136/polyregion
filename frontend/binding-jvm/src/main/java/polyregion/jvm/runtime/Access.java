// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Access implements ByteEnum {
  RW((byte) 1),
  RO((byte) 2),
  WO((byte) 3);

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
