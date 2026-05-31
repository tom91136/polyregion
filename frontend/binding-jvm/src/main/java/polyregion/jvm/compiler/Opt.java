// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
package polyregion.jvm.compiler;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Opt implements ByteEnum {
  O0((byte) 10),
  O1((byte) 11),
  O2((byte) 12),
  O3((byte) 13),
  Ofast((byte) 14);

  public static final Opt[] VALUES = values();

  final byte value;

  Opt(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
