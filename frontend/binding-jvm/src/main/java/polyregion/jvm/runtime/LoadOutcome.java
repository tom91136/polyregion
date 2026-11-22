// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
package polyregion.jvm.runtime;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum LoadOutcome implements ByteEnum {
  LOADED((byte) 0),
  FORMAT((byte) 1),
  LACKS((byte) 2),
  UNPREPARED((byte) 3),
  NO_DEVICE((byte) 4);

  public static final LoadOutcome[] VALUES = values();

  final byte value;

  LoadOutcome(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
