package polyregion.jvm.compiler;

import polyregion.jvm.ByteEnum;

public enum Opt implements ByteEnum {
  O0(Compiler.Opt_O0),
  O1(Compiler.Opt_O1),
  O2(Compiler.Opt_O2),
  O3(Compiler.Opt_O3),
  Ofast(Compiler.Opt_Ofast);

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
