// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
package polyregion.jvm.compiler;

import polyregion.jvm.ByteEnum;

@SuppressWarnings("unused")
public enum Target implements ByteEnum {
  LLVM_HOST((byte) 10),
  LLVM_X86_64((byte) 11),
  LLVM_AARCH64((byte) 12),
  LLVM_ARM((byte) 13),
  LLVM_NVPTX64((byte) 20),
  LLVM_AMDGCN((byte) 21),
  LLVM_SPIRV32_KERNEL((byte) 22),
  LLVM_SPIRV64_KERNEL((byte) 23),
  LLVM_SPIRV_GLCOMPUTE((byte) 24),
  C_C11((byte) 30),
  C_OpenCL1_1((byte) 31),
  C_Metal1_0((byte) 32);

  public static final Target[] VALUES = values();

  final byte value;

  Target(byte value) {
    this.value = value;
  }

  @Override
  public byte value() {
    return value;
  }
}
