package polyregion.jvm.compiler;

import polyregion.jvm.ByteEnum;

public enum Target implements ByteEnum {
  LLVM_HOST(Compiler.Target_Object_LLVM_HOST),
  LLVM_X86_64(Compiler.Target_Object_LLVM_x86_64),
  LLVM_AARCH64(Compiler.Target_Object_LLVM_AArch64),
  LLVM_ARM(Compiler.Target_Object_LLVM_ARM),

  LLVM_NVPTX64(Compiler.Target_Object_LLVM_NVPTX64),
  LLVM_AMDGCN(Compiler.Target_Object_LLVM_AMDGCN),
  LLVM_SPIRV64(Compiler.Target_Object_LLVM_SPIRV64),

  C_C11(Compiler.Target_Source_C_C11),
  C_OpenCL1_1(Compiler.Target_Source_C_OpenCL1_1);

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
