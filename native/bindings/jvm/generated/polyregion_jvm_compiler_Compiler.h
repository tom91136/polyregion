/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class polyregion_jvm_compiler_Compiler */

#ifndef _Included_polyregion_jvm_compiler_Compiler
#define _Included_polyregion_jvm_compiler_Compiler
#ifdef __cplusplus
extern "C" {
#endif
#undef polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1x86_164
#define polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1x86_164 0L
#undef polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1AArch64
#define polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1AArch64 1L
#undef polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1ARM
#define polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1ARM 2L
#undef polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1NVPTX64
#define polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1NVPTX64 3L
#undef polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1AMDGCN
#define polyregion_jvm_compiler_Compiler_TargetObjectLLVM_1AMDGCN 4L
#undef polyregion_jvm_compiler_Compiler_TargetSourceC_1OpenCL1_11
#define polyregion_jvm_compiler_Compiler_TargetSourceC_1OpenCL1_11 5L
#undef polyregion_jvm_compiler_Compiler_TargetSourceC_1C11
#define polyregion_jvm_compiler_Compiler_TargetSourceC_1C11 6L
/*
 * Class:      polyregion_jvm_compiler_Compiler
 * Method:     compile
 * Signature:  ([BZLpolyregion/jvm/compiler/Options;)Lpolyregion/jvm/compiler/Compilation;
 */
JNIEXPORT jobject JNICALL Java_polyregion_jvm_compiler_Compiler_compile
  (JNIEnv *, jclass, jbyteArray, jboolean, jobject);

/*
 * Class:      polyregion_jvm_compiler_Compiler
 * Method:     layoutOf
 * Signature:  ([BLpolyregion/jvm/compiler/Options;)Lpolyregion/jvm/compiler/Layout;
 */
JNIEXPORT jobject JNICALL Java_polyregion_jvm_compiler_Compiler_layoutOf
  (JNIEnv *, jclass, jbyteArray, jobject);

#ifdef __cplusplus
}
#endif
#endif