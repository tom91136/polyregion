// AUTO-GENERATED from PolyAST.Enums via polyregion.ast.CodeGen. DO NOT EDIT.
#pragma once

#include <cstdint>

#include "polyregion/export.h"

namespace polyregion::invoke {

enum class POLYREGION_EXPORT Backend : uint8_t {
  CUDA = 0,
  HIP = 1,
  HSA = 2,
  OpenCL = 3,
  Vulkan = 4,
  Metal = 5,
  SharedObject = 6,
  RelocatableObject = 7,
  LevelZero = 8,
};

enum class Access : uint8_t {
  RW = 1,
  RO = 2,
  WO = 3,
};

} // namespace polyregion::invoke

namespace polyregion::compiletime {

enum class POLYREGION_EXPORT Target : uint8_t {
  Object_LLVM_HOST = 10,
  Object_LLVM_x86_64 = 11,
  Object_LLVM_AArch64 = 12,
  Object_LLVM_ARM = 13,
  Object_LLVM_NVPTX64 = 20,
  Object_LLVM_AMDGCN = 21,
  Object_LLVM_SPIRV32_Kernel = 22,
  Object_LLVM_SPIRV64_Kernel = 23,
  Object_LLVM_SPIRV_GLCompute = 24,
  Source_C_C11 = 30,
  Source_C_OpenCL1_1 = 31,
  Source_C_Metal1_0 = 32,
};

enum class POLYREGION_EXPORT OptLevel : uint8_t {
  O0 = 10,
  O1 = 11,
  O2 = 12,
  O3 = 13,
  Ofast = 14,
};

} // namespace polyregion::compiletime

namespace polyregion::runtime {

enum class POLYREGION_EXPORT Type : uint8_t {
  Void = 1,
  Bool1 = 2,
  IntU8 = 3,
  IntU16 = 4,
  IntU32 = 5,
  IntU64 = 6,
  IntS8 = 7,
  IntS16 = 8,
  IntS32 = 9,
  IntS64 = 10,
  Float16 = 11,
  Float32 = 12,
  Float64 = 13,
  Ptr = 14,
  Scratch = 15,
};

enum class POLYREGION_EXPORT PlatformKind : uint8_t {
  HostThreaded = 1,
  Managed = 2,
};

enum class POLYREGION_EXPORT ModuleFormat : uint8_t {
  Source = 1,
  Object = 2,
  DSO = 3,
  PTX = 4,
  HSACO = 5,
  SPIRV_Kernel = 6,
  SPIRV_GLCompute = 7,
};

} // namespace polyregion::runtime
