#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "export.h"
#include "ast.h"

namespace polyregion::compiler {

using MonoClock = std::chrono::steady_clock;
using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] TimePoint nowMono();
[[nodiscard]] int64_t elapsedNs(const TimePoint &a, const TimePoint &b = nowMono());
[[nodiscard]] int64_t nowMs();

enum class POLYREGION_EXPORT Target : uint8_t {
  Object_LLVM_HOST = 10,
  Object_LLVM_x86_64,
  Object_LLVM_AArch64,
  Object_LLVM_ARM,

  Object_LLVM_NVPTX64 = 20,
  Object_LLVM_AMDGCN,
  Object_LLVM_SPIRV32,
  Object_LLVM_SPIRV64,

  Source_C_C11 = 30,
  Source_C_OpenCL1_1,
  Source_C_Metal1_0,
};

enum class POLYREGION_EXPORT Opt : uint8_t {
  O0 = 10,
  O1,
  O2,
  O3,
  Ofast,
};

POLYREGION_EXPORT std::optional<Target> targetFromOrdinal(std::underlying_type_t<Target> ordinal);

POLYREGION_EXPORT std::optional<Opt> optFromOrdinal(std::underlying_type_t<Opt> ordinal);

enum class POLYREGION_EXPORT CpuArch : uint32_t {

  ARM_Others,
  ARM_A64FX,
  ARM_AppleA7,
  ARM_AppleA10,
  ARM_AppleA11,
  ARM_AppleA12,
  ARM_AppleA13,
  ARM_AppleA14,
  ARM_Carmel,
  ARM_CortexA35,
  ARM_CortexA53,
  ARM_CortexA55,
  ARM_CortexA510,
  ARM_CortexA57,
  ARM_CortexA65,
  ARM_CortexA72,
  ARM_CortexA73,
  ARM_CortexA75,
  ARM_CortexA76,
  ARM_CortexA77,
  ARM_CortexA78,
  ARM_CortexA78C,
  ARM_CortexA710,
  ARM_CortexR82,
  ARM_CortexX1,
  ARM_CortexX1C,
  ARM_CortexX2,
  ARM_ExynosM3,
  ARM_Falkor,
  ARM_Kryo,
  ARM_NeoverseE1,
  ARM_NeoverseN1,
  ARM_NeoverseN2,
  ARM_Neoverse512TVB,
  ARM_NeoverseV1,
  ARM_Saphira,
  ARM_ThunderX2T99,
  ARM_ThunderX,
  ARM_ThunderXT81,
  ARM_ThunderXT83,
  ARM_ThunderXT88,
  ARM_ThunderX3T110,
  ARM_TSV110,

  X86_i686,
  X86_Pentium2,
  X86_Pentium3,
  X86_PentiumM,
  X86_C3_2,
  X86_Yonah,
  X86_Pentium4,
  X86_Prescott,
  X86_Nocona,
  X86_Core2,
  X86_Penryn,
  X86_Bonnell,
  X86_Silvermont,
  X86_Goldmont,
  X86_GoldmontPlus,
  X86_Tremont,
  X86_Nehalem,
  X86_Westmere,
  X86_SandyBridge,
  X86_IvyBridge,
  X86_Haswell,
  X86_Broadwell,
  X86_SkylakeClient,
  X86_SkylakeServer,
  X86_Cascadelake,
  X86_Cooperlake,
  X86_Cannonlake,
  X86_IcelakeClient,
  X86_Rocketlake,
  X86_IcelakeServer,
  X86_Tigerlake,
  X86_SapphireRapids,
  X86_Alderlake,
  X86_KNL,
  X86_KNM,
  X86_Lakemont,
  X86_Athlon,
  X86_AthlonXP,
  X86_K8,
  X86_K8SSE3,
  X86_AMDFAM10,
  X86_BTVER1,
  X86_BTVER2,
  X86_BDVER1,
  X86_BDVER2,
  X86_BDVER3,
  X86_BDVER4,
  X86_ZNVER1,
  X86_ZNVER2,
  X86_ZNVER3,

  // See https://gitlab.com/x86-psABIs/x86-64-ABI/-/commit/77566eb03bc6a326811cb7e9 for the following:
  X86_64,    // CMOV, CMPXCHG8B, FPU, FXSR, MMX, FXSR, SCE, SSE, SSE2
  X86_64_v2, // (close to Nehalem) CMPXCHG16B, LAHF-SAHF, POPCNT, SSE3, SSE4.1, SSE4.2, SSSE3
  X86_64_v3, // (close to Haswell) AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, XSAVE
  X86_64_v4, // AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL

};

enum class POLYREGION_EXPORT GpuArch : uint32_t {

  NVIDIA_SM_20_Fermi,  // GeForce 400, 500, 600, GT-630
  NVIDIA_SM_30_Kepler, // GeForce 700, GT-730
  NVIDIA_SM_35_Kepler, // Tesla K40
  NVIDIA_SM_37_Kepler, // Tesla K80

  NVIDIA_SM_50_Maxwell, // Tesla/Quadro M series
  NVIDIA_SM_52_Maxwell, // Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
  NVIDIA_SM_53_Maxwell, // Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano

  NVIDIA_SM_60_Pascal, //  Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
  NVIDIA_SM_61_Pascal, //  GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030 (GP108), GT 1010 (GP108) Titan Xp, Tesla
                       //  P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
  NVIDIA_SM_62_Pascal, //  Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

  NVIDIA_SM_70_Volta, // DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
  NVIDIA_SM_72_Volta, // Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX

  NVIDIA_SM_75_Turing, //  GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000,
                       //  Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4

  NVIDIA_SM_80_Ampere, // NVIDIA A100 (the name “Tesla” has been dropped – GA100), NVIDIA DGX-A100
  NVIDIA_SM_86_Ampere, // Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, RTX A4000,
  // A5000, A6000, NVIDIA A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX
  // A16, RTX A40, A2 Tensor Core GPU

  NVIDIA_SM_87_Ampere, // ???
  //  NVIDIA_SM_90_Hopper // ???

  AMDGCN_GFX600_SouthernIslands,
  AMDGCN_GFX601_SouthernIslands,
  AMDGCN_GFX602_SouthernIslands,
  AMDGCN_GFX700_SeaIslands,
  AMDGCN_GFX705_SeaIslands,
  AMDGCN_GFX701_SeaIslands,
  AMDGCN_GFX702_SeaIslands,
  AMDGCN_GFX703_SeaIslands,
  AMDGCN_GFX704_SeaIslands,
  AMDGCN_GFX801_VolcanicIslands,
  AMDGCN_GFX802_VolcanicIslands,
  AMDGCN_GFX803_VolcanicIslands,
  AMDGCN_GFX805_VolcanicIslands,
  AMDGCN_GFX810_VolcanicIslands,
  AMDGCN_GFX900_Vega,
  AMDGCN_GFX902_Vega,
  AMDGCN_GFX904_Vega,
  AMDGCN_GFX906_Vega,
  AMDGCN_GFX908_Vega,
  AMDGCN_GFX909_Vega,
  AMDGCN_GFX90A_Vega,
  AMDGCN_GFX90C_Vega,
  AMDGCN_GFX940_Vega,
  AMDGCN_GFX1010_RDNA2,
  AMDGCN_GFX1011_RDNA2,
  AMDGCN_GFX1012_RDNA2,
  AMDGCN_GFX1013_RDNA2,
  AMDGCN_GFX1030_RDNA2,
  AMDGCN_GFX1031_RDNA2,
  AMDGCN_GFX1032_RDNA2,
  AMDGCN_GFX1033_RDNA2,
  AMDGCN_GFX1034_RDNA2,
  AMDGCN_GFX1035_RDNA2,
  AMDGCN_GFX1036_RDNA2,
  //  AMDGCN_GFX1100_RDNA3,
  //  AMDGCN_GFX1101_RDNA3,
  //  AMDGCN_GFX1102_RDNA3,
  //  AMDGCN_GFX1103_RDNA3,

};

struct POLYREGION_EXPORT Member {
  POLYREGION_EXPORT polyast::Named name;
  POLYREGION_EXPORT uint64_t offsetInBytes, sizeInBytes;
  POLYREGION_EXPORT Member(decltype(name) name, decltype(offsetInBytes) offsetInBytes, decltype(sizeInBytes) sizeInBytes)
      : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
};

struct POLYREGION_EXPORT Layout {
  POLYREGION_EXPORT polyast::Sym name;
  POLYREGION_EXPORT uint64_t sizeInBytes, alignment;
  POLYREGION_EXPORT std::vector<Member> members;
  POLYREGION_EXPORT Layout(decltype(name) name, decltype(sizeInBytes) sizeInBytes, decltype(alignment) alignment,
                decltype(members) members)
      : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
};

struct POLYREGION_EXPORT Event {
  POLYREGION_EXPORT int64_t epochMillis, elapsedNanos;
  POLYREGION_EXPORT std::string name, data;
  POLYREGION_EXPORT Event(decltype(epochMillis) epochMillis, decltype(elapsedNanos) elapsedNanos, //
               decltype(name) name, decltype(data) data)                               //
      : epochMillis(epochMillis), elapsedNanos(elapsedNanos), name(std::move(name)), data(std::move(data)) {}
};

using Bytes = std::vector<char>;

struct POLYREGION_EXPORT Compilation {
  POLYREGION_EXPORT std::vector<Layout> layouts;
  POLYREGION_EXPORT std::optional<Bytes> binary;
  POLYREGION_EXPORT std::vector<std::string> features;
  POLYREGION_EXPORT std::vector<Event> events;
  POLYREGION_EXPORT std::string messages;
  POLYREGION_EXPORT Compilation() = default;
  POLYREGION_EXPORT explicit Compilation(decltype(messages) messages) : messages(std::move(messages)) {}
  POLYREGION_EXPORT Compilation(decltype(binary) binary,     //
                     decltype(features) features, //
                     decltype(events) events,     //
                     decltype(messages) messages = "")
      : binary(std::move(binary)), features(std::move(features)), events(std::move(events)),
        messages(std::move(messages)) {}
};

POLYREGION_EXPORT std::ostream &operator<<(std::ostream &, const Member &);
POLYREGION_EXPORT std::ostream &operator<<(std::ostream &, const Layout &);
POLYREGION_EXPORT std::ostream &operator<<(std::ostream &, const Compilation &);

POLYREGION_EXPORT void initialise();

struct POLYREGION_EXPORT Options {
  POLYREGION_EXPORT Target target;
  POLYREGION_EXPORT std::string arch;
};

POLYREGION_EXPORT std::vector<Layout> layoutOf(const std::vector<polyast::StructDef> &sdefs, const Options &options);
POLYREGION_EXPORT std::vector<Layout> layoutOf(const Bytes &bytes, const Options &options);

POLYREGION_EXPORT Compilation compile(const polyast::Program &program, const Options &options, const Opt &opt);
POLYREGION_EXPORT Compilation compile(const Bytes &astBytes, const Options &options, const Opt &opt);

} // namespace polyregion::compiler
