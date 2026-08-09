#pragma once

#include <vector>

#include "aspartame/all.hpp"

#include "polyregion/compat.h"
#ifdef NO_ERROR
  #undef NO_ERROR
#endif

#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/PPCTargetParser.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/X86TargetParser.h"

namespace polyregion::llvm_shared {

using namespace aspartame;

static bool isCPUTargetSupported(const std::string &CPU, //
                                 const llvm::Triple::ArchType &arch) {
  using namespace llvm;
  switch (arch) {
    case Triple::x86_64: return CPU == "native" || (llvm::X86::parseArchX86(CPU) != llvm::X86::CPUKind::CK_None);
    case Triple::arm: return CPU == "native" || llvm::ARM::parseCPUArch(CPU) != llvm::ARM::ArchKind::INVALID;
    case Triple::aarch64: return CPU == "native" || llvm::AArch64::parseCpu(CPU).has_value();
    case Triple::riscv32: return CPU == "native" || llvm::RISCV::parseCPU(CPU, /*IsRV64=*/false);
    case Triple::riscv64: return CPU == "native" || llvm::RISCV::parseCPU(CPU, /*IsRV64=*/true);
    case Triple::ppc64le:
    case Triple::ppc64: return CPU == "native" || llvm::PPC::isValidCPU(CPU);
    case Triple::amdgcn: return llvm::AMDGPU::parseArchAMDGCN(CPU) != llvm::AMDGPU::GPUKind::GK_NONE;
    case Triple::nvptx64: return CPU.rfind("sm_", 0) == 0;

    default: throw std::logic_error("Unexpected arch from triple:" + Triple::getArchTypeName(arch).str());
  }
}

static void collectCPUFeatures(const std::string &CPU,             //
                               const llvm::Triple::ArchType &arch, //
                               std::vector<std::string> &drain) {

  using namespace llvm;
  auto normaliseFeature = [](const std::vector<StringRef> &features, std::vector<std::string> &drain) {
    features                                                                              //
        | filter([](const auto &f) { return !f.empty() && f[0] != '-'; })                 //
        | map([](const auto &f) { return f[0] == '+' ? f.drop_front().str() : f.str(); }) //
        | for_each([&](const auto &f) { drain.emplace_back(std::move(f)); });
  };

  // normalise drain first, stuff could come in with +/- prefix
  drain = drain                                                              //
          | filter([](const auto &f) { return !f.empty() && f[0] != '-'; })  //
          | map([](const auto &f) { return f[0] == '+' ? f.substr(1) : f; }) //
          | to_vector();

  switch (arch) {
    case Triple::x86_64: {
      // Reject unknown CPU names up front - LLVM's getFeaturesForCPU asserts on miss. Callers
      // pass in arch strings that may name a non-x86 target (GPU device names, "vulkan-spirv",
      // etc.) when probing whether an image can run on the host; treat those as zero features.
      if (X86::parseArchX86(CPU) == X86::CK_None) break;
      SmallVector<StringRef> buffer;
      X86::getFeaturesForCPU(CPU, buffer);
      StringMap<bool> implied;
      buffer | for_each([&](const auto &b) {
        drain.push_back(b.str());
        X86::updateImpliedFeatures(b, true, implied);
      });
      implied                                                  //
          | filter([](const auto &i) { return i.getValue(); }) //
          | for_each([&](const auto &i) { drain.emplace_back(i.getKey().str()); });
      break;
    }
    case Triple::arm: {
      std::vector<StringRef> extensions;
      ARM::getExtensionFeatures(ARM::getDefaultExtensions(CPU, ARM::parseCPUArch(CPU)), extensions);
      normaliseFeature(extensions, drain);
      break;
    }
    case Triple::aarch64: {
      std::vector<StringRef> extensions;
      if (auto a = AArch64::getArchForCpu(CPU); a) {
        AArch64::getExtensionFeatures(a->DefaultExts, extensions);
        normaliseFeature(extensions, drain);
      }
      break;
    }
    case Triple::riscv32:
    case Triple::riscv64: {
      if (!RISCV::parseCPU(CPU, arch == Triple::riscv64)) break;
      SmallVector<std::string> buffer;
      RISCV::getFeaturesForCPU(CPU, buffer);
      buffer | for_each([&](const auto &b) { drain.emplace_back(b); });
      break;
    }
    case Triple::ppc64le:
    case Triple::ppc64: {
      if (!PPC::isValidCPU(CPU)) break;
      Triple T;
      T.setArch(arch);
      if (auto features = PPC::getPPCDefaultTargetFeatures(T, CPU)) {
        *features                                              //
            | filter([](const auto &kv) { return kv.second; }) //
            | for_each([&](const auto &kv) { drain.emplace_back(kv.first().str()); });
      }
      break;
    }
    default: throw std::logic_error("Unexpected arch from triple:" + Triple::getArchTypeName(arch).str());
  }

  drain ^= sort();
  drain ^= distinct();
}
} // namespace polyregion::llvm_shared
