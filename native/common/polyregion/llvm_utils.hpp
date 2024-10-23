#pragma once

#include "polyregion/compat.h"


#include <vector>
#ifdef NO_ERROR
#undef NO_ERROR
#endif


#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/X86TargetParser.h"


namespace polyregion::llvm_shared {

static bool isCPUTargetSupported(const std::string &CPU, //
                                 const llvm::Triple::ArchType &arch) {
  using namespace llvm;
  switch (arch) {
    case Triple::x86_64: return CPU == "native" || (llvm::X86::parseArchX86(CPU) != llvm::X86::CPUKind::CK_None);
    case Triple::arm: return llvm::ARM::parseCPUArch(CPU) != llvm::ARM::ArchKind::INVALID;
    case Triple::aarch64: return llvm::AArch64::parseCpu(CPU).has_value();
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
    for (auto &f : features) {
      if (f.empty() || f[0] == '-') continue;
      drain.emplace_back(f[0] == '+' ? f.drop_front().str() : f.str());
    }
  };

  // normalise drain first, stuff could come in with +/- prefix
  for (auto it = drain.begin(); it != drain.end();) {
    if (it->empty() || (*it)[0] == '-') it = drain.erase(it);
    else {
      if ((*it)[0] == '+') it->erase(0, 1);
      ++it;
    }
  }

  switch (arch) {
    case Triple::x86_64: {
      SmallVector<StringRef> buffer;
      X86::getFeaturesForCPU(CPU, buffer);
      StringMap<bool> implied;
      for (auto &b : buffer) {
        drain.push_back(b.str());
        X86::updateImpliedFeatures(b, true, implied);
      }
      for (auto &i : implied)
        if (i.second) drain.emplace_back(i.first().str());
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
    default: throw std::logic_error("Unexpected arch from triple:" + Triple::getArchTypeName(arch).str());
  }

  std::sort(drain.begin(), drain.end());
  drain.erase(unique(drain.begin(), drain.end()), drain.end());
}
} // namespace polyregion::llvm_shared
