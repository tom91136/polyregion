#include "test_utils.h"
#include "llvm_utils.hpp"

std::optional<std::string> polyregion::test_utils::findTestImage(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> &images,
    const polyregion::runtime::Backend &backend, const std::vector<std::string> &features) {

  auto sortedFeatures = features;
  std::sort(sortedFeatures.begin(), sortedFeatures.end());

  //  std::cout << "Got: " << polyregion::mk_string<std::string>(sortedFeatures, std::identity(), ",") << std::endl;

  // HIP accepts HSA kernels
  auto actualBackend = backend == polyregion::runtime::Backend::HIP ? polyregion::runtime::Backend::HSA : backend;
  if (auto it = images.find(nameOfBackend(actualBackend)); it != images.end()) {
    auto entries = it->second;

    if (features.empty() && entries.size() == 1) { // for things like OpenCL which is arch independent
      auto head = entries.begin()->second;
      return std::string(head.begin(), head.end());
    } else {

      // Try direct match first, GPUs would just be the ISA itself
      for (auto &f : sortedFeatures) {
        if (auto archAndCode = entries.find(f); archAndCode != entries.end()) {
          return std::string(archAndCode->second.begin(), archAndCode->second.end());
        }
      }

      // For CPUs, we check if the image's requirement is a subset of the supported features
      for (auto &[arch, image] : entries) {

        std::vector<std::string> required;
        polyregion::llvm_shared::collectCPUFeatures(arch, llvm::Triple::ArchType::x86_64, required);

        std::vector<std::string> missing;
        std::set_difference(required.begin(), required.end(), sortedFeatures.begin(), sortedFeatures.end(),
                            std::back_inserter(missing));
        //        std::cout << "[" << arch << "] missing: " << polyregion::mk_string<std::string>(missing,
        //        std::identity(), ",")
        //                  << std::endl;

        if (missing.empty()) {
          return std::string(image.begin(), image.end());
        }
      }
    }
  }
  return {};
}
