#include "test_utils.h"
#include "llvm_utils.hpp"

std::vector<std::pair<std::string, std::string>>
polyregion::test_utils::findTestImage(const ImageGroups &images, const polyregion::runtime::Backend &backend,
                                      const std::vector<std::string> &features) {

  auto sortedFeatures = features;
  std::sort(sortedFeatures.begin(), sortedFeatures.end());

  //  std::cout << "Got: " << polyregion::mk_string<std::string>(sortedFeatures, std::identity(), ",") << std::endl;

  // HIP accepts HSA kernels
  auto actualBackend = backend == polyregion::runtime::Backend::HIP ? polyregion::runtime::Backend::HSA : backend;
  if (auto it = images.find(std::string(to_string(actualBackend))); it != images.end()) {
    auto entries = it->second;

    if (features.empty()) { // for things like OpenCL/Vulkan which is arch independent
      std::vector<std::pair<std::string, std::string>> out;
      for (auto &[k, v] : entries)
        out.emplace_back(k, std::string(v.begin(), v.end()));
      return out;
    } else {

      // Try direct match first, GPUs would just be the ISA itself
      for (auto &f : sortedFeatures) {
        if (auto archAndCode = entries.find(f); archAndCode != entries.end()) {

          return {{archAndCode->first, std::string(archAndCode->second.begin(), archAndCode->second.end())}};
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
          return {{arch, std::string(image.begin(), image.end())}};
        }
      }
    }
  }
  return {};
}
