#include "test_utils.h"
#include "polyregion/llvm_utils.hpp"

#include "aspartame/all.hpp"

#include <iostream>

using namespace aspartame;

std::unique_ptr<polyregion::runtime::Platform> polyregion::test_utils::makePlatform(const runtime::Backend backend) {
  if (auto errorOrPlatform = runtime::Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
    throw std::runtime_error("Backend " + std::string(to_string(backend)) +
                             " failed to initialise: " + std::get<std::string>(errorOrPlatform));
  } else return std::move(std::get<std::unique_ptr<runtime::Platform>>(errorOrPlatform));
}

std::vector<std::pair<std::string, std::string>> polyregion::test_utils::findTestImage(const ImageGroups &images,
                                                                                       const runtime::Backend &backend,
                                                                                       const std::vector<std::string> &features) {

  const auto sortedFeatures = features ^ sort();

  //  std::cout << "Got: " << polyregion::mk_string<std::string>(sortedFeatures, std::identity(), ",") << std::endl;

  // HIP accepts HSA kernels
  const auto canonicalBackend = backend == runtime::Backend::HIP ? runtime::Backend::HSA : backend;

  return images                                          //
         ^ get_maybe(std::string(to_string(canonicalBackend))) //
         ^ flat_map([&](auto &featureToImage) -> std::optional<std::vector<std::pair<std::string, std::string>>> {
             // For things like OpenCL/Vulkan which is arch independent
             if (sortedFeatures.empty())
               return featureToImage ^ map_values([](auto &x) { return std::string(x.begin(), x.end()); }) ^ to_vector();

             // Try direct match first, GPUs would just be the ISA itself
             if (auto firstWithMatchingFeature =
                     sortedFeatures | collect_first([&](auto &feature) {
                       return featureToImage //
                              ^ get_maybe(feature) //
                              ^ map([&](auto &x) { return std::vector{std::pair{feature, std::string(x.begin(), x.end())}}; });
                     }))
               return *firstWithMatchingFeature;

             // For CPUs, we check if the image's requirement is a subset of the supported features
             for (auto &[arch, image] : featureToImage) {

#if defined(__x86_64__) || defined(_M_X64)
               auto archType = llvm::Triple::ArchType::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
               auto archType = llvm::Triple::ArchType::aarch64;
#elif defined(__arm__) || defined(_M_ARM)
               auto archType = llvm::Triple::ArchType::arm;
#else
  #error "Unsupported architecture"
#endif
               std::vector<std::string> required;
               llvm_shared::collectCPUFeatures(arch, archType, required);

               std::vector<std::string> missing;
               std::set_difference(required.begin(), required.end(), sortedFeatures.begin(), sortedFeatures.end(),
                                   std::back_inserter(missing));
               if (missing.empty()) {
                 return std::vector{std::pair{arch, std::string(image.begin(), image.end())}};
               }
             }
             return {};
           }) ^
         get_or_else(std::vector<std::pair<std::string, std::string>>{});
}
