#include "stream.hpp"
#include "kernels/cpu_stream.hpp"
#include "kernels/gpu_stream.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace polyregion::runtime;

template <typename T, typename I>
void testStream(I images, Type tpe, const std::string &suffix, T relTolerance, //
                std::initializer_list<size_t> sizes,                           //
                std::initializer_list<size_t> groupSizes,                      //
                std::initializer_list<size_t> times,                           //
                std::initializer_list<Backend> backends) {
  auto backend = GENERATE_REF(values(backends));
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << platform->name()) {
    auto size = GENERATE_REF(values(sizes));
    DYNAMIC_SECTION("size=" << size) {
      auto time = GENERATE_REF(values(times));
      DYNAMIC_SECTION("times=" << time) {
        auto Ncore = GENERATE_REF(values(groupSizes));
        DYNAMIC_SECTION("Ncore=" << Ncore) {
          for (auto &d : platform->enumerate()) {
            DYNAMIC_SECTION("device=" << d->name()) {
              if (auto image = polyregion::test_utils::findTestImage(images, backend, d->features()); image) {
                polyregion::stream::runStream<T>(
                    tpe, suffix, size, time, Ncore, std::move(d), *image, false,
                    [](auto actual, auto limit) { CHECK(actual < limit); },
                    [&](auto actual) { CHECK(actual < relTolerance); });
              } else {
                WARN("No kernel test image found for device `"
                     << d->name() << "`(backend=" << nameOfBackend(backend) << ", features="
                     << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
              }
            }
          }
        }
      }
    }
  }
};

TEST_CASE("GPU BabelStream") {
  DYNAMIC_SECTION("float") {
    testStream<float>(generated::gpu::stream, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},   //
                      {32, 64, 128, 256},                                      //
                      {1, 2, 10},                                              //
                      {Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA});
  }
}

TEST_CASE("CPU BabelStream") {
  DYNAMIC_SECTION("double") {
    testStream<double>(generated::cpu::stream, Type::Double64, "_double", 0.0008, //
                       {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},     //
                       {1, 2, 3, 4, 5, 6, 7, 8},                                  //
                       {1, 2, 10},                                                //
                       {Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ});
  }
  DYNAMIC_SECTION("float") {
    testStream<float>(generated::cpu::stream, Type::Float32, "_float", 0.008f, //
                      {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072},   //
                      {1, 2, 3, 4, 5, 6, 7, 8},                                //
                      {1, 2, 10},                                              //
                      {Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ});
  }
}
