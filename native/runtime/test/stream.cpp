#include "kernels/cpu_stream.hpp"
#include "kernels/gpu_stream.hpp"
#include "test_utils.h"
#include "utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace polyregion::runtime;
using namespace polyregion::test_utils;

template <typename N>
constexpr std::tuple<N, N, N, N> expectedResult(size_t times, size_t size, N a, N b, N c, N scalar) {
  N goldA = a;
  N goldB = b;
  N goldC = c;
  N goldSum = {};

  for (size_t i = 0; i < times; i++) {
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }
  for (size_t i = 0; i < size; i++) {
    goldSum += goldA * goldB;
  }
  return {goldA, goldB, goldC, goldSum};
}

template <typename N> static constexpr N StartA = 0.1;
template <typename N> static constexpr N StartB = 0.2;
template <typename N> static constexpr N StartC = 0.0;
template <typename N> static constexpr N StartScalar = 0.4;

TEST_CASE("GPU BabelStream") {
  auto backend = GENERATE(Backend::CUDA, Backend::OpenCL, Backend::HIP, Backend::HSA);
  auto size = GENERATE(as<size_t>{}, 1024, 2048, 4096, 8192, 16384, 32768, 65536);
  auto times = GENERATE(as<size_t>{}, 1, 2, 100);
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << platform->name()) {
    DYNAMIC_SECTION("size=" << size) {
      DYNAMIC_SECTION("times=" << times) {

        for (auto &d : platform->enumerate()) {
          DYNAMIC_SECTION("device=" << d->name()) {
            if (auto image = findTestImage(generated::gpu::stream, backend, d->features()); image) {
              d->loadModule("module", *image);

              size_t dot_wgsize = 256;
              size_t dot_num_groups = size / 256;

              auto a_d = d->mallocTyped<float>(size, Access::RW);
              auto b_d = d->mallocTyped<float>(size, Access::RW);
              auto c_d = d->mallocTyped<float>(size, Access::RW);
              auto sum_d = d->mallocTyped<float>(dot_num_groups, Access::RW);

              auto q = d->createQueue();

              std::vector<float> a(size, StartA<float>);
              std::vector<float> b(size, StartB<float>);
              std::vector<float> c(size, StartC<float>);
              std::vector<float> groupSums(dot_num_groups, {});

              waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(a.data(), a_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(b.data(), b_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(c.data(), c_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(groupSums.data(), sum_d, dot_num_groups, h); });

              float scalar = StartScalar<float>;
              waitAllN(times * 5, [&](auto h) {
                for (size_t i = 0; i < times; i++) {
                  q->enqueueInvokeAsync("module", "stream_copy",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Void, {}}},
                                        {{size, 1, 1}}, h);
                  q->enqueueInvokeAsync("module", "stream_mul",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Float32, &scalar},
                                         {Type::Void, {}}},
                                        {{size, 1, 1}}, h);
                  q->enqueueInvokeAsync("module", "stream_add",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Void, {}}},
                                        {{size, 1, 1}}, h);
                  q->enqueueInvokeAsync("module", "stream_triad",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Float32, &scalar},
                                         {Type::Void, {}}},
                                        {{size, 1, 1}}, h);
                  q->enqueueInvokeAsync(
                      "module", "stream_dot", //
                      {{Type::Ptr, &a_d},
                       {Type::Ptr, &b_d},
                       {Type::Ptr, &c_d},
                       {Type::Ptr, &sum_d},
                       {Type::Scratch, {}},
                       {Type::Int32, &size},
                       {Type::Void, {}}},
                      {{dot_num_groups * dot_wgsize, 1, 1}, {{{dot_wgsize, 1, 1}, sizeof(float) * dot_wgsize}}}, h);
                }
              });

              waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(a_d, a.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(b_d, b.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(c_d, c.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(sum_d, groupSums.data(), dot_num_groups, h); });

              d->freeAll(a_d, b_d, c_d, sum_d);

              auto [expectedA, expectedB, expectedC, expectedSum] =
                  expectedResult(times, size, StartA<float>, StartA<float>, StartC<float>, StartScalar<float>);

              for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(a[i], Catch::Matchers::WithinRel(expectedA));
                REQUIRE_THAT(b[i], Catch::Matchers::WithinRel(expectedB));
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(expectedC));
              }

              float sum = std::reduce(groupSums.begin(), groupSums.end());
              REQUIRE_THAT(sum, Catch::Matchers::WithinRel(expectedSum, 0.0008f));

            } else {
              WARN("No kernel test image found for device `"
                   << d->name() << "`(backend=" << nameOfBackend(backend)
                   << ", features=" << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
            }
          }
        }
      }
    }
  }
}

TEST_CASE("CPU BabelStream") {
  auto backend = GENERATE(Backend::RELOCATABLE_OBJ, Backend::SHARED_OBJ);
  auto size = GENERATE(as<size_t>{}, 1024, 2048, 4096, 8192, 16384, 32768, 65536);
  auto times = GENERATE(as<size_t>{}, 1, 2, 100);
  auto platform = Platform::of(backend);
  DYNAMIC_SECTION("backend=" << platform->name()) {
    DYNAMIC_SECTION("size=" << size) {
      DYNAMIC_SECTION("times=" << times) {

        for (auto &d : platform->enumerate()) {
          DYNAMIC_SECTION("device=" << d->name()) {
            if (auto image = findTestImage(generated::cpu::stream, backend, d->features()); image) {
              d->loadModule("module", *image);

              auto a_d = d->mallocTyped<float>(size, Access::RW);
              auto b_d = d->mallocTyped<float>(size, Access::RW);
              auto c_d = d->mallocTyped<float>(size, Access::RW);
              auto sum_d = d->mallocTyped<float>(1, Access::RW);

              auto q = d->createQueue();

              std::vector<float> a(size, StartA<float>);
              std::vector<float> b(size, StartB<float>);
              std::vector<float> c(size, StartC<float>);
              std::vector<float> sum(1, {});

              waitAll([&](auto &h) { q->enqueueHostToDeviceAsyncTyped(a.data(), a_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(b.data(), b_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(c.data(), c_d, size, h); },
                      [&](auto &h) { q->enqueueHostToDeviceAsyncTyped(sum.data(), sum_d, 1, h); });

              float scalar = StartScalar<float>;
              waitAllN(times * 5, [&](auto h) {
                for (size_t i = 0; i < times; i++) {
                  q->enqueueInvokeAsync("module", "stream_copy",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Int32, &size},
                                         {Type::Void, {}}},
                                        {}, h);
                  q->enqueueInvokeAsync("module", "stream_mul",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Float32, &scalar},
                                         {Type::Int32, &size},
                                         {Type::Void, {}}},
                                        {}, h);
                  q->enqueueInvokeAsync("module", "stream_add",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Int32, &size},
                                         {Type::Void, {}}},
                                        {}, h);
                  q->enqueueInvokeAsync("module", "stream_triad",
                                        {{Type::Ptr, &a_d}, //
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Float32, &scalar},
                                         {Type::Int32, &size},
                                         {Type::Void, {}}},
                                        {}, h);
                  q->enqueueInvokeAsync("module", "stream_dot", //
                                        {{Type::Ptr, &a_d},
                                         {Type::Ptr, &b_d},
                                         {Type::Ptr, &c_d},
                                         {Type::Ptr, &sum_d},
                                         {Type::Int32, &size},
                                         {Type::Void, {}}},
                                        {}, h);
                }
              });

              waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(a_d, a.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(b_d, b.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(c_d, c.data(), size, h); },
                      [&](auto &h) { q->enqueueDeviceToHostAsyncTyped(sum_d, sum.data(), 1, h); });

              d->freeAll(a_d, b_d, c_d, sum_d);

              auto [expectedA, expectedB, expectedC, expectedSum] =
                  expectedResult(times, size, StartA<float>, StartA<float>, StartC<float>, StartScalar<float>);

              for (size_t i = 0; i < size; ++i) {
                REQUIRE_THAT(a[i], Catch::Matchers::WithinRel(expectedA));
                REQUIRE_THAT(b[i], Catch::Matchers::WithinRel(expectedB));
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(expectedC));
              }
              REQUIRE_THAT(sum[0], Catch::Matchers::WithinRel(expectedSum));

            } else {
              WARN("No kernel test image found for device `"
                   << d->name() << "`(backend=" << nameOfBackend(backend)
                   << ", features=" << polyregion::mk_string<std::string>(d->features(), std::identity(), ",") << ")");
            }
          }
        }
      }
    }
  }
}