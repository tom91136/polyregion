#include "concurrency_utils.hpp"
#include "runtime.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace polyregion::stream {

using namespace polyregion::runtime;
using namespace polyregion::concurrency_utils;

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
  //  for (size_t i = 0; i < size; i++) {
  goldSum += goldA * goldB * size;
  //  }
  return {goldA, goldB, goldC, goldSum};
}

template <typename N> static constexpr N StartA = 0.1;
template <typename N> static constexpr N StartB = 0.2;
template <typename N> static constexpr N StartC = 0.0;
template <typename N> static constexpr N StartScalar = 0.4;

template <typename T> struct Kernels {
  T copy, mul, add, triad, dot;
};

template <typename T, typename FValidate, typename FValidateSum>
void runStream(Type tpe, const std::string &suffix, size_t size, size_t times, size_t groups, std::unique_ptr<Device> d,
               const std::string &image, bool verbose, FValidate fValidate, FValidateSum fValidateSum) {
  d->loadModule("module", image);

  bool cpu = d->sharedAddressSpace();

  auto [begin, end] = sequencePair(splitStaticExclusive(0, size, groups));
  auto begins_d = cpu ? d->mallocTyped<int64_t>(begin.size(), Access::RO) : 0;
  auto ends_d = cpu ? d->mallocTyped<int64_t>(end.size(), Access::RO) : 0;
  auto sumGroups = cpu ? groups : 256;

  auto a_d = d->mallocTyped<T>(size, Access::RW);
  auto b_d = d->mallocTyped<T>(size, Access::RW);
  auto c_d = d->mallocTyped<T>(size, Access::RW);
  auto sum_d = d->mallocTyped<T>(sumGroups, Access::RW);

  auto q = d->createQueue();

  std::vector<T> a(size, StartA<T>);
  std::vector<T> b(size, StartB<T>);
  std::vector<T> c(size, StartC<T>);
  std::vector<T> sum(sumGroups, {});

  auto h2dT1 = std::chrono::high_resolution_clock::now();

  waitAllN([&, begin = begin, end = end](auto h) {
    if (cpu) {
      q->enqueueHostToDeviceAsyncTyped(begin.data(), begins_d, begin.size(), h());
      q->enqueueHostToDeviceAsyncTyped(end.data(), ends_d, end.size(), h());
    }
    q->enqueueHostToDeviceAsyncTyped(a.data(), a_d, size, h());
    q->enqueueHostToDeviceAsyncTyped(b.data(), b_d, size, h());
    q->enqueueHostToDeviceAsyncTyped(c.data(), c_d, size, h());
    q->enqueueHostToDeviceAsyncTyped(sum.data(), sum_d, sumGroups, h());
  });
  auto h2dT2 = std::chrono::high_resolution_clock::now();
  auto h2dElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(h2dT2 - h2dT1).count();

  T scalar = StartScalar<T>;

  Kernels<std::vector<double>> elapsed{
      .copy = std::vector<double>(times),
      .mul = std::vector<double>(times),
      .add = std::vector<double>(times),
      .triad = std::vector<double>(times),
      .dot = std::vector<double>(times),
  };
  auto invoke = [&](const auto &acc, const std::string &kernelName, const Policy &policy, const ArgBuffer &buffer) {
    return [&](auto &h) {
      auto _buffer = buffer;
      if (cpu) _buffer.prepend({{Type::Long64, nullptr}, {Type::Ptr, &begins_d}, {Type::Ptr, &ends_d}});

      auto t1 = std::chrono::high_resolution_clock::now();
      q->enqueueInvokeAsync("module", kernelName + suffix, _buffer, policy, [&, t1]() {
        auto t2 = std::chrono::high_resolution_clock::now();
        acc(elapsed, std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
        h();
      });
    };
  };

  Policy forPolicy = cpu ? Policy{{groups, 1, 1}, {}} //
                         : Policy{{size / groups, 1, 1}, {{{groups, 1, 1}, {}}}};
  Policy dotPolicy = cpu ? Policy{{groups, 1, 1}, {}} //
                         : Policy{{sumGroups, 1, 1}, {{{groups, 1, 1}, sizeof(T) * groups}}};

  for (size_t i = 0; i < times; i++) {
    waitAll(10000,
            invoke([=](auto &acc, auto x) { acc.copy[i] = x; }, "stream_copy", forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.mul[i] = x; }, "stream_mul", forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {tpe, &scalar}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.add[i] = x; }, "stream_add", forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.triad[i] = x; }, "stream_triad", forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {tpe, &scalar}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.dot[i] = x; }, "stream_dot", dotPolicy,
                   cpu ? ArgBuffer{{Type::Ptr, &a_d},
                                   {Type::Ptr, &b_d},
                                   {Type::Ptr, &c_d},
                                   {Type::Ptr, &sum_d},
                                   {Type::Void, {}}}
                       : ArgBuffer{{Type::Ptr, &a_d},
                                   {Type::Ptr, &b_d},
                                   {Type::Ptr, &c_d},
                                   {Type::Ptr, &sum_d},
                                   {Type::Scratch, {}},
                                   {Type::Int32, &size},
                                   {Type::Void, {}}})

    );
  }

  auto d2hT1 = std::chrono::high_resolution_clock::now();
  waitAllN([&](auto h) {
    q->enqueueDeviceToHostAsyncTyped(a_d, a.data(), size, h());
    q->enqueueDeviceToHostAsyncTyped(b_d, b.data(), size, h());
    q->enqueueDeviceToHostAsyncTyped(c_d, c.data(), size, h());
    q->enqueueDeviceToHostAsyncTyped(sum_d, sum.data(), sumGroups, h());
  });
  auto d2hT2 = std::chrono::high_resolution_clock::now();
  auto d2hElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(d2hT2 - d2hT1).count();

  d->freeAll(a_d, b_d, c_d, sum_d);
  if (cpu) d->freeAll(begins_d, ends_d);

  auto [expectedA, expectedB, expectedC, expectedSum] =
      expectedResult(times, size, StartA<T>, StartA<T>, StartC<T>, StartScalar<T>);

  auto error = [](auto xs, auto expected) {
    return std::accumulate(xs.begin(), xs.end(), 0.0,
                           [&](double acc, const T x) { return acc + std::fabs(x - expected); }) /
           xs.size();
  };
  auto eps = std::numeric_limits<T>::epsilon() * 100.0;
  fValidate(error(a, expectedA), eps);
  fValidate(error(b, expectedB), eps);
  fValidate(error(c, expectedC), eps);
  auto reducedSum = std::reduce(sum.begin(), sum.end());
  fValidateSum(std::fabs((reducedSum - expectedSum) / expectedSum));

  const auto sizesMB = Kernels<double>{.copy = double(2 * sizeof(T) * size) / 1000 / 1000,  //
                                       .mul = double(2 * sizeof(T) * size) / 1000 / 1000,   //
                                       .add = double(3 * sizeof(T) * size) / 1000 / 1000,   //
                                       .triad = double(3 * sizeof(T) * size) / 1000 / 1000, //
                                       .dot = double(2 * sizeof(T) * size) / 1000 / 1000};

  auto bandwidth = [&](auto &&f) { return *f(sizesMB) / *std::min_element(f(elapsed)->begin(), f(elapsed)->end()); };

  if (verbose) {
    std::cerr                                 //
        << std::fixed << std::setprecision(3) //
        << "===Stream (" << d->name() << " #" << d->id() << " )===\n"
        << "Running kernels " << times << " times\n"
        << "Precision: " << typeName(tpe) << "\n"
        << "Array size: " << sizesMB.copy / 2 << " MB (=" << sizesMB.copy / 2 / 1000 << " GB)\n"
        << "Total size: " << sizesMB.triad << " MB (=" << sizesMB.triad / 1000 << " GB)\n"
        << "D2H = " << d2hElapsed << "s"
        << " H2D = " << h2dElapsed << "s\n"
        << "Function MBytes/sec\n"
        << "Copy     " << bandwidth([](auto &x) { return &x.copy; }) << "\n"
        << "Mul      " << bandwidth([](auto &x) { return &x.mul; }) << "\n"
        << "Add      " << bandwidth([](auto &x) { return &x.add; }) << "\n"
        << "Triad    " << bandwidth([](auto &x) { return &x.triad; }) << "\n"
        << "Dot      " << bandwidth([](auto &x) { return &x.dot; }) << "\n";
  }
}
} // namespace polyregion::stream