#include "concurrency_utils.hpp"
#include "polyrt/runtime.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <algorithm>

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
void validate(size_t size, size_t times, FValidate fValidate, FValidateSum fValidateSum, const std::vector<T> &a, const std::vector<T> &b,
              const std::vector<T> &c, const std::vector<T> &sum) {
  auto [expectedA, expectedB, expectedC, expectedSum] = expectedResult(times, size, StartA<T>, StartA<T>, StartC<T>, StartScalar<T>);

  auto error = [](auto xs, auto expected) {
    return std::accumulate(xs.begin(), xs.end(), 0.0, [&](double acc, const T x) { return acc + std::fabs(x - expected); }) / xs.size();
  };
  auto eps = std::numeric_limits<T>::epsilon() * 100.0;
  fValidate(error(a, expectedA), eps);
  fValidate(error(b, expectedB), eps);
  fValidate(error(c, expectedC), eps);
  auto reducedSum = std::reduce(sum.begin(), sum.end());
  fValidateSum(std::fabs((reducedSum - expectedSum) / expectedSum));
}

template <typename T>
void renderElapsed(std::string title, Type tpe, size_t size, size_t times, const Kernels<std::vector<double>> &elapsed,
                   const std::vector<std::string> &extraLines) {

  const auto sizesMB = Kernels<double>{.copy = double(2 * sizeof(T) * size) / 1000 / 1000,  //
                                       .mul = double(2 * sizeof(T) * size) / 1000 / 1000,   //
                                       .add = double(3 * sizeof(T) * size) / 1000 / 1000,   //
                                       .triad = double(3 * sizeof(T) * size) / 1000 / 1000, //
                                       .dot = double(2 * sizeof(T) * size) / 1000 / 1000};

  auto bandwidth = [&](auto &&f) { return *f(sizesMB) / *std::min_element(f(elapsed)->begin(), f(elapsed)->end()); };

  std::cerr                                 //
      << std::fixed << std::setprecision(3) //
      << "===BabelStream (" << title << " )===\n"
      << "Running kernels " << times << " times\n"
      << "Precision: " << to_string(tpe) << "\n"
      << "Array size: " << sizesMB.copy / 2 << " MB (=" << sizesMB.copy / 2 / 1000 << " GB)\n"
      << "Total size: " << sizesMB.triad << " MB (=" << sizesMB.triad / 1000 << " GB)\n";
  //        << "D2H = " << d2hElapsed << "s"
  //        << " H2D = " << h2dElapsed << "s\n"
  //    title << "; " << d->name() << " #" << d->id()
  for (auto &ln : extraLines)
    std::cerr << ln << "\n";

  std::cerr << "Function MBytes/sec\n"
            << "Copy     " << bandwidth([](auto &x) { return &x.copy; }) << "\n"
            << "Mul      " << bandwidth([](auto &x) { return &x.mul; }) << "\n"
            << "Add      " << bandwidth([](auto &x) { return &x.add; }) << "\n"
            << "Triad    " << bandwidth([](auto &x) { return &x.triad; }) << "\n"
            << "Dot      " << bandwidth([](auto &x) { return &x.dot; }) << "\n";
}

template <typename T>
Kernels<std::vector<double>> dispatch(Type tpe, size_t size, size_t times, size_t groups, bool threaded, DeviceQueue &q,
                                      const Kernels<std::pair<std::string, std::string>> &specs, size_t sumGroups, void *begins_d,
                                      void *ends_d, void *a_d, void *b_d, void *c_d, void *sum_d) {
  T scalar = StartScalar<T>;

  Kernels<std::vector<double>> elapsed{
      .copy = std::vector<double>(times),
      .mul = std::vector<double>(times),
      .add = std::vector<double>(times),
      .triad = std::vector<double>(times),
      .dot = std::vector<double>(times),
  };
  auto invoke = [&](const auto &acc, const std::pair<std::string, std::string> &spec, const Policy &policy, const ArgBuffer &buffer) {
    return [&](auto &h) {
      auto _buffer = buffer;
      if (threaded) _buffer.prepend({{Type::Long64, nullptr}, {Type::Ptr, &begins_d}, {Type::Ptr, &ends_d}});

      auto t1 = std::chrono::high_resolution_clock::now();
      q.enqueueInvokeAsync(spec.first, spec.second, _buffer, policy, [&, t1]() {
        auto t2 = std::chrono::high_resolution_clock::now();
        acc(elapsed, std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
        h();
      });
    };
  };

  Policy forPolicy = threaded ? Policy{{groups, 1, 1}, {}} //
                              : Policy{{size / groups, 1, 1}, {{{groups, 1, 1}, {}}}};
  Policy dotPolicy = threaded ? Policy{{groups, 1, 1}, {}} //
                              : Policy{{sumGroups, 1, 1}, {{{groups, 1, 1}, sizeof(T) * (groups)}}};
  // G=256 * L=2

  for (size_t i = 0; i < times; i++) {
    waitAll(10000,
            invoke([=](auto &acc, auto x) { acc.copy[i] = x; }, specs.copy, forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.mul[i] = x; }, specs.mul, forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {tpe, &scalar}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.add[i] = x; }, specs.add, forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.triad[i] = x; }, specs.triad, forPolicy,
                   {{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {tpe, &scalar}, {Type::Void, {}}}),
            invoke([=](auto &acc, auto x) { acc.dot[i] = x; }, specs.dot, dotPolicy,
                   threaded ? ArgBuffer{{Type::Ptr, &a_d}, {Type::Ptr, &b_d}, {Type::Ptr, &c_d}, {Type::Ptr, &sum_d}, {Type::Void, {}}}
                            : ArgBuffer{{Type::Ptr, &a_d},
                                        {Type::Ptr, &b_d},
                                        {Type::Ptr, &c_d},
                                        {Type::Ptr, &sum_d},
                                        {Type::Scratch, {}},
                                        {Type::Int32, &size},
                                        {Type::Void, {}}})

    );
  }

  return elapsed;
}

template <typename T, typename FValidate, typename FValidateSum>
void runStream(Type tpe, size_t size, size_t times, size_t groups, const std::string &title, PlatformKind kind, Device &d,
               const Kernels<std::pair<std::string, std::string>> &specs, bool verbose, FValidate fValidate, FValidateSum fValidateSum) {

  bool threaded = kind == PlatformKind::HostThreaded;

  auto [begin, end] = sequencePair(splitStaticExclusive<int64_t>(0, int64_t(size), int64_t(groups)));
  auto begins_d = threaded ? d.mallocDeviceTyped<int64_t>(begin.size(), Access::RO) : 0;
  auto ends_d = threaded ? d.mallocDeviceTyped<int64_t>(end.size(), Access::RO) : 0;
  auto sumGroups = threaded ? groups : 256;

  auto a_d = d.mallocDeviceTyped<T>(size, Access::RW);
  auto b_d = d.mallocDeviceTyped<T>(size, Access::RW);
  auto c_d = d.mallocDeviceTyped<T>(size, Access::RW);
  auto sum_d = d.mallocDeviceTyped<T>(sumGroups, Access::RW);

  auto q = d.createQueue();

  std::vector<T> a(size, StartA<T>);
  std::vector<T> b(size, StartB<T>);
  std::vector<T> c(size, StartC<T>);
  std::vector<T> sum(sumGroups, {});

  auto h2dT1 = std::chrono::high_resolution_clock::now();
  {
    waitAllN([&, begin = begin, end = end](auto h) {
      if (threaded) {
        q->enqueueHostToDeviceAsyncTyped(begin.data(), begins_d, begin.size(), h());
        q->enqueueHostToDeviceAsyncTyped(end.data(), ends_d, end.size(), h());
      }
      q->enqueueHostToDeviceAsyncTyped(a.data(), a_d, size, h());
      q->enqueueHostToDeviceAsyncTyped(b.data(), b_d, size, h());
      q->enqueueHostToDeviceAsyncTyped(c.data(), c_d, size, h());
      q->enqueueHostToDeviceAsyncTyped(sum.data(), sum_d, sumGroups, h());
    });
  }
  auto h2dT2 = std::chrono::high_resolution_clock::now();
  auto h2dElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(h2dT2 - h2dT1).count();

  auto elapsed = dispatch<T>(tpe, size, times, groups, threaded, *q, specs, sumGroups,
                             reinterpret_cast<void *>(begins_d), //
                             reinterpret_cast<void *>(ends_d),   //
                             reinterpret_cast<void *>(a_d),      //
                             reinterpret_cast<void *>(b_d),      //
                             reinterpret_cast<void *>(c_d),      //
                             reinterpret_cast<void *>(sum_d)     //
  );

  auto d2hT1 = std::chrono::high_resolution_clock::now();
  {
    waitAllN([&](auto h) {
      q->enqueueDeviceToHostAsyncTyped(a_d, a.data(), size, h());
      q->enqueueDeviceToHostAsyncTyped(b_d, b.data(), size, h());
      q->enqueueDeviceToHostAsyncTyped(c_d, c.data(), size, h());
      q->enqueueDeviceToHostAsyncTyped(sum_d, sum.data(), sumGroups, h());
    });
  }
  auto d2hT2 = std::chrono::high_resolution_clock::now();
  auto d2hElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(d2hT2 - d2hT1).count();

  d.freeAllDevice(a_d, b_d, c_d, sum_d);
  if (threaded) d.freeAllDevice(begins_d, ends_d);

  validate<T, FValidate, FValidateSum>(size, times, fValidate, fValidateSum, a, b, c, sum);

//  for (size_t i = 0; i < sum.size(); ++i) {
//    std::cout << "[" << i << "]" << sum[i] << std::endl;
//  }

  if (verbose) {
    renderElapsed<T>(title + "; " + d.name() + " #" + std::to_string(d.id()), tpe, size, times, elapsed,
                     {"D2H = " + std::to_string(d2hElapsed) + "s", "H2D = " + std::to_string(h2dElapsed) + "s"});
  }
}

template <typename T, typename FValidate, typename FValidateSum>
void runStreamShared(Type tpe, size_t size, size_t times, size_t groups, const std::string &title, PlatformKind kind, Device &d,
                     const Kernels<std::pair<std::string, std::string>> &specs, bool verbose, FValidate fValidate,
                     FValidateSum fValidateSum) {

  if (auto checkAlloc = d.mallocShared(1, Access::RW); checkAlloc) {
    d.freeShared(*checkAlloc);
  } else {
    std::cerr << "shared allocation unsupported for device: " + d.name()  << ", skipping..." << std::endl;
    return;
  }
  bool threaded = kind == PlatformKind::HostThreaded;

  auto [begin, end] = splitStaticExclusive2<int64_t>(0, int64_t(size), int64_t(groups));
  auto begins_d = threaded ? *d.mallocSharedTyped<int64_t>(begin.size(), Access::RO) : 0;
  auto ends_d = threaded ? *d.mallocSharedTyped<int64_t>(end.size(), Access::RO) : 0;
  auto sumGroups = threaded ? groups : 256;

  auto a_d = *d.mallocSharedTyped<T>(size, Access::RW);
  auto b_d = *d.mallocSharedTyped<T>(size, Access::RW);
  auto c_d = *d.mallocSharedTyped<T>(size, Access::RW);
  auto sum_d = *d.mallocSharedTyped<T>(sumGroups, Access::RW);

  auto q = d.createQueue();

  std::vector<T> a(size, StartA<T>);
  std::vector<T> b(size, StartB<T>);
  std::vector<T> c(size, StartC<T>);
  std::vector<T> sum(sumGroups, {});

  auto h2dT1 = std::chrono::high_resolution_clock::now();
  {
    if (threaded) {
      std::memcpy(begins_d, begin.data(), begin.size() * sizeof(T));
      std::memcpy(ends_d, end.data(), end.size() * sizeof(T));
    }
    std::memcpy(a_d, a.data(), size * sizeof(T));
    std::memcpy(b_d, b.data(), size * sizeof(T));
    std::memcpy(c_d, c.data(), size * sizeof(T));
    std::memcpy(sum_d, sum.data(), sumGroups * sizeof(T));
  }
  auto h2dT2 = std::chrono::high_resolution_clock::now();
  auto h2dElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(h2dT2 - h2dT1).count();

  auto elapsed = dispatch<T>(tpe, size, times, groups, threaded, *q, specs, sumGroups, begins_d, ends_d, a_d, b_d, c_d, sum_d);

  auto d2hT1 = std::chrono::high_resolution_clock::now();
  {
    std::memcpy(a.data(), a_d, size * sizeof(T));
    std::memcpy(b.data(), b_d, size * sizeof(T));
    std::memcpy(c.data(), c_d, size * sizeof(T));
    std::memcpy(sum.data(), sum_d, sumGroups * sizeof(T));
  }
  auto d2hT2 = std::chrono::high_resolution_clock::now();
  auto d2hElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(d2hT2 - d2hT1).count();

  d.freeAllShared(a_d, b_d, c_d, sum_d);
  if (threaded) d.freeAllShared(begins_d, ends_d);

  validate<T, FValidate, FValidateSum>(size, times, fValidate, fValidateSum, a, b, c, sum);

//  for (size_t i = 0; i < sum.size(); ++i) {
//    std::cout << "[" << i << "]" << sum[i] << std::endl;
//  }

  if (verbose) {
    renderElapsed<T>(title + "; " + d.name() + " #" + std::to_string(d.id()), tpe, size, times, elapsed,
                     {"D2H = " + std::to_string(d2hElapsed) + "s", "H2D = " + std::to_string(h2dElapsed) + "s"});
  }
}
} // namespace polyregion::stream