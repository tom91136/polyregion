#pragma region case: cxx-new-record
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 1 1 1
#pragma region stderr-count: 0 failed to release

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>

#include "reflect-rt/rt_reflect.hpp"

struct alignas(64) OverAligned {
  char data[64];
};

template <typename T> bool tracked(const T *p) {
  const auto meta = polyregion::rt_reflect::_rt_reflect_p(p);
  return meta.offset == 0 && meta.type == polyregion::rt_reflect::Type::HeapCXXNew;
}

int main() {
  auto *aligned = new OverAligned{};
  auto *nothrow = new (std::nothrow) std::size_t[4];
  const bool alignedTracked = tracked(aligned);
  const bool nothrowTracked = tracked(nothrow);
  const bool alignedHonoured = reinterpret_cast<std::uintptr_t>(aligned) % alignof(OverAligned) == 0;
  delete aligned;
  delete[] nothrow;
  std::printf("%d %d %d", alignedTracked ? 1 : 0, nothrowTracked ? 1 : 0, alignedHonoured ? 1 : 0);
  return 0;
}
