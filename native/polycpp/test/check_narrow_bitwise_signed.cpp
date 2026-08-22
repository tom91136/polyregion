#pragma region case: narrow_bitwise_signed
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: -2 254 1

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  int64_t wide = -2;
  const auto result = __polyregion_offload_f1__([wide]() {
    const auto signedLow = static_cast<int8_t>(wide & 0xFF);
    const auto unsignedLow = static_cast<uint8_t>(static_cast<uint64_t>(wide) & 0xFFULL);
    const auto invertedLow = static_cast<uint8_t>(~static_cast<uint64_t>(wide));
    return (static_cast<uint32_t>(static_cast<uint8_t>(signedLow)) << uint32_t{16}) | (static_cast<uint32_t>(unsignedLow) << uint32_t{8})
           | invertedLow;
  });
  std::printf("%d %u %u", static_cast<int>(static_cast<int8_t>(result >> 16)), static_cast<unsigned>((result >> 8) & 0xFF),
              static_cast<unsigned>(result & 0xFF));
  return 0;
}
