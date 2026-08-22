#pragma region case: narrow_bitwise
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 61185 1 254 61185

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

int main() {
  uint64_t wide = std::uint64_t{0x12345678ABCDEF01ULL};
  const auto result = __polyregion_offload_f1__([wide]() {
    const auto a = static_cast<uint16_t>(wide & 0xFFFFULL);
    const auto b = static_cast<uint8_t>((wide | 0x100ULL) & 0xFFULL);
    const auto c = static_cast<uint8_t>((wide ^ 0xFFULL) & 0xFFULL);
    uint64_t d = wide;
    d &= 0xFFFFULL;
    return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << uint64_t{16}) | (static_cast<uint64_t>(c) << uint64_t{32})
           | (static_cast<uint64_t>(static_cast<uint16_t>(d)) << uint64_t{48});
  });
  std::printf("%u %u %u %u", static_cast<unsigned>(result & 0xFFFF), static_cast<unsigned>((result >> uint64_t{16}) & 0xFFFF),
              static_cast<unsigned>((result >> uint64_t{32}) & 0xFFFF), static_cast<unsigned>((result >> uint64_t{48}) & 0xFFFF));
  return 0;
}
