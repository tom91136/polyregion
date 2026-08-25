#pragma region case: package-source-idioms
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-source-idioms {output}.polyast

#include <cstdint>
#include <variant>

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]

namespace thrust {
template <class T> T *next(T *pointer, long offset = 1) { return pointer + offset; }
} // namespace thrust

struct Pair {
  std::uint32_t left;
  std::uint32_t right;
};

struct EmptyA {};
struct EmptyB {};

POLYREGION_EXPORT_AS("source_idioms.implementation.apply")
std::uint64_t apply(Pair *destination, const Pair *source, int *values, std::variant<EmptyA, EmptyB> choice) {
  __builtin_memcpy(destination, source, sizeof(Pair));
  const auto bits = __builtin_bit_cast(std::uint64_t, *source);
  const auto selected = std::visit([](auto) { return 1; }, choice);
  return bits + std::uint64_t(*thrust::next(values, selected));
}
