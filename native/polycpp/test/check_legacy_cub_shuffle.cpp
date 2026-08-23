#pragma region case: legacy-cub-shuffle
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -fsyntax-only {input}

namespace cub {

unsigned SHFL_DOWN_SYNC(unsigned word, int offset, int flags, unsigned mask) {
  asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;" : "=r"(word) : "r"(word), "r"(offset), "r"(flags), "r"(mask));
  return word;
}

unsigned SHFL_UP_SYNC(unsigned word, int offset, int flags, unsigned mask) {
  asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;" : "=r"(word) : "r"(word), "r"(offset), "r"(flags), "r"(mask));
  return word;
}

} // namespace cub

[[clang::annotate("polyregion_export:foo.implementation.shuffle")]] unsigned shuffle(unsigned value) {
  return cub::SHFL_DOWN_SYNC(value, 1, 31, ~0u) + cub::SHFL_UP_SYNC(value, 1, 31, ~0u);
}
