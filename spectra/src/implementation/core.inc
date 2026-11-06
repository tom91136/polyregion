#define SPECTRA_CORE_VALUE(W, A)                                                                                     \
  struct [[clang::annotate("polyregion_type_variable:T" #W ":size=" #W)]] alignas(A) T##W {                          \
    char _[W];                                                                                                       \
  };

namespace polyregion::spectra {
SPECTRA_CORE_VALUE(16, 8)
SPECTRA_CORE_VALUE(20, 4)
SPECTRA_CORE_VALUE(24, 8)
SPECTRA_CORE_VALUE(28, 4)
SPECTRA_CORE_VALUE(32, 8)
SPECTRA_CORE_VALUE(36, 4)
SPECTRA_CORE_VALUE(40, 8)
SPECTRA_CORE_VALUE(44, 4)
SPECTRA_CORE_VALUE(48, 8)
SPECTRA_CORE_VALUE(52, 4)
SPECTRA_CORE_VALUE(56, 8)
SPECTRA_CORE_VALUE(60, 4)
SPECTRA_CORE_VALUE(64, 8)
} // namespace polyregion::spectra

[[clang::annotate("polyregion_export:spectra.implementation." SPECTRA_VARIANT ".core_for_each"),
  clang::annotate("polyregion_implements:polyregion.core.for_each"),
  clang::annotate("polyregion_requires:spectra." SPECTRA_VARIANT)]] void
spectra_core_for_each(std::int64_t n, fn<void, std::int64_t> body) {
  ::spectra::implementation::core_for_each(n, body);
}

#define SPECTRA_CORE_W(W)                                                                                            \
  (void)&::spectra::implementation::core_transform_reduce<polyregion::spectra::T##W, fn<polyregion::spectra::T##W, std::int64_t>, \
                                                          fn2<polyregion::spectra::T##W, polyregion::spectra::T##W,              \
                                                              polyregion::spectra::T##W>>;

static void instantiate_core() {
  SPECTRA_CORE_W(4)
  SPECTRA_CORE_W(8)
  SPECTRA_CORE_W(12)
  SPECTRA_CORE_W(16)
  SPECTRA_CORE_W(20)
  SPECTRA_CORE_W(24)
  SPECTRA_CORE_W(28)
  SPECTRA_CORE_W(32)
  SPECTRA_CORE_W(36)
  SPECTRA_CORE_W(40)
  SPECTRA_CORE_W(44)
  SPECTRA_CORE_W(48)
  SPECTRA_CORE_W(52)
  SPECTRA_CORE_W(56)
  SPECTRA_CORE_W(60)
  SPECTRA_CORE_W(64)
}

#undef SPECTRA_CORE_W
#undef SPECTRA_CORE_VALUE
