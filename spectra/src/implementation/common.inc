namespace polyregion::spectra {
struct [[clang::annotate("polyregion_type_variable:K4:size=4")]] alignas(4) K4 {
  char _[4];
};
struct [[clang::annotate("polyregion_type_variable:K8:size=8")]] alignas(8) K8 {
  char _[8];
};
struct [[clang::annotate("polyregion_type_variable:K12:size=12")]] alignas(4) K12 {
  char _[12];
};
struct [[clang::annotate("polyregion_type_variable:T4:size=4")]] alignas(4) T4 {
  char _[4];
};
struct [[clang::annotate("polyregion_type_variable:T8:size=8")]] alignas(8) T8 {
  char _[8];
};
struct [[clang::annotate("polyregion_type_variable:T12:size=12")]] alignas(4) T12 {
  char _[12];
};
struct [[clang::annotate("polyregion_type_variable:U4:size=4")]] alignas(4) U4 {
  char _[4];
};
struct [[clang::annotate("polyregion_type_variable:U8:size=8")]] alignas(8) U8 {
  char _[8];
};
struct [[clang::annotate("polyregion_type_variable:U12:size=12")]] alignas(4) U12 {
  char _[12];
};
struct [[clang::annotate("polyregion_type_variable:V4:size=4")]] alignas(4) V4 {
  char _[4];
};
struct [[clang::annotate("polyregion_type_variable:V8:size=8")]] alignas(8) V8 {
  char _[8];
};
struct [[clang::annotate("polyregion_type_variable:V12:size=12")]] alignas(4) V12 {
  char _[12];
};
} // namespace polyregion::spectra

using polyregion::spectra::K12;
using polyregion::spectra::K4;
using polyregion::spectra::K8;
using polyregion::spectra::T12;
using polyregion::spectra::T4;
using polyregion::spectra::T8;
using polyregion::spectra::U12;
using polyregion::spectra::U4;
using polyregion::spectra::U8;
using polyregion::spectra::V12;
using polyregion::spectra::V4;
using polyregion::spectra::V8;
template <class R, class... A> using fn = polyregion::spectra::function<R(A...)>;
template <class R, class... A> using fn2 = polyregion::spectra::function2<R(A...)>;

#define POLYREGION_SPECTRA_EXPORT(name)                                                                               \
  [[clang::annotate("polyregion_export_template:spectra.implementation." SPECTRA_VARIANT "." #name),                 \
    clang::annotate("polyregion_implements:spectra." #name),                                                         \
    clang::annotate("polyregion_requires:spectra." SPECTRA_VARIANT)]]
