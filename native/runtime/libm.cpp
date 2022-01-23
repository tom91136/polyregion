#include "libm.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include <cmath> // we can't use <cmath> here because of the overloads

template <typename T> static void sym(const std::string &name, T (*f)(T)) {
  llvm::sys::DynamicLibrary::AddSymbol(name, (void *)f);
}

template <typename T> static void sym(const std::string &name, T (*f)(T, T)) {
  llvm::sys::DynamicLibrary::AddSymbol(name, (void *)f);
}

template <typename T> static void sym(const std::string &name, T (*f)(T, T, T)) {
  llvm::sys::DynamicLibrary::AddSymbol(name, (void *)f);
}

template <typename T> static void sym(const std::string &name, T f) {
  llvm::sys::DynamicLibrary::AddSymbol(name, (void *)f);
}

template <typename T> static void sincos_local(T x, T *sin, T *cos) {
  *sin = std::sin(x);
  *cos = std::cos(x);
}

void polyregion::libm::exportAll() {

  using I = int;
  using D = double;
  using F = float;
  using LD = long double;
  using L = long;
  using LL = long long;

  // make sure we cover all of https://en.cppreference.com/w/c/numeric/math and use the correct name

  // Non-standard GNU
  sym<void (*)(D, D *, D *)>("sincos", &sincos_local);
  sym<void (*)(F, F *, F *)>("sincosf", &sincos_local);
  sym<void (*)(LD, LD *, LD *)>("sincosl", &sincos_local);

  // Basic
  sym<D>("fabs", &std::fabs);
  sym<F>("fabsf", &std::fabs);
  sym<LD>("fabsl", &std::fabs);

  sym<D>("fmod", &std::fmod);
  sym<F>("fmodf", &std::fmod);
  sym<LD>("fmodl", &std::fmod);

  sym<D>("remainder", &std::remainder);
  sym<F>("remainderf", &std::remainder);
  sym<LD>("remainderl", &std::remainder);

  sym<D (*)(D, D, I *)>("remquo", &std::remquo);
  sym<F (*)(F, F, I *)>("remquof", &std::remquo);
  sym<LD (*)(LD, LD, I *)>("remquol", &std::remquo);

  sym<D>("fma", &std::fma);
  sym<F>("fmaf", &std::fma);
  sym<LD>("fmal", &std::fma);

  sym<D>("fmax", &std::fmax);
  sym<F>("fmaxf", &std::fmax);
  sym<LD>("fmaxl", &std::fmax);

  sym<D>("fmin", &std::fmin);
  sym<F>("fminf", &std::fmin);
  sym<LD>("fminl", &std::fmin);

  sym<D>("fdim", &std::fdim);
  sym<F>("fdimf", &std::fdim);
  sym<LD>("fdiml", &std::fdim);

  sym<D (*)(const char *)>("nan", &std::nan);
  sym<F (*)(const char *)>("nanf", &std::nanf);
  sym<LD (*)(const char *)>("nanl", &std::nanl);

  // Exponential functions
  sym<D>("exp", &std::exp);
  sym<F>("expf", &std::exp);
  sym<LD>("expl", &std::exp);

  sym<D>("exp2", &std::exp2);
  sym<F>("exp2f", &std::exp2);
  sym<LD>("exp2l", &std::exp2);

  sym<D>("expm1", &std::expm1);
  sym<F>("expm1f", &std::expm1);
  sym<LD>("expm1l", &std::expm1);

  sym<D>("log", &std::log);
  sym<F>("logf", &std::log);
  sym<LD>("logl", &std::log);

  sym<D>("log10", &std::log10);
  sym<F>("log10f", &std::log10);
  sym<LD>("log10l", &std::log10);

  sym<D>("log2", &std::log2);
  sym<F>("log2f", &std::log2);
  sym<LD>("log2l", &std::log2);

  sym<D>("log1p", &std::log1p);
  sym<F>("log1pf", &std::log1p);
  sym<LD>("log1pl", &std::log1p);

  // Power functions
  sym<D>("pow", &std::pow);
  sym<F>("powf", &std::pow);
  sym<LD>("powl", &std::pow);

  sym<D>("sqrt", &std::sqrt);
  sym<F>("sqrtf", &std::sqrt);
  sym<LD>("sqrtl", &std::sqrt);

  sym<D>("cbrt", &std::cbrt);
  sym<F>("cbrtf", &std::cbrt);
  sym<LD>("cbrtl", &std::cbrt);

  sym<D (*)(D, D)>("hypot", &std::hypot);
  sym<F (*)(F, F)>("hypotf", &std::hypot);
  sym<LD (*)(LD, LD)>("hypotl", &std::hypot);

  // Trigonometric functions
  sym<D>("sin", &std::sin);
  sym<F>("sinf", &std::sin);
  sym<LD>("sinl", &std::sin);

  sym<D>("cos", &std::cos);
  sym<F>("cosf", &std::cos);
  sym<LD>("cosl", &std::cos);

  sym<D>("tan", &std::tan);
  sym<F>("tanf", &std::tan);
  sym<LD>("tanl", &std::tan);

  sym<D>("asin", &std::asin);
  sym<F>("asinf", &std::asin);
  sym<LD>("asinl", &std::asin);

  sym<D>("acos", &std::acos);
  sym<F>("acosf", &std::acos);
  sym<LD>("acosl", &std::acos);

  sym<D>("atan", &std::atan);
  sym<F>("atanf", &std::atan);
  sym<LD>("atanl", &std::atan);

  sym<D>("atan2", &std::atan2);
  sym<F>("atan2f", &std::atan2);
  sym<LD>("atan2l", &std::atan2);

  // Trigonometric functions
  sym<D>("sinh", &std::sinh);
  sym<F>("sinhf", &std::sinh);
  sym<LD>("sinhl", &std::sinh);

  sym<D>("cosh", &std::cosh);
  sym<F>("coshf", &std::cosh);
  sym<LD>("coshl", &std::cosh);

  sym<D>("tanh", &std::tanh);
  sym<F>("tanhf", &std::tanh);
  sym<LD>("tanhl", &std::tanh);

  sym<D>("asinh", &std::asinh);
  sym<F>("asinf", &std::asinh);
  sym<LD>("asinhl", &std::asinh);

  sym<D>("acosh", &std::acosh);
  sym<F>("acoshf", &std::acosh);
  sym<LD>("acoshl", &std::acosh);

  sym<D>("atanh", &std::atanh);
  sym<F>("atanhf", &std::atanh);
  sym<LD>("atanhl", &std::atanh);

  // Error and gamma functions
  sym<D>("erf", &std::erf);
  sym<F>("erff", &std::erf);
  sym<LD>("erfl", &std::erf);

  sym<D>("erfc", &std::erfc);
  sym<F>("erfcf", &std::erfc);
  sym<LD>("erfcl", &std::erfc);

  sym<D>("tgamma", &std::tgamma);
  sym<F>("tgammaf", &std::tgamma);
  sym<LD>("tgammal", &std::tgamma);

  sym<D>("lgamma", &std::lgamma);
  sym<F>("lgammaf", &std::lgamma);
  sym<LD>("lgammal", &std::lgamma);

  // Nearest integer floating-point operations
  sym<D>("ceil", &std::ceil);
  sym<F>("ceilf", &std::ceil);
  sym<LD>("ceill", &std::ceil);

  sym<D>("floor", &std::floor);
  sym<F>("floorf", &std::floor);
  sym<LD>("floorl", &std::floor);

  sym<D>("trunc", &std::trunc);
  sym<F>("truncf", &std::trunc);
  sym<LD>("truncl", &std::trunc);

  sym<D>("round", &std::round);
  sym<F>("roundf", &std::round);
  sym<LD>("roundl", &std::round);
  sym<L (*)(D)>("lround", &std::lround);
  sym<L (*)(F)>("lroundf", &std::lround);
  sym<L (*)(LD)>("lroundl", &std::lround);
  sym<LL (*)(D)>("llround", &std::llround);
  sym<LL (*)(F)>("llroundf", &std::llround);
  sym<LL (*)(LD)>("llroundl", &std::llround);

  sym<D>("nearbyint", &std::nearbyint);
  sym<F>("nearbyintf", &std::nearbyint);
  sym<LD>("nearbyintl", &std::nearbyint);

  sym<D>("rint", &std::rint);
  sym<F>("rintf", &std::rint);
  sym<LD>("rintl", &std::rint);
  sym<L (*)(D)>("lrint", &std::lrint);
  sym<L (*)(F)>("lrintf", &std::lrint);
  sym<L (*)(LD)>("lrintl", &std::lrint);
  sym<LL (*)(D)>("llrint", &std::llrint);
  sym<LL (*)(F)>("llrintf", &std::llrint);
  sym<LL (*)(LD)>("llrintl", &std::llrint);

  //  Floating-point manipulation functions
  sym<D (*)(D, int *)>("frexp", &std::frexp);
  sym<F (*)(F, int *)>("frexpf", &std::frexp);
  sym<LD (*)(LD, int *)>("frexpl", &std::frexp);

  sym<D (*)(D, int)>("ldexp", &std::ldexp);
  sym<F (*)(F, int)>("ldexpf", &std::ldexp);
  sym<LD (*)(LD, int)>("ldexpl", &std::ldexp);

  sym<D (*)(D, D *)>("modf", &std::modf);
  sym<F (*)(F, F *)>("modff", &std::modf);
  sym<LD (*)(LD, LD *)>("modfl", &std::modf);

  sym<D (*)(D, int)>("scalbn", &std::scalbn);
  sym<F (*)(F, int)>("scalbnf", &std::scalbn);
  sym<LD (*)(LD, int)>("scalbnl", &std::scalbn);
  sym<D (*)(D, L)>("scalbln", &std::scalbln);
  sym<F (*)(F, L)>("scalblnf", &std::scalbln);
  sym<LD (*)(LD, L)>("scalblnl", &std::scalbln);

  sym<I (*)(D)>("ilogb", &std::ilogb);
  sym<I (*)(F)>("ilogbf", &std::ilogb);
  sym<I (*)(LD)>("ilogbl", &std::ilogb);

  sym<D>("logb", &std::logb);
  sym<F>("logbf", &std::logb);
  sym<LD>("logbl", &std::logb);

  sym<D>("nextafter", &std::nextafter);
  sym<F>("nextafterf", &std::nextafter);
  sym<LD>("nextafterl", &std::nextafter);
  sym<D (*)(D, LD)>("nexttoward", &std::nexttoward);
  sym<F (*)(F, LD)>("nexttowardf", &std::nexttoward);
  sym<LD (*)(LD, LD)>("nexttowardl", &std::nexttoward);

  sym<D>("copysign", &std::copysign);
  sym<F>("copysignf", &std::copysign);
  sym<LD>("copysignl", &std::copysign);
}
