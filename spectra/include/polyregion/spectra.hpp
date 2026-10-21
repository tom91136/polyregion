#pragma once

#if defined(SPECTRA_NATIVE_THRUST) && defined(SPECTRA_NATIVE_ONEDPL)
  #error "select exactly one native Spectra backend"
#elif defined(SPECTRA_NATIVE_THRUST)
  #include <polyregion/spectra/implementation/thrust.hpp>
  #define POLYREGION_SPECTRA_IMPLEMENT(function, ...) return ::spectra::implementation::function(__VA_ARGS__)
#elif defined(SPECTRA_NATIVE_ONEDPL)
  #include <polyregion/spectra/implementation/onedpl.hpp>
  #define POLYREGION_SPECTRA_IMPLEMENT(function, ...) return ::spectra::implementation::function(__VA_ARGS__)
#endif

#include <polyregion/spectra_api.hpp>

#if defined(SPECTRA_NATIVE_THRUST) || defined(SPECTRA_NATIVE_ONEDPL)
  #undef POLYREGION_SPECTRA_IMPLEMENT
#endif
