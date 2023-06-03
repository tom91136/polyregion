#pragma once

#if _MSC_VER
//  #define EXPORT __declspec(dllexport)
  #define EXPORT
#else
  #define POLYREGION_EXPORT __attribute__((visibility("default")))
#endif
