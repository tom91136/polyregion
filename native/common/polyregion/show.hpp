#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace polyregion::compiletime {

inline void showHex(std::FILE *file, const size_t width, const char *data) {
  for (size_t i = 0; i < width; ++i) {
    std::fprintf(file, "%02x ", static_cast<unsigned char>(data[i]));
    if (i % 4 == 0) std::fprintf(file, " ");
  }
}

inline void showPtr(std::FILE *file, const size_t width, const char *data) {
  if (width != sizeof(void *)) {
    std::fprintf(file, "(unsupported pointer width %zu) ", width);
    showHex(file, width, data);
  }
  void *p;
  std::memcpy(&p, data, sizeof(void *));
  std::fprintf(file, "%p", p);
}

inline void showInt(std::FILE *file, const bool isUnsigned, const size_t width, const char *data) {
  const auto printVal = [&](auto v, const char *fmt) {
    std::memcpy(&v, data, sizeof(decltype(v)));
    std::fprintf(file, fmt, v);
  };
  switch (width) {
    case 1:
      if (isUnsigned) printVal(static_cast<uint8_t>(0), "%u");
      else printVal(static_cast<int8_t>(0), "%d");
      break;
    case 2:
      if (isUnsigned) printVal(static_cast<uint16_t>(0), "%u");
      else printVal(static_cast<int16_t>(0), "%d");
      break;
    case 4:
      if (isUnsigned) printVal(static_cast<uint32_t>(0), "%u");
      else printVal(static_cast<int32_t>(0), "%d");
      break;
    case 8:
      if (isUnsigned) printVal(static_cast<uint64_t>(0), "%llu");
      else printVal(static_cast<int64_t>(0), "%lld");
      break;
    default: std::fprintf(file, "(unsupported integral width %zu) ", width); showHex(file, width, data);
  }
}

inline void showFloat(std::FILE *file, const size_t width, const char *data) {
  switch (width) {
    case sizeof(float): {
      float v;
      std::memcpy(&v, data, sizeof(v));
      std::fprintf(file, "%f", v);
      break;
    }
    case sizeof(double): {
      double v;
      std::memcpy(&v, data, sizeof(v));
      std::fprintf(file, "%lf", v);
      break;
    }
    default: std::fprintf(file, "(unsupported floating width %zu) ", width); showHex(file, width, data);
  }
}

} // namespace polyregion::compiletime