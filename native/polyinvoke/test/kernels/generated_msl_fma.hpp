// clang-format off
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace generated::msl {

const static uint8_t data_fma__[] = {0x6b,0x65,0x72,0x6e,0x65,0x6c,0x20,0x76,0x6f,0x69,0x64,0x20,0x5f,0x66,0x6d,0x61,0x28,0x64,0x65,0x76,0x69,0x63,0x65,0x20,0x66,0x6c,0x6f,0x61,0x74,0x20,0x26,0x61,0x20,0x5b,0x5b,0x20,0x62,0x75,0x66,0x66,0x65,0x72,0x28,0x30,0x29,0x20,0x5d,0x5d,0x2c,0x0a,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x64,0x65,0x76,0x69,0x63,0x65,0x20,0x66,0x6c,0x6f,0x61,0x74,0x20,0x26,0x62,0x20,0x5b,0x5b,0x20,0x62,0x75,0x66,0x66,0x65,0x72,0x28,0x31,0x29,0x20,0x5d,0x5d,0x2c,0x0a,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x64,0x65,0x76,0x69,0x63,0x65,0x20,0x66,0x6c,0x6f,0x61,0x74,0x20,0x26,0x63,0x20,0x5b,0x5b,0x20,0x62,0x75,0x66,0x66,0x65,0x72,0x28,0x32,0x29,0x20,0x5d,0x5d,0x2c,0x0a,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x64,0x65,0x76,0x69,0x63,0x65,0x20,0x66,0x6c,0x6f,0x61,0x74,0x20,0x2a,0x6f,0x75,0x74,0x20,0x5b,0x5b,0x20,0x62,0x75,0x66,0x66,0x65,0x72,0x28,0x33,0x29,0x20,0x5d,0x5d,0x29,0x20,0x7b,0x0a,0x20,0x20,0x20,0x20,0x6f,0x75,0x74,0x5b,0x30,0x5d,0x20,0x3d,0x20,0x61,0x20,0x2a,0x20,0x62,0x20,0x2b,0x20,0x63,0x3b,0x0a,0x7d,0x0a,};

const static std::unordered_map<std::string, std::unordered_map<std::string, std::vector<uint8_t>>> fma = {
    {"Metal",
      {
        {"", std::vector(std::begin(data_fma__), std::end(data_fma__))}
      }
    }
  };
}
