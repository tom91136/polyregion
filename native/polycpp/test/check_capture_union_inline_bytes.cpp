#pragma region case: capture_union_inline_bytes
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: pass

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <numeric>
#include <vector>

namespace {

struct Padding0 {};

struct StandardLayout {};
struct AlternateLayout {};

template <typename Layout> struct basic_string;

template <> struct basic_string<StandardLayout> {
  struct __long {
    struct __attribute__((__packed__)) {
      std::size_t __is_long_ : 1;
      std::size_t __cap_ : sizeof(std::size_t) * CHAR_BIT - 1;
    };
    std::size_t __size_;
    char *__data_;
  };

  enum { __min_cap = (sizeof(__long) - 1) / sizeof(char) > 2 ? (sizeof(__long) - 1) / sizeof(char) : 2 };

  struct __short {
    struct __attribute__((__packed__)) {
      unsigned char __is_long_ : 1;
      unsigned char __size_ : 7;
    };
    [[no_unique_address]] Padding0 __padding_;
    char __data_[__min_cap];
  };

  union __rep {
    __short __s;
    __long __l;
  } __rep_;
};

template <> struct basic_string<AlternateLayout> {
  struct __long {
    char *__data_;
    std::size_t __size_;
    std::size_t __cap_ : sizeof(std::size_t) * CHAR_BIT - 1;
    std::size_t __is_long_ : 1;
  };

  enum { __min_cap = (sizeof(__long) - 1) / sizeof(char) > 2 ? (sizeof(__long) - 1) / sizeof(char) : 2 };

  struct __short {
    char __data_[__min_cap];
    [[no_unique_address]] Padding0 __padding_;
    unsigned char __size_ : 7;
    unsigned char __is_long_ : 1;
  };

  union __rep {
    __short __s;
    __long __l;
  } __rep_;
};

struct MsvcString {
  union _Bxty {
    char _Buf[16];
    char *_Ptr;
    char _Alias[16];
  } _Bx;
  std::size_t _Mysize;
  std::size_t _Myres;
};

long weigh(const char *data, const std::size_t size) {
  long sum = 0;
  for (std::size_t k = 0; k < size; ++k)
    sum += static_cast<unsigned char>(data[k]) * static_cast<long>(k + 1);
  return sum;
}

long checksum(const basic_string<StandardLayout> &s) {
  const auto *raw = reinterpret_cast<const unsigned char *>(&s);
  const bool isLong = raw[0] & 1;
  return weigh(isLong ? s.__rep_.__l.__data_ : s.__rep_.__s.__data_, isLong ? s.__rep_.__l.__size_ : raw[0] >> 1);
}

long checksum(const basic_string<AlternateLayout> &s) {
  const auto *raw = reinterpret_cast<const unsigned char *>(&s);
  const bool isLong = raw[23] >> 7;
  return weigh(isLong ? s.__rep_.__l.__data_ : s.__rep_.__s.__data_, isLong ? s.__rep_.__l.__size_ : raw[23] & 0x7f);
}

long checksum(const MsvcString &s) { return weigh(s._Myres >= 16 ? s._Bx._Ptr : s._Bx._Buf, s._Mysize); }

} // namespace

int main() {
  const int N = 64;
  std::vector<int> target(4096, 7);
  std::vector<char> text(100);
  for (std::size_t i = 0; i < text.size(); ++i)
    text[i] = static_cast<char>('a' + (i % 26));
  const auto address = reinterpret_cast<std::uintptr_t>(target.data());

  basic_string<StandardLayout> defaultShort{}, defaultLong{};
  defaultShort.__rep_.__s.__size_ = 22;
  for (int i = 0; i < 15; ++i)
    defaultShort.__rep_.__s.__data_[i] = static_cast<char>('A' + i);
  std::memcpy(defaultShort.__rep_.__s.__data_ + 15, &address, sizeof(address));
  defaultLong.__rep_.__l.__is_long_ = 1;
  defaultLong.__rep_.__l.__cap_ = text.size();
  defaultLong.__rep_.__l.__size_ = text.size();
  defaultLong.__rep_.__l.__data_ = text.data();

  basic_string<AlternateLayout> alternateShort{}, alternateLong{};
  alternateShort.__rep_.__s.__size_ = 8;
  std::memcpy(alternateShort.__rep_.__s.__data_, &address, sizeof(address));
  alternateLong.__rep_.__l.__is_long_ = 1;
  alternateLong.__rep_.__l.__cap_ = text.size();
  alternateLong.__rep_.__l.__size_ = text.size();
  alternateLong.__rep_.__l.__data_ = text.data();

  MsvcString msvcShort{}, msvcLong{};
  std::memcpy(msvcShort._Bx._Buf, &address, sizeof(address));
  msvcShort._Mysize = 8;
  msvcShort._Myres = 15;
  msvcLong._Bx._Ptr = text.data();
  msvcLong._Mysize = text.size();
  msvcLong._Myres = text.size();

  const basic_string<StandardLayout> defaultShortBefore = defaultShort, defaultLongBefore = defaultLong;
  const basic_string<AlternateLayout> alternateShortBefore = alternateShort, alternateLongBefore = alternateLong;
  const MsvcString msvcShortBefore = msvcShort, msvcLongBefore = msvcLong;

  std::vector<long> out(N, -1);
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  long *o = out.data();
  const int *t = target.data();

  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](int i) {
    o[i] = checksum(defaultShort) + checksum(defaultLong) + checksum(alternateShort) + checksum(alternateLong) + checksum(msvcShort)
           + checksum(msvcLong) + t[i] + i;
  });

  const long want = checksum(defaultShortBefore) + checksum(defaultLongBefore) + checksum(alternateShortBefore)
                    + checksum(alternateLongBefore) + checksum(msvcShortBefore) + checksum(msvcLongBefore) + 7;
  long bad = 0;
  for (int i = 0; i < N; ++i)
    bad += o[i] != want + i;
  const bool intact = std::memcmp(&defaultShort, &defaultShortBefore, sizeof(basic_string<StandardLayout>)) == 0
                      && std::memcmp(&defaultLong, &defaultLongBefore, sizeof(basic_string<StandardLayout>)) == 0
                      && std::memcmp(&alternateShort, &alternateShortBefore, sizeof(basic_string<AlternateLayout>)) == 0
                      && std::memcmp(&alternateLong, &alternateLongBefore, sizeof(basic_string<AlternateLayout>)) == 0
                      && std::memcmp(&msvcShort, &msvcShortBefore, sizeof(MsvcString)) == 0
                      && std::memcmp(&msvcLong, &msvcLongBefore, sizeof(MsvcString)) == 0;
  const bool ok = bad == 0 && intact;
  std::printf(ok ? "pass" : "fail (bad=%ld hostIntact=%d)", bad, intact);
  return ok ? 0 : 1;
}
