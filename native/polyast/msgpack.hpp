#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polyregion::polyast::msgpack {

constexpr int32_t MsgpackInternedMagic = 0x4d504349; // "MPCI"

class StringInterner {
  std::unordered_map<std::string, int32_t> ids_;
  std::vector<std::string> entries_;

public:
  int32_t id(const std::string &x) {
    if (auto it = ids_.find(x); it != ids_.end()) return it->second;
    if (entries_.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) throw std::runtime_error("String table too large");
    auto next = static_cast<int32_t>(entries_.size());
    entries_.push_back(x);
    ids_.emplace(entries_.back(), next);
    return next;
  }

  [[nodiscard]] const std::vector<std::string> &entries() const { return entries_; }
};

class MsgpackWriter {
  std::vector<uint8_t> bytes_;
  StringInterner *interner_ = nullptr;
  bool collectOnly_ = false;

  void byte(uint8_t x) {
    if (!collectOnly_) bytes_.push_back(x);
  }

  void raw16(uint16_t x) {
    byte(static_cast<uint8_t>(x >> 8));
    byte(static_cast<uint8_t>(x));
  }

  void raw32(uint32_t x) {
    byte(static_cast<uint8_t>(x >> 24));
    byte(static_cast<uint8_t>(x >> 16));
    byte(static_cast<uint8_t>(x >> 8));
    byte(static_cast<uint8_t>(x));
  }

  void raw64(uint64_t x) {
    byte(static_cast<uint8_t>(x >> 56));
    byte(static_cast<uint8_t>(x >> 48));
    byte(static_cast<uint8_t>(x >> 40));
    byte(static_cast<uint8_t>(x >> 32));
    byte(static_cast<uint8_t>(x >> 24));
    byte(static_cast<uint8_t>(x >> 16));
    byte(static_cast<uint8_t>(x >> 8));
    byte(static_cast<uint8_t>(x));
  }

public:
  explicit MsgpackWriter(size_t initialSize = 256, StringInterner *interner = nullptr, bool collectOnly = false)
      : interner_(interner), collectOnly_(collectOnly) {
    if (!collectOnly_) bytes_.reserve(initialSize);
  }

  void setStringInterner(StringInterner *interner) { interner_ = interner; }
  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }

  void writeNil() { byte(0xc0); }
  void writeBoolean(bool x) { byte(x ? 0xc3 : 0xc2); }

  void writeInt32(int32_t x) {
    if (x >= 0 && x <= 0x7f) byte(static_cast<uint8_t>(x));
    else if (x >= -32 && x < 0) byte(static_cast<uint8_t>(x));
    else if (x >= std::numeric_limits<int8_t>::min() && x <= std::numeric_limits<int8_t>::max()) {
      byte(0xd0);
      byte(static_cast<uint8_t>(x));
    } else if (x >= std::numeric_limits<int16_t>::min() && x <= std::numeric_limits<int16_t>::max()) {
      byte(0xd1);
      raw16(static_cast<uint16_t>(x));
    } else {
      byte(0xd2);
      raw32(static_cast<uint32_t>(x));
    }
  }

  void writeInt64(int64_t x) {
    if (x >= std::numeric_limits<int32_t>::min() && x <= std::numeric_limits<int32_t>::max()) writeInt32(static_cast<int32_t>(x));
    else {
      byte(0xd3);
      raw64(static_cast<uint64_t>(x));
    }
  }

  void writeFloat32(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    byte(0xca);
    raw32(bits);
  }

  void writeFloat64(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    byte(0xcb);
    raw64(bits);
  }

  void writeString(const std::string &x) {
    if (interner_) {
      const auto n = interner_->id(x);
      if (!collectOnly_) writeInt32(n);
    } else writeStringLiteral(x);
  }

  void writeStringLiteral(const std::string &x) {
    const auto n = x.size();
    if (n <= 31) byte(static_cast<uint8_t>(0xa0 | n));
    else if (n <= 0xff) {
      byte(0xd9);
      byte(static_cast<uint8_t>(n));
    } else if (n <= 0xffff) {
      byte(0xda);
      raw16(static_cast<uint16_t>(n));
    } else if (n <= std::numeric_limits<uint32_t>::max()) {
      byte(0xdb);
      raw32(static_cast<uint32_t>(n));
    } else throw std::runtime_error("String too large");
    if (!collectOnly_) bytes_.insert(bytes_.end(), x.begin(), x.end());
  }

  void writeArrayHeader(size_t n) {
    if (n <= 15) byte(static_cast<uint8_t>(0x90 | n));
    else if (n <= 0xffff) {
      byte(0xdc);
      raw16(static_cast<uint16_t>(n));
    } else if (n <= std::numeric_limits<uint32_t>::max()) {
      byte(0xdd);
      raw32(static_cast<uint32_t>(n));
    } else throw std::runtime_error("Array too large");
  }
};

class MsgpackReader {
  const uint8_t *begin_;
  const uint8_t *cursor_;
  const uint8_t *end_;
  const std::vector<std::string> *stringTable_ = nullptr;

  [[noreturn]] void fail(const std::string &message) const { throw std::runtime_error(message + " at byte " + std::to_string(offset())); }

  void require(size_t n) const {
    if (static_cast<size_t>(end_ - cursor_) < n) fail("Unexpected end of input");
  }

  uint8_t u8() {
    require(1);
    return *cursor_++;
  }

  int8_t i8() { return static_cast<int8_t>(u8()); }

  uint16_t u16() {
    require(2);
    uint16_t x = (static_cast<uint16_t>(cursor_[0]) << 8) | static_cast<uint16_t>(cursor_[1]);
    cursor_ += 2;
    return x;
  }

  int16_t i16() { return static_cast<int16_t>(u16()); }

  uint32_t u32() {
    require(4);
    uint32_t x = (static_cast<uint32_t>(cursor_[0]) << 24) | (static_cast<uint32_t>(cursor_[1]) << 16)
                 | (static_cast<uint32_t>(cursor_[2]) << 8) | static_cast<uint32_t>(cursor_[3]);
    cursor_ += 4;
    return x;
  }

  int32_t i32() { return static_cast<int32_t>(u32()); }

  uint64_t u64() {
    require(8);
    uint64_t x = (static_cast<uint64_t>(cursor_[0]) << 56) | (static_cast<uint64_t>(cursor_[1]) << 48)
                 | (static_cast<uint64_t>(cursor_[2]) << 40) | (static_cast<uint64_t>(cursor_[3]) << 32)
                 | (static_cast<uint64_t>(cursor_[4]) << 24) | (static_cast<uint64_t>(cursor_[5]) << 16)
                 | (static_cast<uint64_t>(cursor_[6]) << 8) | static_cast<uint64_t>(cursor_[7]);
    cursor_ += 8;
    return x;
  }

  int64_t i64() { return static_cast<int64_t>(u64()); }

  int64_t readIntegralLong() {
    const auto m = u8();
    if (m <= 0x7f) return static_cast<int64_t>(m);
    if (m >= 0xe0) return static_cast<int64_t>(static_cast<int8_t>(m));
    switch (m) {
      case 0xcc: return static_cast<int64_t>(u8());
      case 0xcd: return static_cast<int64_t>(u16());
      case 0xce: return static_cast<int64_t>(u32());
      case 0xcf: {
        const auto x = u64();
        if (x > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail("uint64 value exceeds int64_t");
        return static_cast<int64_t>(x);
      }
      case 0xd0: return static_cast<int64_t>(i8());
      case 0xd1: return static_cast<int64_t>(i16());
      case 0xd2: return static_cast<int64_t>(i32());
      case 0xd3: return i64();
      default: fail("Expected integer");
    }
  }

public:
  MsgpackReader(const uint8_t *begin, const uint8_t *end) : begin_(begin), cursor_(begin), end_(end) {}

  [[nodiscard]] size_t offset() const { return static_cast<size_t>(cursor_ - begin_); }
  [[nodiscard]] bool isAtEnd() const { return cursor_ == end_; }
  void setStringTable(const std::vector<std::string> *table) { stringTable_ = table; }

  [[nodiscard]] bool nextIsArray() const {
    if (cursor_ >= end_) return false;
    const auto m = *cursor_;
    return (m & 0xf0) == 0x90 || m == 0xdc || m == 0xdd;
  }

  void readNil() {
    if (u8() != 0xc0) fail("Expected nil");
  }

  bool tryReadNil() {
    if (cursor_ < end_ && *cursor_ == 0xc0) {
      ++cursor_;
      return true;
    }
    return false;
  }

  bool readBoolean() {
    switch (u8()) {
      case 0xc2: return false;
      case 0xc3: return true;
      default: fail("Expected boolean");
    }
  }

  int32_t readInt32() {
    const auto x = readIntegralLong();
    if (x < std::numeric_limits<int32_t>::min() || x > std::numeric_limits<int32_t>::max()) fail("Integer out of int32_t range");
    return static_cast<int32_t>(x);
  }

  int64_t readInt64() { return readIntegralLong(); }

  float readFloat32() {
    switch (u8()) {
      case 0xca: {
        const auto bits = u32();
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
      }
      case 0xcb: {
        const auto bits = u64();
        double d;
        std::memcpy(&d, &bits, sizeof(d));
        const auto f = static_cast<float>(d);
        if (std::isnan(d) || static_cast<double>(f) == d) return f;
        fail("Float64 to Float32 conversion with loss of precision");
      }
      default: fail("Expected float32/float64");
    }
  }

  double readFloat64() {
    if (u8() != 0xcb) fail("Expected float64");
    const auto bits = u64();
    double out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }

  std::string readString() {
    if (!stringTable_) return readStringLiteral();
    const auto id = readInt32();
    if (id < 0 || static_cast<size_t>(id) >= stringTable_->size()) fail("Bad string table id");
    return stringTable_->at(static_cast<size_t>(id));
  }

  std::string readStringLiteral() {
    const auto m = u8();
    size_t n;
    if ((m & 0xe0) == 0xa0) n = m & 0x1f;
    else {
      switch (m) {
        case 0xd9: n = u8(); break;
        case 0xda: n = u16(); break;
        case 0xdb: n = u32(); break;
        default: fail("Expected string");
      }
    }
    require(n);
    std::string out(reinterpret_cast<const char *>(cursor_), n);
    cursor_ += n;
    return out;
  }

  size_t readArrayHeader() {
    const auto m = u8();
    if ((m & 0xf0) == 0x90) return m & 0x0f;
    switch (m) {
      case 0xdc: return u16();
      case 0xdd: return u32();
      default: fail("Expected array");
    }
  }
};

inline bool isInternedEnvelope(const uint8_t *begin, const uint8_t *end) {
  return end - begin >= 6 && begin[0] == 0x93 && begin[1] == 0xd2 && begin[2] == 0x4d && begin[3] == 0x50 && begin[4] == 0x43
         && begin[5] == 0x49;
}

template <typename F> std::vector<uint8_t> encodeInterned(F &&writeValue) {
  StringInterner table;
  MsgpackWriter collect(16, &table, true);
  writeValue(collect);

  MsgpackWriter w;
  w.writeArrayHeader(3);
  w.writeInt32(MsgpackInternedMagic);
  w.writeArrayHeader(table.entries().size());
  for (const auto &s : table.entries())
    w.writeStringLiteral(s);
  w.setStringInterner(&table);
  writeValue(w);
  return w.take();
}

template <typename F>
auto decodeMaybeInterned(const uint8_t *begin, const uint8_t *end, F &&readValue) -> decltype(readValue(std::declval<MsgpackReader &>())) {
  MsgpackReader r(begin, end);
  if (isInternedEnvelope(begin, end)) {
    const auto n = r.readArrayHeader();
    if (n != 3) throw std::runtime_error("Expected interned envelope array of size 3");
    const auto magic = r.readInt32();
    if (magic != MsgpackInternedMagic) throw std::runtime_error("Bad interned envelope magic");
    const auto tableSize = r.readArrayHeader();
    std::vector<std::string> table;
    table.reserve(tableSize);
    for (size_t i = 0; i < tableSize; ++i)
      table.emplace_back(r.readStringLiteral());
    r.setStringTable(&table);
    auto out = readValue(r);
    if (!r.isAtEnd()) throw std::runtime_error("Trailing bytes after MessagePack value");
    return out;
  }
  auto out = readValue(r);
  if (!r.isAtEnd()) throw std::runtime_error("Trailing bytes after MessagePack value");
  return out;
}

} // namespace polyregion::polyast::msgpack
