#include "polyast_codec.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

constexpr auto AdtHash = "6500e55c5c935d5862d3a1ccb55f916f";

namespace {

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
    uint32_t x = (static_cast<uint32_t>(cursor_[0]) << 24) | (static_cast<uint32_t>(cursor_[1]) << 16) |
                 (static_cast<uint32_t>(cursor_[2]) << 8) | static_cast<uint32_t>(cursor_[3]);
    cursor_ += 4;
    return x;
  }

  int32_t i32() { return static_cast<int32_t>(u32()); }

  uint64_t u64() {
    require(8);
    uint64_t x = (static_cast<uint64_t>(cursor_[0]) << 56) | (static_cast<uint64_t>(cursor_[1]) << 48) |
                 (static_cast<uint64_t>(cursor_[2]) << 40) | (static_cast<uint64_t>(cursor_[3]) << 32) |
                 (static_cast<uint64_t>(cursor_[4]) << 24) | (static_cast<uint64_t>(cursor_[5]) << 16) |
                 (static_cast<uint64_t>(cursor_[6]) << 8) | static_cast<uint64_t>(cursor_[7]);
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

bool isInternedEnvelope(const uint8_t *begin, const uint8_t *end) {
  return end - begin >= 6 && begin[0] == 0x93 && begin[1] == 0xd2 && begin[2] == 0x4d && begin[3] == 0x50 && begin[4] == 0x43 &&
         begin[5] == 0x49;
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

} // namespace

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace polyregion::polyast {

Sym sym_from_json(const json &j_) {
  auto fqn = j_.at(0).get<std::vector<std::string>>();
  return Sym(fqn);
}

json sym_to_json(const Sym &x_) {
  auto fqn = x_.fqn;
  return json::array({fqn});
}

SourcePosition sourceposition_from_json(const json &j_) {
  auto file = j_.at(0).get<std::string>();
  auto line = j_.at(1).get<int32_t>();
  auto col = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<int32_t>());
  return {file, line, col};
}

json sourceposition_to_json(const SourcePosition &x_) {
  auto file = x_.file;
  auto line = x_.line;
  auto col = x_.col ? json(*x_.col) : json();
  return json::array({file, line, col});
}

Named named_from_json(const json &j_) {
  auto symbol = j_.at(0).get<std::string>();
  auto tpe = Type::any_from_json(j_.at(1));
  return {symbol, tpe};
}

json named_to_json(const Named &x_) {
  auto symbol = x_.symbol;
  auto tpe = Type::any_to_json(x_.tpe);
  return json::array({symbol, tpe});
}

TypeKind::None TypeKind::none_from_json(const json &j_) { return {}; }

json TypeKind::none_to_json(const TypeKind::None &x_) { return json::array({}); }

TypeKind::Ref TypeKind::ref_from_json(const json &j_) { return {}; }

json TypeKind::ref_to_json(const TypeKind::Ref &x_) { return json::array({}); }

TypeKind::Integral TypeKind::integral_from_json(const json &j_) { return {}; }

json TypeKind::integral_to_json(const TypeKind::Integral &x_) { return json::array({}); }

TypeKind::Fractional TypeKind::fractional_from_json(const json &j_) { return {}; }

json TypeKind::fractional_to_json(const TypeKind::Fractional &x_) { return json::array({}); }

TypeKind::Any TypeKind::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return TypeKind::none_from_json(t_);
    case 1: return TypeKind::ref_from_json(t_);
    case 2: return TypeKind::integral_from_json(t_);
    case 3: return TypeKind::fractional_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeKind::any_to_json(const TypeKind::Any &x_) {
  return x_.match_total([](const TypeKind::None &y_) -> json { return {0, TypeKind::none_to_json(y_)}; },
                        [](const TypeKind::Ref &y_) -> json { return {1, TypeKind::ref_to_json(y_)}; },
                        [](const TypeKind::Integral &y_) -> json { return {2, TypeKind::integral_to_json(y_)}; },
                        [](const TypeKind::Fractional &y_) -> json { return {3, TypeKind::fractional_to_json(y_)}; });
}

TypeSpace::Global TypeSpace::global_from_json(const json &j_) { return {}; }

json TypeSpace::global_to_json(const TypeSpace::Global &x_) { return json::array({}); }

TypeSpace::Local TypeSpace::local_from_json(const json &j_) { return {}; }

json TypeSpace::local_to_json(const TypeSpace::Local &x_) { return json::array({}); }

TypeSpace::Private TypeSpace::private_from_json(const json &j_) { return {}; }

json TypeSpace::private_to_json(const TypeSpace::Private &x_) { return json::array({}); }

TypeSpace::Constant TypeSpace::constant_from_json(const json &j_) { return {}; }

json TypeSpace::constant_to_json(const TypeSpace::Constant &x_) { return json::array({}); }

TypeSpace::Any TypeSpace::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return TypeSpace::global_from_json(t_);
    case 1: return TypeSpace::local_from_json(t_);
    case 2: return TypeSpace::private_from_json(t_);
    case 3: return TypeSpace::constant_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeSpace::any_to_json(const TypeSpace::Any &x_) {
  return x_.match_total([](const TypeSpace::Global &y_) -> json { return {0, TypeSpace::global_to_json(y_)}; },
                        [](const TypeSpace::Local &y_) -> json { return {1, TypeSpace::local_to_json(y_)}; },
                        [](const TypeSpace::Private &y_) -> json { return {2, TypeSpace::private_to_json(y_)}; },
                        [](const TypeSpace::Constant &y_) -> json { return {3, TypeSpace::constant_to_json(y_)}; });
}

Type::Float16 Type::float16_from_json(const json &j_) { return {}; }

json Type::float16_to_json(const Type::Float16 &x_) { return json::array({}); }

Type::Float32 Type::float32_from_json(const json &j_) { return {}; }

json Type::float32_to_json(const Type::Float32 &x_) { return json::array({}); }

Type::Float64 Type::float64_from_json(const json &j_) { return {}; }

json Type::float64_to_json(const Type::Float64 &x_) { return json::array({}); }

Type::IntU8 Type::intu8_from_json(const json &j_) { return {}; }

json Type::intu8_to_json(const Type::IntU8 &x_) { return json::array({}); }

Type::IntU16 Type::intu16_from_json(const json &j_) { return {}; }

json Type::intu16_to_json(const Type::IntU16 &x_) { return json::array({}); }

Type::IntU32 Type::intu32_from_json(const json &j_) { return {}; }

json Type::intu32_to_json(const Type::IntU32 &x_) { return json::array({}); }

Type::IntU64 Type::intu64_from_json(const json &j_) { return {}; }

json Type::intu64_to_json(const Type::IntU64 &x_) { return json::array({}); }

Type::IntS8 Type::ints8_from_json(const json &j_) { return {}; }

json Type::ints8_to_json(const Type::IntS8 &x_) { return json::array({}); }

Type::IntS16 Type::ints16_from_json(const json &j_) { return {}; }

json Type::ints16_to_json(const Type::IntS16 &x_) { return json::array({}); }

Type::IntS32 Type::ints32_from_json(const json &j_) { return {}; }

json Type::ints32_to_json(const Type::IntS32 &x_) { return json::array({}); }

Type::IntS64 Type::ints64_from_json(const json &j_) { return {}; }

json Type::ints64_to_json(const Type::IntS64 &x_) { return json::array({}); }

Type::Nothing Type::nothing_from_json(const json &j_) { return {}; }

json Type::nothing_to_json(const Type::Nothing &x_) { return json::array({}); }

Type::Unit0 Type::unit0_from_json(const json &j_) { return {}; }

json Type::unit0_to_json(const Type::Unit0 &x_) { return json::array({}); }

Type::Bool1 Type::bool1_from_json(const json &j_) { return {}; }

json Type::bool1_to_json(const Type::Bool1 &x_) { return json::array({}); }

Type::Struct Type::struct_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Any> args;
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  return {name, args};
}

json Type::struct_to_json(const Type::Struct &x_) {
  auto name = sym_to_json(x_.name);
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  return json::array({name, args});
}

Type::Ptr Type::ptr_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto space = TypeSpace::any_from_json(j_.at(1));
  return {comp, space};
}

json Type::ptr_to_json(const Type::Ptr &x_) {
  auto comp = Type::any_to_json(x_.comp);
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, space});
}

Type::Arr Type::arr_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto length = j_.at(1).get<int32_t>();
  auto space = TypeSpace::any_from_json(j_.at(2));
  return {comp, length, space};
}

json Type::arr_to_json(const Type::Arr &x_) {
  auto comp = Type::any_to_json(x_.comp);
  auto length = x_.length;
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, length, space});
}

Type::Var Type::var_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  return Type::Var(name);
}

json Type::var_to_json(const Type::Var &x_) {
  auto name = x_.name;
  return json::array({name});
}

Type::Exec Type::exec_from_json(const json &j_) {
  auto tpeVars = j_.at(0).get<std::vector<std::string>>();
  std::vector<Type::Any> args;
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(2));
  return {tpeVars, args, rtn};
}

json Type::exec_to_json(const Type::Exec &x_) {
  auto tpeVars = x_.tpeVars;
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({tpeVars, args, rtn});
}

Type::Any Type::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Type::float16_from_json(t_);
    case 1: return Type::float32_from_json(t_);
    case 2: return Type::float64_from_json(t_);
    case 3: return Type::intu8_from_json(t_);
    case 4: return Type::intu16_from_json(t_);
    case 5: return Type::intu32_from_json(t_);
    case 6: return Type::intu64_from_json(t_);
    case 7: return Type::ints8_from_json(t_);
    case 8: return Type::ints16_from_json(t_);
    case 9: return Type::ints32_from_json(t_);
    case 10: return Type::ints64_from_json(t_);
    case 11: return Type::nothing_from_json(t_);
    case 12: return Type::unit0_from_json(t_);
    case 13: return Type::bool1_from_json(t_);
    case 14: return Type::struct_from_json(t_);
    case 15: return Type::ptr_from_json(t_);
    case 16: return Type::arr_from_json(t_);
    case 17: return Type::var_from_json(t_);
    case 18: return Type::exec_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Type::any_to_json(const Type::Any &x_) {
  return x_.match_total([](const Type::Float16 &y_) -> json { return {0, Type::float16_to_json(y_)}; },
                        [](const Type::Float32 &y_) -> json { return {1, Type::float32_to_json(y_)}; },
                        [](const Type::Float64 &y_) -> json { return {2, Type::float64_to_json(y_)}; },
                        [](const Type::IntU8 &y_) -> json { return {3, Type::intu8_to_json(y_)}; },
                        [](const Type::IntU16 &y_) -> json { return {4, Type::intu16_to_json(y_)}; },
                        [](const Type::IntU32 &y_) -> json { return {5, Type::intu32_to_json(y_)}; },
                        [](const Type::IntU64 &y_) -> json { return {6, Type::intu64_to_json(y_)}; },
                        [](const Type::IntS8 &y_) -> json { return {7, Type::ints8_to_json(y_)}; },
                        [](const Type::IntS16 &y_) -> json { return {8, Type::ints16_to_json(y_)}; },
                        [](const Type::IntS32 &y_) -> json { return {9, Type::ints32_to_json(y_)}; },
                        [](const Type::IntS64 &y_) -> json { return {10, Type::ints64_to_json(y_)}; },
                        [](const Type::Nothing &y_) -> json { return {11, Type::nothing_to_json(y_)}; },
                        [](const Type::Unit0 &y_) -> json { return {12, Type::unit0_to_json(y_)}; },
                        [](const Type::Bool1 &y_) -> json { return {13, Type::bool1_to_json(y_)}; },
                        [](const Type::Struct &y_) -> json { return {14, Type::struct_to_json(y_)}; },
                        [](const Type::Ptr &y_) -> json { return {15, Type::ptr_to_json(y_)}; },
                        [](const Type::Arr &y_) -> json { return {16, Type::arr_to_json(y_)}; },
                        [](const Type::Var &y_) -> json { return {17, Type::var_to_json(y_)}; },
                        [](const Type::Exec &y_) -> json { return {18, Type::exec_to_json(y_)}; });
}

PathStep::Field PathStep::field_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  return PathStep::Field(name);
}

json PathStep::field_to_json(const PathStep::Field &x_) {
  auto name = x_.name;
  return json::array({name});
}

PathStep::Deref PathStep::deref_from_json(const json &j_) { return {}; }

json PathStep::deref_to_json(const PathStep::Deref &x_) { return json::array({}); }

PathStep::Index PathStep::index_from_json(const json &j_) {
  auto idx = j_.at(0).get<int32_t>();
  return PathStep::Index(idx);
}

json PathStep::index_to_json(const PathStep::Index &x_) {
  auto idx = x_.idx;
  return json::array({idx});
}

PathStep::IndexDyn PathStep::indexdyn_from_json(const json &j_) {
  auto idx = Term::any_from_json(j_.at(0));
  return PathStep::IndexDyn(idx);
}

json PathStep::indexdyn_to_json(const PathStep::IndexDyn &x_) {
  auto idx = Term::any_to_json(x_.idx);
  return json::array({idx});
}

PathStep::Any PathStep::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return PathStep::field_from_json(t_);
    case 1: return PathStep::deref_from_json(t_);
    case 2: return PathStep::index_from_json(t_);
    case 3: return PathStep::indexdyn_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json PathStep::any_to_json(const PathStep::Any &x_) {
  return x_.match_total([](const PathStep::Field &y_) -> json { return {0, PathStep::field_to_json(y_)}; },
                        [](const PathStep::Deref &y_) -> json { return {1, PathStep::deref_to_json(y_)}; },
                        [](const PathStep::Index &y_) -> json { return {2, PathStep::index_to_json(y_)}; },
                        [](const PathStep::IndexDyn &y_) -> json { return {3, PathStep::indexdyn_to_json(y_)}; });
}

Region::Rooted Region::rooted_from_json(const json &j_) {
  auto root = named_from_json(j_.at(0));
  return Region::Rooted(root);
}

json Region::rooted_to_json(const Region::Rooted &x_) {
  auto root = named_to_json(x_.root);
  return json::array({root});
}

Region::Opaque Region::opaque_from_json(const json &j_) { return {}; }

json Region::opaque_to_json(const Region::Opaque &x_) { return json::array({}); }

Region::Any Region::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Region::rooted_from_json(t_);
    case 1: return Region::opaque_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Region::any_to_json(const Region::Any &x_) {
  return x_.match_total([](const Region::Rooted &y_) -> json { return {0, Region::rooted_to_json(y_)}; },
                        [](const Region::Opaque &y_) -> json { return {1, Region::opaque_to_json(y_)}; });
}

Term::Float16Const Term::float16const_from_json(const json &j_) {
  auto value = j_.at(0).get<float>();
  return Term::Float16Const(value);
}

json Term::float16const_to_json(const Term::Float16Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::Float32Const Term::float32const_from_json(const json &j_) {
  auto value = j_.at(0).get<float>();
  return Term::Float32Const(value);
}

json Term::float32const_to_json(const Term::Float32Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::Float64Const Term::float64const_from_json(const json &j_) {
  auto value = j_.at(0).get<double>();
  return Term::Float64Const(value);
}

json Term::float64const_to_json(const Term::Float64Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntU8Const Term::intu8const_from_json(const json &j_) {
  auto value = j_.at(0).get<int8_t>();
  return Term::IntU8Const(value);
}

json Term::intu8const_to_json(const Term::IntU8Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntU16Const Term::intu16const_from_json(const json &j_) {
  auto value = j_.at(0).get<uint16_t>();
  return Term::IntU16Const(value);
}

json Term::intu16const_to_json(const Term::IntU16Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntU32Const Term::intu32const_from_json(const json &j_) {
  auto value = j_.at(0).get<int32_t>();
  return Term::IntU32Const(value);
}

json Term::intu32const_to_json(const Term::IntU32Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntU64Const Term::intu64const_from_json(const json &j_) {
  auto value = j_.at(0).get<int64_t>();
  return Term::IntU64Const(value);
}

json Term::intu64const_to_json(const Term::IntU64Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntS8Const Term::ints8const_from_json(const json &j_) {
  auto value = j_.at(0).get<int8_t>();
  return Term::IntS8Const(value);
}

json Term::ints8const_to_json(const Term::IntS8Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntS16Const Term::ints16const_from_json(const json &j_) {
  auto value = j_.at(0).get<int16_t>();
  return Term::IntS16Const(value);
}

json Term::ints16const_to_json(const Term::IntS16Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntS32Const Term::ints32const_from_json(const json &j_) {
  auto value = j_.at(0).get<int32_t>();
  return Term::IntS32Const(value);
}

json Term::ints32const_to_json(const Term::IntS32Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::IntS64Const Term::ints64const_from_json(const json &j_) {
  auto value = j_.at(0).get<int64_t>();
  return Term::IntS64Const(value);
}

json Term::ints64const_to_json(const Term::IntS64Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::Unit0Const Term::unit0const_from_json(const json &j_) { return {}; }

json Term::unit0const_to_json(const Term::Unit0Const &x_) { return json::array({}); }

Term::Bool1Const Term::bool1const_from_json(const json &j_) {
  auto value = j_.at(0).get<bool>();
  return Term::Bool1Const(value);
}

json Term::bool1const_to_json(const Term::Bool1Const &x_) {
  auto value = x_.value;
  return json::array({value});
}

Term::NullPtrConst Term::nullptrconst_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto space = TypeSpace::any_from_json(j_.at(1));
  auto region = Region::any_from_json(j_.at(2));
  return {comp, space, region};
}

json Term::nullptrconst_to_json(const Term::NullPtrConst &x_) {
  auto comp = Type::any_to_json(x_.comp);
  auto space = TypeSpace::any_to_json(x_.space);
  auto region = Region::any_to_json(x_.region);
  return json::array({comp, space, region});
}

Term::Poison Term::poison_from_json(const json &j_) {
  auto t = Type::any_from_json(j_.at(0));
  return Term::Poison(t);
}

json Term::poison_to_json(const Term::Poison &x_) {
  auto t = Type::any_to_json(x_.t);
  return json::array({t});
}

Term::Select Term::select_from_json(const json &j_) {
  auto root = named_from_json(j_.at(0));
  std::vector<PathStep::Any> steps;
  for (const auto &v_ : j_.at(1)) {
    steps.emplace_back(PathStep::any_from_json(v_));
  }
  auto tpe = Type::any_from_json(j_.at(2));
  return {root, steps, tpe};
}

json Term::select_to_json(const Term::Select &x_) {
  auto root = named_to_json(x_.root);
  std::vector<json> steps;
  for (const auto &v_ : x_.steps) {
    steps.emplace_back(PathStep::any_to_json(v_));
  }
  auto tpe = Type::any_to_json(x_.tpe);
  return json::array({root, steps, tpe});
}

Term::Any Term::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Term::float16const_from_json(t_);
    case 1: return Term::float32const_from_json(t_);
    case 2: return Term::float64const_from_json(t_);
    case 3: return Term::intu8const_from_json(t_);
    case 4: return Term::intu16const_from_json(t_);
    case 5: return Term::intu32const_from_json(t_);
    case 6: return Term::intu64const_from_json(t_);
    case 7: return Term::ints8const_from_json(t_);
    case 8: return Term::ints16const_from_json(t_);
    case 9: return Term::ints32const_from_json(t_);
    case 10: return Term::ints64const_from_json(t_);
    case 11: return Term::unit0const_from_json(t_);
    case 12: return Term::bool1const_from_json(t_);
    case 13: return Term::nullptrconst_from_json(t_);
    case 14: return Term::poison_from_json(t_);
    case 15: return Term::select_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Term::any_to_json(const Term::Any &x_) {
  return x_.match_total([](const Term::Float16Const &y_) -> json { return {0, Term::float16const_to_json(y_)}; },
                        [](const Term::Float32Const &y_) -> json { return {1, Term::float32const_to_json(y_)}; },
                        [](const Term::Float64Const &y_) -> json { return {2, Term::float64const_to_json(y_)}; },
                        [](const Term::IntU8Const &y_) -> json { return {3, Term::intu8const_to_json(y_)}; },
                        [](const Term::IntU16Const &y_) -> json { return {4, Term::intu16const_to_json(y_)}; },
                        [](const Term::IntU32Const &y_) -> json { return {5, Term::intu32const_to_json(y_)}; },
                        [](const Term::IntU64Const &y_) -> json { return {6, Term::intu64const_to_json(y_)}; },
                        [](const Term::IntS8Const &y_) -> json { return {7, Term::ints8const_to_json(y_)}; },
                        [](const Term::IntS16Const &y_) -> json { return {8, Term::ints16const_to_json(y_)}; },
                        [](const Term::IntS32Const &y_) -> json { return {9, Term::ints32const_to_json(y_)}; },
                        [](const Term::IntS64Const &y_) -> json { return {10, Term::ints64const_to_json(y_)}; },
                        [](const Term::Unit0Const &y_) -> json { return {11, Term::unit0const_to_json(y_)}; },
                        [](const Term::Bool1Const &y_) -> json { return {12, Term::bool1const_to_json(y_)}; },
                        [](const Term::NullPtrConst &y_) -> json { return {13, Term::nullptrconst_to_json(y_)}; },
                        [](const Term::Poison &y_) -> json { return {14, Term::poison_to_json(y_)}; },
                        [](const Term::Select &y_) -> json { return {15, Term::select_to_json(y_)}; });
}

Expr::Alias Expr::alias_from_json(const json &j_) {
  auto ref = Term::any_from_json(j_.at(0));
  return Expr::Alias(ref);
}

json Expr::alias_to_json(const Expr::Alias &x_) {
  auto ref = Term::any_to_json(x_.ref);
  return json::array({ref});
}

Expr::SpecOp Expr::specop_from_json(const json &j_) {
  auto op = Spec::any_from_json(j_.at(0));
  return Expr::SpecOp(op);
}

json Expr::specop_to_json(const Expr::SpecOp &x_) {
  auto op = Spec::any_to_json(x_.op);
  return json::array({op});
}

Expr::MathOp Expr::mathop_from_json(const json &j_) {
  auto op = Math::any_from_json(j_.at(0));
  return Expr::MathOp(op);
}

json Expr::mathop_to_json(const Expr::MathOp &x_) {
  auto op = Math::any_to_json(x_.op);
  return json::array({op});
}

Expr::IntrOp Expr::introp_from_json(const json &j_) {
  auto op = Intr::any_from_json(j_.at(0));
  return Expr::IntrOp(op);
}

json Expr::introp_to_json(const Expr::IntrOp &x_) {
  auto op = Intr::any_to_json(x_.op);
  return json::array({op});
}

Expr::Cast Expr::cast_from_json(const json &j_) {
  auto from = Term::any_from_json(j_.at(0));
  auto as = Type::any_from_json(j_.at(1));
  return {from, as};
}

json Expr::cast_to_json(const Expr::Cast &x_) {
  auto from = Term::any_to_json(x_.from);
  auto as = Type::any_to_json(x_.as);
  return json::array({from, as});
}

Expr::Index Expr::index_from_json(const json &j_) {
  auto lhs = Term::any_from_json(j_.at(0));
  auto idx = Term::any_from_json(j_.at(1));
  auto comp = Type::any_from_json(j_.at(2));
  return {lhs, idx, comp};
}

json Expr::index_to_json(const Expr::Index &x_) {
  auto lhs = Term::any_to_json(x_.lhs);
  auto idx = Term::any_to_json(x_.idx);
  auto comp = Type::any_to_json(x_.comp);
  return json::array({lhs, idx, comp});
}

Expr::RefTo Expr::refto_from_json(const json &j_) {
  auto lhs = Term::any_from_json(j_.at(0));
  auto idx = j_.at(1).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j_.at(1)));
  auto comp = Type::any_from_json(j_.at(2));
  auto space = TypeSpace::any_from_json(j_.at(3));
  auto region = Region::any_from_json(j_.at(4));
  return {lhs, idx, comp, space, region};
}

json Expr::refto_to_json(const Expr::RefTo &x_) {
  auto lhs = Term::any_to_json(x_.lhs);
  auto idx = x_.idx ? Term::any_to_json(*x_.idx) : json();
  auto comp = Type::any_to_json(x_.comp);
  auto space = TypeSpace::any_to_json(x_.space);
  auto region = Region::any_to_json(x_.region);
  return json::array({lhs, idx, comp, space, region});
}

Expr::Alloc Expr::alloc_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto size = Term::any_from_json(j_.at(1));
  auto space = TypeSpace::any_from_json(j_.at(2));
  auto region = Region::any_from_json(j_.at(3));
  return {comp, size, space, region};
}

json Expr::alloc_to_json(const Expr::Alloc &x_) {
  auto comp = Type::any_to_json(x_.comp);
  auto size = Term::any_to_json(x_.size);
  auto space = TypeSpace::any_to_json(x_.space);
  auto region = Region::any_to_json(x_.region);
  return json::array({comp, size, space, region});
}

Expr::Invoke Expr::invoke_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Any> tpeArgs;
  for (const auto &v_ : j_.at(1)) {
    tpeArgs.emplace_back(Type::any_from_json(v_));
  }
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j_.at(2)));
  std::vector<Term::Any> args;
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(Term::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(4));
  return {name, tpeArgs, receiver, args, rtn};
}

json Expr::invoke_to_json(const Expr::Invoke &x_) {
  auto name = sym_to_json(x_.name);
  std::vector<json> tpeArgs;
  for (const auto &v_ : x_.tpeArgs) {
    tpeArgs.emplace_back(Type::any_to_json(v_));
  }
  auto receiver = x_.receiver ? Term::any_to_json(*x_.receiver) : json();
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Term::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, tpeArgs, receiver, args, rtn});
}

Expr::ForeignCall Expr::foreigncall_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  std::vector<Term::Any> args;
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Term::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(2));
  return {name, args, rtn};
}

json Expr::foreigncall_to_json(const Expr::ForeignCall &x_) {
  auto name = x_.name;
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Term::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, args, rtn});
}

Expr::OffsetOf Expr::offsetof_from_json(const json &j_) {
  auto structTpe = Type::any_from_json(j_.at(0));
  auto field = j_.at(1).get<std::string>();
  return {structTpe, field};
}

json Expr::offsetof_to_json(const Expr::OffsetOf &x_) {
  auto structTpe = Type::any_to_json(x_.structTpe);
  auto field = x_.field;
  return json::array({structTpe, field});
}

Expr::SizeOf Expr::sizeof_from_json(const json &j_) {
  auto forTpe = Type::any_from_json(j_.at(0));
  return Expr::SizeOf(forTpe);
}

json Expr::sizeof_to_json(const Expr::SizeOf &x_) {
  auto forTpe = Type::any_to_json(x_.forTpe);
  return json::array({forTpe});
}

Expr::Any Expr::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Expr::alias_from_json(t_);
    case 1: return Expr::specop_from_json(t_);
    case 2: return Expr::mathop_from_json(t_);
    case 3: return Expr::introp_from_json(t_);
    case 4: return Expr::cast_from_json(t_);
    case 5: return Expr::index_from_json(t_);
    case 6: return Expr::refto_from_json(t_);
    case 7: return Expr::alloc_from_json(t_);
    case 8: return Expr::invoke_from_json(t_);
    case 9: return Expr::foreigncall_from_json(t_);
    case 10: return Expr::offsetof_from_json(t_);
    case 11: return Expr::sizeof_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Expr::any_to_json(const Expr::Any &x_) {
  return x_.match_total([](const Expr::Alias &y_) -> json { return {0, Expr::alias_to_json(y_)}; },
                        [](const Expr::SpecOp &y_) -> json { return {1, Expr::specop_to_json(y_)}; },
                        [](const Expr::MathOp &y_) -> json { return {2, Expr::mathop_to_json(y_)}; },
                        [](const Expr::IntrOp &y_) -> json { return {3, Expr::introp_to_json(y_)}; },
                        [](const Expr::Cast &y_) -> json { return {4, Expr::cast_to_json(y_)}; },
                        [](const Expr::Index &y_) -> json { return {5, Expr::index_to_json(y_)}; },
                        [](const Expr::RefTo &y_) -> json { return {6, Expr::refto_to_json(y_)}; },
                        [](const Expr::Alloc &y_) -> json { return {7, Expr::alloc_to_json(y_)}; },
                        [](const Expr::Invoke &y_) -> json { return {8, Expr::invoke_to_json(y_)}; },
                        [](const Expr::ForeignCall &y_) -> json { return {9, Expr::foreigncall_to_json(y_)}; },
                        [](const Expr::OffsetOf &y_) -> json { return {10, Expr::offsetof_to_json(y_)}; },
                        [](const Expr::SizeOf &y_) -> json { return {11, Expr::sizeof_to_json(y_)}; });
}

Overload overload_from_json(const json &j_) {
  std::vector<Type::Any> args;
  for (const auto &v_ : j_.at(0)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(1));
  return {args, rtn};
}

json overload_to_json(const Overload &x_) {
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({args, rtn});
}

Spec::Assert Spec::assert_from_json(const json &j_) { return {}; }

json Spec::assert_to_json(const Spec::Assert &x_) { return json::array({}); }

Spec::GpuBarrierGlobal Spec::gpubarrierglobal_from_json(const json &j_) { return {}; }

json Spec::gpubarrierglobal_to_json(const Spec::GpuBarrierGlobal &x_) { return json::array({}); }

Spec::GpuBarrierLocal Spec::gpubarrierlocal_from_json(const json &j_) { return {}; }

json Spec::gpubarrierlocal_to_json(const Spec::GpuBarrierLocal &x_) { return json::array({}); }

Spec::GpuBarrierAll Spec::gpubarrierall_from_json(const json &j_) { return {}; }

json Spec::gpubarrierall_to_json(const Spec::GpuBarrierAll &x_) { return json::array({}); }

Spec::GpuFenceGlobal Spec::gpufenceglobal_from_json(const json &j_) { return {}; }

json Spec::gpufenceglobal_to_json(const Spec::GpuFenceGlobal &x_) { return json::array({}); }

Spec::GpuFenceLocal Spec::gpufencelocal_from_json(const json &j_) { return {}; }

json Spec::gpufencelocal_to_json(const Spec::GpuFenceLocal &x_) { return json::array({}); }

Spec::GpuFenceAll Spec::gpufenceall_from_json(const json &j_) { return {}; }

json Spec::gpufenceall_to_json(const Spec::GpuFenceAll &x_) { return json::array({}); }

Spec::GpuGlobalIdx Spec::gpuglobalidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGlobalIdx(dim);
}

json Spec::gpuglobalidx_to_json(const Spec::GpuGlobalIdx &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGlobalSize Spec::gpuglobalsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGlobalSize(dim);
}

json Spec::gpuglobalsize_to_json(const Spec::GpuGlobalSize &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupIdx Spec::gpugroupidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGroupIdx(dim);
}

json Spec::gpugroupidx_to_json(const Spec::GpuGroupIdx &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupSize Spec::gpugroupsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGroupSize(dim);
}

json Spec::gpugroupsize_to_json(const Spec::GpuGroupSize &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalIdx Spec::gpulocalidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuLocalIdx(dim);
}

json Spec::gpulocalidx_to_json(const Spec::GpuLocalIdx &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalSize Spec::gpulocalsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuLocalSize(dim);
}

json Spec::gpulocalsize_to_json(const Spec::GpuLocalSize &x_) {
  auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::Any Spec::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Spec::assert_from_json(t_);
    case 1: return Spec::gpubarrierglobal_from_json(t_);
    case 2: return Spec::gpubarrierlocal_from_json(t_);
    case 3: return Spec::gpubarrierall_from_json(t_);
    case 4: return Spec::gpufenceglobal_from_json(t_);
    case 5: return Spec::gpufencelocal_from_json(t_);
    case 6: return Spec::gpufenceall_from_json(t_);
    case 7: return Spec::gpuglobalidx_from_json(t_);
    case 8: return Spec::gpuglobalsize_from_json(t_);
    case 9: return Spec::gpugroupidx_from_json(t_);
    case 10: return Spec::gpugroupsize_from_json(t_);
    case 11: return Spec::gpulocalidx_from_json(t_);
    case 12: return Spec::gpulocalsize_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Spec::any_to_json(const Spec::Any &x_) {
  return x_.match_total([](const Spec::Assert &y_) -> json { return {0, Spec::assert_to_json(y_)}; },
                        [](const Spec::GpuBarrierGlobal &y_) -> json { return {1, Spec::gpubarrierglobal_to_json(y_)}; },
                        [](const Spec::GpuBarrierLocal &y_) -> json { return {2, Spec::gpubarrierlocal_to_json(y_)}; },
                        [](const Spec::GpuBarrierAll &y_) -> json { return {3, Spec::gpubarrierall_to_json(y_)}; },
                        [](const Spec::GpuFenceGlobal &y_) -> json { return {4, Spec::gpufenceglobal_to_json(y_)}; },
                        [](const Spec::GpuFenceLocal &y_) -> json { return {5, Spec::gpufencelocal_to_json(y_)}; },
                        [](const Spec::GpuFenceAll &y_) -> json { return {6, Spec::gpufenceall_to_json(y_)}; },
                        [](const Spec::GpuGlobalIdx &y_) -> json { return {7, Spec::gpuglobalidx_to_json(y_)}; },
                        [](const Spec::GpuGlobalSize &y_) -> json { return {8, Spec::gpuglobalsize_to_json(y_)}; },
                        [](const Spec::GpuGroupIdx &y_) -> json { return {9, Spec::gpugroupidx_to_json(y_)}; },
                        [](const Spec::GpuGroupSize &y_) -> json { return {10, Spec::gpugroupsize_to_json(y_)}; },
                        [](const Spec::GpuLocalIdx &y_) -> json { return {11, Spec::gpulocalidx_to_json(y_)}; },
                        [](const Spec::GpuLocalSize &y_) -> json { return {12, Spec::gpulocalsize_to_json(y_)}; });
}

Intr::BNot Intr::bnot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::bnot_to_json(const Intr::BNot &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::LogicNot Intr::logicnot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  return Intr::LogicNot(x);
}

json Intr::logicnot_to_json(const Intr::LogicNot &x_) {
  auto x = Term::any_to_json(x_.x);
  return json::array({x});
}

Intr::Pos Intr::pos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::pos_to_json(const Intr::Pos &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Neg Intr::neg_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::neg_to_json(const Intr::Neg &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Add Intr::add_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::add_to_json(const Intr::Add &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Sub Intr::sub_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::sub_to_json(const Intr::Sub &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Mul Intr::mul_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::mul_to_json(const Intr::Mul &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Div Intr::div_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::div_to_json(const Intr::Div &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Rem Intr::rem_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::rem_to_json(const Intr::Rem &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Min Intr::min_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::min_to_json(const Intr::Min &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Max Intr::max_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::max_to_json(const Intr::Max &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BAnd Intr::band_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::band_to_json(const Intr::BAnd &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BOr Intr::bor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bor_to_json(const Intr::BOr &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BXor Intr::bxor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bxor_to_json(const Intr::BXor &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSL Intr::bsl_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bsl_to_json(const Intr::BSL &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSR Intr::bsr_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bsr_to_json(const Intr::BSR &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BZSR Intr::bzsr_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bzsr_to_json(const Intr::BZSR &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::LogicAnd Intr::logicand_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicand_to_json(const Intr::LogicAnd &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicOr Intr::logicor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicor_to_json(const Intr::LogicOr &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicEq Intr::logiceq_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiceq_to_json(const Intr::LogicEq &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicNeq Intr::logicneq_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicneq_to_json(const Intr::LogicNeq &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLte Intr::logiclte_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiclte_to_json(const Intr::LogicLte &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGte Intr::logicgte_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicgte_to_json(const Intr::LogicGte &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLt Intr::logiclt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiclt_to_json(const Intr::LogicLt &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGt Intr::logicgt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicgt_to_json(const Intr::LogicGt &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::Any Intr::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Intr::bnot_from_json(t_);
    case 1: return Intr::logicnot_from_json(t_);
    case 2: return Intr::pos_from_json(t_);
    case 3: return Intr::neg_from_json(t_);
    case 4: return Intr::add_from_json(t_);
    case 5: return Intr::sub_from_json(t_);
    case 6: return Intr::mul_from_json(t_);
    case 7: return Intr::div_from_json(t_);
    case 8: return Intr::rem_from_json(t_);
    case 9: return Intr::min_from_json(t_);
    case 10: return Intr::max_from_json(t_);
    case 11: return Intr::band_from_json(t_);
    case 12: return Intr::bor_from_json(t_);
    case 13: return Intr::bxor_from_json(t_);
    case 14: return Intr::bsl_from_json(t_);
    case 15: return Intr::bsr_from_json(t_);
    case 16: return Intr::bzsr_from_json(t_);
    case 17: return Intr::logicand_from_json(t_);
    case 18: return Intr::logicor_from_json(t_);
    case 19: return Intr::logiceq_from_json(t_);
    case 20: return Intr::logicneq_from_json(t_);
    case 21: return Intr::logiclte_from_json(t_);
    case 22: return Intr::logicgte_from_json(t_);
    case 23: return Intr::logiclt_from_json(t_);
    case 24: return Intr::logicgt_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Intr::any_to_json(const Intr::Any &x_) {
  return x_.match_total([](const Intr::BNot &y_) -> json { return {0, Intr::bnot_to_json(y_)}; },
                        [](const Intr::LogicNot &y_) -> json { return {1, Intr::logicnot_to_json(y_)}; },
                        [](const Intr::Pos &y_) -> json { return {2, Intr::pos_to_json(y_)}; },
                        [](const Intr::Neg &y_) -> json { return {3, Intr::neg_to_json(y_)}; },
                        [](const Intr::Add &y_) -> json { return {4, Intr::add_to_json(y_)}; },
                        [](const Intr::Sub &y_) -> json { return {5, Intr::sub_to_json(y_)}; },
                        [](const Intr::Mul &y_) -> json { return {6, Intr::mul_to_json(y_)}; },
                        [](const Intr::Div &y_) -> json { return {7, Intr::div_to_json(y_)}; },
                        [](const Intr::Rem &y_) -> json { return {8, Intr::rem_to_json(y_)}; },
                        [](const Intr::Min &y_) -> json { return {9, Intr::min_to_json(y_)}; },
                        [](const Intr::Max &y_) -> json { return {10, Intr::max_to_json(y_)}; },
                        [](const Intr::BAnd &y_) -> json { return {11, Intr::band_to_json(y_)}; },
                        [](const Intr::BOr &y_) -> json { return {12, Intr::bor_to_json(y_)}; },
                        [](const Intr::BXor &y_) -> json { return {13, Intr::bxor_to_json(y_)}; },
                        [](const Intr::BSL &y_) -> json { return {14, Intr::bsl_to_json(y_)}; },
                        [](const Intr::BSR &y_) -> json { return {15, Intr::bsr_to_json(y_)}; },
                        [](const Intr::BZSR &y_) -> json { return {16, Intr::bzsr_to_json(y_)}; },
                        [](const Intr::LogicAnd &y_) -> json { return {17, Intr::logicand_to_json(y_)}; },
                        [](const Intr::LogicOr &y_) -> json { return {18, Intr::logicor_to_json(y_)}; },
                        [](const Intr::LogicEq &y_) -> json { return {19, Intr::logiceq_to_json(y_)}; },
                        [](const Intr::LogicNeq &y_) -> json { return {20, Intr::logicneq_to_json(y_)}; },
                        [](const Intr::LogicLte &y_) -> json { return {21, Intr::logiclte_to_json(y_)}; },
                        [](const Intr::LogicGte &y_) -> json { return {22, Intr::logicgte_to_json(y_)}; },
                        [](const Intr::LogicLt &y_) -> json { return {23, Intr::logiclt_to_json(y_)}; },
                        [](const Intr::LogicGt &y_) -> json { return {24, Intr::logicgt_to_json(y_)}; });
}

Math::Abs Math::abs_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::abs_to_json(const Math::Abs &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sin Math::sin_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sin_to_json(const Math::Sin &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cos Math::cos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cos_to_json(const Math::Cos &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tan Math::tan_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::tan_to_json(const Math::Tan &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Asin Math::asin_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::asin_to_json(const Math::Asin &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Acos Math::acos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::acos_to_json(const Math::Acos &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Atan Math::atan_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::atan_to_json(const Math::Atan &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sinh Math::sinh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sinh_to_json(const Math::Sinh &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cosh Math::cosh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cosh_to_json(const Math::Cosh &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tanh Math::tanh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::tanh_to_json(const Math::Tanh &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Signum Math::signum_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::signum_to_json(const Math::Signum &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Round Math::round_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::round_to_json(const Math::Round &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Ceil Math::ceil_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::ceil_to_json(const Math::Ceil &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Floor Math::floor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::floor_to_json(const Math::Floor &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Rint Math::rint_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::rint_to_json(const Math::Rint &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sqrt Math::sqrt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sqrt_to_json(const Math::Sqrt &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cbrt Math::cbrt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cbrt_to_json(const Math::Cbrt &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Exp Math::exp_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::exp_to_json(const Math::Exp &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Expm1 Math::expm1_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::expm1_to_json(const Math::Expm1 &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log Math::log_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log_to_json(const Math::Log &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log1p Math::log1p_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log1p_to_json(const Math::Log1p &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log10 Math::log10_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log10_to_json(const Math::Log10 &x_) {
  auto x = Term::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Pow Math::pow_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::pow_to_json(const Math::Pow &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Atan2 Math::atan2_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::atan2_to_json(const Math::Atan2 &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Hypot Math::hypot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::hypot_to_json(const Math::Hypot &x_) {
  auto x = Term::any_to_json(x_.x);
  auto y = Term::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Any Math::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Math::abs_from_json(t_);
    case 1: return Math::sin_from_json(t_);
    case 2: return Math::cos_from_json(t_);
    case 3: return Math::tan_from_json(t_);
    case 4: return Math::asin_from_json(t_);
    case 5: return Math::acos_from_json(t_);
    case 6: return Math::atan_from_json(t_);
    case 7: return Math::sinh_from_json(t_);
    case 8: return Math::cosh_from_json(t_);
    case 9: return Math::tanh_from_json(t_);
    case 10: return Math::signum_from_json(t_);
    case 11: return Math::round_from_json(t_);
    case 12: return Math::ceil_from_json(t_);
    case 13: return Math::floor_from_json(t_);
    case 14: return Math::rint_from_json(t_);
    case 15: return Math::sqrt_from_json(t_);
    case 16: return Math::cbrt_from_json(t_);
    case 17: return Math::exp_from_json(t_);
    case 18: return Math::expm1_from_json(t_);
    case 19: return Math::log_from_json(t_);
    case 20: return Math::log1p_from_json(t_);
    case 21: return Math::log10_from_json(t_);
    case 22: return Math::pow_from_json(t_);
    case 23: return Math::atan2_from_json(t_);
    case 24: return Math::hypot_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Math::any_to_json(const Math::Any &x_) {
  return x_.match_total([](const Math::Abs &y_) -> json { return {0, Math::abs_to_json(y_)}; },
                        [](const Math::Sin &y_) -> json { return {1, Math::sin_to_json(y_)}; },
                        [](const Math::Cos &y_) -> json { return {2, Math::cos_to_json(y_)}; },
                        [](const Math::Tan &y_) -> json { return {3, Math::tan_to_json(y_)}; },
                        [](const Math::Asin &y_) -> json { return {4, Math::asin_to_json(y_)}; },
                        [](const Math::Acos &y_) -> json { return {5, Math::acos_to_json(y_)}; },
                        [](const Math::Atan &y_) -> json { return {6, Math::atan_to_json(y_)}; },
                        [](const Math::Sinh &y_) -> json { return {7, Math::sinh_to_json(y_)}; },
                        [](const Math::Cosh &y_) -> json { return {8, Math::cosh_to_json(y_)}; },
                        [](const Math::Tanh &y_) -> json { return {9, Math::tanh_to_json(y_)}; },
                        [](const Math::Signum &y_) -> json { return {10, Math::signum_to_json(y_)}; },
                        [](const Math::Round &y_) -> json { return {11, Math::round_to_json(y_)}; },
                        [](const Math::Ceil &y_) -> json { return {12, Math::ceil_to_json(y_)}; },
                        [](const Math::Floor &y_) -> json { return {13, Math::floor_to_json(y_)}; },
                        [](const Math::Rint &y_) -> json { return {14, Math::rint_to_json(y_)}; },
                        [](const Math::Sqrt &y_) -> json { return {15, Math::sqrt_to_json(y_)}; },
                        [](const Math::Cbrt &y_) -> json { return {16, Math::cbrt_to_json(y_)}; },
                        [](const Math::Exp &y_) -> json { return {17, Math::exp_to_json(y_)}; },
                        [](const Math::Expm1 &y_) -> json { return {18, Math::expm1_to_json(y_)}; },
                        [](const Math::Log &y_) -> json { return {19, Math::log_to_json(y_)}; },
                        [](const Math::Log1p &y_) -> json { return {20, Math::log1p_to_json(y_)}; },
                        [](const Math::Log10 &y_) -> json { return {21, Math::log10_to_json(y_)}; },
                        [](const Math::Pow &y_) -> json { return {22, Math::pow_to_json(y_)}; },
                        [](const Math::Atan2 &y_) -> json { return {23, Math::atan2_to_json(y_)}; },
                        [](const Math::Hypot &y_) -> json { return {24, Math::hypot_to_json(y_)}; });
}

Stmt::Var Stmt::var_from_json(const json &j_) {
  auto name = named_from_json(j_.at(0));
  auto expr = j_.at(1).is_null() ? std::nullopt : std::make_optional(Expr::any_from_json(j_.at(1)));
  auto isMutable = j_.at(2).get<bool>();
  return {name, expr, isMutable};
}

json Stmt::var_to_json(const Stmt::Var &x_) {
  auto name = named_to_json(x_.name);
  auto expr = x_.expr ? Expr::any_to_json(*x_.expr) : json();
  auto isMutable = x_.isMutable;
  return json::array({name, expr, isMutable});
}

Stmt::Mut Stmt::mut_from_json(const json &j_) {
  auto name = Term::select_from_json(j_.at(0));
  auto expr = Expr::any_from_json(j_.at(1));
  return {name, expr};
}

json Stmt::mut_to_json(const Stmt::Mut &x_) {
  auto name = Term::select_to_json(x_.name);
  auto expr = Expr::any_to_json(x_.expr);
  return json::array({name, expr});
}

Stmt::Update Stmt::update_from_json(const json &j_) {
  auto lhs = Term::select_from_json(j_.at(0));
  auto idx = Term::any_from_json(j_.at(1));
  auto value = Term::any_from_json(j_.at(2));
  return {lhs, idx, value};
}

json Stmt::update_to_json(const Stmt::Update &x_) {
  auto lhs = Term::select_to_json(x_.lhs);
  auto idx = Term::any_to_json(x_.idx);
  auto value = Term::any_to_json(x_.value);
  return json::array({lhs, idx, value});
}

Stmt::While Stmt::while_from_json(const json &j_) {
  auto cond = Term::any_from_json(j_.at(0));
  std::vector<Stmt::Any> body;
  for (const auto &v_ : j_.at(1)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  return {cond, body};
}

json Stmt::while_to_json(const Stmt::While &x_) {
  auto cond = Term::any_to_json(x_.cond);
  std::vector<json> body;
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({cond, body});
}

Stmt::ForRange Stmt::forrange_from_json(const json &j_) {
  auto induction = named_from_json(j_.at(0));
  auto lbIncl = Term::any_from_json(j_.at(1));
  auto ubExcl = Term::any_from_json(j_.at(2));
  auto step = Term::any_from_json(j_.at(3));
  std::vector<Stmt::Any> body;
  for (const auto &v_ : j_.at(4)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  return {induction, lbIncl, ubExcl, step, body};
}

json Stmt::forrange_to_json(const Stmt::ForRange &x_) {
  auto induction = named_to_json(x_.induction);
  auto lbIncl = Term::any_to_json(x_.lbIncl);
  auto ubExcl = Term::any_to_json(x_.ubExcl);
  auto step = Term::any_to_json(x_.step);
  std::vector<json> body;
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({induction, lbIncl, ubExcl, step, body});
}

Stmt::Break Stmt::break_from_json(const json &j_) { return {}; }

json Stmt::break_to_json(const Stmt::Break &x_) { return json::array({}); }

Stmt::Cont Stmt::cont_from_json(const json &j_) { return {}; }

json Stmt::cont_to_json(const Stmt::Cont &x_) { return json::array({}); }

Stmt::Cond Stmt::cond_from_json(const json &j_) {
  auto cond = Term::any_from_json(j_.at(0));
  std::vector<Stmt::Any> trueBr;
  for (const auto &v_ : j_.at(1)) {
    trueBr.emplace_back(Stmt::any_from_json(v_));
  }
  std::vector<Stmt::Any> falseBr;
  for (const auto &v_ : j_.at(2)) {
    falseBr.emplace_back(Stmt::any_from_json(v_));
  }
  return {cond, trueBr, falseBr};
}

json Stmt::cond_to_json(const Stmt::Cond &x_) {
  auto cond = Term::any_to_json(x_.cond);
  std::vector<json> trueBr;
  for (const auto &v_ : x_.trueBr) {
    trueBr.emplace_back(Stmt::any_to_json(v_));
  }
  std::vector<json> falseBr;
  for (const auto &v_ : x_.falseBr) {
    falseBr.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({cond, trueBr, falseBr});
}

Stmt::Return Stmt::return_from_json(const json &j_) {
  auto value = Expr::any_from_json(j_.at(0));
  return Stmt::Return(value);
}

json Stmt::return_to_json(const Stmt::Return &x_) {
  auto value = Expr::any_to_json(x_.value);
  return json::array({value});
}

Stmt::Annotated Stmt::annotated_from_json(const json &j_) {
  auto inner = Stmt::any_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto comment = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<std::string>());
  return {inner, pos, comment};
}

json Stmt::annotated_to_json(const Stmt::Annotated &x_) {
  auto inner = Stmt::any_to_json(x_.inner);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  auto comment = x_.comment ? json(*x_.comment) : json();
  return json::array({inner, pos, comment});
}

Stmt::Any Stmt::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Stmt::var_from_json(t_);
    case 1: return Stmt::mut_from_json(t_);
    case 2: return Stmt::update_from_json(t_);
    case 3: return Stmt::while_from_json(t_);
    case 4: return Stmt::forrange_from_json(t_);
    case 5: return Stmt::break_from_json(t_);
    case 6: return Stmt::cont_from_json(t_);
    case 7: return Stmt::cond_from_json(t_);
    case 8: return Stmt::return_from_json(t_);
    case 9: return Stmt::annotated_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Stmt::any_to_json(const Stmt::Any &x_) {
  return x_.match_total([](const Stmt::Var &y_) -> json { return {0, Stmt::var_to_json(y_)}; },
                        [](const Stmt::Mut &y_) -> json { return {1, Stmt::mut_to_json(y_)}; },
                        [](const Stmt::Update &y_) -> json { return {2, Stmt::update_to_json(y_)}; },
                        [](const Stmt::While &y_) -> json { return {3, Stmt::while_to_json(y_)}; },
                        [](const Stmt::ForRange &y_) -> json { return {4, Stmt::forrange_to_json(y_)}; },
                        [](const Stmt::Break &y_) -> json { return {5, Stmt::break_to_json(y_)}; },
                        [](const Stmt::Cont &y_) -> json { return {6, Stmt::cont_to_json(y_)}; },
                        [](const Stmt::Cond &y_) -> json { return {7, Stmt::cond_to_json(y_)}; },
                        [](const Stmt::Return &y_) -> json { return {8, Stmt::return_to_json(y_)}; },
                        [](const Stmt::Annotated &y_) -> json { return {9, Stmt::annotated_to_json(y_)}; });
}

Signature signature_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  auto tpeVars = j_.at(1).get<std::vector<std::string>>();
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(Type::any_from_json(j_.at(2)));
  std::vector<Type::Any> args;
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  std::vector<Type::Any> moduleCaptures;
  for (const auto &v_ : j_.at(4)) {
    moduleCaptures.emplace_back(Type::any_from_json(v_));
  }
  std::vector<Type::Any> termCaptures;
  for (const auto &v_ : j_.at(5)) {
    termCaptures.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(6));
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn};
}

json signature_to_json(const Signature &x_) {
  auto name = sym_to_json(x_.name);
  auto tpeVars = x_.tpeVars;
  auto receiver = x_.receiver ? Type::any_to_json(*x_.receiver) : json();
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  std::vector<json> moduleCaptures;
  for (const auto &v_ : x_.moduleCaptures) {
    moduleCaptures.emplace_back(Type::any_to_json(v_));
  }
  std::vector<json> termCaptures;
  for (const auto &v_ : x_.termCaptures) {
    termCaptures.emplace_back(Type::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn});
}

InvokeSignature invokesignature_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Any> tpeVars;
  for (const auto &v_ : j_.at(1)) {
    tpeVars.emplace_back(Type::any_from_json(v_));
  }
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(Type::any_from_json(j_.at(2)));
  std::vector<Type::Any> args;
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(4));
  return {name, tpeVars, receiver, args, rtn};
}

json invokesignature_to_json(const InvokeSignature &x_) {
  auto name = sym_to_json(x_.name);
  std::vector<json> tpeVars;
  for (const auto &v_ : x_.tpeVars) {
    tpeVars.emplace_back(Type::any_to_json(v_));
  }
  auto receiver = x_.receiver ? Type::any_to_json(*x_.receiver) : json();
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, tpeVars, receiver, args, rtn});
}

FunctionVisibility::Internal FunctionVisibility::internal_from_json(const json &j_) { return {}; }

json FunctionVisibility::internal_to_json(const FunctionVisibility::Internal &x_) { return json::array({}); }

FunctionVisibility::Exported FunctionVisibility::exported_from_json(const json &j_) { return {}; }

json FunctionVisibility::exported_to_json(const FunctionVisibility::Exported &x_) { return json::array({}); }

FunctionVisibility::Any FunctionVisibility::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionVisibility::internal_from_json(t_);
    case 1: return FunctionVisibility::exported_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionVisibility::any_to_json(const FunctionVisibility::Any &x_) {
  return x_.match_total([](const FunctionVisibility::Internal &y_) -> json { return {0, FunctionVisibility::internal_to_json(y_)}; },
                        [](const FunctionVisibility::Exported &y_) -> json { return {1, FunctionVisibility::exported_to_json(y_)}; });
}

FunctionFpMode::Relaxed FunctionFpMode::relaxed_from_json(const json &j_) { return {}; }

json FunctionFpMode::relaxed_to_json(const FunctionFpMode::Relaxed &x_) { return json::array({}); }

FunctionFpMode::Strict FunctionFpMode::strict_from_json(const json &j_) { return {}; }

json FunctionFpMode::strict_to_json(const FunctionFpMode::Strict &x_) { return json::array({}); }

FunctionFpMode::Any FunctionFpMode::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionFpMode::relaxed_from_json(t_);
    case 1: return FunctionFpMode::strict_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionFpMode::any_to_json(const FunctionFpMode::Any &x_) {
  return x_.match_total([](const FunctionFpMode::Relaxed &y_) -> json { return {0, FunctionFpMode::relaxed_to_json(y_)}; },
                        [](const FunctionFpMode::Strict &y_) -> json { return {1, FunctionFpMode::strict_to_json(y_)}; });
}

FunctionAffinity::Offload FunctionAffinity::offload_from_json(const json &j_) { return {}; }

json FunctionAffinity::offload_to_json(const FunctionAffinity::Offload &x_) { return json::array({}); }

FunctionAffinity::Host FunctionAffinity::host_from_json(const json &j_) { return {}; }

json FunctionAffinity::host_to_json(const FunctionAffinity::Host &x_) { return json::array({}); }

FunctionAffinity::Any FunctionAffinity::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionAffinity::offload_from_json(t_);
    case 1: return FunctionAffinity::host_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionAffinity::any_to_json(const FunctionAffinity::Any &x_) {
  return x_.match_total([](const FunctionAffinity::Offload &y_) -> json { return {0, FunctionAffinity::offload_to_json(y_)}; },
                        [](const FunctionAffinity::Host &y_) -> json { return {1, FunctionAffinity::host_to_json(y_)}; });
}

Arg arg_from_json(const json &j_) {
  auto named = named_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  return {named, pos};
}

json arg_to_json(const Arg &x_) {
  auto named = named_to_json(x_.named);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  return json::array({named, pos});
}

Function function_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  auto tpeVars = j_.at(1).get<std::vector<std::string>>();
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(arg_from_json(j_.at(2)));
  std::vector<Arg> args;
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(arg_from_json(v_));
  }
  std::vector<Arg> moduleCaptures;
  for (const auto &v_ : j_.at(4)) {
    moduleCaptures.emplace_back(arg_from_json(v_));
  }
  std::vector<Arg> termCaptures;
  for (const auto &v_ : j_.at(5)) {
    termCaptures.emplace_back(arg_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(6));
  std::vector<Stmt::Any> body;
  for (const auto &v_ : j_.at(7)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  auto visibility = FunctionVisibility::any_from_json(j_.at(8));
  auto fpMode = FunctionFpMode::any_from_json(j_.at(9));
  auto isEntry = j_.at(10).get<bool>();
  auto affinity = FunctionAffinity::any_from_json(j_.at(11));
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity};
}

json function_to_json(const Function &x_) {
  auto name = sym_to_json(x_.name);
  auto tpeVars = x_.tpeVars;
  auto receiver = x_.receiver ? arg_to_json(*x_.receiver) : json();
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(arg_to_json(v_));
  }
  std::vector<json> moduleCaptures;
  for (const auto &v_ : x_.moduleCaptures) {
    moduleCaptures.emplace_back(arg_to_json(v_));
  }
  std::vector<json> termCaptures;
  for (const auto &v_ : x_.termCaptures) {
    termCaptures.emplace_back(arg_to_json(v_));
  }
  auto rtn = Type::any_to_json(x_.rtn);
  std::vector<json> body;
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  auto visibility = FunctionVisibility::any_to_json(x_.visibility);
  auto fpMode = FunctionFpMode::any_to_json(x_.fpMode);
  auto isEntry = x_.isEntry;
  auto affinity = FunctionAffinity::any_to_json(x_.affinity);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity});
}

StructDef structdef_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  auto tpeVars = j_.at(1).get<std::vector<std::string>>();
  std::vector<Named> members;
  for (const auto &v_ : j_.at(2)) {
    members.emplace_back(named_from_json(v_));
  }
  std::vector<Type::Struct> parents;
  for (const auto &v_ : j_.at(3)) {
    parents.emplace_back(Type::struct_from_json(v_));
  }
  return {name, tpeVars, members, parents};
}

json structdef_to_json(const StructDef &x_) {
  auto name = sym_to_json(x_.name);
  auto tpeVars = x_.tpeVars;
  std::vector<json> members;
  for (const auto &v_ : x_.members) {
    members.emplace_back(named_to_json(v_));
  }
  std::vector<json> parents;
  for (const auto &v_ : x_.parents) {
    parents.emplace_back(Type::struct_to_json(v_));
  }
  return json::array({name, tpeVars, members, parents});
}

Mirror mirror_from_json(const json &j_) {
  auto source = sym_from_json(j_.at(0));
  std::vector<Sym> sourceParents;
  for (const auto &v_ : j_.at(1)) {
    sourceParents.emplace_back(sym_from_json(v_));
  }
  auto structDef = structdef_from_json(j_.at(2));
  std::vector<Function> functions;
  for (const auto &v_ : j_.at(3)) {
    functions.emplace_back(function_from_json(v_));
  }
  std::vector<StructDef> dependencies;
  for (const auto &v_ : j_.at(4)) {
    dependencies.emplace_back(structdef_from_json(v_));
  }
  return {source, sourceParents, structDef, functions, dependencies};
}

json mirror_to_json(const Mirror &x_) {
  auto source = sym_to_json(x_.source);
  std::vector<json> sourceParents;
  for (const auto &v_ : x_.sourceParents) {
    sourceParents.emplace_back(sym_to_json(v_));
  }
  auto structDef = structdef_to_json(x_.structDef);
  std::vector<json> functions;
  for (const auto &v_ : x_.functions) {
    functions.emplace_back(function_to_json(v_));
  }
  std::vector<json> dependencies;
  for (const auto &v_ : x_.dependencies) {
    dependencies.emplace_back(structdef_to_json(v_));
  }
  return json::array({source, sourceParents, structDef, functions, dependencies});
}

PassPhase::Initial PassPhase::initial_from_json(const json &j_) { return {}; }

json PassPhase::initial_to_json(const PassPhase::Initial &x_) { return json::array({}); }

PassPhase::PostMono PassPhase::postmono_from_json(const json &j_) { return {}; }

json PassPhase::postmono_to_json(const PassPhase::PostMono &x_) { return json::array({}); }

PassPhase::Any PassPhase::any_from_json(const json &j_) {
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return PassPhase::initial_from_json(t_);
    case 1: return PassPhase::postmono_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json PassPhase::any_to_json(const PassPhase::Any &x_) {
  return x_.match_total([](const PassPhase::Initial &y_) -> json { return {0, PassPhase::initial_to_json(y_)}; },
                        [](const PassPhase::PostMono &y_) -> json { return {1, PassPhase::postmono_to_json(y_)}; });
}

MetaEntry metaentry_from_json(const json &j_) {
  auto key = j_.at(0).get<std::string>();
  auto value = j_.at(1).get<std::string>();
  return {key, value};
}

json metaentry_to_json(const MetaEntry &x_) {
  auto key = x_.key;
  auto value = x_.value;
  return json::array({key, value});
}

Program program_from_json(const json &j_) {
  auto entry = function_from_json(j_.at(0));
  std::vector<Function> functions;
  for (const auto &v_ : j_.at(1)) {
    functions.emplace_back(function_from_json(v_));
  }
  std::vector<StructDef> defs;
  for (const auto &v_ : j_.at(2)) {
    defs.emplace_back(structdef_from_json(v_));
  }
  auto phase = PassPhase::any_from_json(j_.at(3));
  std::vector<MetaEntry> metadata;
  for (const auto &v_ : j_.at(4)) {
    metadata.emplace_back(metaentry_from_json(v_));
  }
  return {entry, functions, defs, phase, metadata};
}

json program_to_json(const Program &x_) {
  auto entry = function_to_json(x_.entry);
  std::vector<json> functions;
  for (const auto &v_ : x_.functions) {
    functions.emplace_back(function_to_json(v_));
  }
  std::vector<json> defs;
  for (const auto &v_ : x_.defs) {
    defs.emplace_back(structdef_to_json(v_));
  }
  auto phase = PassPhase::any_to_json(x_.phase);
  std::vector<json> metadata;
  for (const auto &v_ : x_.metadata) {
    metadata.emplace_back(metaentry_to_json(v_));
  }
  return json::array({entry, functions, defs, phase, metadata});
}

StructLayoutMember structlayoutmember_from_json(const json &j_) {
  auto name = named_from_json(j_.at(0));
  auto offsetInBytes = j_.at(1).get<int64_t>();
  auto sizeInBytes = j_.at(2).get<int64_t>();
  return {name, offsetInBytes, sizeInBytes};
}

json structlayoutmember_to_json(const StructLayoutMember &x_) {
  auto name = named_to_json(x_.name);
  auto offsetInBytes = x_.offsetInBytes;
  auto sizeInBytes = x_.sizeInBytes;
  return json::array({name, offsetInBytes, sizeInBytes});
}

StructLayout structlayout_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  auto sizeInBytes = j_.at(1).get<int64_t>();
  auto alignment = j_.at(2).get<int64_t>();
  std::vector<StructLayoutMember> members;
  for (const auto &v_ : j_.at(3)) {
    members.emplace_back(structlayoutmember_from_json(v_));
  }
  return {name, sizeInBytes, alignment, members};
}

json structlayout_to_json(const StructLayout &x_) {
  auto name = x_.name;
  auto sizeInBytes = x_.sizeInBytes;
  auto alignment = x_.alignment;
  std::vector<json> members;
  for (const auto &v_ : x_.members) {
    members.emplace_back(structlayoutmember_to_json(v_));
  }
  return json::array({name, sizeInBytes, alignment, members});
}

CompileEvent compileevent_from_json(const json &j_) {
  auto epochMillis = j_.at(0).get<int64_t>();
  auto elapsedNanos = j_.at(1).get<int64_t>();
  auto name = j_.at(2).get<std::string>();
  auto data = j_.at(3).get<std::string>();
  std::vector<CompileEvent> items;
  for (const auto &v_ : j_.at(4)) {
    items.emplace_back(compileevent_from_json(v_));
  }
  return {epochMillis, elapsedNanos, name, data, items};
}

json compileevent_to_json(const CompileEvent &x_) {
  auto epochMillis = x_.epochMillis;
  auto elapsedNanos = x_.elapsedNanos;
  auto name = x_.name;
  auto data = x_.data;
  std::vector<json> items;
  for (const auto &v_ : x_.items) {
    items.emplace_back(compileevent_to_json(v_));
  }
  return json::array({epochMillis, elapsedNanos, name, data, items});
}

PassArg passarg_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  auto value = j_.at(1).get<std::string>();
  return {name, value};
}

json passarg_to_json(const PassArg &x_) {
  auto name = x_.name;
  auto value = x_.value;
  return json::array({name, value});
}

PassSpec passspec_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  std::vector<PassArg> args;
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(passarg_from_json(v_));
  }
  return {name, args};
}

json passspec_to_json(const PassSpec &x_) {
  auto name = x_.name;
  std::vector<json> args;
  for (const auto &v_ : x_.args) {
    args.emplace_back(passarg_to_json(v_));
  }
  return json::array({name, args});
}

PassPipeline passpipeline_from_json(const json &j_) {
  std::vector<PassSpec> steps;
  for (const auto &v_ : j_.at(0)) {
    steps.emplace_back(passspec_from_json(v_));
  }
  return PassPipeline(steps);
}

json passpipeline_to_json(const PassPipeline &x_) {
  std::vector<json> steps;
  for (const auto &v_ : x_.steps) {
    steps.emplace_back(passspec_to_json(v_));
  }
  return json::array({steps});
}

PassRunResult passrunresult_from_json(const json &j_) {
  auto program = program_from_json(j_.at(0));
  auto event = compileevent_from_json(j_.at(1));
  return {program, event};
}

json passrunresult_to_json(const PassRunResult &x_) {
  auto program = program_to_json(x_.program);
  auto event = compileevent_to_json(x_.event);
  return json::array({program, event});
}

CompileResult compileresult_from_json(const json &j_) {
  auto binary = j_.at(0).is_null() ? std::nullopt : std::make_optional(j_.at(0).get<std::vector<int8_t>>());
  auto features = j_.at(1).get<std::vector<std::string>>();
  std::vector<CompileEvent> events;
  for (const auto &v_ : j_.at(2)) {
    events.emplace_back(compileevent_from_json(v_));
  }
  std::vector<StructLayout> layouts;
  for (const auto &v_ : j_.at(3)) {
    layouts.emplace_back(structlayout_from_json(v_));
  }
  auto messages = j_.at(4).get<std::string>();
  return {binary, features, events, layouts, messages};
}

json compileresult_to_json(const CompileResult &x_) {
  auto binary = x_.binary ? json(*x_.binary) : json();
  auto features = x_.features;
  std::vector<json> events;
  for (const auto &v_ : x_.events) {
    events.emplace_back(compileevent_to_json(v_));
  }
  std::vector<json> layouts;
  for (const auto &v_ : x_.layouts) {
    layouts.emplace_back(structlayout_to_json(v_));
  }
  auto messages = x_.messages;
  return json::array({binary, features, events, layouts, messages});
}
json hashed_from_json(const json &j_) {
  auto hash_ = j_.at(0).get<std::string>();
  auto data_ = j_.at(1);
  if (hash_ != AdtHash) {
    throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
  }
  return data_;
}

json hashed_to_json(const json &x_) { return json::array({AdtHash, x_}); }

namespace Intr {
Intr::BNot bnot_fields_from_msgpack(MsgpackReader &, size_t);
void bnot_fields_to_msgpack(MsgpackWriter &, const Intr::BNot &);
Intr::BNot bnot_from_msgpack(MsgpackReader &);
void bnot_to_msgpack(MsgpackWriter &, const Intr::BNot &);
Intr::LogicNot logicnot_fields_from_msgpack(MsgpackReader &, size_t);
void logicnot_fields_to_msgpack(MsgpackWriter &, const Intr::LogicNot &);
Intr::LogicNot logicnot_from_msgpack(MsgpackReader &);
void logicnot_to_msgpack(MsgpackWriter &, const Intr::LogicNot &);
Intr::Pos pos_fields_from_msgpack(MsgpackReader &, size_t);
void pos_fields_to_msgpack(MsgpackWriter &, const Intr::Pos &);
Intr::Pos pos_from_msgpack(MsgpackReader &);
void pos_to_msgpack(MsgpackWriter &, const Intr::Pos &);
Intr::Neg neg_fields_from_msgpack(MsgpackReader &, size_t);
void neg_fields_to_msgpack(MsgpackWriter &, const Intr::Neg &);
Intr::Neg neg_from_msgpack(MsgpackReader &);
void neg_to_msgpack(MsgpackWriter &, const Intr::Neg &);
Intr::Add add_fields_from_msgpack(MsgpackReader &, size_t);
void add_fields_to_msgpack(MsgpackWriter &, const Intr::Add &);
Intr::Add add_from_msgpack(MsgpackReader &);
void add_to_msgpack(MsgpackWriter &, const Intr::Add &);
Intr::Sub sub_fields_from_msgpack(MsgpackReader &, size_t);
void sub_fields_to_msgpack(MsgpackWriter &, const Intr::Sub &);
Intr::Sub sub_from_msgpack(MsgpackReader &);
void sub_to_msgpack(MsgpackWriter &, const Intr::Sub &);
Intr::Mul mul_fields_from_msgpack(MsgpackReader &, size_t);
void mul_fields_to_msgpack(MsgpackWriter &, const Intr::Mul &);
Intr::Mul mul_from_msgpack(MsgpackReader &);
void mul_to_msgpack(MsgpackWriter &, const Intr::Mul &);
Intr::Div div_fields_from_msgpack(MsgpackReader &, size_t);
void div_fields_to_msgpack(MsgpackWriter &, const Intr::Div &);
Intr::Div div_from_msgpack(MsgpackReader &);
void div_to_msgpack(MsgpackWriter &, const Intr::Div &);
Intr::Rem rem_fields_from_msgpack(MsgpackReader &, size_t);
void rem_fields_to_msgpack(MsgpackWriter &, const Intr::Rem &);
Intr::Rem rem_from_msgpack(MsgpackReader &);
void rem_to_msgpack(MsgpackWriter &, const Intr::Rem &);
Intr::Min min_fields_from_msgpack(MsgpackReader &, size_t);
void min_fields_to_msgpack(MsgpackWriter &, const Intr::Min &);
Intr::Min min_from_msgpack(MsgpackReader &);
void min_to_msgpack(MsgpackWriter &, const Intr::Min &);
Intr::Max max_fields_from_msgpack(MsgpackReader &, size_t);
void max_fields_to_msgpack(MsgpackWriter &, const Intr::Max &);
Intr::Max max_from_msgpack(MsgpackReader &);
void max_to_msgpack(MsgpackWriter &, const Intr::Max &);
Intr::BAnd band_fields_from_msgpack(MsgpackReader &, size_t);
void band_fields_to_msgpack(MsgpackWriter &, const Intr::BAnd &);
Intr::BAnd band_from_msgpack(MsgpackReader &);
void band_to_msgpack(MsgpackWriter &, const Intr::BAnd &);
Intr::BOr bor_fields_from_msgpack(MsgpackReader &, size_t);
void bor_fields_to_msgpack(MsgpackWriter &, const Intr::BOr &);
Intr::BOr bor_from_msgpack(MsgpackReader &);
void bor_to_msgpack(MsgpackWriter &, const Intr::BOr &);
Intr::BXor bxor_fields_from_msgpack(MsgpackReader &, size_t);
void bxor_fields_to_msgpack(MsgpackWriter &, const Intr::BXor &);
Intr::BXor bxor_from_msgpack(MsgpackReader &);
void bxor_to_msgpack(MsgpackWriter &, const Intr::BXor &);
Intr::BSL bsl_fields_from_msgpack(MsgpackReader &, size_t);
void bsl_fields_to_msgpack(MsgpackWriter &, const Intr::BSL &);
Intr::BSL bsl_from_msgpack(MsgpackReader &);
void bsl_to_msgpack(MsgpackWriter &, const Intr::BSL &);
Intr::BSR bsr_fields_from_msgpack(MsgpackReader &, size_t);
void bsr_fields_to_msgpack(MsgpackWriter &, const Intr::BSR &);
Intr::BSR bsr_from_msgpack(MsgpackReader &);
void bsr_to_msgpack(MsgpackWriter &, const Intr::BSR &);
Intr::BZSR bzsr_fields_from_msgpack(MsgpackReader &, size_t);
void bzsr_fields_to_msgpack(MsgpackWriter &, const Intr::BZSR &);
Intr::BZSR bzsr_from_msgpack(MsgpackReader &);
void bzsr_to_msgpack(MsgpackWriter &, const Intr::BZSR &);
Intr::LogicAnd logicand_fields_from_msgpack(MsgpackReader &, size_t);
void logicand_fields_to_msgpack(MsgpackWriter &, const Intr::LogicAnd &);
Intr::LogicAnd logicand_from_msgpack(MsgpackReader &);
void logicand_to_msgpack(MsgpackWriter &, const Intr::LogicAnd &);
Intr::LogicOr logicor_fields_from_msgpack(MsgpackReader &, size_t);
void logicor_fields_to_msgpack(MsgpackWriter &, const Intr::LogicOr &);
Intr::LogicOr logicor_from_msgpack(MsgpackReader &);
void logicor_to_msgpack(MsgpackWriter &, const Intr::LogicOr &);
Intr::LogicEq logiceq_fields_from_msgpack(MsgpackReader &, size_t);
void logiceq_fields_to_msgpack(MsgpackWriter &, const Intr::LogicEq &);
Intr::LogicEq logiceq_from_msgpack(MsgpackReader &);
void logiceq_to_msgpack(MsgpackWriter &, const Intr::LogicEq &);
Intr::LogicNeq logicneq_fields_from_msgpack(MsgpackReader &, size_t);
void logicneq_fields_to_msgpack(MsgpackWriter &, const Intr::LogicNeq &);
Intr::LogicNeq logicneq_from_msgpack(MsgpackReader &);
void logicneq_to_msgpack(MsgpackWriter &, const Intr::LogicNeq &);
Intr::LogicLte logiclte_fields_from_msgpack(MsgpackReader &, size_t);
void logiclte_fields_to_msgpack(MsgpackWriter &, const Intr::LogicLte &);
Intr::LogicLte logiclte_from_msgpack(MsgpackReader &);
void logiclte_to_msgpack(MsgpackWriter &, const Intr::LogicLte &);
Intr::LogicGte logicgte_fields_from_msgpack(MsgpackReader &, size_t);
void logicgte_fields_to_msgpack(MsgpackWriter &, const Intr::LogicGte &);
Intr::LogicGte logicgte_from_msgpack(MsgpackReader &);
void logicgte_to_msgpack(MsgpackWriter &, const Intr::LogicGte &);
Intr::LogicLt logiclt_fields_from_msgpack(MsgpackReader &, size_t);
void logiclt_fields_to_msgpack(MsgpackWriter &, const Intr::LogicLt &);
Intr::LogicLt logiclt_from_msgpack(MsgpackReader &);
void logiclt_to_msgpack(MsgpackWriter &, const Intr::LogicLt &);
Intr::LogicGt logicgt_fields_from_msgpack(MsgpackReader &, size_t);
void logicgt_fields_to_msgpack(MsgpackWriter &, const Intr::LogicGt &);
Intr::LogicGt logicgt_from_msgpack(MsgpackReader &);
void logicgt_to_msgpack(MsgpackWriter &, const Intr::LogicGt &);
Intr::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Intr::Any &);
} // namespace Intr
namespace FunctionFpMode {
FunctionFpMode::Relaxed relaxed_fields_from_msgpack(MsgpackReader &, size_t);
void relaxed_fields_to_msgpack(MsgpackWriter &, const FunctionFpMode::Relaxed &);
FunctionFpMode::Relaxed relaxed_from_msgpack(MsgpackReader &);
void relaxed_to_msgpack(MsgpackWriter &, const FunctionFpMode::Relaxed &);
FunctionFpMode::Strict strict_fields_from_msgpack(MsgpackReader &, size_t);
void strict_fields_to_msgpack(MsgpackWriter &, const FunctionFpMode::Strict &);
FunctionFpMode::Strict strict_from_msgpack(MsgpackReader &);
void strict_to_msgpack(MsgpackWriter &, const FunctionFpMode::Strict &);
FunctionFpMode::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionFpMode::Any &);
} // namespace FunctionFpMode
namespace Expr {
Expr::Alias alias_fields_from_msgpack(MsgpackReader &, size_t);
void alias_fields_to_msgpack(MsgpackWriter &, const Expr::Alias &);
Expr::Alias alias_from_msgpack(MsgpackReader &);
void alias_to_msgpack(MsgpackWriter &, const Expr::Alias &);
Expr::SpecOp specop_fields_from_msgpack(MsgpackReader &, size_t);
void specop_fields_to_msgpack(MsgpackWriter &, const Expr::SpecOp &);
Expr::SpecOp specop_from_msgpack(MsgpackReader &);
void specop_to_msgpack(MsgpackWriter &, const Expr::SpecOp &);
Expr::MathOp mathop_fields_from_msgpack(MsgpackReader &, size_t);
void mathop_fields_to_msgpack(MsgpackWriter &, const Expr::MathOp &);
Expr::MathOp mathop_from_msgpack(MsgpackReader &);
void mathop_to_msgpack(MsgpackWriter &, const Expr::MathOp &);
Expr::IntrOp introp_fields_from_msgpack(MsgpackReader &, size_t);
void introp_fields_to_msgpack(MsgpackWriter &, const Expr::IntrOp &);
Expr::IntrOp introp_from_msgpack(MsgpackReader &);
void introp_to_msgpack(MsgpackWriter &, const Expr::IntrOp &);
Expr::Cast cast_fields_from_msgpack(MsgpackReader &, size_t);
void cast_fields_to_msgpack(MsgpackWriter &, const Expr::Cast &);
Expr::Cast cast_from_msgpack(MsgpackReader &);
void cast_to_msgpack(MsgpackWriter &, const Expr::Cast &);
Expr::Index index_fields_from_msgpack(MsgpackReader &, size_t);
void index_fields_to_msgpack(MsgpackWriter &, const Expr::Index &);
Expr::Index index_from_msgpack(MsgpackReader &);
void index_to_msgpack(MsgpackWriter &, const Expr::Index &);
Expr::RefTo refto_fields_from_msgpack(MsgpackReader &, size_t);
void refto_fields_to_msgpack(MsgpackWriter &, const Expr::RefTo &);
Expr::RefTo refto_from_msgpack(MsgpackReader &);
void refto_to_msgpack(MsgpackWriter &, const Expr::RefTo &);
Expr::Alloc alloc_fields_from_msgpack(MsgpackReader &, size_t);
void alloc_fields_to_msgpack(MsgpackWriter &, const Expr::Alloc &);
Expr::Alloc alloc_from_msgpack(MsgpackReader &);
void alloc_to_msgpack(MsgpackWriter &, const Expr::Alloc &);
Expr::Invoke invoke_fields_from_msgpack(MsgpackReader &, size_t);
void invoke_fields_to_msgpack(MsgpackWriter &, const Expr::Invoke &);
Expr::Invoke invoke_from_msgpack(MsgpackReader &);
void invoke_to_msgpack(MsgpackWriter &, const Expr::Invoke &);
Expr::ForeignCall foreigncall_fields_from_msgpack(MsgpackReader &, size_t);
void foreigncall_fields_to_msgpack(MsgpackWriter &, const Expr::ForeignCall &);
Expr::ForeignCall foreigncall_from_msgpack(MsgpackReader &);
void foreigncall_to_msgpack(MsgpackWriter &, const Expr::ForeignCall &);
Expr::OffsetOf offsetof_fields_from_msgpack(MsgpackReader &, size_t);
void offsetof_fields_to_msgpack(MsgpackWriter &, const Expr::OffsetOf &);
Expr::OffsetOf offsetof_from_msgpack(MsgpackReader &);
void offsetof_to_msgpack(MsgpackWriter &, const Expr::OffsetOf &);
Expr::SizeOf sizeof_fields_from_msgpack(MsgpackReader &, size_t);
void sizeof_fields_to_msgpack(MsgpackWriter &, const Expr::SizeOf &);
Expr::SizeOf sizeof_from_msgpack(MsgpackReader &);
void sizeof_to_msgpack(MsgpackWriter &, const Expr::SizeOf &);
Expr::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Expr::Any &);
} // namespace Expr
namespace Region {
Region::Rooted rooted_fields_from_msgpack(MsgpackReader &, size_t);
void rooted_fields_to_msgpack(MsgpackWriter &, const Region::Rooted &);
Region::Rooted rooted_from_msgpack(MsgpackReader &);
void rooted_to_msgpack(MsgpackWriter &, const Region::Rooted &);
Region::Opaque opaque_fields_from_msgpack(MsgpackReader &, size_t);
void opaque_fields_to_msgpack(MsgpackWriter &, const Region::Opaque &);
Region::Opaque opaque_from_msgpack(MsgpackReader &);
void opaque_to_msgpack(MsgpackWriter &, const Region::Opaque &);
Region::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Region::Any &);
} // namespace Region
namespace PathStep {
PathStep::Field field_fields_from_msgpack(MsgpackReader &, size_t);
void field_fields_to_msgpack(MsgpackWriter &, const PathStep::Field &);
PathStep::Field field_from_msgpack(MsgpackReader &);
void field_to_msgpack(MsgpackWriter &, const PathStep::Field &);
PathStep::Deref deref_fields_from_msgpack(MsgpackReader &, size_t);
void deref_fields_to_msgpack(MsgpackWriter &, const PathStep::Deref &);
PathStep::Deref deref_from_msgpack(MsgpackReader &);
void deref_to_msgpack(MsgpackWriter &, const PathStep::Deref &);
PathStep::Index index_fields_from_msgpack(MsgpackReader &, size_t);
void index_fields_to_msgpack(MsgpackWriter &, const PathStep::Index &);
PathStep::Index index_from_msgpack(MsgpackReader &);
void index_to_msgpack(MsgpackWriter &, const PathStep::Index &);
PathStep::IndexDyn indexdyn_fields_from_msgpack(MsgpackReader &, size_t);
void indexdyn_fields_to_msgpack(MsgpackWriter &, const PathStep::IndexDyn &);
PathStep::IndexDyn indexdyn_from_msgpack(MsgpackReader &);
void indexdyn_to_msgpack(MsgpackWriter &, const PathStep::IndexDyn &);
PathStep::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const PathStep::Any &);
} // namespace PathStep
namespace Stmt {
Stmt::Var var_fields_from_msgpack(MsgpackReader &, size_t);
void var_fields_to_msgpack(MsgpackWriter &, const Stmt::Var &);
Stmt::Var var_from_msgpack(MsgpackReader &);
void var_to_msgpack(MsgpackWriter &, const Stmt::Var &);
Stmt::Mut mut_fields_from_msgpack(MsgpackReader &, size_t);
void mut_fields_to_msgpack(MsgpackWriter &, const Stmt::Mut &);
Stmt::Mut mut_from_msgpack(MsgpackReader &);
void mut_to_msgpack(MsgpackWriter &, const Stmt::Mut &);
Stmt::Update update_fields_from_msgpack(MsgpackReader &, size_t);
void update_fields_to_msgpack(MsgpackWriter &, const Stmt::Update &);
Stmt::Update update_from_msgpack(MsgpackReader &);
void update_to_msgpack(MsgpackWriter &, const Stmt::Update &);
Stmt::While while_fields_from_msgpack(MsgpackReader &, size_t);
void while_fields_to_msgpack(MsgpackWriter &, const Stmt::While &);
Stmt::While while_from_msgpack(MsgpackReader &);
void while_to_msgpack(MsgpackWriter &, const Stmt::While &);
Stmt::ForRange forrange_fields_from_msgpack(MsgpackReader &, size_t);
void forrange_fields_to_msgpack(MsgpackWriter &, const Stmt::ForRange &);
Stmt::ForRange forrange_from_msgpack(MsgpackReader &);
void forrange_to_msgpack(MsgpackWriter &, const Stmt::ForRange &);
Stmt::Break break_fields_from_msgpack(MsgpackReader &, size_t);
void break_fields_to_msgpack(MsgpackWriter &, const Stmt::Break &);
Stmt::Break break_from_msgpack(MsgpackReader &);
void break_to_msgpack(MsgpackWriter &, const Stmt::Break &);
Stmt::Cont cont_fields_from_msgpack(MsgpackReader &, size_t);
void cont_fields_to_msgpack(MsgpackWriter &, const Stmt::Cont &);
Stmt::Cont cont_from_msgpack(MsgpackReader &);
void cont_to_msgpack(MsgpackWriter &, const Stmt::Cont &);
Stmt::Cond cond_fields_from_msgpack(MsgpackReader &, size_t);
void cond_fields_to_msgpack(MsgpackWriter &, const Stmt::Cond &);
Stmt::Cond cond_from_msgpack(MsgpackReader &);
void cond_to_msgpack(MsgpackWriter &, const Stmt::Cond &);
Stmt::Return return_fields_from_msgpack(MsgpackReader &, size_t);
void return_fields_to_msgpack(MsgpackWriter &, const Stmt::Return &);
Stmt::Return return_from_msgpack(MsgpackReader &);
void return_to_msgpack(MsgpackWriter &, const Stmt::Return &);
Stmt::Annotated annotated_fields_from_msgpack(MsgpackReader &, size_t);
void annotated_fields_to_msgpack(MsgpackWriter &, const Stmt::Annotated &);
Stmt::Annotated annotated_from_msgpack(MsgpackReader &);
void annotated_to_msgpack(MsgpackWriter &, const Stmt::Annotated &);
Stmt::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Stmt::Any &);
} // namespace Stmt
namespace Math {
Math::Abs abs_fields_from_msgpack(MsgpackReader &, size_t);
void abs_fields_to_msgpack(MsgpackWriter &, const Math::Abs &);
Math::Abs abs_from_msgpack(MsgpackReader &);
void abs_to_msgpack(MsgpackWriter &, const Math::Abs &);
Math::Sin sin_fields_from_msgpack(MsgpackReader &, size_t);
void sin_fields_to_msgpack(MsgpackWriter &, const Math::Sin &);
Math::Sin sin_from_msgpack(MsgpackReader &);
void sin_to_msgpack(MsgpackWriter &, const Math::Sin &);
Math::Cos cos_fields_from_msgpack(MsgpackReader &, size_t);
void cos_fields_to_msgpack(MsgpackWriter &, const Math::Cos &);
Math::Cos cos_from_msgpack(MsgpackReader &);
void cos_to_msgpack(MsgpackWriter &, const Math::Cos &);
Math::Tan tan_fields_from_msgpack(MsgpackReader &, size_t);
void tan_fields_to_msgpack(MsgpackWriter &, const Math::Tan &);
Math::Tan tan_from_msgpack(MsgpackReader &);
void tan_to_msgpack(MsgpackWriter &, const Math::Tan &);
Math::Asin asin_fields_from_msgpack(MsgpackReader &, size_t);
void asin_fields_to_msgpack(MsgpackWriter &, const Math::Asin &);
Math::Asin asin_from_msgpack(MsgpackReader &);
void asin_to_msgpack(MsgpackWriter &, const Math::Asin &);
Math::Acos acos_fields_from_msgpack(MsgpackReader &, size_t);
void acos_fields_to_msgpack(MsgpackWriter &, const Math::Acos &);
Math::Acos acos_from_msgpack(MsgpackReader &);
void acos_to_msgpack(MsgpackWriter &, const Math::Acos &);
Math::Atan atan_fields_from_msgpack(MsgpackReader &, size_t);
void atan_fields_to_msgpack(MsgpackWriter &, const Math::Atan &);
Math::Atan atan_from_msgpack(MsgpackReader &);
void atan_to_msgpack(MsgpackWriter &, const Math::Atan &);
Math::Sinh sinh_fields_from_msgpack(MsgpackReader &, size_t);
void sinh_fields_to_msgpack(MsgpackWriter &, const Math::Sinh &);
Math::Sinh sinh_from_msgpack(MsgpackReader &);
void sinh_to_msgpack(MsgpackWriter &, const Math::Sinh &);
Math::Cosh cosh_fields_from_msgpack(MsgpackReader &, size_t);
void cosh_fields_to_msgpack(MsgpackWriter &, const Math::Cosh &);
Math::Cosh cosh_from_msgpack(MsgpackReader &);
void cosh_to_msgpack(MsgpackWriter &, const Math::Cosh &);
Math::Tanh tanh_fields_from_msgpack(MsgpackReader &, size_t);
void tanh_fields_to_msgpack(MsgpackWriter &, const Math::Tanh &);
Math::Tanh tanh_from_msgpack(MsgpackReader &);
void tanh_to_msgpack(MsgpackWriter &, const Math::Tanh &);
Math::Signum signum_fields_from_msgpack(MsgpackReader &, size_t);
void signum_fields_to_msgpack(MsgpackWriter &, const Math::Signum &);
Math::Signum signum_from_msgpack(MsgpackReader &);
void signum_to_msgpack(MsgpackWriter &, const Math::Signum &);
Math::Round round_fields_from_msgpack(MsgpackReader &, size_t);
void round_fields_to_msgpack(MsgpackWriter &, const Math::Round &);
Math::Round round_from_msgpack(MsgpackReader &);
void round_to_msgpack(MsgpackWriter &, const Math::Round &);
Math::Ceil ceil_fields_from_msgpack(MsgpackReader &, size_t);
void ceil_fields_to_msgpack(MsgpackWriter &, const Math::Ceil &);
Math::Ceil ceil_from_msgpack(MsgpackReader &);
void ceil_to_msgpack(MsgpackWriter &, const Math::Ceil &);
Math::Floor floor_fields_from_msgpack(MsgpackReader &, size_t);
void floor_fields_to_msgpack(MsgpackWriter &, const Math::Floor &);
Math::Floor floor_from_msgpack(MsgpackReader &);
void floor_to_msgpack(MsgpackWriter &, const Math::Floor &);
Math::Rint rint_fields_from_msgpack(MsgpackReader &, size_t);
void rint_fields_to_msgpack(MsgpackWriter &, const Math::Rint &);
Math::Rint rint_from_msgpack(MsgpackReader &);
void rint_to_msgpack(MsgpackWriter &, const Math::Rint &);
Math::Sqrt sqrt_fields_from_msgpack(MsgpackReader &, size_t);
void sqrt_fields_to_msgpack(MsgpackWriter &, const Math::Sqrt &);
Math::Sqrt sqrt_from_msgpack(MsgpackReader &);
void sqrt_to_msgpack(MsgpackWriter &, const Math::Sqrt &);
Math::Cbrt cbrt_fields_from_msgpack(MsgpackReader &, size_t);
void cbrt_fields_to_msgpack(MsgpackWriter &, const Math::Cbrt &);
Math::Cbrt cbrt_from_msgpack(MsgpackReader &);
void cbrt_to_msgpack(MsgpackWriter &, const Math::Cbrt &);
Math::Exp exp_fields_from_msgpack(MsgpackReader &, size_t);
void exp_fields_to_msgpack(MsgpackWriter &, const Math::Exp &);
Math::Exp exp_from_msgpack(MsgpackReader &);
void exp_to_msgpack(MsgpackWriter &, const Math::Exp &);
Math::Expm1 expm1_fields_from_msgpack(MsgpackReader &, size_t);
void expm1_fields_to_msgpack(MsgpackWriter &, const Math::Expm1 &);
Math::Expm1 expm1_from_msgpack(MsgpackReader &);
void expm1_to_msgpack(MsgpackWriter &, const Math::Expm1 &);
Math::Log log_fields_from_msgpack(MsgpackReader &, size_t);
void log_fields_to_msgpack(MsgpackWriter &, const Math::Log &);
Math::Log log_from_msgpack(MsgpackReader &);
void log_to_msgpack(MsgpackWriter &, const Math::Log &);
Math::Log1p log1p_fields_from_msgpack(MsgpackReader &, size_t);
void log1p_fields_to_msgpack(MsgpackWriter &, const Math::Log1p &);
Math::Log1p log1p_from_msgpack(MsgpackReader &);
void log1p_to_msgpack(MsgpackWriter &, const Math::Log1p &);
Math::Log10 log10_fields_from_msgpack(MsgpackReader &, size_t);
void log10_fields_to_msgpack(MsgpackWriter &, const Math::Log10 &);
Math::Log10 log10_from_msgpack(MsgpackReader &);
void log10_to_msgpack(MsgpackWriter &, const Math::Log10 &);
Math::Pow pow_fields_from_msgpack(MsgpackReader &, size_t);
void pow_fields_to_msgpack(MsgpackWriter &, const Math::Pow &);
Math::Pow pow_from_msgpack(MsgpackReader &);
void pow_to_msgpack(MsgpackWriter &, const Math::Pow &);
Math::Atan2 atan2_fields_from_msgpack(MsgpackReader &, size_t);
void atan2_fields_to_msgpack(MsgpackWriter &, const Math::Atan2 &);
Math::Atan2 atan2_from_msgpack(MsgpackReader &);
void atan2_to_msgpack(MsgpackWriter &, const Math::Atan2 &);
Math::Hypot hypot_fields_from_msgpack(MsgpackReader &, size_t);
void hypot_fields_to_msgpack(MsgpackWriter &, const Math::Hypot &);
Math::Hypot hypot_from_msgpack(MsgpackReader &);
void hypot_to_msgpack(MsgpackWriter &, const Math::Hypot &);
Math::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Math::Any &);
} // namespace Math
namespace TypeSpace {
TypeSpace::Global global_fields_from_msgpack(MsgpackReader &, size_t);
void global_fields_to_msgpack(MsgpackWriter &, const TypeSpace::Global &);
TypeSpace::Global global_from_msgpack(MsgpackReader &);
void global_to_msgpack(MsgpackWriter &, const TypeSpace::Global &);
TypeSpace::Local local_fields_from_msgpack(MsgpackReader &, size_t);
void local_fields_to_msgpack(MsgpackWriter &, const TypeSpace::Local &);
TypeSpace::Local local_from_msgpack(MsgpackReader &);
void local_to_msgpack(MsgpackWriter &, const TypeSpace::Local &);
TypeSpace::Private private_fields_from_msgpack(MsgpackReader &, size_t);
void private_fields_to_msgpack(MsgpackWriter &, const TypeSpace::Private &);
TypeSpace::Private private_from_msgpack(MsgpackReader &);
void private_to_msgpack(MsgpackWriter &, const TypeSpace::Private &);
TypeSpace::Constant constant_fields_from_msgpack(MsgpackReader &, size_t);
void constant_fields_to_msgpack(MsgpackWriter &, const TypeSpace::Constant &);
TypeSpace::Constant constant_from_msgpack(MsgpackReader &);
void constant_to_msgpack(MsgpackWriter &, const TypeSpace::Constant &);
TypeSpace::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const TypeSpace::Any &);
} // namespace TypeSpace
namespace PassPhase {
PassPhase::Initial initial_fields_from_msgpack(MsgpackReader &, size_t);
void initial_fields_to_msgpack(MsgpackWriter &, const PassPhase::Initial &);
PassPhase::Initial initial_from_msgpack(MsgpackReader &);
void initial_to_msgpack(MsgpackWriter &, const PassPhase::Initial &);
PassPhase::PostMono postmono_fields_from_msgpack(MsgpackReader &, size_t);
void postmono_fields_to_msgpack(MsgpackWriter &, const PassPhase::PostMono &);
PassPhase::PostMono postmono_from_msgpack(MsgpackReader &);
void postmono_to_msgpack(MsgpackWriter &, const PassPhase::PostMono &);
PassPhase::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const PassPhase::Any &);
} // namespace PassPhase
namespace Spec {
Spec::Assert assert_fields_from_msgpack(MsgpackReader &, size_t);
void assert_fields_to_msgpack(MsgpackWriter &, const Spec::Assert &);
Spec::Assert assert_from_msgpack(MsgpackReader &);
void assert_to_msgpack(MsgpackWriter &, const Spec::Assert &);
Spec::GpuBarrierGlobal gpubarrierglobal_fields_from_msgpack(MsgpackReader &, size_t);
void gpubarrierglobal_fields_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierGlobal &);
Spec::GpuBarrierGlobal gpubarrierglobal_from_msgpack(MsgpackReader &);
void gpubarrierglobal_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierGlobal &);
Spec::GpuBarrierLocal gpubarrierlocal_fields_from_msgpack(MsgpackReader &, size_t);
void gpubarrierlocal_fields_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierLocal &);
Spec::GpuBarrierLocal gpubarrierlocal_from_msgpack(MsgpackReader &);
void gpubarrierlocal_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierLocal &);
Spec::GpuBarrierAll gpubarrierall_fields_from_msgpack(MsgpackReader &, size_t);
void gpubarrierall_fields_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierAll &);
Spec::GpuBarrierAll gpubarrierall_from_msgpack(MsgpackReader &);
void gpubarrierall_to_msgpack(MsgpackWriter &, const Spec::GpuBarrierAll &);
Spec::GpuFenceGlobal gpufenceglobal_fields_from_msgpack(MsgpackReader &, size_t);
void gpufenceglobal_fields_to_msgpack(MsgpackWriter &, const Spec::GpuFenceGlobal &);
Spec::GpuFenceGlobal gpufenceglobal_from_msgpack(MsgpackReader &);
void gpufenceglobal_to_msgpack(MsgpackWriter &, const Spec::GpuFenceGlobal &);
Spec::GpuFenceLocal gpufencelocal_fields_from_msgpack(MsgpackReader &, size_t);
void gpufencelocal_fields_to_msgpack(MsgpackWriter &, const Spec::GpuFenceLocal &);
Spec::GpuFenceLocal gpufencelocal_from_msgpack(MsgpackReader &);
void gpufencelocal_to_msgpack(MsgpackWriter &, const Spec::GpuFenceLocal &);
Spec::GpuFenceAll gpufenceall_fields_from_msgpack(MsgpackReader &, size_t);
void gpufenceall_fields_to_msgpack(MsgpackWriter &, const Spec::GpuFenceAll &);
Spec::GpuFenceAll gpufenceall_from_msgpack(MsgpackReader &);
void gpufenceall_to_msgpack(MsgpackWriter &, const Spec::GpuFenceAll &);
Spec::GpuGlobalIdx gpuglobalidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpuglobalidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalIdx &);
Spec::GpuGlobalIdx gpuglobalidx_from_msgpack(MsgpackReader &);
void gpuglobalidx_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalIdx &);
Spec::GpuGlobalSize gpuglobalsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpuglobalsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalSize &);
Spec::GpuGlobalSize gpuglobalsize_from_msgpack(MsgpackReader &);
void gpuglobalsize_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalSize &);
Spec::GpuGroupIdx gpugroupidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupIdx &);
Spec::GpuGroupIdx gpugroupidx_from_msgpack(MsgpackReader &);
void gpugroupidx_to_msgpack(MsgpackWriter &, const Spec::GpuGroupIdx &);
Spec::GpuGroupSize gpugroupsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupSize &);
Spec::GpuGroupSize gpugroupsize_from_msgpack(MsgpackReader &);
void gpugroupsize_to_msgpack(MsgpackWriter &, const Spec::GpuGroupSize &);
Spec::GpuLocalIdx gpulocalidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpulocalidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuLocalIdx &);
Spec::GpuLocalIdx gpulocalidx_from_msgpack(MsgpackReader &);
void gpulocalidx_to_msgpack(MsgpackWriter &, const Spec::GpuLocalIdx &);
Spec::GpuLocalSize gpulocalsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpulocalsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuLocalSize &);
Spec::GpuLocalSize gpulocalsize_from_msgpack(MsgpackReader &);
void gpulocalsize_to_msgpack(MsgpackWriter &, const Spec::GpuLocalSize &);
Spec::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Spec::Any &);
} // namespace Spec
namespace FunctionAffinity {
FunctionAffinity::Offload offload_fields_from_msgpack(MsgpackReader &, size_t);
void offload_fields_to_msgpack(MsgpackWriter &, const FunctionAffinity::Offload &);
FunctionAffinity::Offload offload_from_msgpack(MsgpackReader &);
void offload_to_msgpack(MsgpackWriter &, const FunctionAffinity::Offload &);
FunctionAffinity::Host host_fields_from_msgpack(MsgpackReader &, size_t);
void host_fields_to_msgpack(MsgpackWriter &, const FunctionAffinity::Host &);
FunctionAffinity::Host host_from_msgpack(MsgpackReader &);
void host_to_msgpack(MsgpackWriter &, const FunctionAffinity::Host &);
FunctionAffinity::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionAffinity::Any &);
} // namespace FunctionAffinity
namespace Type {
Type::Float16 float16_fields_from_msgpack(MsgpackReader &, size_t);
void float16_fields_to_msgpack(MsgpackWriter &, const Type::Float16 &);
Type::Float16 float16_from_msgpack(MsgpackReader &);
void float16_to_msgpack(MsgpackWriter &, const Type::Float16 &);
Type::Float32 float32_fields_from_msgpack(MsgpackReader &, size_t);
void float32_fields_to_msgpack(MsgpackWriter &, const Type::Float32 &);
Type::Float32 float32_from_msgpack(MsgpackReader &);
void float32_to_msgpack(MsgpackWriter &, const Type::Float32 &);
Type::Float64 float64_fields_from_msgpack(MsgpackReader &, size_t);
void float64_fields_to_msgpack(MsgpackWriter &, const Type::Float64 &);
Type::Float64 float64_from_msgpack(MsgpackReader &);
void float64_to_msgpack(MsgpackWriter &, const Type::Float64 &);
Type::IntU8 intu8_fields_from_msgpack(MsgpackReader &, size_t);
void intu8_fields_to_msgpack(MsgpackWriter &, const Type::IntU8 &);
Type::IntU8 intu8_from_msgpack(MsgpackReader &);
void intu8_to_msgpack(MsgpackWriter &, const Type::IntU8 &);
Type::IntU16 intu16_fields_from_msgpack(MsgpackReader &, size_t);
void intu16_fields_to_msgpack(MsgpackWriter &, const Type::IntU16 &);
Type::IntU16 intu16_from_msgpack(MsgpackReader &);
void intu16_to_msgpack(MsgpackWriter &, const Type::IntU16 &);
Type::IntU32 intu32_fields_from_msgpack(MsgpackReader &, size_t);
void intu32_fields_to_msgpack(MsgpackWriter &, const Type::IntU32 &);
Type::IntU32 intu32_from_msgpack(MsgpackReader &);
void intu32_to_msgpack(MsgpackWriter &, const Type::IntU32 &);
Type::IntU64 intu64_fields_from_msgpack(MsgpackReader &, size_t);
void intu64_fields_to_msgpack(MsgpackWriter &, const Type::IntU64 &);
Type::IntU64 intu64_from_msgpack(MsgpackReader &);
void intu64_to_msgpack(MsgpackWriter &, const Type::IntU64 &);
Type::IntS8 ints8_fields_from_msgpack(MsgpackReader &, size_t);
void ints8_fields_to_msgpack(MsgpackWriter &, const Type::IntS8 &);
Type::IntS8 ints8_from_msgpack(MsgpackReader &);
void ints8_to_msgpack(MsgpackWriter &, const Type::IntS8 &);
Type::IntS16 ints16_fields_from_msgpack(MsgpackReader &, size_t);
void ints16_fields_to_msgpack(MsgpackWriter &, const Type::IntS16 &);
Type::IntS16 ints16_from_msgpack(MsgpackReader &);
void ints16_to_msgpack(MsgpackWriter &, const Type::IntS16 &);
Type::IntS32 ints32_fields_from_msgpack(MsgpackReader &, size_t);
void ints32_fields_to_msgpack(MsgpackWriter &, const Type::IntS32 &);
Type::IntS32 ints32_from_msgpack(MsgpackReader &);
void ints32_to_msgpack(MsgpackWriter &, const Type::IntS32 &);
Type::IntS64 ints64_fields_from_msgpack(MsgpackReader &, size_t);
void ints64_fields_to_msgpack(MsgpackWriter &, const Type::IntS64 &);
Type::IntS64 ints64_from_msgpack(MsgpackReader &);
void ints64_to_msgpack(MsgpackWriter &, const Type::IntS64 &);
Type::Nothing nothing_fields_from_msgpack(MsgpackReader &, size_t);
void nothing_fields_to_msgpack(MsgpackWriter &, const Type::Nothing &);
Type::Nothing nothing_from_msgpack(MsgpackReader &);
void nothing_to_msgpack(MsgpackWriter &, const Type::Nothing &);
Type::Unit0 unit0_fields_from_msgpack(MsgpackReader &, size_t);
void unit0_fields_to_msgpack(MsgpackWriter &, const Type::Unit0 &);
Type::Unit0 unit0_from_msgpack(MsgpackReader &);
void unit0_to_msgpack(MsgpackWriter &, const Type::Unit0 &);
Type::Bool1 bool1_fields_from_msgpack(MsgpackReader &, size_t);
void bool1_fields_to_msgpack(MsgpackWriter &, const Type::Bool1 &);
Type::Bool1 bool1_from_msgpack(MsgpackReader &);
void bool1_to_msgpack(MsgpackWriter &, const Type::Bool1 &);
Type::Struct struct_fields_from_msgpack(MsgpackReader &, size_t);
void struct_fields_to_msgpack(MsgpackWriter &, const Type::Struct &);
Type::Struct struct_from_msgpack(MsgpackReader &);
void struct_to_msgpack(MsgpackWriter &, const Type::Struct &);
Type::Ptr ptr_fields_from_msgpack(MsgpackReader &, size_t);
void ptr_fields_to_msgpack(MsgpackWriter &, const Type::Ptr &);
Type::Ptr ptr_from_msgpack(MsgpackReader &);
void ptr_to_msgpack(MsgpackWriter &, const Type::Ptr &);
Type::Arr arr_fields_from_msgpack(MsgpackReader &, size_t);
void arr_fields_to_msgpack(MsgpackWriter &, const Type::Arr &);
Type::Arr arr_from_msgpack(MsgpackReader &);
void arr_to_msgpack(MsgpackWriter &, const Type::Arr &);
Type::Var var_fields_from_msgpack(MsgpackReader &, size_t);
void var_fields_to_msgpack(MsgpackWriter &, const Type::Var &);
Type::Var var_from_msgpack(MsgpackReader &);
void var_to_msgpack(MsgpackWriter &, const Type::Var &);
Type::Exec exec_fields_from_msgpack(MsgpackReader &, size_t);
void exec_fields_to_msgpack(MsgpackWriter &, const Type::Exec &);
Type::Exec exec_from_msgpack(MsgpackReader &);
void exec_to_msgpack(MsgpackWriter &, const Type::Exec &);
Type::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Type::Any &);
} // namespace Type
namespace Term {
Term::Float16Const float16const_fields_from_msgpack(MsgpackReader &, size_t);
void float16const_fields_to_msgpack(MsgpackWriter &, const Term::Float16Const &);
Term::Float16Const float16const_from_msgpack(MsgpackReader &);
void float16const_to_msgpack(MsgpackWriter &, const Term::Float16Const &);
Term::Float32Const float32const_fields_from_msgpack(MsgpackReader &, size_t);
void float32const_fields_to_msgpack(MsgpackWriter &, const Term::Float32Const &);
Term::Float32Const float32const_from_msgpack(MsgpackReader &);
void float32const_to_msgpack(MsgpackWriter &, const Term::Float32Const &);
Term::Float64Const float64const_fields_from_msgpack(MsgpackReader &, size_t);
void float64const_fields_to_msgpack(MsgpackWriter &, const Term::Float64Const &);
Term::Float64Const float64const_from_msgpack(MsgpackReader &);
void float64const_to_msgpack(MsgpackWriter &, const Term::Float64Const &);
Term::IntU8Const intu8const_fields_from_msgpack(MsgpackReader &, size_t);
void intu8const_fields_to_msgpack(MsgpackWriter &, const Term::IntU8Const &);
Term::IntU8Const intu8const_from_msgpack(MsgpackReader &);
void intu8const_to_msgpack(MsgpackWriter &, const Term::IntU8Const &);
Term::IntU16Const intu16const_fields_from_msgpack(MsgpackReader &, size_t);
void intu16const_fields_to_msgpack(MsgpackWriter &, const Term::IntU16Const &);
Term::IntU16Const intu16const_from_msgpack(MsgpackReader &);
void intu16const_to_msgpack(MsgpackWriter &, const Term::IntU16Const &);
Term::IntU32Const intu32const_fields_from_msgpack(MsgpackReader &, size_t);
void intu32const_fields_to_msgpack(MsgpackWriter &, const Term::IntU32Const &);
Term::IntU32Const intu32const_from_msgpack(MsgpackReader &);
void intu32const_to_msgpack(MsgpackWriter &, const Term::IntU32Const &);
Term::IntU64Const intu64const_fields_from_msgpack(MsgpackReader &, size_t);
void intu64const_fields_to_msgpack(MsgpackWriter &, const Term::IntU64Const &);
Term::IntU64Const intu64const_from_msgpack(MsgpackReader &);
void intu64const_to_msgpack(MsgpackWriter &, const Term::IntU64Const &);
Term::IntS8Const ints8const_fields_from_msgpack(MsgpackReader &, size_t);
void ints8const_fields_to_msgpack(MsgpackWriter &, const Term::IntS8Const &);
Term::IntS8Const ints8const_from_msgpack(MsgpackReader &);
void ints8const_to_msgpack(MsgpackWriter &, const Term::IntS8Const &);
Term::IntS16Const ints16const_fields_from_msgpack(MsgpackReader &, size_t);
void ints16const_fields_to_msgpack(MsgpackWriter &, const Term::IntS16Const &);
Term::IntS16Const ints16const_from_msgpack(MsgpackReader &);
void ints16const_to_msgpack(MsgpackWriter &, const Term::IntS16Const &);
Term::IntS32Const ints32const_fields_from_msgpack(MsgpackReader &, size_t);
void ints32const_fields_to_msgpack(MsgpackWriter &, const Term::IntS32Const &);
Term::IntS32Const ints32const_from_msgpack(MsgpackReader &);
void ints32const_to_msgpack(MsgpackWriter &, const Term::IntS32Const &);
Term::IntS64Const ints64const_fields_from_msgpack(MsgpackReader &, size_t);
void ints64const_fields_to_msgpack(MsgpackWriter &, const Term::IntS64Const &);
Term::IntS64Const ints64const_from_msgpack(MsgpackReader &);
void ints64const_to_msgpack(MsgpackWriter &, const Term::IntS64Const &);
Term::Unit0Const unit0const_fields_from_msgpack(MsgpackReader &, size_t);
void unit0const_fields_to_msgpack(MsgpackWriter &, const Term::Unit0Const &);
Term::Unit0Const unit0const_from_msgpack(MsgpackReader &);
void unit0const_to_msgpack(MsgpackWriter &, const Term::Unit0Const &);
Term::Bool1Const bool1const_fields_from_msgpack(MsgpackReader &, size_t);
void bool1const_fields_to_msgpack(MsgpackWriter &, const Term::Bool1Const &);
Term::Bool1Const bool1const_from_msgpack(MsgpackReader &);
void bool1const_to_msgpack(MsgpackWriter &, const Term::Bool1Const &);
Term::NullPtrConst nullptrconst_fields_from_msgpack(MsgpackReader &, size_t);
void nullptrconst_fields_to_msgpack(MsgpackWriter &, const Term::NullPtrConst &);
Term::NullPtrConst nullptrconst_from_msgpack(MsgpackReader &);
void nullptrconst_to_msgpack(MsgpackWriter &, const Term::NullPtrConst &);
Term::Poison poison_fields_from_msgpack(MsgpackReader &, size_t);
void poison_fields_to_msgpack(MsgpackWriter &, const Term::Poison &);
Term::Poison poison_from_msgpack(MsgpackReader &);
void poison_to_msgpack(MsgpackWriter &, const Term::Poison &);
Term::Select select_fields_from_msgpack(MsgpackReader &, size_t);
void select_fields_to_msgpack(MsgpackWriter &, const Term::Select &);
Term::Select select_from_msgpack(MsgpackReader &);
void select_to_msgpack(MsgpackWriter &, const Term::Select &);
Term::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Term::Any &);
} // namespace Term
Sym sym_fields_from_msgpack(MsgpackReader &, size_t);
void sym_fields_to_msgpack(MsgpackWriter &, const Sym &);
Sym sym_from_msgpack(MsgpackReader &);
void sym_to_msgpack(MsgpackWriter &, const Sym &);
SourcePosition sourceposition_fields_from_msgpack(MsgpackReader &, size_t);
void sourceposition_fields_to_msgpack(MsgpackWriter &, const SourcePosition &);
SourcePosition sourceposition_from_msgpack(MsgpackReader &);
void sourceposition_to_msgpack(MsgpackWriter &, const SourcePosition &);
Named named_fields_from_msgpack(MsgpackReader &, size_t);
void named_fields_to_msgpack(MsgpackWriter &, const Named &);
Named named_from_msgpack(MsgpackReader &);
void named_to_msgpack(MsgpackWriter &, const Named &);
Overload overload_fields_from_msgpack(MsgpackReader &, size_t);
void overload_fields_to_msgpack(MsgpackWriter &, const Overload &);
Overload overload_from_msgpack(MsgpackReader &);
void overload_to_msgpack(MsgpackWriter &, const Overload &);
Signature signature_fields_from_msgpack(MsgpackReader &, size_t);
void signature_fields_to_msgpack(MsgpackWriter &, const Signature &);
Signature signature_from_msgpack(MsgpackReader &);
void signature_to_msgpack(MsgpackWriter &, const Signature &);
InvokeSignature invokesignature_fields_from_msgpack(MsgpackReader &, size_t);
void invokesignature_fields_to_msgpack(MsgpackWriter &, const InvokeSignature &);
InvokeSignature invokesignature_from_msgpack(MsgpackReader &);
void invokesignature_to_msgpack(MsgpackWriter &, const InvokeSignature &);
Arg arg_fields_from_msgpack(MsgpackReader &, size_t);
void arg_fields_to_msgpack(MsgpackWriter &, const Arg &);
Arg arg_from_msgpack(MsgpackReader &);
void arg_to_msgpack(MsgpackWriter &, const Arg &);
Function function_fields_from_msgpack(MsgpackReader &, size_t);
void function_fields_to_msgpack(MsgpackWriter &, const Function &);
Function function_from_msgpack(MsgpackReader &);
void function_to_msgpack(MsgpackWriter &, const Function &);
StructDef structdef_fields_from_msgpack(MsgpackReader &, size_t);
void structdef_fields_to_msgpack(MsgpackWriter &, const StructDef &);
StructDef structdef_from_msgpack(MsgpackReader &);
void structdef_to_msgpack(MsgpackWriter &, const StructDef &);
Mirror mirror_fields_from_msgpack(MsgpackReader &, size_t);
void mirror_fields_to_msgpack(MsgpackWriter &, const Mirror &);
Mirror mirror_from_msgpack(MsgpackReader &);
void mirror_to_msgpack(MsgpackWriter &, const Mirror &);
MetaEntry metaentry_fields_from_msgpack(MsgpackReader &, size_t);
void metaentry_fields_to_msgpack(MsgpackWriter &, const MetaEntry &);
MetaEntry metaentry_from_msgpack(MsgpackReader &);
void metaentry_to_msgpack(MsgpackWriter &, const MetaEntry &);
Program program_fields_from_msgpack(MsgpackReader &, size_t);
void program_fields_to_msgpack(MsgpackWriter &, const Program &);
Program program_from_msgpack(MsgpackReader &);
void program_to_msgpack(MsgpackWriter &, const Program &);
StructLayoutMember structlayoutmember_fields_from_msgpack(MsgpackReader &, size_t);
void structlayoutmember_fields_to_msgpack(MsgpackWriter &, const StructLayoutMember &);
StructLayoutMember structlayoutmember_from_msgpack(MsgpackReader &);
void structlayoutmember_to_msgpack(MsgpackWriter &, const StructLayoutMember &);
StructLayout structlayout_fields_from_msgpack(MsgpackReader &, size_t);
void structlayout_fields_to_msgpack(MsgpackWriter &, const StructLayout &);
StructLayout structlayout_from_msgpack(MsgpackReader &);
void structlayout_to_msgpack(MsgpackWriter &, const StructLayout &);
CompileEvent compileevent_fields_from_msgpack(MsgpackReader &, size_t);
void compileevent_fields_to_msgpack(MsgpackWriter &, const CompileEvent &);
CompileEvent compileevent_from_msgpack(MsgpackReader &);
void compileevent_to_msgpack(MsgpackWriter &, const CompileEvent &);
PassArg passarg_fields_from_msgpack(MsgpackReader &, size_t);
void passarg_fields_to_msgpack(MsgpackWriter &, const PassArg &);
PassArg passarg_from_msgpack(MsgpackReader &);
void passarg_to_msgpack(MsgpackWriter &, const PassArg &);
PassSpec passspec_fields_from_msgpack(MsgpackReader &, size_t);
void passspec_fields_to_msgpack(MsgpackWriter &, const PassSpec &);
PassSpec passspec_from_msgpack(MsgpackReader &);
void passspec_to_msgpack(MsgpackWriter &, const PassSpec &);
PassPipeline passpipeline_fields_from_msgpack(MsgpackReader &, size_t);
void passpipeline_fields_to_msgpack(MsgpackWriter &, const PassPipeline &);
PassPipeline passpipeline_from_msgpack(MsgpackReader &);
void passpipeline_to_msgpack(MsgpackWriter &, const PassPipeline &);
PassRunResult passrunresult_fields_from_msgpack(MsgpackReader &, size_t);
void passrunresult_fields_to_msgpack(MsgpackWriter &, const PassRunResult &);
PassRunResult passrunresult_from_msgpack(MsgpackReader &);
void passrunresult_to_msgpack(MsgpackWriter &, const PassRunResult &);
CompileResult compileresult_fields_from_msgpack(MsgpackReader &, size_t);
void compileresult_fields_to_msgpack(MsgpackWriter &, const CompileResult &);
CompileResult compileresult_from_msgpack(MsgpackReader &);
void compileresult_to_msgpack(MsgpackWriter &, const CompileResult &);
namespace TypeKind {
TypeKind::None none_fields_from_msgpack(MsgpackReader &, size_t);
void none_fields_to_msgpack(MsgpackWriter &, const TypeKind::None &);
TypeKind::None none_from_msgpack(MsgpackReader &);
void none_to_msgpack(MsgpackWriter &, const TypeKind::None &);
TypeKind::Ref ref_fields_from_msgpack(MsgpackReader &, size_t);
void ref_fields_to_msgpack(MsgpackWriter &, const TypeKind::Ref &);
TypeKind::Ref ref_from_msgpack(MsgpackReader &);
void ref_to_msgpack(MsgpackWriter &, const TypeKind::Ref &);
TypeKind::Integral integral_fields_from_msgpack(MsgpackReader &, size_t);
void integral_fields_to_msgpack(MsgpackWriter &, const TypeKind::Integral &);
TypeKind::Integral integral_from_msgpack(MsgpackReader &);
void integral_to_msgpack(MsgpackWriter &, const TypeKind::Integral &);
TypeKind::Fractional fractional_fields_from_msgpack(MsgpackReader &, size_t);
void fractional_fields_to_msgpack(MsgpackWriter &, const TypeKind::Fractional &);
TypeKind::Fractional fractional_from_msgpack(MsgpackReader &);
void fractional_to_msgpack(MsgpackWriter &, const TypeKind::Fractional &);
TypeKind::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const TypeKind::Any &);
} // namespace TypeKind
namespace FunctionVisibility {
FunctionVisibility::Internal internal_fields_from_msgpack(MsgpackReader &, size_t);
void internal_fields_to_msgpack(MsgpackWriter &, const FunctionVisibility::Internal &);
FunctionVisibility::Internal internal_from_msgpack(MsgpackReader &);
void internal_to_msgpack(MsgpackWriter &, const FunctionVisibility::Internal &);
FunctionVisibility::Exported exported_fields_from_msgpack(MsgpackReader &, size_t);
void exported_fields_to_msgpack(MsgpackWriter &, const FunctionVisibility::Exported &);
FunctionVisibility::Exported exported_from_msgpack(MsgpackReader &);
void exported_to_msgpack(MsgpackWriter &, const FunctionVisibility::Exported &);
FunctionVisibility::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionVisibility::Any &);
} // namespace FunctionVisibility

Sym sym_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Sym with 1 field(s)");
  std::vector<std::string> fqn;
  {
    auto fqn_size = r_.readArrayHeader();
    fqn.reserve(fqn_size);
    for (size_t fqn_idx = 0; fqn_idx < fqn_size; ++fqn_idx) {
      auto fqn_elem = r_.readString();
      fqn.emplace_back(std::move(fqn_elem));
    }
  }
  return Sym(fqn);
}

void sym_fields_to_msgpack(MsgpackWriter &w_, const Sym &x_) {
  w_.writeArrayHeader(x_.fqn.size());
  for (const auto &v0_ : x_.fqn) {
    w_.writeString(v0_);
  }
}

Sym sym_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return sym_fields_from_msgpack(r_, n_);
}

void sym_to_msgpack(MsgpackWriter &w_, const Sym &x_) {
  w_.writeArrayHeader(1);
  sym_fields_to_msgpack(w_, x_);
}

SourcePosition sourceposition_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected SourcePosition with 3 field(s)");
  auto file = r_.readString();
  auto line = static_cast<int32_t>(r_.readInt32());
  std::optional<int32_t> col;
  if (!r_.tryReadNil()) {
    auto col_value = static_cast<int32_t>(r_.readInt32());
    col = std::move(col_value);
  }
  return {file, line, col};
}

void sourceposition_fields_to_msgpack(MsgpackWriter &w_, const SourcePosition &x_) {
  w_.writeString(x_.file);
  w_.writeInt32(static_cast<int32_t>(x_.line));
  if (x_.col) {
    w_.writeInt32(static_cast<int32_t>((*x_.col)));
  } else {
    w_.writeNil();
  }
}

SourcePosition sourceposition_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return sourceposition_fields_from_msgpack(r_, n_);
}

void sourceposition_to_msgpack(MsgpackWriter &w_, const SourcePosition &x_) {
  w_.writeArrayHeader(3);
  sourceposition_fields_to_msgpack(w_, x_);
}

Named named_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Named with 2 field(s)");
  auto symbol = r_.readString();
  auto tpe = Type::any_from_msgpack(r_);
  return {symbol, tpe};
}

void named_fields_to_msgpack(MsgpackWriter &w_, const Named &x_) {
  w_.writeString(x_.symbol);
  Type::any_to_msgpack(w_, x_.tpe);
}

Named named_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return named_fields_from_msgpack(r_, n_);
}

void named_to_msgpack(MsgpackWriter &w_, const Named &x_) {
  w_.writeArrayHeader(2);
  named_fields_to_msgpack(w_, x_);
}

TypeKind::None TypeKind::none_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeKind::None with 0 field(s)");
  return {};
}

void TypeKind::none_fields_to_msgpack(MsgpackWriter &w_, const TypeKind::None &x_) {}

TypeKind::None TypeKind::none_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeKind::none_fields_from_msgpack(r_, n_);
}

void TypeKind::none_to_msgpack(MsgpackWriter &w_, const TypeKind::None &x_) {
  w_.writeArrayHeader(0);
  TypeKind::none_fields_to_msgpack(w_, x_);
}

TypeKind::Ref TypeKind::ref_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeKind::Ref with 0 field(s)");
  return {};
}

void TypeKind::ref_fields_to_msgpack(MsgpackWriter &w_, const TypeKind::Ref &x_) {}

TypeKind::Ref TypeKind::ref_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeKind::ref_fields_from_msgpack(r_, n_);
}

void TypeKind::ref_to_msgpack(MsgpackWriter &w_, const TypeKind::Ref &x_) {
  w_.writeArrayHeader(0);
  TypeKind::ref_fields_to_msgpack(w_, x_);
}

TypeKind::Integral TypeKind::integral_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeKind::Integral with 0 field(s)");
  return {};
}

void TypeKind::integral_fields_to_msgpack(MsgpackWriter &w_, const TypeKind::Integral &x_) {}

TypeKind::Integral TypeKind::integral_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeKind::integral_fields_from_msgpack(r_, n_);
}

void TypeKind::integral_to_msgpack(MsgpackWriter &w_, const TypeKind::Integral &x_) {
  w_.writeArrayHeader(0);
  TypeKind::integral_fields_to_msgpack(w_, x_);
}

TypeKind::Fractional TypeKind::fractional_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeKind::Fractional with 0 field(s)");
  return {};
}

void TypeKind::fractional_fields_to_msgpack(MsgpackWriter &w_, const TypeKind::Fractional &x_) {}

TypeKind::Fractional TypeKind::fractional_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeKind::fractional_fields_from_msgpack(r_, n_);
}

void TypeKind::fractional_to_msgpack(MsgpackWriter &w_, const TypeKind::Fractional &x_) {
  w_.writeArrayHeader(0);
  TypeKind::fractional_fields_to_msgpack(w_, x_);
}

TypeKind::Any TypeKind::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeKind::none_fields_from_msgpack(r_, n_ - 1);
      case 1: return TypeKind::ref_fields_from_msgpack(r_, n_ - 1);
      case 2: return TypeKind::integral_fields_from_msgpack(r_, n_ - 1);
      case 3: return TypeKind::fractional_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeKind::none_fields_from_msgpack(r_, 0);
      case 1: return TypeKind::ref_fields_from_msgpack(r_, 0);
      case 2: return TypeKind::integral_fields_from_msgpack(r_, 0);
      case 3: return TypeKind::fractional_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void TypeKind::any_to_msgpack(MsgpackWriter &w_, const TypeKind::Any &x_) {
  x_.match_total([&](const TypeKind::None &y_) -> void { w_.writeInt32(0); }, [&](const TypeKind::Ref &y_) -> void { w_.writeInt32(1); },
                 [&](const TypeKind::Integral &y_) -> void { w_.writeInt32(2); },
                 [&](const TypeKind::Fractional &y_) -> void { w_.writeInt32(3); });
}

TypeSpace::Global TypeSpace::global_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Global with 0 field(s)");
  return {};
}

void TypeSpace::global_fields_to_msgpack(MsgpackWriter &w_, const TypeSpace::Global &x_) {}

TypeSpace::Global TypeSpace::global_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeSpace::global_fields_from_msgpack(r_, n_);
}

void TypeSpace::global_to_msgpack(MsgpackWriter &w_, const TypeSpace::Global &x_) {
  w_.writeArrayHeader(0);
  TypeSpace::global_fields_to_msgpack(w_, x_);
}

TypeSpace::Local TypeSpace::local_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Local with 0 field(s)");
  return {};
}

void TypeSpace::local_fields_to_msgpack(MsgpackWriter &w_, const TypeSpace::Local &x_) {}

TypeSpace::Local TypeSpace::local_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeSpace::local_fields_from_msgpack(r_, n_);
}

void TypeSpace::local_to_msgpack(MsgpackWriter &w_, const TypeSpace::Local &x_) {
  w_.writeArrayHeader(0);
  TypeSpace::local_fields_to_msgpack(w_, x_);
}

TypeSpace::Private TypeSpace::private_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Private with 0 field(s)");
  return {};
}

void TypeSpace::private_fields_to_msgpack(MsgpackWriter &w_, const TypeSpace::Private &x_) {}

TypeSpace::Private TypeSpace::private_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeSpace::private_fields_from_msgpack(r_, n_);
}

void TypeSpace::private_to_msgpack(MsgpackWriter &w_, const TypeSpace::Private &x_) {
  w_.writeArrayHeader(0);
  TypeSpace::private_fields_to_msgpack(w_, x_);
}

TypeSpace::Constant TypeSpace::constant_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Constant with 0 field(s)");
  return {};
}

void TypeSpace::constant_fields_to_msgpack(MsgpackWriter &w_, const TypeSpace::Constant &x_) {}

TypeSpace::Constant TypeSpace::constant_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return TypeSpace::constant_fields_from_msgpack(r_, n_);
}

void TypeSpace::constant_to_msgpack(MsgpackWriter &w_, const TypeSpace::Constant &x_) {
  w_.writeArrayHeader(0);
  TypeSpace::constant_fields_to_msgpack(w_, x_);
}

TypeSpace::Any TypeSpace::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeSpace::global_fields_from_msgpack(r_, n_ - 1);
      case 1: return TypeSpace::local_fields_from_msgpack(r_, n_ - 1);
      case 2: return TypeSpace::private_fields_from_msgpack(r_, n_ - 1);
      case 3: return TypeSpace::constant_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeSpace::global_fields_from_msgpack(r_, 0);
      case 1: return TypeSpace::local_fields_from_msgpack(r_, 0);
      case 2: return TypeSpace::private_fields_from_msgpack(r_, 0);
      case 3: return TypeSpace::constant_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void TypeSpace::any_to_msgpack(MsgpackWriter &w_, const TypeSpace::Any &x_) {
  x_.match_total(
      [&](const TypeSpace::Global &y_) -> void { w_.writeInt32(0); }, [&](const TypeSpace::Local &y_) -> void { w_.writeInt32(1); },
      [&](const TypeSpace::Private &y_) -> void { w_.writeInt32(2); }, [&](const TypeSpace::Constant &y_) -> void { w_.writeInt32(3); });
}

Type::Float16 Type::float16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float16 with 0 field(s)");
  return {};
}

void Type::float16_fields_to_msgpack(MsgpackWriter &w_, const Type::Float16 &x_) {}

Type::Float16 Type::float16_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::float16_fields_from_msgpack(r_, n_);
}

void Type::float16_to_msgpack(MsgpackWriter &w_, const Type::Float16 &x_) {
  w_.writeArrayHeader(0);
  Type::float16_fields_to_msgpack(w_, x_);
}

Type::Float32 Type::float32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float32 with 0 field(s)");
  return {};
}

void Type::float32_fields_to_msgpack(MsgpackWriter &w_, const Type::Float32 &x_) {}

Type::Float32 Type::float32_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::float32_fields_from_msgpack(r_, n_);
}

void Type::float32_to_msgpack(MsgpackWriter &w_, const Type::Float32 &x_) {
  w_.writeArrayHeader(0);
  Type::float32_fields_to_msgpack(w_, x_);
}

Type::Float64 Type::float64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float64 with 0 field(s)");
  return {};
}

void Type::float64_fields_to_msgpack(MsgpackWriter &w_, const Type::Float64 &x_) {}

Type::Float64 Type::float64_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::float64_fields_from_msgpack(r_, n_);
}

void Type::float64_to_msgpack(MsgpackWriter &w_, const Type::Float64 &x_) {
  w_.writeArrayHeader(0);
  Type::float64_fields_to_msgpack(w_, x_);
}

Type::IntU8 Type::intu8_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU8 with 0 field(s)");
  return {};
}

void Type::intu8_fields_to_msgpack(MsgpackWriter &w_, const Type::IntU8 &x_) {}

Type::IntU8 Type::intu8_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::intu8_fields_from_msgpack(r_, n_);
}

void Type::intu8_to_msgpack(MsgpackWriter &w_, const Type::IntU8 &x_) {
  w_.writeArrayHeader(0);
  Type::intu8_fields_to_msgpack(w_, x_);
}

Type::IntU16 Type::intu16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU16 with 0 field(s)");
  return {};
}

void Type::intu16_fields_to_msgpack(MsgpackWriter &w_, const Type::IntU16 &x_) {}

Type::IntU16 Type::intu16_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::intu16_fields_from_msgpack(r_, n_);
}

void Type::intu16_to_msgpack(MsgpackWriter &w_, const Type::IntU16 &x_) {
  w_.writeArrayHeader(0);
  Type::intu16_fields_to_msgpack(w_, x_);
}

Type::IntU32 Type::intu32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU32 with 0 field(s)");
  return {};
}

void Type::intu32_fields_to_msgpack(MsgpackWriter &w_, const Type::IntU32 &x_) {}

Type::IntU32 Type::intu32_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::intu32_fields_from_msgpack(r_, n_);
}

void Type::intu32_to_msgpack(MsgpackWriter &w_, const Type::IntU32 &x_) {
  w_.writeArrayHeader(0);
  Type::intu32_fields_to_msgpack(w_, x_);
}

Type::IntU64 Type::intu64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU64 with 0 field(s)");
  return {};
}

void Type::intu64_fields_to_msgpack(MsgpackWriter &w_, const Type::IntU64 &x_) {}

Type::IntU64 Type::intu64_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::intu64_fields_from_msgpack(r_, n_);
}

void Type::intu64_to_msgpack(MsgpackWriter &w_, const Type::IntU64 &x_) {
  w_.writeArrayHeader(0);
  Type::intu64_fields_to_msgpack(w_, x_);
}

Type::IntS8 Type::ints8_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS8 with 0 field(s)");
  return {};
}

void Type::ints8_fields_to_msgpack(MsgpackWriter &w_, const Type::IntS8 &x_) {}

Type::IntS8 Type::ints8_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::ints8_fields_from_msgpack(r_, n_);
}

void Type::ints8_to_msgpack(MsgpackWriter &w_, const Type::IntS8 &x_) {
  w_.writeArrayHeader(0);
  Type::ints8_fields_to_msgpack(w_, x_);
}

Type::IntS16 Type::ints16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS16 with 0 field(s)");
  return {};
}

void Type::ints16_fields_to_msgpack(MsgpackWriter &w_, const Type::IntS16 &x_) {}

Type::IntS16 Type::ints16_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::ints16_fields_from_msgpack(r_, n_);
}

void Type::ints16_to_msgpack(MsgpackWriter &w_, const Type::IntS16 &x_) {
  w_.writeArrayHeader(0);
  Type::ints16_fields_to_msgpack(w_, x_);
}

Type::IntS32 Type::ints32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS32 with 0 field(s)");
  return {};
}

void Type::ints32_fields_to_msgpack(MsgpackWriter &w_, const Type::IntS32 &x_) {}

Type::IntS32 Type::ints32_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::ints32_fields_from_msgpack(r_, n_);
}

void Type::ints32_to_msgpack(MsgpackWriter &w_, const Type::IntS32 &x_) {
  w_.writeArrayHeader(0);
  Type::ints32_fields_to_msgpack(w_, x_);
}

Type::IntS64 Type::ints64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS64 with 0 field(s)");
  return {};
}

void Type::ints64_fields_to_msgpack(MsgpackWriter &w_, const Type::IntS64 &x_) {}

Type::IntS64 Type::ints64_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::ints64_fields_from_msgpack(r_, n_);
}

void Type::ints64_to_msgpack(MsgpackWriter &w_, const Type::IntS64 &x_) {
  w_.writeArrayHeader(0);
  Type::ints64_fields_to_msgpack(w_, x_);
}

Type::Nothing Type::nothing_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Nothing with 0 field(s)");
  return {};
}

void Type::nothing_fields_to_msgpack(MsgpackWriter &w_, const Type::Nothing &x_) {}

Type::Nothing Type::nothing_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::nothing_fields_from_msgpack(r_, n_);
}

void Type::nothing_to_msgpack(MsgpackWriter &w_, const Type::Nothing &x_) {
  w_.writeArrayHeader(0);
  Type::nothing_fields_to_msgpack(w_, x_);
}

Type::Unit0 Type::unit0_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Unit0 with 0 field(s)");
  return {};
}

void Type::unit0_fields_to_msgpack(MsgpackWriter &w_, const Type::Unit0 &x_) {}

Type::Unit0 Type::unit0_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::unit0_fields_from_msgpack(r_, n_);
}

void Type::unit0_to_msgpack(MsgpackWriter &w_, const Type::Unit0 &x_) {
  w_.writeArrayHeader(0);
  Type::unit0_fields_to_msgpack(w_, x_);
}

Type::Bool1 Type::bool1_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Bool1 with 0 field(s)");
  return {};
}

void Type::bool1_fields_to_msgpack(MsgpackWriter &w_, const Type::Bool1 &x_) {}

Type::Bool1 Type::bool1_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::bool1_fields_from_msgpack(r_, n_);
}

void Type::bool1_to_msgpack(MsgpackWriter &w_, const Type::Bool1 &x_) {
  w_.writeArrayHeader(0);
  Type::bool1_fields_to_msgpack(w_, x_);
}

Type::Struct Type::struct_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Type::Struct with 2 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  return {name, args};
}

void Type::struct_fields_to_msgpack(MsgpackWriter &w_, const Type::Struct &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
}

Type::Struct Type::struct_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::struct_fields_from_msgpack(r_, n_);
}

void Type::struct_to_msgpack(MsgpackWriter &w_, const Type::Struct &x_) {
  w_.writeArrayHeader(2);
  Type::struct_fields_to_msgpack(w_, x_);
}

Type::Ptr Type::ptr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Type::Ptr with 2 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  return {comp, space};
}

void Type::ptr_fields_to_msgpack(MsgpackWriter &w_, const Type::Ptr &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
}

Type::Ptr Type::ptr_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::ptr_fields_from_msgpack(r_, n_);
}

void Type::ptr_to_msgpack(MsgpackWriter &w_, const Type::Ptr &x_) {
  w_.writeArrayHeader(2);
  Type::ptr_fields_to_msgpack(w_, x_);
}

Type::Arr Type::arr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Type::Arr with 3 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto length = static_cast<int32_t>(r_.readInt32());
  auto space = TypeSpace::any_from_msgpack(r_);
  return {comp, length, space};
}

void Type::arr_fields_to_msgpack(MsgpackWriter &w_, const Type::Arr &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  w_.writeInt32(static_cast<int32_t>(x_.length));
  TypeSpace::any_to_msgpack(w_, x_.space);
}

Type::Arr Type::arr_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::arr_fields_from_msgpack(r_, n_);
}

void Type::arr_to_msgpack(MsgpackWriter &w_, const Type::Arr &x_) {
  w_.writeArrayHeader(3);
  Type::arr_fields_to_msgpack(w_, x_);
}

Type::Var Type::var_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Type::Var with 1 field(s)");
  auto name = r_.readString();
  return Type::Var(name);
}

void Type::var_fields_to_msgpack(MsgpackWriter &w_, const Type::Var &x_) { w_.writeString(x_.name); }

Type::Var Type::var_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::var_fields_from_msgpack(r_, n_);
}

void Type::var_to_msgpack(MsgpackWriter &w_, const Type::Var &x_) {
  w_.writeArrayHeader(1);
  Type::var_fields_to_msgpack(w_, x_);
}

Type::Exec Type::exec_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Type::Exec with 3 field(s)");
  std::vector<std::string> tpeVars;
  {
    auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = r_.readString();
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::vector<Type::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {tpeVars, args, rtn};
}

void Type::exec_fields_to_msgpack(MsgpackWriter &w_, const Type::Exec &x_) {
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Type::Exec Type::exec_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Type::exec_fields_from_msgpack(r_, n_);
}

void Type::exec_to_msgpack(MsgpackWriter &w_, const Type::Exec &x_) {
  w_.writeArrayHeader(3);
  Type::exec_fields_to_msgpack(w_, x_);
}

Type::Any Type::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Type::float16_fields_from_msgpack(r_, n_ - 1);
      case 1: return Type::float32_fields_from_msgpack(r_, n_ - 1);
      case 2: return Type::float64_fields_from_msgpack(r_, n_ - 1);
      case 3: return Type::intu8_fields_from_msgpack(r_, n_ - 1);
      case 4: return Type::intu16_fields_from_msgpack(r_, n_ - 1);
      case 5: return Type::intu32_fields_from_msgpack(r_, n_ - 1);
      case 6: return Type::intu64_fields_from_msgpack(r_, n_ - 1);
      case 7: return Type::ints8_fields_from_msgpack(r_, n_ - 1);
      case 8: return Type::ints16_fields_from_msgpack(r_, n_ - 1);
      case 9: return Type::ints32_fields_from_msgpack(r_, n_ - 1);
      case 10: return Type::ints64_fields_from_msgpack(r_, n_ - 1);
      case 11: return Type::nothing_fields_from_msgpack(r_, n_ - 1);
      case 12: return Type::unit0_fields_from_msgpack(r_, n_ - 1);
      case 13: return Type::bool1_fields_from_msgpack(r_, n_ - 1);
      case 14: return Type::struct_fields_from_msgpack(r_, n_ - 1);
      case 15: return Type::ptr_fields_from_msgpack(r_, n_ - 1);
      case 16: return Type::arr_fields_from_msgpack(r_, n_ - 1);
      case 17: return Type::var_fields_from_msgpack(r_, n_ - 1);
      case 18: return Type::exec_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Type::float16_fields_from_msgpack(r_, 0);
      case 1: return Type::float32_fields_from_msgpack(r_, 0);
      case 2: return Type::float64_fields_from_msgpack(r_, 0);
      case 3: return Type::intu8_fields_from_msgpack(r_, 0);
      case 4: return Type::intu16_fields_from_msgpack(r_, 0);
      case 5: return Type::intu32_fields_from_msgpack(r_, 0);
      case 6: return Type::intu64_fields_from_msgpack(r_, 0);
      case 7: return Type::ints8_fields_from_msgpack(r_, 0);
      case 8: return Type::ints16_fields_from_msgpack(r_, 0);
      case 9: return Type::ints32_fields_from_msgpack(r_, 0);
      case 10: return Type::ints64_fields_from_msgpack(r_, 0);
      case 11: return Type::nothing_fields_from_msgpack(r_, 0);
      case 12: return Type::unit0_fields_from_msgpack(r_, 0);
      case 13: return Type::bool1_fields_from_msgpack(r_, 0);
      case 14: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 15: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 16: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 17: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 18: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Type::any_to_msgpack(MsgpackWriter &w_, const Type::Any &x_) {
  x_.match_total([&](const Type::Float16 &y_) -> void { w_.writeInt32(0); }, [&](const Type::Float32 &y_) -> void { w_.writeInt32(1); },
                 [&](const Type::Float64 &y_) -> void { w_.writeInt32(2); }, [&](const Type::IntU8 &y_) -> void { w_.writeInt32(3); },
                 [&](const Type::IntU16 &y_) -> void { w_.writeInt32(4); }, [&](const Type::IntU32 &y_) -> void { w_.writeInt32(5); },
                 [&](const Type::IntU64 &y_) -> void { w_.writeInt32(6); }, [&](const Type::IntS8 &y_) -> void { w_.writeInt32(7); },
                 [&](const Type::IntS16 &y_) -> void { w_.writeInt32(8); }, [&](const Type::IntS32 &y_) -> void { w_.writeInt32(9); },
                 [&](const Type::IntS64 &y_) -> void { w_.writeInt32(10); }, [&](const Type::Nothing &y_) -> void { w_.writeInt32(11); },
                 [&](const Type::Unit0 &y_) -> void { w_.writeInt32(12); }, [&](const Type::Bool1 &y_) -> void { w_.writeInt32(13); },
                 [&](const Type::Struct &y_) -> void {
                   w_.writeArrayHeader(3);
                   w_.writeInt32(14);
                   Type::struct_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Ptr &y_) -> void {
                   w_.writeArrayHeader(3);
                   w_.writeInt32(15);
                   Type::ptr_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Arr &y_) -> void {
                   w_.writeArrayHeader(4);
                   w_.writeInt32(16);
                   Type::arr_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Var &y_) -> void {
                   w_.writeArrayHeader(2);
                   w_.writeInt32(17);
                   Type::var_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Exec &y_) -> void {
                   w_.writeArrayHeader(4);
                   w_.writeInt32(18);
                   Type::exec_fields_to_msgpack(w_, y_);
                 });
}

PathStep::Field PathStep::field_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::Field with 1 field(s)");
  auto name = r_.readString();
  return PathStep::Field(name);
}

void PathStep::field_fields_to_msgpack(MsgpackWriter &w_, const PathStep::Field &x_) { w_.writeString(x_.name); }

PathStep::Field PathStep::field_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PathStep::field_fields_from_msgpack(r_, n_);
}

void PathStep::field_to_msgpack(MsgpackWriter &w_, const PathStep::Field &x_) {
  w_.writeArrayHeader(1);
  PathStep::field_fields_to_msgpack(w_, x_);
}

PathStep::Deref PathStep::deref_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PathStep::Deref with 0 field(s)");
  return {};
}

void PathStep::deref_fields_to_msgpack(MsgpackWriter &w_, const PathStep::Deref &x_) {}

PathStep::Deref PathStep::deref_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PathStep::deref_fields_from_msgpack(r_, n_);
}

void PathStep::deref_to_msgpack(MsgpackWriter &w_, const PathStep::Deref &x_) {
  w_.writeArrayHeader(0);
  PathStep::deref_fields_to_msgpack(w_, x_);
}

PathStep::Index PathStep::index_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::Index with 1 field(s)");
  auto idx = static_cast<int32_t>(r_.readInt32());
  return PathStep::Index(idx);
}

void PathStep::index_fields_to_msgpack(MsgpackWriter &w_, const PathStep::Index &x_) { w_.writeInt32(static_cast<int32_t>(x_.idx)); }

PathStep::Index PathStep::index_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PathStep::index_fields_from_msgpack(r_, n_);
}

void PathStep::index_to_msgpack(MsgpackWriter &w_, const PathStep::Index &x_) {
  w_.writeArrayHeader(1);
  PathStep::index_fields_to_msgpack(w_, x_);
}

PathStep::IndexDyn PathStep::indexdyn_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::IndexDyn with 1 field(s)");
  auto idx = Term::any_from_msgpack(r_);
  return PathStep::IndexDyn(idx);
}

void PathStep::indexdyn_fields_to_msgpack(MsgpackWriter &w_, const PathStep::IndexDyn &x_) { Term::any_to_msgpack(w_, x_.idx); }

PathStep::IndexDyn PathStep::indexdyn_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PathStep::indexdyn_fields_from_msgpack(r_, n_);
}

void PathStep::indexdyn_to_msgpack(MsgpackWriter &w_, const PathStep::IndexDyn &x_) {
  w_.writeArrayHeader(1);
  PathStep::indexdyn_fields_to_msgpack(w_, x_);
}

PathStep::Any PathStep::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PathStep::field_fields_from_msgpack(r_, n_ - 1);
      case 1: return PathStep::deref_fields_from_msgpack(r_, n_ - 1);
      case 2: return PathStep::index_fields_from_msgpack(r_, n_ - 1);
      case 3: return PathStep::indexdyn_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: return PathStep::deref_fields_from_msgpack(r_, 0);
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void PathStep::any_to_msgpack(MsgpackWriter &w_, const PathStep::Any &x_) {
  x_.match_total(
      [&](const PathStep::Field &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        PathStep::field_fields_to_msgpack(w_, y_);
      },
      [&](const PathStep::Deref &y_) -> void { w_.writeInt32(1); },
      [&](const PathStep::Index &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        PathStep::index_fields_to_msgpack(w_, y_);
      },
      [&](const PathStep::IndexDyn &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        PathStep::indexdyn_fields_to_msgpack(w_, y_);
      });
}

Region::Rooted Region::rooted_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Region::Rooted with 1 field(s)");
  auto root = named_from_msgpack(r_);
  return Region::Rooted(root);
}

void Region::rooted_fields_to_msgpack(MsgpackWriter &w_, const Region::Rooted &x_) { named_to_msgpack(w_, x_.root); }

Region::Rooted Region::rooted_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Region::rooted_fields_from_msgpack(r_, n_);
}

void Region::rooted_to_msgpack(MsgpackWriter &w_, const Region::Rooted &x_) {
  w_.writeArrayHeader(1);
  Region::rooted_fields_to_msgpack(w_, x_);
}

Region::Opaque Region::opaque_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Region::Opaque with 0 field(s)");
  return {};
}

void Region::opaque_fields_to_msgpack(MsgpackWriter &w_, const Region::Opaque &x_) {}

Region::Opaque Region::opaque_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Region::opaque_fields_from_msgpack(r_, n_);
}

void Region::opaque_to_msgpack(MsgpackWriter &w_, const Region::Opaque &x_) {
  w_.writeArrayHeader(0);
  Region::opaque_fields_to_msgpack(w_, x_);
}

Region::Any Region::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Region::rooted_fields_from_msgpack(r_, n_ - 1);
      case 1: return Region::opaque_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: return Region::opaque_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Region::any_to_msgpack(MsgpackWriter &w_, const Region::Any &x_) {
  x_.match_total(
      [&](const Region::Rooted &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Region::rooted_fields_to_msgpack(w_, y_);
      },
      [&](const Region::Opaque &y_) -> void { w_.writeInt32(1); });
}

Term::Float16Const Term::float16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float16Const with 1 field(s)");
  auto value = r_.readFloat32();
  return Term::Float16Const(value);
}

void Term::float16const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float16Const &x_) { w_.writeFloat32(x_.value); }

Term::Float16Const Term::float16const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::float16const_fields_from_msgpack(r_, n_);
}

void Term::float16const_to_msgpack(MsgpackWriter &w_, const Term::Float16Const &x_) {
  w_.writeArrayHeader(1);
  Term::float16const_fields_to_msgpack(w_, x_);
}

Term::Float32Const Term::float32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float32Const with 1 field(s)");
  auto value = r_.readFloat32();
  return Term::Float32Const(value);
}

void Term::float32const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float32Const &x_) { w_.writeFloat32(x_.value); }

Term::Float32Const Term::float32const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::float32const_fields_from_msgpack(r_, n_);
}

void Term::float32const_to_msgpack(MsgpackWriter &w_, const Term::Float32Const &x_) {
  w_.writeArrayHeader(1);
  Term::float32const_fields_to_msgpack(w_, x_);
}

Term::Float64Const Term::float64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float64Const with 1 field(s)");
  auto value = r_.readFloat64();
  return Term::Float64Const(value);
}

void Term::float64const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float64Const &x_) { w_.writeFloat64(x_.value); }

Term::Float64Const Term::float64const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::float64const_fields_from_msgpack(r_, n_);
}

void Term::float64const_to_msgpack(MsgpackWriter &w_, const Term::Float64Const &x_) {
  w_.writeArrayHeader(1);
  Term::float64const_fields_to_msgpack(w_, x_);
}

Term::IntU8Const Term::intu8const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU8Const with 1 field(s)");
  auto value = static_cast<int8_t>(r_.readInt32());
  return Term::IntU8Const(value);
}

void Term::intu8const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU8Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntU8Const Term::intu8const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::intu8const_fields_from_msgpack(r_, n_);
}

void Term::intu8const_to_msgpack(MsgpackWriter &w_, const Term::IntU8Const &x_) {
  w_.writeArrayHeader(1);
  Term::intu8const_fields_to_msgpack(w_, x_);
}

Term::IntU16Const Term::intu16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU16Const with 1 field(s)");
  auto value = static_cast<uint16_t>(r_.readInt32());
  return Term::IntU16Const(value);
}

void Term::intu16const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU16Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntU16Const Term::intu16const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::intu16const_fields_from_msgpack(r_, n_);
}

void Term::intu16const_to_msgpack(MsgpackWriter &w_, const Term::IntU16Const &x_) {
  w_.writeArrayHeader(1);
  Term::intu16const_fields_to_msgpack(w_, x_);
}

Term::IntU32Const Term::intu32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU32Const with 1 field(s)");
  auto value = static_cast<int32_t>(r_.readInt32());
  return Term::IntU32Const(value);
}

void Term::intu32const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU32Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntU32Const Term::intu32const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::intu32const_fields_from_msgpack(r_, n_);
}

void Term::intu32const_to_msgpack(MsgpackWriter &w_, const Term::IntU32Const &x_) {
  w_.writeArrayHeader(1);
  Term::intu32const_fields_to_msgpack(w_, x_);
}

Term::IntU64Const Term::intu64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU64Const with 1 field(s)");
  auto value = r_.readInt64();
  return Term::IntU64Const(value);
}

void Term::intu64const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU64Const &x_) { w_.writeInt64(x_.value); }

Term::IntU64Const Term::intu64const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::intu64const_fields_from_msgpack(r_, n_);
}

void Term::intu64const_to_msgpack(MsgpackWriter &w_, const Term::IntU64Const &x_) {
  w_.writeArrayHeader(1);
  Term::intu64const_fields_to_msgpack(w_, x_);
}

Term::IntS8Const Term::ints8const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS8Const with 1 field(s)");
  auto value = static_cast<int8_t>(r_.readInt32());
  return Term::IntS8Const(value);
}

void Term::ints8const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS8Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntS8Const Term::ints8const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::ints8const_fields_from_msgpack(r_, n_);
}

void Term::ints8const_to_msgpack(MsgpackWriter &w_, const Term::IntS8Const &x_) {
  w_.writeArrayHeader(1);
  Term::ints8const_fields_to_msgpack(w_, x_);
}

Term::IntS16Const Term::ints16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS16Const with 1 field(s)");
  auto value = static_cast<int16_t>(r_.readInt32());
  return Term::IntS16Const(value);
}

void Term::ints16const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS16Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntS16Const Term::ints16const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::ints16const_fields_from_msgpack(r_, n_);
}

void Term::ints16const_to_msgpack(MsgpackWriter &w_, const Term::IntS16Const &x_) {
  w_.writeArrayHeader(1);
  Term::ints16const_fields_to_msgpack(w_, x_);
}

Term::IntS32Const Term::ints32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS32Const with 1 field(s)");
  auto value = static_cast<int32_t>(r_.readInt32());
  return Term::IntS32Const(value);
}

void Term::ints32const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS32Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntS32Const Term::ints32const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::ints32const_fields_from_msgpack(r_, n_);
}

void Term::ints32const_to_msgpack(MsgpackWriter &w_, const Term::IntS32Const &x_) {
  w_.writeArrayHeader(1);
  Term::ints32const_fields_to_msgpack(w_, x_);
}

Term::IntS64Const Term::ints64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS64Const with 1 field(s)");
  auto value = r_.readInt64();
  return Term::IntS64Const(value);
}

void Term::ints64const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS64Const &x_) { w_.writeInt64(x_.value); }

Term::IntS64Const Term::ints64const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::ints64const_fields_from_msgpack(r_, n_);
}

void Term::ints64const_to_msgpack(MsgpackWriter &w_, const Term::IntS64Const &x_) {
  w_.writeArrayHeader(1);
  Term::ints64const_fields_to_msgpack(w_, x_);
}

Term::Unit0Const Term::unit0const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Term::Unit0Const with 0 field(s)");
  return {};
}

void Term::unit0const_fields_to_msgpack(MsgpackWriter &w_, const Term::Unit0Const &x_) {}

Term::Unit0Const Term::unit0const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::unit0const_fields_from_msgpack(r_, n_);
}

void Term::unit0const_to_msgpack(MsgpackWriter &w_, const Term::Unit0Const &x_) {
  w_.writeArrayHeader(0);
  Term::unit0const_fields_to_msgpack(w_, x_);
}

Term::Bool1Const Term::bool1const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Bool1Const with 1 field(s)");
  auto value = r_.readBoolean();
  return Term::Bool1Const(value);
}

void Term::bool1const_fields_to_msgpack(MsgpackWriter &w_, const Term::Bool1Const &x_) { w_.writeBoolean(x_.value); }

Term::Bool1Const Term::bool1const_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::bool1const_fields_from_msgpack(r_, n_);
}

void Term::bool1const_to_msgpack(MsgpackWriter &w_, const Term::Bool1Const &x_) {
  w_.writeArrayHeader(1);
  Term::bool1const_fields_to_msgpack(w_, x_);
}

Term::NullPtrConst Term::nullptrconst_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Term::NullPtrConst with 3 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {comp, space, region};
}

void Term::nullptrconst_fields_to_msgpack(MsgpackWriter &w_, const Term::NullPtrConst &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Term::NullPtrConst Term::nullptrconst_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::nullptrconst_fields_from_msgpack(r_, n_);
}

void Term::nullptrconst_to_msgpack(MsgpackWriter &w_, const Term::NullPtrConst &x_) {
  w_.writeArrayHeader(3);
  Term::nullptrconst_fields_to_msgpack(w_, x_);
}

Term::Poison Term::poison_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Poison with 1 field(s)");
  auto t = Type::any_from_msgpack(r_);
  return Term::Poison(t);
}

void Term::poison_fields_to_msgpack(MsgpackWriter &w_, const Term::Poison &x_) { Type::any_to_msgpack(w_, x_.t); }

Term::Poison Term::poison_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::poison_fields_from_msgpack(r_, n_);
}

void Term::poison_to_msgpack(MsgpackWriter &w_, const Term::Poison &x_) {
  w_.writeArrayHeader(1);
  Term::poison_fields_to_msgpack(w_, x_);
}

Term::Select Term::select_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Term::Select with 3 field(s)");
  auto root = named_from_msgpack(r_);
  std::vector<PathStep::Any> steps;
  {
    auto steps_size = r_.readArrayHeader();
    steps.reserve(steps_size);
    for (size_t steps_idx = 0; steps_idx < steps_size; ++steps_idx) {
      auto steps_elem = PathStep::any_from_msgpack(r_);
      steps.emplace_back(std::move(steps_elem));
    }
  }
  auto tpe = Type::any_from_msgpack(r_);
  return {root, steps, tpe};
}

void Term::select_fields_to_msgpack(MsgpackWriter &w_, const Term::Select &x_) {
  named_to_msgpack(w_, x_.root);
  w_.writeArrayHeader(x_.steps.size());
  for (const auto &v0_ : x_.steps) {
    PathStep::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.tpe);
}

Term::Select Term::select_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Term::select_fields_from_msgpack(r_, n_);
}

void Term::select_to_msgpack(MsgpackWriter &w_, const Term::Select &x_) {
  w_.writeArrayHeader(3);
  Term::select_fields_to_msgpack(w_, x_);
}

Term::Any Term::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Term::float16const_fields_from_msgpack(r_, n_ - 1);
      case 1: return Term::float32const_fields_from_msgpack(r_, n_ - 1);
      case 2: return Term::float64const_fields_from_msgpack(r_, n_ - 1);
      case 3: return Term::intu8const_fields_from_msgpack(r_, n_ - 1);
      case 4: return Term::intu16const_fields_from_msgpack(r_, n_ - 1);
      case 5: return Term::intu32const_fields_from_msgpack(r_, n_ - 1);
      case 6: return Term::intu64const_fields_from_msgpack(r_, n_ - 1);
      case 7: return Term::ints8const_fields_from_msgpack(r_, n_ - 1);
      case 8: return Term::ints16const_fields_from_msgpack(r_, n_ - 1);
      case 9: return Term::ints32const_fields_from_msgpack(r_, n_ - 1);
      case 10: return Term::ints64const_fields_from_msgpack(r_, n_ - 1);
      case 11: return Term::unit0const_fields_from_msgpack(r_, n_ - 1);
      case 12: return Term::bool1const_fields_from_msgpack(r_, n_ - 1);
      case 13: return Term::nullptrconst_fields_from_msgpack(r_, n_ - 1);
      case 14: return Term::poison_fields_from_msgpack(r_, n_ - 1);
      case 15: return Term::select_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 4: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 6: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 10: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: return Term::unit0const_fields_from_msgpack(r_, 0);
      case 12: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 13: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 14: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 15: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Term::any_to_msgpack(MsgpackWriter &w_, const Term::Any &x_) {
  x_.match_total(
      [&](const Term::Float16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Term::float16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Float32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Term::float32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Float64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        Term::float64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU8Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        Term::intu8const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(4);
        Term::intu16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(5);
        Term::intu32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(6);
        Term::intu64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS8Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(7);
        Term::ints8const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Term::ints16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(9);
        Term::ints32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(10);
        Term::ints64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Unit0Const &y_) -> void { w_.writeInt32(11); },
      [&](const Term::Bool1Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(12);
        Term::bool1const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::NullPtrConst &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(13);
        Term::nullptrconst_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Poison &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(14);
        Term::poison_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Select &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(15);
        Term::select_fields_to_msgpack(w_, y_);
      });
}

Expr::Alias Expr::alias_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::Alias with 1 field(s)");
  auto ref = Term::any_from_msgpack(r_);
  return Expr::Alias(ref);
}

void Expr::alias_fields_to_msgpack(MsgpackWriter &w_, const Expr::Alias &x_) { Term::any_to_msgpack(w_, x_.ref); }

Expr::Alias Expr::alias_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::alias_fields_from_msgpack(r_, n_);
}

void Expr::alias_to_msgpack(MsgpackWriter &w_, const Expr::Alias &x_) {
  w_.writeArrayHeader(1);
  Expr::alias_fields_to_msgpack(w_, x_);
}

Expr::SpecOp Expr::specop_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::SpecOp with 1 field(s)");
  auto op = Spec::any_from_msgpack(r_);
  return Expr::SpecOp(op);
}

void Expr::specop_fields_to_msgpack(MsgpackWriter &w_, const Expr::SpecOp &x_) { Spec::any_to_msgpack(w_, x_.op); }

Expr::SpecOp Expr::specop_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::specop_fields_from_msgpack(r_, n_);
}

void Expr::specop_to_msgpack(MsgpackWriter &w_, const Expr::SpecOp &x_) {
  w_.writeArrayHeader(1);
  Expr::specop_fields_to_msgpack(w_, x_);
}

Expr::MathOp Expr::mathop_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::MathOp with 1 field(s)");
  auto op = Math::any_from_msgpack(r_);
  return Expr::MathOp(op);
}

void Expr::mathop_fields_to_msgpack(MsgpackWriter &w_, const Expr::MathOp &x_) { Math::any_to_msgpack(w_, x_.op); }

Expr::MathOp Expr::mathop_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::mathop_fields_from_msgpack(r_, n_);
}

void Expr::mathop_to_msgpack(MsgpackWriter &w_, const Expr::MathOp &x_) {
  w_.writeArrayHeader(1);
  Expr::mathop_fields_to_msgpack(w_, x_);
}

Expr::IntrOp Expr::introp_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::IntrOp with 1 field(s)");
  auto op = Intr::any_from_msgpack(r_);
  return Expr::IntrOp(op);
}

void Expr::introp_fields_to_msgpack(MsgpackWriter &w_, const Expr::IntrOp &x_) { Intr::any_to_msgpack(w_, x_.op); }

Expr::IntrOp Expr::introp_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::introp_fields_from_msgpack(r_, n_);
}

void Expr::introp_to_msgpack(MsgpackWriter &w_, const Expr::IntrOp &x_) {
  w_.writeArrayHeader(1);
  Expr::introp_fields_to_msgpack(w_, x_);
}

Expr::Cast Expr::cast_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Expr::Cast with 2 field(s)");
  auto from = Term::any_from_msgpack(r_);
  auto as = Type::any_from_msgpack(r_);
  return {from, as};
}

void Expr::cast_fields_to_msgpack(MsgpackWriter &w_, const Expr::Cast &x_) {
  Term::any_to_msgpack(w_, x_.from);
  Type::any_to_msgpack(w_, x_.as);
}

Expr::Cast Expr::cast_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::cast_fields_from_msgpack(r_, n_);
}

void Expr::cast_to_msgpack(MsgpackWriter &w_, const Expr::Cast &x_) {
  w_.writeArrayHeader(2);
  Expr::cast_fields_to_msgpack(w_, x_);
}

Expr::Index Expr::index_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Expr::Index with 3 field(s)");
  auto lhs = Term::any_from_msgpack(r_);
  auto idx = Term::any_from_msgpack(r_);
  auto comp = Type::any_from_msgpack(r_);
  return {lhs, idx, comp};
}

void Expr::index_fields_to_msgpack(MsgpackWriter &w_, const Expr::Index &x_) {
  Term::any_to_msgpack(w_, x_.lhs);
  Term::any_to_msgpack(w_, x_.idx);
  Type::any_to_msgpack(w_, x_.comp);
}

Expr::Index Expr::index_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::index_fields_from_msgpack(r_, n_);
}

void Expr::index_to_msgpack(MsgpackWriter &w_, const Expr::Index &x_) {
  w_.writeArrayHeader(3);
  Expr::index_fields_to_msgpack(w_, x_);
}

Expr::RefTo Expr::refto_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Expr::RefTo with 5 field(s)");
  auto lhs = Term::any_from_msgpack(r_);
  std::optional<Term::Any> idx;
  if (!r_.tryReadNil()) {
    auto idx_value = Term::any_from_msgpack(r_);
    idx = std::move(idx_value);
  }
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {lhs, idx, comp, space, region};
}

void Expr::refto_fields_to_msgpack(MsgpackWriter &w_, const Expr::RefTo &x_) {
  Term::any_to_msgpack(w_, x_.lhs);
  if (x_.idx) {
    Term::any_to_msgpack(w_, (*x_.idx));
  } else {
    w_.writeNil();
  }
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Expr::RefTo Expr::refto_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::refto_fields_from_msgpack(r_, n_);
}

void Expr::refto_to_msgpack(MsgpackWriter &w_, const Expr::RefTo &x_) {
  w_.writeArrayHeader(5);
  Expr::refto_fields_to_msgpack(w_, x_);
}

Expr::Alloc Expr::alloc_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected Expr::Alloc with 4 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto size = Term::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {comp, size, space, region};
}

void Expr::alloc_fields_to_msgpack(MsgpackWriter &w_, const Expr::Alloc &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  Term::any_to_msgpack(w_, x_.size);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Expr::Alloc Expr::alloc_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::alloc_fields_from_msgpack(r_, n_);
}

void Expr::alloc_to_msgpack(MsgpackWriter &w_, const Expr::Alloc &x_) {
  w_.writeArrayHeader(4);
  Expr::alloc_fields_to_msgpack(w_, x_);
}

Expr::Invoke Expr::invoke_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Expr::Invoke with 5 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Any> tpeArgs;
  {
    auto tpeArgs_size = r_.readArrayHeader();
    tpeArgs.reserve(tpeArgs_size);
    for (size_t tpeArgs_idx = 0; tpeArgs_idx < tpeArgs_size; ++tpeArgs_idx) {
      auto tpeArgs_elem = Type::any_from_msgpack(r_);
      tpeArgs.emplace_back(std::move(tpeArgs_elem));
    }
  }
  std::optional<Term::Any> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = Term::any_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Term::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Term::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {name, tpeArgs, receiver, args, rtn};
}

void Expr::invoke_fields_to_msgpack(MsgpackWriter &w_, const Expr::Invoke &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeArgs.size());
  for (const auto &v0_ : x_.tpeArgs) {
    Type::any_to_msgpack(w_, v0_);
  }
  if (x_.receiver) {
    Term::any_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Term::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Expr::Invoke Expr::invoke_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::invoke_fields_from_msgpack(r_, n_);
}

void Expr::invoke_to_msgpack(MsgpackWriter &w_, const Expr::Invoke &x_) {
  w_.writeArrayHeader(5);
  Expr::invoke_fields_to_msgpack(w_, x_);
}

Expr::ForeignCall Expr::foreigncall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Expr::ForeignCall with 3 field(s)");
  auto name = r_.readString();
  std::vector<Term::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Term::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {name, args, rtn};
}

void Expr::foreigncall_fields_to_msgpack(MsgpackWriter &w_, const Expr::ForeignCall &x_) {
  w_.writeString(x_.name);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Term::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Expr::ForeignCall Expr::foreigncall_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::foreigncall_fields_from_msgpack(r_, n_);
}

void Expr::foreigncall_to_msgpack(MsgpackWriter &w_, const Expr::ForeignCall &x_) {
  w_.writeArrayHeader(3);
  Expr::foreigncall_fields_to_msgpack(w_, x_);
}

Expr::OffsetOf Expr::offsetof_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Expr::OffsetOf with 2 field(s)");
  auto structTpe = Type::any_from_msgpack(r_);
  auto field = r_.readString();
  return {structTpe, field};
}

void Expr::offsetof_fields_to_msgpack(MsgpackWriter &w_, const Expr::OffsetOf &x_) {
  Type::any_to_msgpack(w_, x_.structTpe);
  w_.writeString(x_.field);
}

Expr::OffsetOf Expr::offsetof_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::offsetof_fields_from_msgpack(r_, n_);
}

void Expr::offsetof_to_msgpack(MsgpackWriter &w_, const Expr::OffsetOf &x_) {
  w_.writeArrayHeader(2);
  Expr::offsetof_fields_to_msgpack(w_, x_);
}

Expr::SizeOf Expr::sizeof_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::SizeOf with 1 field(s)");
  auto forTpe = Type::any_from_msgpack(r_);
  return Expr::SizeOf(forTpe);
}

void Expr::sizeof_fields_to_msgpack(MsgpackWriter &w_, const Expr::SizeOf &x_) { Type::any_to_msgpack(w_, x_.forTpe); }

Expr::SizeOf Expr::sizeof_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Expr::sizeof_fields_from_msgpack(r_, n_);
}

void Expr::sizeof_to_msgpack(MsgpackWriter &w_, const Expr::SizeOf &x_) {
  w_.writeArrayHeader(1);
  Expr::sizeof_fields_to_msgpack(w_, x_);
}

Expr::Any Expr::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Expr::alias_fields_from_msgpack(r_, n_ - 1);
      case 1: return Expr::specop_fields_from_msgpack(r_, n_ - 1);
      case 2: return Expr::mathop_fields_from_msgpack(r_, n_ - 1);
      case 3: return Expr::introp_fields_from_msgpack(r_, n_ - 1);
      case 4: return Expr::cast_fields_from_msgpack(r_, n_ - 1);
      case 5: return Expr::index_fields_from_msgpack(r_, n_ - 1);
      case 6: return Expr::refto_fields_from_msgpack(r_, n_ - 1);
      case 7: return Expr::alloc_fields_from_msgpack(r_, n_ - 1);
      case 8: return Expr::invoke_fields_from_msgpack(r_, n_ - 1);
      case 9: return Expr::foreigncall_fields_from_msgpack(r_, n_ - 1);
      case 10: return Expr::offsetof_fields_from_msgpack(r_, n_ - 1);
      case 11: return Expr::sizeof_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 4: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 6: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 10: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Expr::any_to_msgpack(MsgpackWriter &w_, const Expr::Any &x_) {
  x_.match_total(
      [&](const Expr::Alias &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Expr::alias_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::SpecOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Expr::specop_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::MathOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        Expr::mathop_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::IntrOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        Expr::introp_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Cast &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(4);
        Expr::cast_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Index &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(5);
        Expr::index_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::RefTo &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(6);
        Expr::refto_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Alloc &y_) -> void {
        w_.writeArrayHeader(5);
        w_.writeInt32(7);
        Expr::alloc_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Invoke &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(8);
        Expr::invoke_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::ForeignCall &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(9);
        Expr::foreigncall_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::OffsetOf &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(10);
        Expr::offsetof_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::SizeOf &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(11);
        Expr::sizeof_fields_to_msgpack(w_, y_);
      });
}

Overload overload_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Overload with 2 field(s)");
  std::vector<Type::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {args, rtn};
}

void overload_fields_to_msgpack(MsgpackWriter &w_, const Overload &x_) {
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Overload overload_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return overload_fields_from_msgpack(r_, n_);
}

void overload_to_msgpack(MsgpackWriter &w_, const Overload &x_) {
  w_.writeArrayHeader(2);
  overload_fields_to_msgpack(w_, x_);
}

Spec::Assert Spec::assert_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::Assert with 0 field(s)");
  return {};
}

void Spec::assert_fields_to_msgpack(MsgpackWriter &w_, const Spec::Assert &x_) {}

Spec::Assert Spec::assert_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::assert_fields_from_msgpack(r_, n_);
}

void Spec::assert_to_msgpack(MsgpackWriter &w_, const Spec::Assert &x_) {
  w_.writeArrayHeader(0);
  Spec::assert_fields_to_msgpack(w_, x_);
}

Spec::GpuBarrierGlobal Spec::gpubarrierglobal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierGlobal with 0 field(s)");
  return {};
}

void Spec::gpubarrierglobal_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierGlobal &x_) {}

Spec::GpuBarrierGlobal Spec::gpubarrierglobal_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpubarrierglobal_fields_from_msgpack(r_, n_);
}

void Spec::gpubarrierglobal_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierGlobal &x_) {
  w_.writeArrayHeader(0);
  Spec::gpubarrierglobal_fields_to_msgpack(w_, x_);
}

Spec::GpuBarrierLocal Spec::gpubarrierlocal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierLocal with 0 field(s)");
  return {};
}

void Spec::gpubarrierlocal_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierLocal &x_) {}

Spec::GpuBarrierLocal Spec::gpubarrierlocal_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpubarrierlocal_fields_from_msgpack(r_, n_);
}

void Spec::gpubarrierlocal_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierLocal &x_) {
  w_.writeArrayHeader(0);
  Spec::gpubarrierlocal_fields_to_msgpack(w_, x_);
}

Spec::GpuBarrierAll Spec::gpubarrierall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierAll with 0 field(s)");
  return {};
}

void Spec::gpubarrierall_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierAll &x_) {}

Spec::GpuBarrierAll Spec::gpubarrierall_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpubarrierall_fields_from_msgpack(r_, n_);
}

void Spec::gpubarrierall_to_msgpack(MsgpackWriter &w_, const Spec::GpuBarrierAll &x_) {
  w_.writeArrayHeader(0);
  Spec::gpubarrierall_fields_to_msgpack(w_, x_);
}

Spec::GpuFenceGlobal Spec::gpufenceglobal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceGlobal with 0 field(s)");
  return {};
}

void Spec::gpufenceglobal_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceGlobal &x_) {}

Spec::GpuFenceGlobal Spec::gpufenceglobal_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpufenceglobal_fields_from_msgpack(r_, n_);
}

void Spec::gpufenceglobal_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceGlobal &x_) {
  w_.writeArrayHeader(0);
  Spec::gpufenceglobal_fields_to_msgpack(w_, x_);
}

Spec::GpuFenceLocal Spec::gpufencelocal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceLocal with 0 field(s)");
  return {};
}

void Spec::gpufencelocal_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceLocal &x_) {}

Spec::GpuFenceLocal Spec::gpufencelocal_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpufencelocal_fields_from_msgpack(r_, n_);
}

void Spec::gpufencelocal_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceLocal &x_) {
  w_.writeArrayHeader(0);
  Spec::gpufencelocal_fields_to_msgpack(w_, x_);
}

Spec::GpuFenceAll Spec::gpufenceall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceAll with 0 field(s)");
  return {};
}

void Spec::gpufenceall_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceAll &x_) {}

Spec::GpuFenceAll Spec::gpufenceall_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpufenceall_fields_from_msgpack(r_, n_);
}

void Spec::gpufenceall_to_msgpack(MsgpackWriter &w_, const Spec::GpuFenceAll &x_) {
  w_.writeArrayHeader(0);
  Spec::gpufenceall_fields_to_msgpack(w_, x_);
}

Spec::GpuGlobalIdx Spec::gpuglobalidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGlobalIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGlobalIdx(dim);
}

void Spec::gpuglobalidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGlobalIdx Spec::gpuglobalidx_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpuglobalidx_fields_from_msgpack(r_, n_);
}

void Spec::gpuglobalidx_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalIdx &x_) {
  w_.writeArrayHeader(1);
  Spec::gpuglobalidx_fields_to_msgpack(w_, x_);
}

Spec::GpuGlobalSize Spec::gpuglobalsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGlobalSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGlobalSize(dim);
}

void Spec::gpuglobalsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGlobalSize Spec::gpuglobalsize_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpuglobalsize_fields_from_msgpack(r_, n_);
}

void Spec::gpuglobalsize_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalSize &x_) {
  w_.writeArrayHeader(1);
  Spec::gpuglobalsize_fields_to_msgpack(w_, x_);
}

Spec::GpuGroupIdx Spec::gpugroupidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGroupIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGroupIdx(dim);
}

void Spec::gpugroupidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGroupIdx Spec::gpugroupidx_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpugroupidx_fields_from_msgpack(r_, n_);
}

void Spec::gpugroupidx_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupIdx &x_) {
  w_.writeArrayHeader(1);
  Spec::gpugroupidx_fields_to_msgpack(w_, x_);
}

Spec::GpuGroupSize Spec::gpugroupsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGroupSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGroupSize(dim);
}

void Spec::gpugroupsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGroupSize Spec::gpugroupsize_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpugroupsize_fields_from_msgpack(r_, n_);
}

void Spec::gpugroupsize_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupSize &x_) {
  w_.writeArrayHeader(1);
  Spec::gpugroupsize_fields_to_msgpack(w_, x_);
}

Spec::GpuLocalIdx Spec::gpulocalidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuLocalIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuLocalIdx(dim);
}

void Spec::gpulocalidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuLocalIdx Spec::gpulocalidx_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpulocalidx_fields_from_msgpack(r_, n_);
}

void Spec::gpulocalidx_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalIdx &x_) {
  w_.writeArrayHeader(1);
  Spec::gpulocalidx_fields_to_msgpack(w_, x_);
}

Spec::GpuLocalSize Spec::gpulocalsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuLocalSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuLocalSize(dim);
}

void Spec::gpulocalsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuLocalSize Spec::gpulocalsize_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Spec::gpulocalsize_fields_from_msgpack(r_, n_);
}

void Spec::gpulocalsize_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalSize &x_) {
  w_.writeArrayHeader(1);
  Spec::gpulocalsize_fields_to_msgpack(w_, x_);
}

Spec::Any Spec::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Spec::assert_fields_from_msgpack(r_, n_ - 1);
      case 1: return Spec::gpubarrierglobal_fields_from_msgpack(r_, n_ - 1);
      case 2: return Spec::gpubarrierlocal_fields_from_msgpack(r_, n_ - 1);
      case 3: return Spec::gpubarrierall_fields_from_msgpack(r_, n_ - 1);
      case 4: return Spec::gpufenceglobal_fields_from_msgpack(r_, n_ - 1);
      case 5: return Spec::gpufencelocal_fields_from_msgpack(r_, n_ - 1);
      case 6: return Spec::gpufenceall_fields_from_msgpack(r_, n_ - 1);
      case 7: return Spec::gpuglobalidx_fields_from_msgpack(r_, n_ - 1);
      case 8: return Spec::gpuglobalsize_fields_from_msgpack(r_, n_ - 1);
      case 9: return Spec::gpugroupidx_fields_from_msgpack(r_, n_ - 1);
      case 10: return Spec::gpugroupsize_fields_from_msgpack(r_, n_ - 1);
      case 11: return Spec::gpulocalidx_fields_from_msgpack(r_, n_ - 1);
      case 12: return Spec::gpulocalsize_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Spec::assert_fields_from_msgpack(r_, 0);
      case 1: return Spec::gpubarrierglobal_fields_from_msgpack(r_, 0);
      case 2: return Spec::gpubarrierlocal_fields_from_msgpack(r_, 0);
      case 3: return Spec::gpubarrierall_fields_from_msgpack(r_, 0);
      case 4: return Spec::gpufenceglobal_fields_from_msgpack(r_, 0);
      case 5: return Spec::gpufencelocal_fields_from_msgpack(r_, 0);
      case 6: return Spec::gpufenceall_fields_from_msgpack(r_, 0);
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 10: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 12: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Spec::any_to_msgpack(MsgpackWriter &w_, const Spec::Any &x_) {
  x_.match_total(
      [&](const Spec::Assert &y_) -> void { w_.writeInt32(0); }, [&](const Spec::GpuBarrierGlobal &y_) -> void { w_.writeInt32(1); },
      [&](const Spec::GpuBarrierLocal &y_) -> void { w_.writeInt32(2); }, [&](const Spec::GpuBarrierAll &y_) -> void { w_.writeInt32(3); },
      [&](const Spec::GpuFenceGlobal &y_) -> void { w_.writeInt32(4); }, [&](const Spec::GpuFenceLocal &y_) -> void { w_.writeInt32(5); },
      [&](const Spec::GpuFenceAll &y_) -> void { w_.writeInt32(6); },
      [&](const Spec::GpuGlobalIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(7);
        Spec::gpuglobalidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGlobalSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Spec::gpuglobalsize_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(9);
        Spec::gpugroupidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(10);
        Spec::gpugroupsize_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuLocalIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(11);
        Spec::gpulocalidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuLocalSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(12);
        Spec::gpulocalsize_fields_to_msgpack(w_, y_);
      });
}

Intr::BNot Intr::bnot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::BNot with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Intr::bnot_fields_to_msgpack(MsgpackWriter &w_, const Intr::BNot &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BNot Intr::bnot_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bnot_fields_from_msgpack(r_, n_);
}

void Intr::bnot_to_msgpack(MsgpackWriter &w_, const Intr::BNot &x_) {
  w_.writeArrayHeader(2);
  Intr::bnot_fields_to_msgpack(w_, x_);
}

Intr::LogicNot Intr::logicnot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Intr::LogicNot with 1 field(s)");
  auto x = Term::any_from_msgpack(r_);
  return Intr::LogicNot(x);
}

void Intr::logicnot_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicNot &x_) { Term::any_to_msgpack(w_, x_.x); }

Intr::LogicNot Intr::logicnot_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicnot_fields_from_msgpack(r_, n_);
}

void Intr::logicnot_to_msgpack(MsgpackWriter &w_, const Intr::LogicNot &x_) {
  w_.writeArrayHeader(1);
  Intr::logicnot_fields_to_msgpack(w_, x_);
}

Intr::Pos Intr::pos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::Pos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Intr::pos_fields_to_msgpack(MsgpackWriter &w_, const Intr::Pos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Pos Intr::pos_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::pos_fields_from_msgpack(r_, n_);
}

void Intr::pos_to_msgpack(MsgpackWriter &w_, const Intr::Pos &x_) {
  w_.writeArrayHeader(2);
  Intr::pos_fields_to_msgpack(w_, x_);
}

Intr::Neg Intr::neg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::Neg with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Intr::neg_fields_to_msgpack(MsgpackWriter &w_, const Intr::Neg &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Neg Intr::neg_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::neg_fields_from_msgpack(r_, n_);
}

void Intr::neg_to_msgpack(MsgpackWriter &w_, const Intr::Neg &x_) {
  w_.writeArrayHeader(2);
  Intr::neg_fields_to_msgpack(w_, x_);
}

Intr::Add Intr::add_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Add with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::add_fields_to_msgpack(MsgpackWriter &w_, const Intr::Add &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Add Intr::add_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::add_fields_from_msgpack(r_, n_);
}

void Intr::add_to_msgpack(MsgpackWriter &w_, const Intr::Add &x_) {
  w_.writeArrayHeader(3);
  Intr::add_fields_to_msgpack(w_, x_);
}

Intr::Sub Intr::sub_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Sub with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::sub_fields_to_msgpack(MsgpackWriter &w_, const Intr::Sub &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Sub Intr::sub_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::sub_fields_from_msgpack(r_, n_);
}

void Intr::sub_to_msgpack(MsgpackWriter &w_, const Intr::Sub &x_) {
  w_.writeArrayHeader(3);
  Intr::sub_fields_to_msgpack(w_, x_);
}

Intr::Mul Intr::mul_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Mul with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::mul_fields_to_msgpack(MsgpackWriter &w_, const Intr::Mul &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Mul Intr::mul_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::mul_fields_from_msgpack(r_, n_);
}

void Intr::mul_to_msgpack(MsgpackWriter &w_, const Intr::Mul &x_) {
  w_.writeArrayHeader(3);
  Intr::mul_fields_to_msgpack(w_, x_);
}

Intr::Div Intr::div_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Div with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::div_fields_to_msgpack(MsgpackWriter &w_, const Intr::Div &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Div Intr::div_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::div_fields_from_msgpack(r_, n_);
}

void Intr::div_to_msgpack(MsgpackWriter &w_, const Intr::Div &x_) {
  w_.writeArrayHeader(3);
  Intr::div_fields_to_msgpack(w_, x_);
}

Intr::Rem Intr::rem_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Rem with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::rem_fields_to_msgpack(MsgpackWriter &w_, const Intr::Rem &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Rem Intr::rem_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::rem_fields_from_msgpack(r_, n_);
}

void Intr::rem_to_msgpack(MsgpackWriter &w_, const Intr::Rem &x_) {
  w_.writeArrayHeader(3);
  Intr::rem_fields_to_msgpack(w_, x_);
}

Intr::Min Intr::min_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Min with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::min_fields_to_msgpack(MsgpackWriter &w_, const Intr::Min &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Min Intr::min_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::min_fields_from_msgpack(r_, n_);
}

void Intr::min_to_msgpack(MsgpackWriter &w_, const Intr::Min &x_) {
  w_.writeArrayHeader(3);
  Intr::min_fields_to_msgpack(w_, x_);
}

Intr::Max Intr::max_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Max with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::max_fields_to_msgpack(MsgpackWriter &w_, const Intr::Max &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Max Intr::max_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::max_fields_from_msgpack(r_, n_);
}

void Intr::max_to_msgpack(MsgpackWriter &w_, const Intr::Max &x_) {
  w_.writeArrayHeader(3);
  Intr::max_fields_to_msgpack(w_, x_);
}

Intr::BAnd Intr::band_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BAnd with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::band_fields_to_msgpack(MsgpackWriter &w_, const Intr::BAnd &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BAnd Intr::band_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::band_fields_from_msgpack(r_, n_);
}

void Intr::band_to_msgpack(MsgpackWriter &w_, const Intr::BAnd &x_) {
  w_.writeArrayHeader(3);
  Intr::band_fields_to_msgpack(w_, x_);
}

Intr::BOr Intr::bor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BOr with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::bor_fields_to_msgpack(MsgpackWriter &w_, const Intr::BOr &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BOr Intr::bor_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bor_fields_from_msgpack(r_, n_);
}

void Intr::bor_to_msgpack(MsgpackWriter &w_, const Intr::BOr &x_) {
  w_.writeArrayHeader(3);
  Intr::bor_fields_to_msgpack(w_, x_);
}

Intr::BXor Intr::bxor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BXor with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::bxor_fields_to_msgpack(MsgpackWriter &w_, const Intr::BXor &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BXor Intr::bxor_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bxor_fields_from_msgpack(r_, n_);
}

void Intr::bxor_to_msgpack(MsgpackWriter &w_, const Intr::BXor &x_) {
  w_.writeArrayHeader(3);
  Intr::bxor_fields_to_msgpack(w_, x_);
}

Intr::BSL Intr::bsl_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BSL with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::bsl_fields_to_msgpack(MsgpackWriter &w_, const Intr::BSL &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BSL Intr::bsl_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bsl_fields_from_msgpack(r_, n_);
}

void Intr::bsl_to_msgpack(MsgpackWriter &w_, const Intr::BSL &x_) {
  w_.writeArrayHeader(3);
  Intr::bsl_fields_to_msgpack(w_, x_);
}

Intr::BSR Intr::bsr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BSR with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::bsr_fields_to_msgpack(MsgpackWriter &w_, const Intr::BSR &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BSR Intr::bsr_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bsr_fields_from_msgpack(r_, n_);
}

void Intr::bsr_to_msgpack(MsgpackWriter &w_, const Intr::BSR &x_) {
  w_.writeArrayHeader(3);
  Intr::bsr_fields_to_msgpack(w_, x_);
}

Intr::BZSR Intr::bzsr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BZSR with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Intr::bzsr_fields_to_msgpack(MsgpackWriter &w_, const Intr::BZSR &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BZSR Intr::bzsr_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::bzsr_fields_from_msgpack(r_, n_);
}

void Intr::bzsr_to_msgpack(MsgpackWriter &w_, const Intr::BZSR &x_) {
  w_.writeArrayHeader(3);
  Intr::bzsr_fields_to_msgpack(w_, x_);
}

Intr::LogicAnd Intr::logicand_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicAnd with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logicand_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicAnd &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicAnd Intr::logicand_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicand_fields_from_msgpack(r_, n_);
}

void Intr::logicand_to_msgpack(MsgpackWriter &w_, const Intr::LogicAnd &x_) {
  w_.writeArrayHeader(2);
  Intr::logicand_fields_to_msgpack(w_, x_);
}

Intr::LogicOr Intr::logicor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicOr with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logicor_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicOr &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicOr Intr::logicor_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicor_fields_from_msgpack(r_, n_);
}

void Intr::logicor_to_msgpack(MsgpackWriter &w_, const Intr::LogicOr &x_) {
  w_.writeArrayHeader(2);
  Intr::logicor_fields_to_msgpack(w_, x_);
}

Intr::LogicEq Intr::logiceq_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicEq with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logiceq_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicEq &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicEq Intr::logiceq_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logiceq_fields_from_msgpack(r_, n_);
}

void Intr::logiceq_to_msgpack(MsgpackWriter &w_, const Intr::LogicEq &x_) {
  w_.writeArrayHeader(2);
  Intr::logiceq_fields_to_msgpack(w_, x_);
}

Intr::LogicNeq Intr::logicneq_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicNeq with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logicneq_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicNeq &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicNeq Intr::logicneq_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicneq_fields_from_msgpack(r_, n_);
}

void Intr::logicneq_to_msgpack(MsgpackWriter &w_, const Intr::LogicNeq &x_) {
  w_.writeArrayHeader(2);
  Intr::logicneq_fields_to_msgpack(w_, x_);
}

Intr::LogicLte Intr::logiclte_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicLte with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logiclte_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicLte &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicLte Intr::logiclte_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logiclte_fields_from_msgpack(r_, n_);
}

void Intr::logiclte_to_msgpack(MsgpackWriter &w_, const Intr::LogicLte &x_) {
  w_.writeArrayHeader(2);
  Intr::logiclte_fields_to_msgpack(w_, x_);
}

Intr::LogicGte Intr::logicgte_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicGte with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logicgte_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicGte &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicGte Intr::logicgte_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicgte_fields_from_msgpack(r_, n_);
}

void Intr::logicgte_to_msgpack(MsgpackWriter &w_, const Intr::LogicGte &x_) {
  w_.writeArrayHeader(2);
  Intr::logicgte_fields_to_msgpack(w_, x_);
}

Intr::LogicLt Intr::logiclt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicLt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logiclt_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicLt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicLt Intr::logiclt_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logiclt_fields_from_msgpack(r_, n_);
}

void Intr::logiclt_to_msgpack(MsgpackWriter &w_, const Intr::LogicLt &x_) {
  w_.writeArrayHeader(2);
  Intr::logiclt_fields_to_msgpack(w_, x_);
}

Intr::LogicGt Intr::logicgt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicGt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {x, y};
}

void Intr::logicgt_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicGt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicGt Intr::logicgt_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Intr::logicgt_fields_from_msgpack(r_, n_);
}

void Intr::logicgt_to_msgpack(MsgpackWriter &w_, const Intr::LogicGt &x_) {
  w_.writeArrayHeader(2);
  Intr::logicgt_fields_to_msgpack(w_, x_);
}

Intr::Any Intr::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Intr::bnot_fields_from_msgpack(r_, n_ - 1);
      case 1: return Intr::logicnot_fields_from_msgpack(r_, n_ - 1);
      case 2: return Intr::pos_fields_from_msgpack(r_, n_ - 1);
      case 3: return Intr::neg_fields_from_msgpack(r_, n_ - 1);
      case 4: return Intr::add_fields_from_msgpack(r_, n_ - 1);
      case 5: return Intr::sub_fields_from_msgpack(r_, n_ - 1);
      case 6: return Intr::mul_fields_from_msgpack(r_, n_ - 1);
      case 7: return Intr::div_fields_from_msgpack(r_, n_ - 1);
      case 8: return Intr::rem_fields_from_msgpack(r_, n_ - 1);
      case 9: return Intr::min_fields_from_msgpack(r_, n_ - 1);
      case 10: return Intr::max_fields_from_msgpack(r_, n_ - 1);
      case 11: return Intr::band_fields_from_msgpack(r_, n_ - 1);
      case 12: return Intr::bor_fields_from_msgpack(r_, n_ - 1);
      case 13: return Intr::bxor_fields_from_msgpack(r_, n_ - 1);
      case 14: return Intr::bsl_fields_from_msgpack(r_, n_ - 1);
      case 15: return Intr::bsr_fields_from_msgpack(r_, n_ - 1);
      case 16: return Intr::bzsr_fields_from_msgpack(r_, n_ - 1);
      case 17: return Intr::logicand_fields_from_msgpack(r_, n_ - 1);
      case 18: return Intr::logicor_fields_from_msgpack(r_, n_ - 1);
      case 19: return Intr::logiceq_fields_from_msgpack(r_, n_ - 1);
      case 20: return Intr::logicneq_fields_from_msgpack(r_, n_ - 1);
      case 21: return Intr::logiclte_fields_from_msgpack(r_, n_ - 1);
      case 22: return Intr::logicgte_fields_from_msgpack(r_, n_ - 1);
      case 23: return Intr::logiclt_fields_from_msgpack(r_, n_ - 1);
      case 24: return Intr::logicgt_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 4: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 6: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 10: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 12: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 13: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 14: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 15: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 16: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 17: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 18: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 19: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 20: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 21: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 22: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 23: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 24: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Intr::any_to_msgpack(MsgpackWriter &w_, const Intr::Any &x_) {
  x_.match_total(
      [&](const Intr::BNot &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(0);
        Intr::bnot_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicNot &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Intr::logicnot_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Pos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(2);
        Intr::pos_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Neg &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Intr::neg_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Add &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(4);
        Intr::add_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Sub &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(5);
        Intr::sub_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Mul &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(6);
        Intr::mul_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Div &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(7);
        Intr::div_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Rem &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(8);
        Intr::rem_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Min &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(9);
        Intr::min_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Max &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(10);
        Intr::max_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BAnd &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(11);
        Intr::band_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BOr &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(12);
        Intr::bor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BXor &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(13);
        Intr::bxor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BSL &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(14);
        Intr::bsl_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BSR &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(15);
        Intr::bsr_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BZSR &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(16);
        Intr::bzsr_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicAnd &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(17);
        Intr::logicand_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicOr &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(18);
        Intr::logicor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicEq &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(19);
        Intr::logiceq_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicNeq &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(20);
        Intr::logicneq_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicLte &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(21);
        Intr::logiclte_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicGte &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(22);
        Intr::logicgte_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicLt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(23);
        Intr::logiclt_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicGt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(24);
        Intr::logicgt_fields_to_msgpack(w_, y_);
      });
}

Math::Abs Math::abs_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Abs with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::abs_fields_to_msgpack(MsgpackWriter &w_, const Math::Abs &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Abs Math::abs_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::abs_fields_from_msgpack(r_, n_);
}

void Math::abs_to_msgpack(MsgpackWriter &w_, const Math::Abs &x_) {
  w_.writeArrayHeader(2);
  Math::abs_fields_to_msgpack(w_, x_);
}

Math::Sin Math::sin_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sin with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::sin_fields_to_msgpack(MsgpackWriter &w_, const Math::Sin &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sin Math::sin_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::sin_fields_from_msgpack(r_, n_);
}

void Math::sin_to_msgpack(MsgpackWriter &w_, const Math::Sin &x_) {
  w_.writeArrayHeader(2);
  Math::sin_fields_to_msgpack(w_, x_);
}

Math::Cos Math::cos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::cos_fields_to_msgpack(MsgpackWriter &w_, const Math::Cos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cos Math::cos_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::cos_fields_from_msgpack(r_, n_);
}

void Math::cos_to_msgpack(MsgpackWriter &w_, const Math::Cos &x_) {
  w_.writeArrayHeader(2);
  Math::cos_fields_to_msgpack(w_, x_);
}

Math::Tan Math::tan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Tan with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::tan_fields_to_msgpack(MsgpackWriter &w_, const Math::Tan &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Tan Math::tan_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::tan_fields_from_msgpack(r_, n_);
}

void Math::tan_to_msgpack(MsgpackWriter &w_, const Math::Tan &x_) {
  w_.writeArrayHeader(2);
  Math::tan_fields_to_msgpack(w_, x_);
}

Math::Asin Math::asin_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Asin with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::asin_fields_to_msgpack(MsgpackWriter &w_, const Math::Asin &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Asin Math::asin_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::asin_fields_from_msgpack(r_, n_);
}

void Math::asin_to_msgpack(MsgpackWriter &w_, const Math::Asin &x_) {
  w_.writeArrayHeader(2);
  Math::asin_fields_to_msgpack(w_, x_);
}

Math::Acos Math::acos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Acos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::acos_fields_to_msgpack(MsgpackWriter &w_, const Math::Acos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Acos Math::acos_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::acos_fields_from_msgpack(r_, n_);
}

void Math::acos_to_msgpack(MsgpackWriter &w_, const Math::Acos &x_) {
  w_.writeArrayHeader(2);
  Math::acos_fields_to_msgpack(w_, x_);
}

Math::Atan Math::atan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Atan with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::atan_fields_to_msgpack(MsgpackWriter &w_, const Math::Atan &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Atan Math::atan_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::atan_fields_from_msgpack(r_, n_);
}

void Math::atan_to_msgpack(MsgpackWriter &w_, const Math::Atan &x_) {
  w_.writeArrayHeader(2);
  Math::atan_fields_to_msgpack(w_, x_);
}

Math::Sinh Math::sinh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sinh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::sinh_fields_to_msgpack(MsgpackWriter &w_, const Math::Sinh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sinh Math::sinh_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::sinh_fields_from_msgpack(r_, n_);
}

void Math::sinh_to_msgpack(MsgpackWriter &w_, const Math::Sinh &x_) {
  w_.writeArrayHeader(2);
  Math::sinh_fields_to_msgpack(w_, x_);
}

Math::Cosh Math::cosh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cosh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::cosh_fields_to_msgpack(MsgpackWriter &w_, const Math::Cosh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cosh Math::cosh_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::cosh_fields_from_msgpack(r_, n_);
}

void Math::cosh_to_msgpack(MsgpackWriter &w_, const Math::Cosh &x_) {
  w_.writeArrayHeader(2);
  Math::cosh_fields_to_msgpack(w_, x_);
}

Math::Tanh Math::tanh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Tanh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::tanh_fields_to_msgpack(MsgpackWriter &w_, const Math::Tanh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Tanh Math::tanh_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::tanh_fields_from_msgpack(r_, n_);
}

void Math::tanh_to_msgpack(MsgpackWriter &w_, const Math::Tanh &x_) {
  w_.writeArrayHeader(2);
  Math::tanh_fields_to_msgpack(w_, x_);
}

Math::Signum Math::signum_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Signum with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::signum_fields_to_msgpack(MsgpackWriter &w_, const Math::Signum &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Signum Math::signum_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::signum_fields_from_msgpack(r_, n_);
}

void Math::signum_to_msgpack(MsgpackWriter &w_, const Math::Signum &x_) {
  w_.writeArrayHeader(2);
  Math::signum_fields_to_msgpack(w_, x_);
}

Math::Round Math::round_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Round with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::round_fields_to_msgpack(MsgpackWriter &w_, const Math::Round &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Round Math::round_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::round_fields_from_msgpack(r_, n_);
}

void Math::round_to_msgpack(MsgpackWriter &w_, const Math::Round &x_) {
  w_.writeArrayHeader(2);
  Math::round_fields_to_msgpack(w_, x_);
}

Math::Ceil Math::ceil_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Ceil with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::ceil_fields_to_msgpack(MsgpackWriter &w_, const Math::Ceil &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Ceil Math::ceil_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::ceil_fields_from_msgpack(r_, n_);
}

void Math::ceil_to_msgpack(MsgpackWriter &w_, const Math::Ceil &x_) {
  w_.writeArrayHeader(2);
  Math::ceil_fields_to_msgpack(w_, x_);
}

Math::Floor Math::floor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Floor with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::floor_fields_to_msgpack(MsgpackWriter &w_, const Math::Floor &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Floor Math::floor_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::floor_fields_from_msgpack(r_, n_);
}

void Math::floor_to_msgpack(MsgpackWriter &w_, const Math::Floor &x_) {
  w_.writeArrayHeader(2);
  Math::floor_fields_to_msgpack(w_, x_);
}

Math::Rint Math::rint_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Rint with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::rint_fields_to_msgpack(MsgpackWriter &w_, const Math::Rint &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Rint Math::rint_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::rint_fields_from_msgpack(r_, n_);
}

void Math::rint_to_msgpack(MsgpackWriter &w_, const Math::Rint &x_) {
  w_.writeArrayHeader(2);
  Math::rint_fields_to_msgpack(w_, x_);
}

Math::Sqrt Math::sqrt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sqrt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::sqrt_fields_to_msgpack(MsgpackWriter &w_, const Math::Sqrt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sqrt Math::sqrt_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::sqrt_fields_from_msgpack(r_, n_);
}

void Math::sqrt_to_msgpack(MsgpackWriter &w_, const Math::Sqrt &x_) {
  w_.writeArrayHeader(2);
  Math::sqrt_fields_to_msgpack(w_, x_);
}

Math::Cbrt Math::cbrt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cbrt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::cbrt_fields_to_msgpack(MsgpackWriter &w_, const Math::Cbrt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cbrt Math::cbrt_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::cbrt_fields_from_msgpack(r_, n_);
}

void Math::cbrt_to_msgpack(MsgpackWriter &w_, const Math::Cbrt &x_) {
  w_.writeArrayHeader(2);
  Math::cbrt_fields_to_msgpack(w_, x_);
}

Math::Exp Math::exp_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Exp with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::exp_fields_to_msgpack(MsgpackWriter &w_, const Math::Exp &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Exp Math::exp_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::exp_fields_from_msgpack(r_, n_);
}

void Math::exp_to_msgpack(MsgpackWriter &w_, const Math::Exp &x_) {
  w_.writeArrayHeader(2);
  Math::exp_fields_to_msgpack(w_, x_);
}

Math::Expm1 Math::expm1_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Expm1 with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::expm1_fields_to_msgpack(MsgpackWriter &w_, const Math::Expm1 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Expm1 Math::expm1_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::expm1_fields_from_msgpack(r_, n_);
}

void Math::expm1_to_msgpack(MsgpackWriter &w_, const Math::Expm1 &x_) {
  w_.writeArrayHeader(2);
  Math::expm1_fields_to_msgpack(w_, x_);
}

Math::Log Math::log_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::log_fields_to_msgpack(MsgpackWriter &w_, const Math::Log &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log Math::log_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::log_fields_from_msgpack(r_, n_);
}

void Math::log_to_msgpack(MsgpackWriter &w_, const Math::Log &x_) {
  w_.writeArrayHeader(2);
  Math::log_fields_to_msgpack(w_, x_);
}

Math::Log1p Math::log1p_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log1p with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::log1p_fields_to_msgpack(MsgpackWriter &w_, const Math::Log1p &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log1p Math::log1p_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::log1p_fields_from_msgpack(r_, n_);
}

void Math::log1p_to_msgpack(MsgpackWriter &w_, const Math::Log1p &x_) {
  w_.writeArrayHeader(2);
  Math::log1p_fields_to_msgpack(w_, x_);
}

Math::Log10 Math::log10_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log10 with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, rtn};
}

void Math::log10_fields_to_msgpack(MsgpackWriter &w_, const Math::Log10 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log10 Math::log10_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::log10_fields_from_msgpack(r_, n_);
}

void Math::log10_to_msgpack(MsgpackWriter &w_, const Math::Log10 &x_) {
  w_.writeArrayHeader(2);
  Math::log10_fields_to_msgpack(w_, x_);
}

Math::Pow Math::pow_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Pow with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Math::pow_fields_to_msgpack(MsgpackWriter &w_, const Math::Pow &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Pow Math::pow_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::pow_fields_from_msgpack(r_, n_);
}

void Math::pow_to_msgpack(MsgpackWriter &w_, const Math::Pow &x_) {
  w_.writeArrayHeader(3);
  Math::pow_fields_to_msgpack(w_, x_);
}

Math::Atan2 Math::atan2_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Atan2 with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Math::atan2_fields_to_msgpack(MsgpackWriter &w_, const Math::Atan2 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Atan2 Math::atan2_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::atan2_fields_from_msgpack(r_, n_);
}

void Math::atan2_to_msgpack(MsgpackWriter &w_, const Math::Atan2 &x_) {
  w_.writeArrayHeader(3);
  Math::atan2_fields_to_msgpack(w_, x_);
}

Math::Hypot Math::hypot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Hypot with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {x, y, rtn};
}

void Math::hypot_fields_to_msgpack(MsgpackWriter &w_, const Math::Hypot &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Hypot Math::hypot_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Math::hypot_fields_from_msgpack(r_, n_);
}

void Math::hypot_to_msgpack(MsgpackWriter &w_, const Math::Hypot &x_) {
  w_.writeArrayHeader(3);
  Math::hypot_fields_to_msgpack(w_, x_);
}

Math::Any Math::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Math::abs_fields_from_msgpack(r_, n_ - 1);
      case 1: return Math::sin_fields_from_msgpack(r_, n_ - 1);
      case 2: return Math::cos_fields_from_msgpack(r_, n_ - 1);
      case 3: return Math::tan_fields_from_msgpack(r_, n_ - 1);
      case 4: return Math::asin_fields_from_msgpack(r_, n_ - 1);
      case 5: return Math::acos_fields_from_msgpack(r_, n_ - 1);
      case 6: return Math::atan_fields_from_msgpack(r_, n_ - 1);
      case 7: return Math::sinh_fields_from_msgpack(r_, n_ - 1);
      case 8: return Math::cosh_fields_from_msgpack(r_, n_ - 1);
      case 9: return Math::tanh_fields_from_msgpack(r_, n_ - 1);
      case 10: return Math::signum_fields_from_msgpack(r_, n_ - 1);
      case 11: return Math::round_fields_from_msgpack(r_, n_ - 1);
      case 12: return Math::ceil_fields_from_msgpack(r_, n_ - 1);
      case 13: return Math::floor_fields_from_msgpack(r_, n_ - 1);
      case 14: return Math::rint_fields_from_msgpack(r_, n_ - 1);
      case 15: return Math::sqrt_fields_from_msgpack(r_, n_ - 1);
      case 16: return Math::cbrt_fields_from_msgpack(r_, n_ - 1);
      case 17: return Math::exp_fields_from_msgpack(r_, n_ - 1);
      case 18: return Math::expm1_fields_from_msgpack(r_, n_ - 1);
      case 19: return Math::log_fields_from_msgpack(r_, n_ - 1);
      case 20: return Math::log1p_fields_from_msgpack(r_, n_ - 1);
      case 21: return Math::log10_fields_from_msgpack(r_, n_ - 1);
      case 22: return Math::pow_fields_from_msgpack(r_, n_ - 1);
      case 23: return Math::atan2_fields_from_msgpack(r_, n_ - 1);
      case 24: return Math::hypot_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 4: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 6: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 10: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 12: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 13: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 14: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 15: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 16: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 17: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 18: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 19: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 20: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 21: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 22: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 23: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 24: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Math::any_to_msgpack(MsgpackWriter &w_, const Math::Any &x_) {
  x_.match_total(
      [&](const Math::Abs &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(0);
        Math::abs_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sin &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(1);
        Math::sin_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(2);
        Math::cos_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Tan &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Math::tan_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Asin &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(4);
        Math::asin_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Acos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(5);
        Math::acos_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Atan &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(6);
        Math::atan_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sinh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(7);
        Math::sinh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cosh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(8);
        Math::cosh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Tanh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(9);
        Math::tanh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Signum &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(10);
        Math::signum_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Round &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(11);
        Math::round_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Ceil &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(12);
        Math::ceil_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Floor &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(13);
        Math::floor_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Rint &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(14);
        Math::rint_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sqrt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(15);
        Math::sqrt_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cbrt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(16);
        Math::cbrt_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Exp &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(17);
        Math::exp_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Expm1 &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(18);
        Math::expm1_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(19);
        Math::log_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log1p &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(20);
        Math::log1p_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log10 &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(21);
        Math::log10_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Pow &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(22);
        Math::pow_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Atan2 &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(23);
        Math::atan2_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Hypot &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(24);
        Math::hypot_fields_to_msgpack(w_, y_);
      });
}

Stmt::Var Stmt::var_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Var with 3 field(s)");
  auto name = named_from_msgpack(r_);
  std::optional<Expr::Any> expr;
  if (!r_.tryReadNil()) {
    auto expr_value = Expr::any_from_msgpack(r_);
    expr = std::move(expr_value);
  }
  auto isMutable = r_.readBoolean();
  return {name, expr, isMutable};
}

void Stmt::var_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Var &x_) {
  named_to_msgpack(w_, x_.name);
  if (x_.expr) {
    Expr::any_to_msgpack(w_, (*x_.expr));
  } else {
    w_.writeNil();
  }
  w_.writeBoolean(x_.isMutable);
}

Stmt::Var Stmt::var_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::var_fields_from_msgpack(r_, n_);
}

void Stmt::var_to_msgpack(MsgpackWriter &w_, const Stmt::Var &x_) {
  w_.writeArrayHeader(3);
  Stmt::var_fields_to_msgpack(w_, x_);
}

Stmt::Mut Stmt::mut_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Stmt::Mut with 2 field(s)");
  auto name = Term::select_from_msgpack(r_);
  auto expr = Expr::any_from_msgpack(r_);
  return {name, expr};
}

void Stmt::mut_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Mut &x_) {
  Term::select_to_msgpack(w_, x_.name);
  Expr::any_to_msgpack(w_, x_.expr);
}

Stmt::Mut Stmt::mut_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::mut_fields_from_msgpack(r_, n_);
}

void Stmt::mut_to_msgpack(MsgpackWriter &w_, const Stmt::Mut &x_) {
  w_.writeArrayHeader(2);
  Stmt::mut_fields_to_msgpack(w_, x_);
}

Stmt::Update Stmt::update_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Update with 3 field(s)");
  auto lhs = Term::select_from_msgpack(r_);
  auto idx = Term::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  return {lhs, idx, value};
}

void Stmt::update_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Update &x_) {
  Term::select_to_msgpack(w_, x_.lhs);
  Term::any_to_msgpack(w_, x_.idx);
  Term::any_to_msgpack(w_, x_.value);
}

Stmt::Update Stmt::update_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::update_fields_from_msgpack(r_, n_);
}

void Stmt::update_to_msgpack(MsgpackWriter &w_, const Stmt::Update &x_) {
  w_.writeArrayHeader(3);
  Stmt::update_fields_to_msgpack(w_, x_);
}

Stmt::While Stmt::while_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Stmt::While with 2 field(s)");
  auto cond = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  return {cond, body};
}

void Stmt::while_fields_to_msgpack(MsgpackWriter &w_, const Stmt::While &x_) {
  Term::any_to_msgpack(w_, x_.cond);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::While Stmt::while_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::while_fields_from_msgpack(r_, n_);
}

void Stmt::while_to_msgpack(MsgpackWriter &w_, const Stmt::While &x_) {
  w_.writeArrayHeader(2);
  Stmt::while_fields_to_msgpack(w_, x_);
}

Stmt::ForRange Stmt::forrange_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Stmt::ForRange with 5 field(s)");
  auto induction = named_from_msgpack(r_);
  auto lbIncl = Term::any_from_msgpack(r_);
  auto ubExcl = Term::any_from_msgpack(r_);
  auto step = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  return {induction, lbIncl, ubExcl, step, body};
}

void Stmt::forrange_fields_to_msgpack(MsgpackWriter &w_, const Stmt::ForRange &x_) {
  named_to_msgpack(w_, x_.induction);
  Term::any_to_msgpack(w_, x_.lbIncl);
  Term::any_to_msgpack(w_, x_.ubExcl);
  Term::any_to_msgpack(w_, x_.step);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::ForRange Stmt::forrange_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::forrange_fields_from_msgpack(r_, n_);
}

void Stmt::forrange_to_msgpack(MsgpackWriter &w_, const Stmt::ForRange &x_) {
  w_.writeArrayHeader(5);
  Stmt::forrange_fields_to_msgpack(w_, x_);
}

Stmt::Break Stmt::break_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Stmt::Break with 0 field(s)");
  return {};
}

void Stmt::break_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Break &x_) {}

Stmt::Break Stmt::break_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::break_fields_from_msgpack(r_, n_);
}

void Stmt::break_to_msgpack(MsgpackWriter &w_, const Stmt::Break &x_) {
  w_.writeArrayHeader(0);
  Stmt::break_fields_to_msgpack(w_, x_);
}

Stmt::Cont Stmt::cont_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Stmt::Cont with 0 field(s)");
  return {};
}

void Stmt::cont_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Cont &x_) {}

Stmt::Cont Stmt::cont_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::cont_fields_from_msgpack(r_, n_);
}

void Stmt::cont_to_msgpack(MsgpackWriter &w_, const Stmt::Cont &x_) {
  w_.writeArrayHeader(0);
  Stmt::cont_fields_to_msgpack(w_, x_);
}

Stmt::Cond Stmt::cond_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Cond with 3 field(s)");
  auto cond = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> trueBr;
  {
    auto trueBr_size = r_.readArrayHeader();
    trueBr.reserve(trueBr_size);
    for (size_t trueBr_idx = 0; trueBr_idx < trueBr_size; ++trueBr_idx) {
      auto trueBr_elem = Stmt::any_from_msgpack(r_);
      trueBr.emplace_back(std::move(trueBr_elem));
    }
  }
  std::vector<Stmt::Any> falseBr;
  {
    auto falseBr_size = r_.readArrayHeader();
    falseBr.reserve(falseBr_size);
    for (size_t falseBr_idx = 0; falseBr_idx < falseBr_size; ++falseBr_idx) {
      auto falseBr_elem = Stmt::any_from_msgpack(r_);
      falseBr.emplace_back(std::move(falseBr_elem));
    }
  }
  return {cond, trueBr, falseBr};
}

void Stmt::cond_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Cond &x_) {
  Term::any_to_msgpack(w_, x_.cond);
  w_.writeArrayHeader(x_.trueBr.size());
  for (const auto &v0_ : x_.trueBr) {
    Stmt::any_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.falseBr.size());
  for (const auto &v0_ : x_.falseBr) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::Cond Stmt::cond_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::cond_fields_from_msgpack(r_, n_);
}

void Stmt::cond_to_msgpack(MsgpackWriter &w_, const Stmt::Cond &x_) {
  w_.writeArrayHeader(3);
  Stmt::cond_fields_to_msgpack(w_, x_);
}

Stmt::Return Stmt::return_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Stmt::Return with 1 field(s)");
  auto value = Expr::any_from_msgpack(r_);
  return Stmt::Return(value);
}

void Stmt::return_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Return &x_) { Expr::any_to_msgpack(w_, x_.value); }

Stmt::Return Stmt::return_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::return_fields_from_msgpack(r_, n_);
}

void Stmt::return_to_msgpack(MsgpackWriter &w_, const Stmt::Return &x_) {
  w_.writeArrayHeader(1);
  Stmt::return_fields_to_msgpack(w_, x_);
}

Stmt::Annotated Stmt::annotated_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Annotated with 3 field(s)");
  auto inner = Stmt::any_from_msgpack(r_);
  std::optional<SourcePosition> pos;
  if (!r_.tryReadNil()) {
    auto pos_value = sourceposition_from_msgpack(r_);
    pos = std::move(pos_value);
  }
  std::optional<std::string> comment;
  if (!r_.tryReadNil()) {
    auto comment_value = r_.readString();
    comment = std::move(comment_value);
  }
  return {inner, pos, comment};
}

void Stmt::annotated_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Annotated &x_) {
  Stmt::any_to_msgpack(w_, x_.inner);
  if (x_.pos) {
    sourceposition_to_msgpack(w_, (*x_.pos));
  } else {
    w_.writeNil();
  }
  if (x_.comment) {
    w_.writeString((*x_.comment));
  } else {
    w_.writeNil();
  }
}

Stmt::Annotated Stmt::annotated_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return Stmt::annotated_fields_from_msgpack(r_, n_);
}

void Stmt::annotated_to_msgpack(MsgpackWriter &w_, const Stmt::Annotated &x_) {
  w_.writeArrayHeader(3);
  Stmt::annotated_fields_to_msgpack(w_, x_);
}

Stmt::Any Stmt::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Stmt::var_fields_from_msgpack(r_, n_ - 1);
      case 1: return Stmt::mut_fields_from_msgpack(r_, n_ - 1);
      case 2: return Stmt::update_fields_from_msgpack(r_, n_ - 1);
      case 3: return Stmt::while_fields_from_msgpack(r_, n_ - 1);
      case 4: return Stmt::forrange_fields_from_msgpack(r_, n_ - 1);
      case 5: return Stmt::break_fields_from_msgpack(r_, n_ - 1);
      case 6: return Stmt::cont_fields_from_msgpack(r_, n_ - 1);
      case 7: return Stmt::cond_fields_from_msgpack(r_, n_ - 1);
      case 8: return Stmt::return_fields_from_msgpack(r_, n_ - 1);
      case 9: return Stmt::annotated_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 2: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 4: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: return Stmt::break_fields_from_msgpack(r_, 0);
      case 6: return Stmt::cont_fields_from_msgpack(r_, 0);
      case 7: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 8: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 9: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void Stmt::any_to_msgpack(MsgpackWriter &w_, const Stmt::Any &x_) {
  x_.match_total(
      [&](const Stmt::Var &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(0);
        Stmt::var_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Mut &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(1);
        Stmt::mut_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Update &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(2);
        Stmt::update_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::While &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Stmt::while_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::ForRange &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(4);
        Stmt::forrange_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Break &y_) -> void { w_.writeInt32(5); }, [&](const Stmt::Cont &y_) -> void { w_.writeInt32(6); },
      [&](const Stmt::Cond &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(7);
        Stmt::cond_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Return &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Stmt::return_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Annotated &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(9);
        Stmt::annotated_fields_to_msgpack(w_, y_);
      });
}

Signature signature_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 7) throw std::runtime_error("Expected Signature with 7 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<std::string> tpeVars;
  {
    auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = r_.readString();
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::optional<Type::Any> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = Type::any_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Type::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  std::vector<Type::Any> moduleCaptures;
  {
    auto moduleCaptures_size = r_.readArrayHeader();
    moduleCaptures.reserve(moduleCaptures_size);
    for (size_t moduleCaptures_idx = 0; moduleCaptures_idx < moduleCaptures_size; ++moduleCaptures_idx) {
      auto moduleCaptures_elem = Type::any_from_msgpack(r_);
      moduleCaptures.emplace_back(std::move(moduleCaptures_elem));
    }
  }
  std::vector<Type::Any> termCaptures;
  {
    auto termCaptures_size = r_.readArrayHeader();
    termCaptures.reserve(termCaptures_size);
    for (size_t termCaptures_idx = 0; termCaptures_idx < termCaptures_size; ++termCaptures_idx) {
      auto termCaptures_elem = Type::any_from_msgpack(r_);
      termCaptures.emplace_back(std::move(termCaptures_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn};
}

void signature_fields_to_msgpack(MsgpackWriter &w_, const Signature &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    w_.writeString(v0_);
  }
  if (x_.receiver) {
    Type::any_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.moduleCaptures.size());
  for (const auto &v0_ : x_.moduleCaptures) {
    Type::any_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.termCaptures.size());
  for (const auto &v0_ : x_.termCaptures) {
    Type::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Signature signature_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return signature_fields_from_msgpack(r_, n_);
}

void signature_to_msgpack(MsgpackWriter &w_, const Signature &x_) {
  w_.writeArrayHeader(7);
  signature_fields_to_msgpack(w_, x_);
}

InvokeSignature invokesignature_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected InvokeSignature with 5 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Any> tpeVars;
  {
    auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = Type::any_from_msgpack(r_);
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::optional<Type::Any> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = Type::any_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Type::Any> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {name, tpeVars, receiver, args, rtn};
}

void invokesignature_fields_to_msgpack(MsgpackWriter &w_, const InvokeSignature &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    Type::any_to_msgpack(w_, v0_);
  }
  if (x_.receiver) {
    Type::any_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

InvokeSignature invokesignature_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return invokesignature_fields_from_msgpack(r_, n_);
}

void invokesignature_to_msgpack(MsgpackWriter &w_, const InvokeSignature &x_) {
  w_.writeArrayHeader(5);
  invokesignature_fields_to_msgpack(w_, x_);
}

FunctionVisibility::Internal FunctionVisibility::internal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionVisibility::Internal with 0 field(s)");
  return {};
}

void FunctionVisibility::internal_fields_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Internal &x_) {}

FunctionVisibility::Internal FunctionVisibility::internal_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionVisibility::internal_fields_from_msgpack(r_, n_);
}

void FunctionVisibility::internal_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Internal &x_) {
  w_.writeArrayHeader(0);
  FunctionVisibility::internal_fields_to_msgpack(w_, x_);
}

FunctionVisibility::Exported FunctionVisibility::exported_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionVisibility::Exported with 0 field(s)");
  return {};
}

void FunctionVisibility::exported_fields_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Exported &x_) {}

FunctionVisibility::Exported FunctionVisibility::exported_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionVisibility::exported_fields_from_msgpack(r_, n_);
}

void FunctionVisibility::exported_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Exported &x_) {
  w_.writeArrayHeader(0);
  FunctionVisibility::exported_fields_to_msgpack(w_, x_);
}

FunctionVisibility::Any FunctionVisibility::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionVisibility::internal_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionVisibility::exported_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionVisibility::internal_fields_from_msgpack(r_, 0);
      case 1: return FunctionVisibility::exported_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void FunctionVisibility::any_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Any &x_) {
  x_.match_total([&](const FunctionVisibility::Internal &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionVisibility::Exported &y_) -> void { w_.writeInt32(1); });
}

FunctionFpMode::Relaxed FunctionFpMode::relaxed_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionFpMode::Relaxed with 0 field(s)");
  return {};
}

void FunctionFpMode::relaxed_fields_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Relaxed &x_) {}

FunctionFpMode::Relaxed FunctionFpMode::relaxed_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionFpMode::relaxed_fields_from_msgpack(r_, n_);
}

void FunctionFpMode::relaxed_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Relaxed &x_) {
  w_.writeArrayHeader(0);
  FunctionFpMode::relaxed_fields_to_msgpack(w_, x_);
}

FunctionFpMode::Strict FunctionFpMode::strict_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionFpMode::Strict with 0 field(s)");
  return {};
}

void FunctionFpMode::strict_fields_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Strict &x_) {}

FunctionFpMode::Strict FunctionFpMode::strict_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionFpMode::strict_fields_from_msgpack(r_, n_);
}

void FunctionFpMode::strict_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Strict &x_) {
  w_.writeArrayHeader(0);
  FunctionFpMode::strict_fields_to_msgpack(w_, x_);
}

FunctionFpMode::Any FunctionFpMode::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionFpMode::relaxed_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionFpMode::strict_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionFpMode::relaxed_fields_from_msgpack(r_, 0);
      case 1: return FunctionFpMode::strict_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void FunctionFpMode::any_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Any &x_) {
  x_.match_total([&](const FunctionFpMode::Relaxed &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionFpMode::Strict &y_) -> void { w_.writeInt32(1); });
}

FunctionAffinity::Offload FunctionAffinity::offload_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionAffinity::Offload with 0 field(s)");
  return {};
}

void FunctionAffinity::offload_fields_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Offload &x_) {}

FunctionAffinity::Offload FunctionAffinity::offload_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionAffinity::offload_fields_from_msgpack(r_, n_);
}

void FunctionAffinity::offload_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Offload &x_) {
  w_.writeArrayHeader(0);
  FunctionAffinity::offload_fields_to_msgpack(w_, x_);
}

FunctionAffinity::Host FunctionAffinity::host_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionAffinity::Host with 0 field(s)");
  return {};
}

void FunctionAffinity::host_fields_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Host &x_) {}

FunctionAffinity::Host FunctionAffinity::host_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return FunctionAffinity::host_fields_from_msgpack(r_, n_);
}

void FunctionAffinity::host_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Host &x_) {
  w_.writeArrayHeader(0);
  FunctionAffinity::host_fields_to_msgpack(w_, x_);
}

FunctionAffinity::Any FunctionAffinity::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionAffinity::offload_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionAffinity::host_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionAffinity::offload_fields_from_msgpack(r_, 0);
      case 1: return FunctionAffinity::host_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void FunctionAffinity::any_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Any &x_) {
  x_.match_total([&](const FunctionAffinity::Offload &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionAffinity::Host &y_) -> void { w_.writeInt32(1); });
}

Arg arg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Arg with 2 field(s)");
  auto named = named_from_msgpack(r_);
  std::optional<SourcePosition> pos;
  if (!r_.tryReadNil()) {
    auto pos_value = sourceposition_from_msgpack(r_);
    pos = std::move(pos_value);
  }
  return {named, pos};
}

void arg_fields_to_msgpack(MsgpackWriter &w_, const Arg &x_) {
  named_to_msgpack(w_, x_.named);
  if (x_.pos) {
    sourceposition_to_msgpack(w_, (*x_.pos));
  } else {
    w_.writeNil();
  }
}

Arg arg_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return arg_fields_from_msgpack(r_, n_);
}

void arg_to_msgpack(MsgpackWriter &w_, const Arg &x_) {
  w_.writeArrayHeader(2);
  arg_fields_to_msgpack(w_, x_);
}

Function function_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 12) throw std::runtime_error("Expected Function with 12 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<std::string> tpeVars;
  {
    auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = r_.readString();
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::optional<Arg> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = arg_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Arg> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = arg_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  std::vector<Arg> moduleCaptures;
  {
    auto moduleCaptures_size = r_.readArrayHeader();
    moduleCaptures.reserve(moduleCaptures_size);
    for (size_t moduleCaptures_idx = 0; moduleCaptures_idx < moduleCaptures_size; ++moduleCaptures_idx) {
      auto moduleCaptures_elem = arg_from_msgpack(r_);
      moduleCaptures.emplace_back(std::move(moduleCaptures_elem));
    }
  }
  std::vector<Arg> termCaptures;
  {
    auto termCaptures_size = r_.readArrayHeader();
    termCaptures.reserve(termCaptures_size);
    for (size_t termCaptures_idx = 0; termCaptures_idx < termCaptures_size; ++termCaptures_idx) {
      auto termCaptures_elem = arg_from_msgpack(r_);
      termCaptures.emplace_back(std::move(termCaptures_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  auto visibility = FunctionVisibility::any_from_msgpack(r_);
  auto fpMode = FunctionFpMode::any_from_msgpack(r_);
  auto isEntry = r_.readBoolean();
  auto affinity = FunctionAffinity::any_from_msgpack(r_);
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity};
}

void function_fields_to_msgpack(MsgpackWriter &w_, const Function &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    w_.writeString(v0_);
  }
  if (x_.receiver) {
    arg_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    arg_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.moduleCaptures.size());
  for (const auto &v0_ : x_.moduleCaptures) {
    arg_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.termCaptures.size());
  for (const auto &v0_ : x_.termCaptures) {
    arg_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
  FunctionVisibility::any_to_msgpack(w_, x_.visibility);
  FunctionFpMode::any_to_msgpack(w_, x_.fpMode);
  w_.writeBoolean(x_.isEntry);
  FunctionAffinity::any_to_msgpack(w_, x_.affinity);
}

Function function_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return function_fields_from_msgpack(r_, n_);
}

void function_to_msgpack(MsgpackWriter &w_, const Function &x_) {
  w_.writeArrayHeader(12);
  function_fields_to_msgpack(w_, x_);
}

StructDef structdef_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected StructDef with 4 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<std::string> tpeVars;
  {
    auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = r_.readString();
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::vector<Named> members;
  {
    auto members_size = r_.readArrayHeader();
    members.reserve(members_size);
    for (size_t members_idx = 0; members_idx < members_size; ++members_idx) {
      auto members_elem = named_from_msgpack(r_);
      members.emplace_back(std::move(members_elem));
    }
  }
  std::vector<Type::Struct> parents;
  {
    auto parents_size = r_.readArrayHeader();
    parents.reserve(parents_size);
    for (size_t parents_idx = 0; parents_idx < parents_size; ++parents_idx) {
      auto parents_elem = Type::struct_from_msgpack(r_);
      parents.emplace_back(std::move(parents_elem));
    }
  }
  return {name, tpeVars, members, parents};
}

void structdef_fields_to_msgpack(MsgpackWriter &w_, const StructDef &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.members.size());
  for (const auto &v0_ : x_.members) {
    named_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.parents.size());
  for (const auto &v0_ : x_.parents) {
    Type::struct_to_msgpack(w_, v0_);
  }
}

StructDef structdef_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return structdef_fields_from_msgpack(r_, n_);
}

void structdef_to_msgpack(MsgpackWriter &w_, const StructDef &x_) {
  w_.writeArrayHeader(4);
  structdef_fields_to_msgpack(w_, x_);
}

Mirror mirror_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Mirror with 5 field(s)");
  auto source = sym_from_msgpack(r_);
  std::vector<Sym> sourceParents;
  {
    auto sourceParents_size = r_.readArrayHeader();
    sourceParents.reserve(sourceParents_size);
    for (size_t sourceParents_idx = 0; sourceParents_idx < sourceParents_size; ++sourceParents_idx) {
      auto sourceParents_elem = sym_from_msgpack(r_);
      sourceParents.emplace_back(std::move(sourceParents_elem));
    }
  }
  auto structDef = structdef_from_msgpack(r_);
  std::vector<Function> functions;
  {
    auto functions_size = r_.readArrayHeader();
    functions.reserve(functions_size);
    for (size_t functions_idx = 0; functions_idx < functions_size; ++functions_idx) {
      auto functions_elem = function_from_msgpack(r_);
      functions.emplace_back(std::move(functions_elem));
    }
  }
  std::vector<StructDef> dependencies;
  {
    auto dependencies_size = r_.readArrayHeader();
    dependencies.reserve(dependencies_size);
    for (size_t dependencies_idx = 0; dependencies_idx < dependencies_size; ++dependencies_idx) {
      auto dependencies_elem = structdef_from_msgpack(r_);
      dependencies.emplace_back(std::move(dependencies_elem));
    }
  }
  return {source, sourceParents, structDef, functions, dependencies};
}

void mirror_fields_to_msgpack(MsgpackWriter &w_, const Mirror &x_) {
  sym_to_msgpack(w_, x_.source);
  w_.writeArrayHeader(x_.sourceParents.size());
  for (const auto &v0_ : x_.sourceParents) {
    sym_to_msgpack(w_, v0_);
  }
  structdef_to_msgpack(w_, x_.structDef);
  w_.writeArrayHeader(x_.functions.size());
  for (const auto &v0_ : x_.functions) {
    function_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.dependencies.size());
  for (const auto &v0_ : x_.dependencies) {
    structdef_to_msgpack(w_, v0_);
  }
}

Mirror mirror_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return mirror_fields_from_msgpack(r_, n_);
}

void mirror_to_msgpack(MsgpackWriter &w_, const Mirror &x_) {
  w_.writeArrayHeader(5);
  mirror_fields_to_msgpack(w_, x_);
}

PassPhase::Initial PassPhase::initial_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PassPhase::Initial with 0 field(s)");
  return {};
}

void PassPhase::initial_fields_to_msgpack(MsgpackWriter &w_, const PassPhase::Initial &x_) {}

PassPhase::Initial PassPhase::initial_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PassPhase::initial_fields_from_msgpack(r_, n_);
}

void PassPhase::initial_to_msgpack(MsgpackWriter &w_, const PassPhase::Initial &x_) {
  w_.writeArrayHeader(0);
  PassPhase::initial_fields_to_msgpack(w_, x_);
}

PassPhase::PostMono PassPhase::postmono_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PassPhase::PostMono with 0 field(s)");
  return {};
}

void PassPhase::postmono_fields_to_msgpack(MsgpackWriter &w_, const PassPhase::PostMono &x_) {}

PassPhase::PostMono PassPhase::postmono_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return PassPhase::postmono_fields_from_msgpack(r_, n_);
}

void PassPhase::postmono_to_msgpack(MsgpackWriter &w_, const PassPhase::PostMono &x_) {
  w_.writeArrayHeader(0);
  PassPhase::postmono_fields_to_msgpack(w_, x_);
}

PassPhase::Any PassPhase::any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PassPhase::initial_fields_from_msgpack(r_, n_ - 1);
      case 1: return PassPhase::postmono_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PassPhase::initial_fields_from_msgpack(r_, 0);
      case 1: return PassPhase::postmono_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void PassPhase::any_to_msgpack(MsgpackWriter &w_, const PassPhase::Any &x_) {
  x_.match_total([&](const PassPhase::Initial &y_) -> void { w_.writeInt32(0); },
                 [&](const PassPhase::PostMono &y_) -> void { w_.writeInt32(1); });
}

MetaEntry metaentry_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected MetaEntry with 2 field(s)");
  auto key = r_.readString();
  auto value = r_.readString();
  return {key, value};
}

void metaentry_fields_to_msgpack(MsgpackWriter &w_, const MetaEntry &x_) {
  w_.writeString(x_.key);
  w_.writeString(x_.value);
}

MetaEntry metaentry_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return metaentry_fields_from_msgpack(r_, n_);
}

void metaentry_to_msgpack(MsgpackWriter &w_, const MetaEntry &x_) {
  w_.writeArrayHeader(2);
  metaentry_fields_to_msgpack(w_, x_);
}

Program program_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Program with 5 field(s)");
  auto entry = function_from_msgpack(r_);
  std::vector<Function> functions;
  {
    auto functions_size = r_.readArrayHeader();
    functions.reserve(functions_size);
    for (size_t functions_idx = 0; functions_idx < functions_size; ++functions_idx) {
      auto functions_elem = function_from_msgpack(r_);
      functions.emplace_back(std::move(functions_elem));
    }
  }
  std::vector<StructDef> defs;
  {
    auto defs_size = r_.readArrayHeader();
    defs.reserve(defs_size);
    for (size_t defs_idx = 0; defs_idx < defs_size; ++defs_idx) {
      auto defs_elem = structdef_from_msgpack(r_);
      defs.emplace_back(std::move(defs_elem));
    }
  }
  auto phase = PassPhase::any_from_msgpack(r_);
  std::vector<MetaEntry> metadata;
  {
    auto metadata_size = r_.readArrayHeader();
    metadata.reserve(metadata_size);
    for (size_t metadata_idx = 0; metadata_idx < metadata_size; ++metadata_idx) {
      auto metadata_elem = metaentry_from_msgpack(r_);
      metadata.emplace_back(std::move(metadata_elem));
    }
  }
  return {entry, functions, defs, phase, metadata};
}

void program_fields_to_msgpack(MsgpackWriter &w_, const Program &x_) {
  function_to_msgpack(w_, x_.entry);
  w_.writeArrayHeader(x_.functions.size());
  for (const auto &v0_ : x_.functions) {
    function_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.defs.size());
  for (const auto &v0_ : x_.defs) {
    structdef_to_msgpack(w_, v0_);
  }
  PassPhase::any_to_msgpack(w_, x_.phase);
  w_.writeArrayHeader(x_.metadata.size());
  for (const auto &v0_ : x_.metadata) {
    metaentry_to_msgpack(w_, v0_);
  }
}

Program program_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return program_fields_from_msgpack(r_, n_);
}

void program_to_msgpack(MsgpackWriter &w_, const Program &x_) {
  w_.writeArrayHeader(5);
  program_fields_to_msgpack(w_, x_);
}

StructLayoutMember structlayoutmember_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected StructLayoutMember with 3 field(s)");
  auto name = named_from_msgpack(r_);
  auto offsetInBytes = r_.readInt64();
  auto sizeInBytes = r_.readInt64();
  return {name, offsetInBytes, sizeInBytes};
}

void structlayoutmember_fields_to_msgpack(MsgpackWriter &w_, const StructLayoutMember &x_) {
  named_to_msgpack(w_, x_.name);
  w_.writeInt64(x_.offsetInBytes);
  w_.writeInt64(x_.sizeInBytes);
}

StructLayoutMember structlayoutmember_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return structlayoutmember_fields_from_msgpack(r_, n_);
}

void structlayoutmember_to_msgpack(MsgpackWriter &w_, const StructLayoutMember &x_) {
  w_.writeArrayHeader(3);
  structlayoutmember_fields_to_msgpack(w_, x_);
}

StructLayout structlayout_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected StructLayout with 4 field(s)");
  auto name = r_.readString();
  auto sizeInBytes = r_.readInt64();
  auto alignment = r_.readInt64();
  std::vector<StructLayoutMember> members;
  {
    auto members_size = r_.readArrayHeader();
    members.reserve(members_size);
    for (size_t members_idx = 0; members_idx < members_size; ++members_idx) {
      auto members_elem = structlayoutmember_from_msgpack(r_);
      members.emplace_back(std::move(members_elem));
    }
  }
  return {name, sizeInBytes, alignment, members};
}

void structlayout_fields_to_msgpack(MsgpackWriter &w_, const StructLayout &x_) {
  w_.writeString(x_.name);
  w_.writeInt64(x_.sizeInBytes);
  w_.writeInt64(x_.alignment);
  w_.writeArrayHeader(x_.members.size());
  for (const auto &v0_ : x_.members) {
    structlayoutmember_to_msgpack(w_, v0_);
  }
}

StructLayout structlayout_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return structlayout_fields_from_msgpack(r_, n_);
}

void structlayout_to_msgpack(MsgpackWriter &w_, const StructLayout &x_) {
  w_.writeArrayHeader(4);
  structlayout_fields_to_msgpack(w_, x_);
}

CompileEvent compileevent_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected CompileEvent with 5 field(s)");
  auto epochMillis = r_.readInt64();
  auto elapsedNanos = r_.readInt64();
  auto name = r_.readString();
  auto data = r_.readString();
  std::vector<CompileEvent> items;
  {
    auto items_size = r_.readArrayHeader();
    items.reserve(items_size);
    for (size_t items_idx = 0; items_idx < items_size; ++items_idx) {
      auto items_elem = compileevent_from_msgpack(r_);
      items.emplace_back(std::move(items_elem));
    }
  }
  return {epochMillis, elapsedNanos, name, data, items};
}

void compileevent_fields_to_msgpack(MsgpackWriter &w_, const CompileEvent &x_) {
  w_.writeInt64(x_.epochMillis);
  w_.writeInt64(x_.elapsedNanos);
  w_.writeString(x_.name);
  w_.writeString(x_.data);
  w_.writeArrayHeader(x_.items.size());
  for (const auto &v0_ : x_.items) {
    compileevent_to_msgpack(w_, v0_);
  }
}

CompileEvent compileevent_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return compileevent_fields_from_msgpack(r_, n_);
}

void compileevent_to_msgpack(MsgpackWriter &w_, const CompileEvent &x_) {
  w_.writeArrayHeader(5);
  compileevent_fields_to_msgpack(w_, x_);
}

PassArg passarg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected PassArg with 2 field(s)");
  auto name = r_.readString();
  auto value = r_.readString();
  return {name, value};
}

void passarg_fields_to_msgpack(MsgpackWriter &w_, const PassArg &x_) {
  w_.writeString(x_.name);
  w_.writeString(x_.value);
}

PassArg passarg_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return passarg_fields_from_msgpack(r_, n_);
}

void passarg_to_msgpack(MsgpackWriter &w_, const PassArg &x_) {
  w_.writeArrayHeader(2);
  passarg_fields_to_msgpack(w_, x_);
}

PassSpec passspec_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected PassSpec with 2 field(s)");
  auto name = r_.readString();
  std::vector<PassArg> args;
  {
    auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = passarg_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  return {name, args};
}

void passspec_fields_to_msgpack(MsgpackWriter &w_, const PassSpec &x_) {
  w_.writeString(x_.name);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    passarg_to_msgpack(w_, v0_);
  }
}

PassSpec passspec_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return passspec_fields_from_msgpack(r_, n_);
}

void passspec_to_msgpack(MsgpackWriter &w_, const PassSpec &x_) {
  w_.writeArrayHeader(2);
  passspec_fields_to_msgpack(w_, x_);
}

PassPipeline passpipeline_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PassPipeline with 1 field(s)");
  std::vector<PassSpec> steps;
  {
    auto steps_size = r_.readArrayHeader();
    steps.reserve(steps_size);
    for (size_t steps_idx = 0; steps_idx < steps_size; ++steps_idx) {
      auto steps_elem = passspec_from_msgpack(r_);
      steps.emplace_back(std::move(steps_elem));
    }
  }
  return PassPipeline(steps);
}

void passpipeline_fields_to_msgpack(MsgpackWriter &w_, const PassPipeline &x_) {
  w_.writeArrayHeader(x_.steps.size());
  for (const auto &v0_ : x_.steps) {
    passspec_to_msgpack(w_, v0_);
  }
}

PassPipeline passpipeline_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return passpipeline_fields_from_msgpack(r_, n_);
}

void passpipeline_to_msgpack(MsgpackWriter &w_, const PassPipeline &x_) {
  w_.writeArrayHeader(1);
  passpipeline_fields_to_msgpack(w_, x_);
}

PassRunResult passrunresult_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected PassRunResult with 2 field(s)");
  auto program = program_from_msgpack(r_);
  auto event = compileevent_from_msgpack(r_);
  return {program, event};
}

void passrunresult_fields_to_msgpack(MsgpackWriter &w_, const PassRunResult &x_) {
  program_to_msgpack(w_, x_.program);
  compileevent_to_msgpack(w_, x_.event);
}

PassRunResult passrunresult_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return passrunresult_fields_from_msgpack(r_, n_);
}

void passrunresult_to_msgpack(MsgpackWriter &w_, const PassRunResult &x_) {
  w_.writeArrayHeader(2);
  passrunresult_fields_to_msgpack(w_, x_);
}

CompileResult compileresult_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected CompileResult with 5 field(s)");
  std::optional<std::vector<int8_t>> binary;
  if (!r_.tryReadNil()) {
    std::vector<int8_t> binary_value;
    {
      auto binary_value_size = r_.readArrayHeader();
      binary_value.reserve(binary_value_size);
      for (size_t binary_value_idx = 0; binary_value_idx < binary_value_size; ++binary_value_idx) {
        auto binary_value_elem = static_cast<int8_t>(r_.readInt32());
        binary_value.emplace_back(std::move(binary_value_elem));
      }
    }
    binary = std::move(binary_value);
  }
  std::vector<std::string> features;
  {
    auto features_size = r_.readArrayHeader();
    features.reserve(features_size);
    for (size_t features_idx = 0; features_idx < features_size; ++features_idx) {
      auto features_elem = r_.readString();
      features.emplace_back(std::move(features_elem));
    }
  }
  std::vector<CompileEvent> events;
  {
    auto events_size = r_.readArrayHeader();
    events.reserve(events_size);
    for (size_t events_idx = 0; events_idx < events_size; ++events_idx) {
      auto events_elem = compileevent_from_msgpack(r_);
      events.emplace_back(std::move(events_elem));
    }
  }
  std::vector<StructLayout> layouts;
  {
    auto layouts_size = r_.readArrayHeader();
    layouts.reserve(layouts_size);
    for (size_t layouts_idx = 0; layouts_idx < layouts_size; ++layouts_idx) {
      auto layouts_elem = structlayout_from_msgpack(r_);
      layouts.emplace_back(std::move(layouts_elem));
    }
  }
  auto messages = r_.readString();
  return {binary, features, events, layouts, messages};
}

void compileresult_fields_to_msgpack(MsgpackWriter &w_, const CompileResult &x_) {
  if (x_.binary) {
    w_.writeArrayHeader((*x_.binary).size());
    for (const auto &v1_ : (*x_.binary)) {
      w_.writeInt32(static_cast<int32_t>(v1_));
    }
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.features.size());
  for (const auto &v0_ : x_.features) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.events.size());
  for (const auto &v0_ : x_.events) {
    compileevent_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.layouts.size());
  for (const auto &v0_ : x_.layouts) {
    structlayout_to_msgpack(w_, v0_);
  }
  w_.writeString(x_.messages);
}

CompileResult compileresult_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  return compileresult_fields_from_msgpack(r_, n_);
}

void compileresult_to_msgpack(MsgpackWriter &w_, const CompileResult &x_) {
  w_.writeArrayHeader(5);
  compileresult_fields_to_msgpack(w_, x_);
}

static void structdefs_value_to_msgpack(MsgpackWriter &w_, const std::vector<StructDef> &xs_) {
  w_.writeArrayHeader(xs_.size());
  for (const auto &x_ : xs_)
    structdef_to_msgpack(w_, x_);
}

static std::vector<StructDef> structdefs_value_from_msgpack(MsgpackReader &r_) {
  auto n_ = r_.readArrayHeader();
  std::vector<StructDef> xs_;
  xs_.reserve(n_);
  for (size_t i_ = 0; i_ < n_; ++i_)
    xs_.emplace_back(structdef_from_msgpack(r_));
  return xs_;
}

std::vector<uint8_t> program_to_msgpack(const Program &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { program_to_msgpack(w_, x_); });
}

Program program_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) { return program_from_msgpack(r_); });
}

Program program_from_msgpack(const std::vector<uint8_t> &xs_) { return program_from_msgpack(xs_.data(), xs_.data() + xs_.size()); }

std::vector<uint8_t> hashed_program_to_msgpack(const Program &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(AdtHash));
    program_to_msgpack(w_, x_);
  });
}

Program hashed_program_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned Program array of size 2");
    auto hash_ = r_.readString();
    if (hash_ != AdtHash) throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
    return program_from_msgpack(r_);
  });
}

Program hashed_program_from_msgpack(const std::vector<uint8_t> &xs_) {
  return hashed_program_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> structdefs_to_msgpack(const std::vector<StructDef> &xs_) {
  return encodeInterned([&](MsgpackWriter &w_) { structdefs_value_to_msgpack(w_, xs_); });
}

std::vector<StructDef> structdefs_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) { return structdefs_value_from_msgpack(r_); });
}

std::vector<StructDef> structdefs_from_msgpack(const std::vector<uint8_t> &xs_) {
  return structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> hashed_structdefs_to_msgpack(const std::vector<StructDef> &xs_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(AdtHash));
    structdefs_value_to_msgpack(w_, xs_);
  });
}

std::vector<StructDef> hashed_structdefs_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned StructDef list array of size 2");
    auto hash_ = r_.readString();
    if (hash_ != AdtHash) throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
    return structdefs_value_from_msgpack(r_);
  });
}

std::vector<StructDef> hashed_structdefs_from_msgpack(const std::vector<uint8_t> &xs_) {
  return hashed_structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> compileresult_to_msgpack(const CompileResult &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { compileresult_to_msgpack(w_, x_); });
}

CompileResult compileresult_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) { return compileresult_from_msgpack(r_); });
}

CompileResult compileresult_from_msgpack(const std::vector<uint8_t> &xs_) {
  return compileresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> passrunresult_to_msgpack(const PassRunResult &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { passrunresult_to_msgpack(w_, x_); });
}

PassRunResult passrunresult_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) { return passrunresult_from_msgpack(r_); });
}

PassRunResult passrunresult_from_msgpack(const std::vector<uint8_t> &xs_) {
  return passrunresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

} // namespace polyregion::polyast
