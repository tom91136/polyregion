#include "polyast.h"

namespace polyregion::polyast {

SourcePosition::SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept : file(std::move(file)), line(line), col(std::move(col)) {}
size_t SourcePosition::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(file)>()(file) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(line)>()(line) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(col)>()(col) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const SourcePosition &x) { return x.dump(os); }
std::ostream &SourcePosition::dump(std::ostream &os) const {
  os << "SourcePosition(";
  os << '"' << file << '"';
  os << ',';
  os << line;
  os << ',';
  os << '{';
  if (col) {
    os << (*col);
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool SourcePosition::operator==(const SourcePosition& rhs) const {
  return (file == rhs.file) && (line == rhs.line) && (col == rhs.col);
}

Named::Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
size_t Named::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(symbol)>()(symbol) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpe)>()(tpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Named &x) { return x.dump(os); }
std::ostream &Named::dump(std::ostream &os) const {
  os << "Named(";
  os << '"' << symbol << '"';
  os << ',';
  os << tpe;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Named::operator==(const Named& rhs) const {
  return (symbol == rhs.symbol) && (tpe == rhs.tpe);
}

TypeKind::Base::Base() = default;
uint32_t TypeKind::Any::id() const { return _v->id(); }
size_t TypeKind::Any::hash_code() const { return _v->hash_code(); }
std::ostream &TypeKind::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace TypeKind { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool TypeKind::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool TypeKind::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool TypeKind::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

TypeKind::None::None() noexcept : TypeKind::Base() {}
uint32_t TypeKind::None::id() const { return variant_id; };
size_t TypeKind::None::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeKind { std::ostream &operator<<(std::ostream &os, const TypeKind::None &x) { return x.dump(os); } }
std::ostream &TypeKind::None::dump(std::ostream &os) const {
  os << "None(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeKind::None::operator==(const TypeKind::None& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeKind::None::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::None::operator<(const TypeKind::None& rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::None::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeKind::None::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<None>(*this)); }
TypeKind::Any TypeKind::None::widen() const { return Any(*this); };

TypeKind::Ref::Ref() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Ref::id() const { return variant_id; };
size_t TypeKind::Ref::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeKind { std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &x) { return x.dump(os); } }
std::ostream &TypeKind::Ref::dump(std::ostream &os) const {
  os << "Ref(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeKind::Ref::operator==(const TypeKind::Ref& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeKind::Ref::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Ref::operator<(const TypeKind::Ref& rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Ref::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Ref::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ref>(*this)); }
TypeKind::Any TypeKind::Ref::widen() const { return Any(*this); };

TypeKind::Integral::Integral() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Integral::id() const { return variant_id; };
size_t TypeKind::Integral::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeKind { std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &x) { return x.dump(os); } }
std::ostream &TypeKind::Integral::dump(std::ostream &os) const {
  os << "Integral(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeKind::Integral::operator==(const TypeKind::Integral& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeKind::Integral::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Integral::operator<(const TypeKind::Integral& rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Integral::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Integral::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Integral>(*this)); }
TypeKind::Any TypeKind::Integral::widen() const { return Any(*this); };

TypeKind::Fractional::Fractional() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Fractional::id() const { return variant_id; };
size_t TypeKind::Fractional::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeKind { std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &x) { return x.dump(os); } }
std::ostream &TypeKind::Fractional::dump(std::ostream &os) const {
  os << "Fractional(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeKind::Fractional::operator==(const TypeKind::Fractional& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeKind::Fractional::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Fractional::operator<(const TypeKind::Fractional& rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Fractional::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Fractional::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Fractional>(*this)); }
TypeKind::Any TypeKind::Fractional::widen() const { return Any(*this); };

TypeSpace::Base::Base() = default;
uint32_t TypeSpace::Any::id() const { return _v->id(); }
size_t TypeSpace::Any::hash_code() const { return _v->hash_code(); }
std::ostream &TypeSpace::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace TypeSpace { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool TypeSpace::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool TypeSpace::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool TypeSpace::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

TypeSpace::Global::Global() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Global::id() const { return variant_id; };
size_t TypeSpace::Global::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeSpace { std::ostream &operator<<(std::ostream &os, const TypeSpace::Global &x) { return x.dump(os); } }
std::ostream &TypeSpace::Global::dump(std::ostream &os) const {
  os << "Global(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeSpace::Global::operator==(const TypeSpace::Global& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Global::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Global::operator<(const TypeSpace::Global& rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Global::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Global::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Global>(*this)); }
TypeSpace::Any TypeSpace::Global::widen() const { return Any(*this); };

TypeSpace::Local::Local() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Local::id() const { return variant_id; };
size_t TypeSpace::Local::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace TypeSpace { std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &x) { return x.dump(os); } }
std::ostream &TypeSpace::Local::dump(std::ostream &os) const {
  os << "Local(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool TypeSpace::Local::operator==(const TypeSpace::Local& rhs) const {
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Local::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Local::operator<(const TypeSpace::Local& rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Local::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Local::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Local>(*this)); }
TypeSpace::Any TypeSpace::Local::widen() const { return Any(*this); };

Type::Base::Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
uint32_t Type::Any::id() const { return _v->id(); }
size_t Type::Any::hash_code() const { return _v->hash_code(); }
TypeKind::Any Type::Any::kind() const { return _v->kind; }
std::ostream &Type::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Type { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Type::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Type::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Type::Float16::Float16() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float16::id() const { return variant_id; };
size_t Type::Float16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Float16 &x) { return x.dump(os); } }
std::ostream &Type::Float16::dump(std::ostream &os) const {
  os << "Float16(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Float16::operator==(const Type::Float16& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Float16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float16::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float16>(*this)); }
Type::Any Type::Float16::widen() const { return Any(*this); };

Type::Float32::Float32() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float32::id() const { return variant_id; };
size_t Type::Float32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Float32 &x) { return x.dump(os); } }
std::ostream &Type::Float32::dump(std::ostream &os) const {
  os << "Float32(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Float32::operator==(const Type::Float32& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Float32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float32::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float32>(*this)); }
Type::Any Type::Float32::widen() const { return Any(*this); };

Type::Float64::Float64() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float64::id() const { return variant_id; };
size_t Type::Float64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Float64 &x) { return x.dump(os); } }
std::ostream &Type::Float64::dump(std::ostream &os) const {
  os << "Float64(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Float64::operator==(const Type::Float64& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Float64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float64::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float64>(*this)); }
Type::Any Type::Float64::widen() const { return Any(*this); };

Type::IntU8::IntU8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU8::id() const { return variant_id; };
size_t Type::IntU8::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntU8 &x) { return x.dump(os); } }
std::ostream &Type::IntU8::dump(std::ostream &os) const {
  os << "IntU8(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntU8::operator==(const Type::IntU8& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntU8::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU8::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU8>(*this)); }
Type::Any Type::IntU8::widen() const { return Any(*this); };

Type::IntU16::IntU16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU16::id() const { return variant_id; };
size_t Type::IntU16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntU16 &x) { return x.dump(os); } }
std::ostream &Type::IntU16::dump(std::ostream &os) const {
  os << "IntU16(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntU16::operator==(const Type::IntU16& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntU16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU16::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU16>(*this)); }
Type::Any Type::IntU16::widen() const { return Any(*this); };

Type::IntU32::IntU32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU32::id() const { return variant_id; };
size_t Type::IntU32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntU32 &x) { return x.dump(os); } }
std::ostream &Type::IntU32::dump(std::ostream &os) const {
  os << "IntU32(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntU32::operator==(const Type::IntU32& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntU32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU32::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU32>(*this)); }
Type::Any Type::IntU32::widen() const { return Any(*this); };

Type::IntU64::IntU64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU64::id() const { return variant_id; };
size_t Type::IntU64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntU64 &x) { return x.dump(os); } }
std::ostream &Type::IntU64::dump(std::ostream &os) const {
  os << "IntU64(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntU64::operator==(const Type::IntU64& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntU64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU64::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU64>(*this)); }
Type::Any Type::IntU64::widen() const { return Any(*this); };

Type::IntS8::IntS8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS8::id() const { return variant_id; };
size_t Type::IntS8::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntS8 &x) { return x.dump(os); } }
std::ostream &Type::IntS8::dump(std::ostream &os) const {
  os << "IntS8(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntS8::operator==(const Type::IntS8& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntS8::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS8::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS8>(*this)); }
Type::Any Type::IntS8::widen() const { return Any(*this); };

Type::IntS16::IntS16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS16::id() const { return variant_id; };
size_t Type::IntS16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntS16 &x) { return x.dump(os); } }
std::ostream &Type::IntS16::dump(std::ostream &os) const {
  os << "IntS16(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntS16::operator==(const Type::IntS16& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntS16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS16::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS16>(*this)); }
Type::Any Type::IntS16::widen() const { return Any(*this); };

Type::IntS32::IntS32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS32::id() const { return variant_id; };
size_t Type::IntS32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntS32 &x) { return x.dump(os); } }
std::ostream &Type::IntS32::dump(std::ostream &os) const {
  os << "IntS32(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntS32::operator==(const Type::IntS32& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntS32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS32::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS32>(*this)); }
Type::Any Type::IntS32::widen() const { return Any(*this); };

Type::IntS64::IntS64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS64::id() const { return variant_id; };
size_t Type::IntS64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::IntS64 &x) { return x.dump(os); } }
std::ostream &Type::IntS64::dump(std::ostream &os) const {
  os << "IntS64(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::IntS64::operator==(const Type::IntS64& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::IntS64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS64::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS64>(*this)); }
Type::Any Type::IntS64::widen() const { return Any(*this); };

Type::Nothing::Nothing() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Nothing::id() const { return variant_id; };
size_t Type::Nothing::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Nothing &x) { return x.dump(os); } }
std::ostream &Type::Nothing::dump(std::ostream &os) const {
  os << "Nothing(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Nothing::operator==(const Type::Nothing& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Nothing::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Nothing::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Nothing>(*this)); }
Type::Any Type::Nothing::widen() const { return Any(*this); };

Type::Unit0::Unit0() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Unit0::id() const { return variant_id; };
size_t Type::Unit0::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Unit0 &x) { return x.dump(os); } }
std::ostream &Type::Unit0::dump(std::ostream &os) const {
  os << "Unit0(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Unit0::operator==(const Type::Unit0& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Unit0::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Unit0::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Unit0>(*this)); }
Type::Any Type::Unit0::widen() const { return Any(*this); };

Type::Bool1::Bool1() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::Bool1::id() const { return variant_id; };
size_t Type::Bool1::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Bool1 &x) { return x.dump(os); } }
std::ostream &Type::Bool1::dump(std::ostream &os) const {
  os << "Bool1(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Bool1::operator==(const Type::Bool1& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Type::Bool1::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Bool1::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Bool1>(*this)); }
Type::Any Type::Bool1::widen() const { return Any(*this); };

Type::Struct::Struct(std::string name) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)) {}
uint32_t Type::Struct::id() const { return variant_id; };
size_t Type::Struct::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Struct &x) { return x.dump(os); } }
std::ostream &Type::Struct::dump(std::ostream &os) const {
  os << "Struct(";
  os << '"' << name << '"';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Struct::operator==(const Type::Struct& rhs) const {
  return (this->name == rhs.name);
}
POLYREGION_EXPORT bool Type::Struct::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Struct&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Struct::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Struct>(*this)); }
Type::Any Type::Struct::widen() const { return Any(*this); };

Type::Ptr::Ptr(Type::Any component, std::optional<int32_t> length, TypeSpace::Any space) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)), length(std::move(length)), space(std::move(space)) {}
uint32_t Type::Ptr::id() const { return variant_id; };
size_t Type::Ptr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(length)>()(length) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Ptr &x) { return x.dump(os); } }
std::ostream &Type::Ptr::dump(std::ostream &os) const {
  os << "Ptr(";
  os << component;
  os << ',';
  os << '{';
  if (length) {
    os << (*length);
  }
  os << '}';
  os << ',';
  os << space;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Ptr::operator==(const Type::Ptr& rhs) const {
  return (this->component == rhs.component) && (this->length == rhs.length) && (this->space == rhs.space);
}
POLYREGION_EXPORT bool Type::Ptr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Ptr&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Ptr::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ptr>(*this)); }
Type::Any Type::Ptr::widen() const { return Any(*this); };

Type::Annotated::Annotated(Type::Any tpe, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept : Type::Base(tpe.kind()), tpe(std::move(tpe)), pos(std::move(pos)), comment(std::move(comment)) {}
uint32_t Type::Annotated::id() const { return variant_id; };
size_t Type::Annotated::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(tpe)>()(tpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comment)>()(comment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Type { std::ostream &operator<<(std::ostream &os, const Type::Annotated &x) { return x.dump(os); } }
std::ostream &Type::Annotated::dump(std::ostream &os) const {
  os << "Annotated(";
  os << tpe;
  os << ',';
  os << '{';
  if (pos) {
    os << (*pos);
  }
  os << '}';
  os << ',';
  os << '{';
  if (comment) {
    os << '"' << (*comment) << '"';
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Type::Annotated::operator==(const Type::Annotated& rhs) const {
  return (this->tpe == rhs.tpe) && (this->pos == rhs.pos) && (this->comment == rhs.comment);
}
POLYREGION_EXPORT bool Type::Annotated::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Annotated&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Annotated::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Annotated>(*this)); }
Type::Any Type::Annotated::widen() const { return Any(*this); };

Expr::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
uint32_t Expr::Any::id() const { return _v->id(); }
size_t Expr::Any::hash_code() const { return _v->hash_code(); }
Type::Any Expr::Any::tpe() const { return _v->tpe; }
std::ostream &Expr::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Expr { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Expr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Expr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Expr::Float16Const::Float16Const(float value) noexcept : Expr::Base(Type::Float16()), value(value) {}
uint32_t Expr::Float16Const::id() const { return variant_id; };
size_t Expr::Float16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Float16Const &x) { return x.dump(os); } }
std::ostream &Expr::Float16Const::dump(std::ostream &os) const {
  os << "Float16Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Float16Const::operator==(const Expr::Float16Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::Float16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Float16Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Float16Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float16Const>(*this)); }
Expr::Any Expr::Float16Const::widen() const { return Any(*this); };

Expr::Float32Const::Float32Const(float value) noexcept : Expr::Base(Type::Float32()), value(value) {}
uint32_t Expr::Float32Const::id() const { return variant_id; };
size_t Expr::Float32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Float32Const &x) { return x.dump(os); } }
std::ostream &Expr::Float32Const::dump(std::ostream &os) const {
  os << "Float32Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Float32Const::operator==(const Expr::Float32Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::Float32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Float32Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Float32Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float32Const>(*this)); }
Expr::Any Expr::Float32Const::widen() const { return Any(*this); };

Expr::Float64Const::Float64Const(double value) noexcept : Expr::Base(Type::Float64()), value(value) {}
uint32_t Expr::Float64Const::id() const { return variant_id; };
size_t Expr::Float64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Float64Const &x) { return x.dump(os); } }
std::ostream &Expr::Float64Const::dump(std::ostream &os) const {
  os << "Float64Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Float64Const::operator==(const Expr::Float64Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::Float64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Float64Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Float64Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float64Const>(*this)); }
Expr::Any Expr::Float64Const::widen() const { return Any(*this); };

Expr::IntU8Const::IntU8Const(int8_t value) noexcept : Expr::Base(Type::IntU8()), value(value) {}
uint32_t Expr::IntU8Const::id() const { return variant_id; };
size_t Expr::IntU8Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntU8Const &x) { return x.dump(os); } }
std::ostream &Expr::IntU8Const::dump(std::ostream &os) const {
  os << "IntU8Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntU8Const::operator==(const Expr::IntU8Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntU8Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntU8Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntU8Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU8Const>(*this)); }
Expr::Any Expr::IntU8Const::widen() const { return Any(*this); };

Expr::IntU16Const::IntU16Const(uint16_t value) noexcept : Expr::Base(Type::IntU16()), value(value) {}
uint32_t Expr::IntU16Const::id() const { return variant_id; };
size_t Expr::IntU16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntU16Const &x) { return x.dump(os); } }
std::ostream &Expr::IntU16Const::dump(std::ostream &os) const {
  os << "IntU16Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntU16Const::operator==(const Expr::IntU16Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntU16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntU16Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntU16Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU16Const>(*this)); }
Expr::Any Expr::IntU16Const::widen() const { return Any(*this); };

Expr::IntU32Const::IntU32Const(int32_t value) noexcept : Expr::Base(Type::IntU32()), value(value) {}
uint32_t Expr::IntU32Const::id() const { return variant_id; };
size_t Expr::IntU32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntU32Const &x) { return x.dump(os); } }
std::ostream &Expr::IntU32Const::dump(std::ostream &os) const {
  os << "IntU32Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntU32Const::operator==(const Expr::IntU32Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntU32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntU32Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntU32Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU32Const>(*this)); }
Expr::Any Expr::IntU32Const::widen() const { return Any(*this); };

Expr::IntU64Const::IntU64Const(int64_t value) noexcept : Expr::Base(Type::IntU64()), value(value) {}
uint32_t Expr::IntU64Const::id() const { return variant_id; };
size_t Expr::IntU64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntU64Const &x) { return x.dump(os); } }
std::ostream &Expr::IntU64Const::dump(std::ostream &os) const {
  os << "IntU64Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntU64Const::operator==(const Expr::IntU64Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntU64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntU64Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntU64Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU64Const>(*this)); }
Expr::Any Expr::IntU64Const::widen() const { return Any(*this); };

Expr::IntS8Const::IntS8Const(int8_t value) noexcept : Expr::Base(Type::IntS8()), value(value) {}
uint32_t Expr::IntS8Const::id() const { return variant_id; };
size_t Expr::IntS8Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntS8Const &x) { return x.dump(os); } }
std::ostream &Expr::IntS8Const::dump(std::ostream &os) const {
  os << "IntS8Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntS8Const::operator==(const Expr::IntS8Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntS8Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntS8Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntS8Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS8Const>(*this)); }
Expr::Any Expr::IntS8Const::widen() const { return Any(*this); };

Expr::IntS16Const::IntS16Const(int16_t value) noexcept : Expr::Base(Type::IntS16()), value(value) {}
uint32_t Expr::IntS16Const::id() const { return variant_id; };
size_t Expr::IntS16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntS16Const &x) { return x.dump(os); } }
std::ostream &Expr::IntS16Const::dump(std::ostream &os) const {
  os << "IntS16Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntS16Const::operator==(const Expr::IntS16Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntS16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntS16Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntS16Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS16Const>(*this)); }
Expr::Any Expr::IntS16Const::widen() const { return Any(*this); };

Expr::IntS32Const::IntS32Const(int32_t value) noexcept : Expr::Base(Type::IntS32()), value(value) {}
uint32_t Expr::IntS32Const::id() const { return variant_id; };
size_t Expr::IntS32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntS32Const &x) { return x.dump(os); } }
std::ostream &Expr::IntS32Const::dump(std::ostream &os) const {
  os << "IntS32Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntS32Const::operator==(const Expr::IntS32Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntS32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntS32Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntS32Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS32Const>(*this)); }
Expr::Any Expr::IntS32Const::widen() const { return Any(*this); };

Expr::IntS64Const::IntS64Const(int64_t value) noexcept : Expr::Base(Type::IntS64()), value(value) {}
uint32_t Expr::IntS64Const::id() const { return variant_id; };
size_t Expr::IntS64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntS64Const &x) { return x.dump(os); } }
std::ostream &Expr::IntS64Const::dump(std::ostream &os) const {
  os << "IntS64Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntS64Const::operator==(const Expr::IntS64Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::IntS64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntS64Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntS64Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS64Const>(*this)); }
Expr::Any Expr::IntS64Const::widen() const { return Any(*this); };

Expr::Unit0Const::Unit0Const() noexcept : Expr::Base(Type::Unit0()) {}
uint32_t Expr::Unit0Const::id() const { return variant_id; };
size_t Expr::Unit0Const::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Unit0Const &x) { return x.dump(os); } }
std::ostream &Expr::Unit0Const::dump(std::ostream &os) const {
  os << "Unit0Const(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Unit0Const::operator==(const Expr::Unit0Const& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Expr::Unit0Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Expr::Unit0Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Unit0Const>(*this)); }
Expr::Any Expr::Unit0Const::widen() const { return Any(*this); };

Expr::Bool1Const::Bool1Const(bool value) noexcept : Expr::Base(Type::Bool1()), value(value) {}
uint32_t Expr::Bool1Const::id() const { return variant_id; };
size_t Expr::Bool1Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Bool1Const &x) { return x.dump(os); } }
std::ostream &Expr::Bool1Const::dump(std::ostream &os) const {
  os << "Bool1Const(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Bool1Const::operator==(const Expr::Bool1Const& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Expr::Bool1Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Bool1Const&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Bool1Const::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Bool1Const>(*this)); }
Expr::Any Expr::Bool1Const::widen() const { return Any(*this); };

Expr::SpecOp::SpecOp(Spec::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::SpecOp::id() const { return variant_id; };
size_t Expr::SpecOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::SpecOp &x) { return x.dump(os); } }
std::ostream &Expr::SpecOp::dump(std::ostream &os) const {
  os << "SpecOp(";
  os << op;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::SpecOp::operator==(const Expr::SpecOp& rhs) const {
  return (this->op == rhs.op);
}
POLYREGION_EXPORT bool Expr::SpecOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::SpecOp&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::SpecOp::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<SpecOp>(*this)); }
Expr::Any Expr::SpecOp::widen() const { return Any(*this); };

Expr::MathOp::MathOp(Math::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::MathOp::id() const { return variant_id; };
size_t Expr::MathOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::MathOp &x) { return x.dump(os); } }
std::ostream &Expr::MathOp::dump(std::ostream &os) const {
  os << "MathOp(";
  os << op;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::MathOp::operator==(const Expr::MathOp& rhs) const {
  return (this->op == rhs.op);
}
POLYREGION_EXPORT bool Expr::MathOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::MathOp&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::MathOp::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<MathOp>(*this)); }
Expr::Any Expr::MathOp::widen() const { return Any(*this); };

Expr::IntrOp::IntrOp(Intr::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::IntrOp::id() const { return variant_id; };
size_t Expr::IntrOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::IntrOp &x) { return x.dump(os); } }
std::ostream &Expr::IntrOp::dump(std::ostream &os) const {
  os << "IntrOp(";
  os << op;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::IntrOp::operator==(const Expr::IntrOp& rhs) const {
  return (this->op == rhs.op);
}
POLYREGION_EXPORT bool Expr::IntrOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntrOp&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntrOp::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntrOp>(*this)); }
Expr::Any Expr::IntrOp::widen() const { return Any(*this); };

Expr::Select::Select(std::vector<Named> init, Named last) noexcept : Expr::Base(last.tpe), init(std::move(init)), last(std::move(last)) {}
uint32_t Expr::Select::id() const { return variant_id; };
size_t Expr::Select::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(init)>()(init) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(last)>()(last) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Select &x) { return x.dump(os); } }
std::ostream &Expr::Select::dump(std::ostream &os) const {
  os << "Select(";
  os << '{';
  if (!init.empty()) {
    std::for_each(init.begin(), std::prev(init.end()), [&os](auto &&x) { os << x; os << ','; });
    os << init.back();
  }
  os << '}';
  os << ',';
  os << last;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Select::operator==(const Expr::Select& rhs) const {
  return (this->init == rhs.init) && (this->last == rhs.last);
}
POLYREGION_EXPORT bool Expr::Select::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Select&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Select::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Select>(*this)); }
Expr::Any Expr::Select::widen() const { return Any(*this); };

Expr::Poison::Poison(Type::Any t) noexcept : Expr::Base(t), t(std::move(t)) {}
uint32_t Expr::Poison::id() const { return variant_id; };
size_t Expr::Poison::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(t)>()(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Poison &x) { return x.dump(os); } }
std::ostream &Expr::Poison::dump(std::ostream &os) const {
  os << "Poison(";
  os << t;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Poison::operator==(const Expr::Poison& rhs) const {
  return (this->t == rhs.t);
}
POLYREGION_EXPORT bool Expr::Poison::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Poison&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Poison::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Poison>(*this)); }
Expr::Any Expr::Poison::widen() const { return Any(*this); };

Expr::Cast::Cast(Expr::Any from, Type::Any as) noexcept : Expr::Base(as), from(std::move(from)), as(std::move(as)) {}
uint32_t Expr::Cast::id() const { return variant_id; };
size_t Expr::Cast::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(from)>()(from) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(as)>()(as) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Cast &x) { return x.dump(os); } }
std::ostream &Expr::Cast::dump(std::ostream &os) const {
  os << "Cast(";
  os << from;
  os << ',';
  os << as;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Cast::operator==(const Expr::Cast& rhs) const {
  return (this->from == rhs.from) && (this->as == rhs.as);
}
POLYREGION_EXPORT bool Expr::Cast::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Cast&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Cast::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cast>(*this)); }
Expr::Any Expr::Cast::widen() const { return Any(*this); };

Expr::Index::Index(Expr::Any lhs, Expr::Any idx, Type::Any component) noexcept : Expr::Base(component), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::Index::id() const { return variant_id; };
size_t Expr::Index::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Index &x) { return x.dump(os); } }
std::ostream &Expr::Index::dump(std::ostream &os) const {
  os << "Index(";
  os << lhs;
  os << ',';
  os << idx;
  os << ',';
  os << component;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Index::operator==(const Expr::Index& rhs) const {
  return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->component == rhs.component);
}
POLYREGION_EXPORT bool Expr::Index::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Index&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Index::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Index>(*this)); }
Expr::Any Expr::Index::widen() const { return Any(*this); };

Expr::RefTo::RefTo(Expr::Any lhs, std::optional<Expr::Any> idx, Type::Any component) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::RefTo::id() const { return variant_id; };
size_t Expr::RefTo::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::RefTo &x) { return x.dump(os); } }
std::ostream &Expr::RefTo::dump(std::ostream &os) const {
  os << "RefTo(";
  os << lhs;
  os << ',';
  os << '{';
  if (idx) {
    os << (*idx);
  }
  os << '}';
  os << ',';
  os << component;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::RefTo::operator==(const Expr::RefTo& rhs) const {
  return (this->lhs == rhs.lhs) && ( (!this->idx && !rhs.idx) || (this->idx && rhs.idx && *this->idx == *rhs.idx) ) && (this->component == rhs.component);
}
POLYREGION_EXPORT bool Expr::RefTo::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::RefTo&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::RefTo::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<RefTo>(*this)); }
Expr::Any Expr::RefTo::widen() const { return Any(*this); };

Expr::Alloc::Alloc(Type::Any component, Expr::Any size) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), component(std::move(component)), size(std::move(size)) {}
uint32_t Expr::Alloc::id() const { return variant_id; };
size_t Expr::Alloc::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(size)>()(size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Alloc &x) { return x.dump(os); } }
std::ostream &Expr::Alloc::dump(std::ostream &os) const {
  os << "Alloc(";
  os << component;
  os << ',';
  os << size;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Alloc::operator==(const Expr::Alloc& rhs) const {
  return (this->component == rhs.component) && (this->size == rhs.size);
}
POLYREGION_EXPORT bool Expr::Alloc::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Alloc&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Alloc::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Alloc>(*this)); }
Expr::Any Expr::Alloc::widen() const { return Any(*this); };

Expr::Invoke::Invoke(std::string name, std::vector<Expr::Any> args, Type::Any rtn) noexcept : Expr::Base(rtn), name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)) {}
uint32_t Expr::Invoke::id() const { return variant_id; };
size_t Expr::Invoke::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Invoke &x) { return x.dump(os); } }
std::ostream &Expr::Invoke::dump(std::ostream &os) const {
  os << "Invoke(";
  os << '"' << name << '"';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Invoke::operator==(const Expr::Invoke& rhs) const {
  return (this->name == rhs.name) && std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Expr::Invoke::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Invoke&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Invoke::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Invoke>(*this)); }
Expr::Any Expr::Invoke::widen() const { return Any(*this); };

Expr::Annotated::Annotated(Expr::Any expr, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept : Expr::Base(expr.tpe()), expr(std::move(expr)), pos(std::move(pos)), comment(std::move(comment)) {}
uint32_t Expr::Annotated::id() const { return variant_id; };
size_t Expr::Annotated::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comment)>()(comment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Expr { std::ostream &operator<<(std::ostream &os, const Expr::Annotated &x) { return x.dump(os); } }
std::ostream &Expr::Annotated::dump(std::ostream &os) const {
  os << "Annotated(";
  os << expr;
  os << ',';
  os << '{';
  if (pos) {
    os << (*pos);
  }
  os << '}';
  os << ',';
  os << '{';
  if (comment) {
    os << '"' << (*comment) << '"';
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Expr::Annotated::operator==(const Expr::Annotated& rhs) const {
  return (this->expr == rhs.expr) && (this->pos == rhs.pos) && (this->comment == rhs.comment);
}
POLYREGION_EXPORT bool Expr::Annotated::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Annotated&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Annotated::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Annotated>(*this)); }
Expr::Any Expr::Annotated::widen() const { return Any(*this); };

Overload::Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept : args(std::move(args)), rtn(std::move(rtn)) {}
size_t Overload::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Overload &x) { return x.dump(os); }
std::ostream &Overload::dump(std::ostream &os) const {
  os << "Overload(";
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Overload::operator==(const Overload& rhs) const {
  return std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

Spec::Base::Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept : overloads(std::move(overloads)), exprs(std::move(exprs)), tpe(std::move(tpe)) {}
uint32_t Spec::Any::id() const { return _v->id(); }
size_t Spec::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Spec::Any::overloads() const { return _v->overloads; }
std::vector<Expr::Any> Spec::Any::exprs() const { return _v->exprs; }
Type::Any Spec::Any::tpe() const { return _v->tpe; }
std::ostream &Spec::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Spec { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Spec::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Spec::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Spec::Assert::Assert() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Nothing()) {}
uint32_t Spec::Assert::id() const { return variant_id; };
size_t Spec::Assert::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::Assert &x) { return x.dump(os); } }
std::ostream &Spec::Assert::dump(std::ostream &os) const {
  os << "Assert(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::Assert::operator==(const Spec::Assert& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::Assert::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::Assert::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Assert>(*this)); }
Spec::Any Spec::Assert::widen() const { return Any(*this); };

Spec::GpuBarrierGlobal::GpuBarrierGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierGlobal::id() const { return variant_id; };
size_t Spec::GpuBarrierGlobal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &x) { return x.dump(os); } }
std::ostream &Spec::GpuBarrierGlobal::dump(std::ostream &os) const {
  os << "GpuBarrierGlobal(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuBarrierGlobal::operator==(const Spec::GpuBarrierGlobal& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuBarrierGlobal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierGlobal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierGlobal>(*this)); }
Spec::Any Spec::GpuBarrierGlobal::widen() const { return Any(*this); };

Spec::GpuBarrierLocal::GpuBarrierLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierLocal::id() const { return variant_id; };
size_t Spec::GpuBarrierLocal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierLocal &x) { return x.dump(os); } }
std::ostream &Spec::GpuBarrierLocal::dump(std::ostream &os) const {
  os << "GpuBarrierLocal(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuBarrierLocal::operator==(const Spec::GpuBarrierLocal& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuBarrierLocal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierLocal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierLocal>(*this)); }
Spec::Any Spec::GpuBarrierLocal::widen() const { return Any(*this); };

Spec::GpuBarrierAll::GpuBarrierAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierAll::id() const { return variant_id; };
size_t Spec::GpuBarrierAll::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierAll &x) { return x.dump(os); } }
std::ostream &Spec::GpuBarrierAll::dump(std::ostream &os) const {
  os << "GpuBarrierAll(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuBarrierAll::operator==(const Spec::GpuBarrierAll& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuBarrierAll::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierAll::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierAll>(*this)); }
Spec::Any Spec::GpuBarrierAll::widen() const { return Any(*this); };

Spec::GpuFenceGlobal::GpuFenceGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceGlobal::id() const { return variant_id; };
size_t Spec::GpuFenceGlobal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceGlobal &x) { return x.dump(os); } }
std::ostream &Spec::GpuFenceGlobal::dump(std::ostream &os) const {
  os << "GpuFenceGlobal(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuFenceGlobal::operator==(const Spec::GpuFenceGlobal& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuFenceGlobal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceGlobal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceGlobal>(*this)); }
Spec::Any Spec::GpuFenceGlobal::widen() const { return Any(*this); };

Spec::GpuFenceLocal::GpuFenceLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceLocal::id() const { return variant_id; };
size_t Spec::GpuFenceLocal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceLocal &x) { return x.dump(os); } }
std::ostream &Spec::GpuFenceLocal::dump(std::ostream &os) const {
  os << "GpuFenceLocal(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuFenceLocal::operator==(const Spec::GpuFenceLocal& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuFenceLocal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceLocal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceLocal>(*this)); }
Spec::Any Spec::GpuFenceLocal::widen() const { return Any(*this); };

Spec::GpuFenceAll::GpuFenceAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceAll::id() const { return variant_id; };
size_t Spec::GpuFenceAll::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceAll &x) { return x.dump(os); } }
std::ostream &Spec::GpuFenceAll::dump(std::ostream &os) const {
  os << "GpuFenceAll(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuFenceAll::operator==(const Spec::GpuFenceAll& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Spec::GpuFenceAll::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceAll::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceAll>(*this)); }
Spec::Any Spec::GpuFenceAll::widen() const { return Any(*this); };

Spec::GpuGlobalIdx::GpuGlobalIdx(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalIdx::id() const { return variant_id; };
size_t Spec::GpuGlobalIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalIdx &x) { return x.dump(os); } }
std::ostream &Spec::GpuGlobalIdx::dump(std::ostream &os) const {
  os << "GpuGlobalIdx(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuGlobalIdx::operator==(const Spec::GpuGlobalIdx& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuGlobalIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGlobalIdx&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGlobalIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGlobalIdx>(*this)); }
Spec::Any Spec::GpuGlobalIdx::widen() const { return Any(*this); };

Spec::GpuGlobalSize::GpuGlobalSize(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalSize::id() const { return variant_id; };
size_t Spec::GpuGlobalSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalSize &x) { return x.dump(os); } }
std::ostream &Spec::GpuGlobalSize::dump(std::ostream &os) const {
  os << "GpuGlobalSize(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuGlobalSize::operator==(const Spec::GpuGlobalSize& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuGlobalSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGlobalSize&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGlobalSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGlobalSize>(*this)); }
Spec::Any Spec::GpuGlobalSize::widen() const { return Any(*this); };

Spec::GpuGroupIdx::GpuGroupIdx(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupIdx::id() const { return variant_id; };
size_t Spec::GpuGroupIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupIdx &x) { return x.dump(os); } }
std::ostream &Spec::GpuGroupIdx::dump(std::ostream &os) const {
  os << "GpuGroupIdx(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuGroupIdx::operator==(const Spec::GpuGroupIdx& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuGroupIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGroupIdx&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGroupIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGroupIdx>(*this)); }
Spec::Any Spec::GpuGroupIdx::widen() const { return Any(*this); };

Spec::GpuGroupSize::GpuGroupSize(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupSize::id() const { return variant_id; };
size_t Spec::GpuGroupSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupSize &x) { return x.dump(os); } }
std::ostream &Spec::GpuGroupSize::dump(std::ostream &os) const {
  os << "GpuGroupSize(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuGroupSize::operator==(const Spec::GpuGroupSize& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuGroupSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGroupSize&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGroupSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGroupSize>(*this)); }
Spec::Any Spec::GpuGroupSize::widen() const { return Any(*this); };

Spec::GpuLocalIdx::GpuLocalIdx(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalIdx::id() const { return variant_id; };
size_t Spec::GpuLocalIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalIdx &x) { return x.dump(os); } }
std::ostream &Spec::GpuLocalIdx::dump(std::ostream &os) const {
  os << "GpuLocalIdx(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuLocalIdx::operator==(const Spec::GpuLocalIdx& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuLocalIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuLocalIdx&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuLocalIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuLocalIdx>(*this)); }
Spec::Any Spec::GpuLocalIdx::widen() const { return Any(*this); };

Spec::GpuLocalSize::GpuLocalSize(Expr::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalSize::id() const { return variant_id; };
size_t Spec::GpuLocalSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Spec { std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalSize &x) { return x.dump(os); } }
std::ostream &Spec::GpuLocalSize::dump(std::ostream &os) const {
  os << "GpuLocalSize(";
  os << dim;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Spec::GpuLocalSize::operator==(const Spec::GpuLocalSize& rhs) const {
  return (this->dim == rhs.dim);
}
POLYREGION_EXPORT bool Spec::GpuLocalSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuLocalSize&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuLocalSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuLocalSize>(*this)); }
Spec::Any Spec::GpuLocalSize::widen() const { return Any(*this); };

Intr::Base::Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept : overloads(std::move(overloads)), exprs(std::move(exprs)), tpe(std::move(tpe)) {}
uint32_t Intr::Any::id() const { return _v->id(); }
size_t Intr::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Intr::Any::overloads() const { return _v->overloads; }
std::vector<Expr::Any> Intr::Any::exprs() const { return _v->exprs; }
Type::Any Intr::Any::tpe() const { return _v->tpe; }
std::ostream &Intr::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Intr { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Intr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Intr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Intr::BNot::BNot(Expr::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8()},Type::IntU8()),Overload({Type::IntU16()},Type::IntU16()),Overload({Type::IntU32()},Type::IntU32()),Overload({Type::IntU64()},Type::IntU64()),Overload({Type::IntS8()},Type::IntS8()),Overload({Type::IntS16()},Type::IntS16()),Overload({Type::IntS32()},Type::IntS32()),Overload({Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::BNot::id() const { return variant_id; };
size_t Intr::BNot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BNot &x) { return x.dump(os); } }
std::ostream &Intr::BNot::dump(std::ostream &os) const {
  os << "BNot(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BNot::operator==(const Intr::BNot& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BNot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BNot&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BNot::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BNot>(*this)); }
Intr::Any Intr::BNot::widen() const { return Any(*this); };

Intr::LogicNot::LogicNot(Expr::Any x) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x}, Type::Bool1()), x(std::move(x)) {}
uint32_t Intr::LogicNot::id() const { return variant_id; };
size_t Intr::LogicNot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicNot &x) { return x.dump(os); } }
std::ostream &Intr::LogicNot::dump(std::ostream &os) const {
  os << "LogicNot(";
  os << x;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicNot::operator==(const Intr::LogicNot& rhs) const {
  return (this->x == rhs.x);
}
POLYREGION_EXPORT bool Intr::LogicNot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicNot&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicNot::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicNot>(*this)); }
Intr::Any Intr::LogicNot::widen() const { return Any(*this); };

Intr::Pos::Pos(Expr::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Pos::id() const { return variant_id; };
size_t Intr::Pos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Pos &x) { return x.dump(os); } }
std::ostream &Intr::Pos::dump(std::ostream &os) const {
  os << "Pos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Pos::operator==(const Intr::Pos& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Pos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Pos&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Pos::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Pos>(*this)); }
Intr::Any Intr::Pos::widen() const { return Any(*this); };

Intr::Neg::Neg(Expr::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Neg::id() const { return variant_id; };
size_t Intr::Neg::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Neg &x) { return x.dump(os); } }
std::ostream &Intr::Neg::dump(std::ostream &os) const {
  os << "Neg(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Neg::operator==(const Intr::Neg& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Neg::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Neg&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Neg::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Neg>(*this)); }
Intr::Any Intr::Neg::widen() const { return Any(*this); };

Intr::Add::Add(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Add::id() const { return variant_id; };
size_t Intr::Add::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Add &x) { return x.dump(os); } }
std::ostream &Intr::Add::dump(std::ostream &os) const {
  os << "Add(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Add::operator==(const Intr::Add& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Add::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Add&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Add::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Add>(*this)); }
Intr::Any Intr::Add::widen() const { return Any(*this); };

Intr::Sub::Sub(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Sub::id() const { return variant_id; };
size_t Intr::Sub::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Sub &x) { return x.dump(os); } }
std::ostream &Intr::Sub::dump(std::ostream &os) const {
  os << "Sub(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Sub::operator==(const Intr::Sub& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Sub::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Sub&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Sub::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sub>(*this)); }
Intr::Any Intr::Sub::widen() const { return Any(*this); };

Intr::Mul::Mul(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Mul::id() const { return variant_id; };
size_t Intr::Mul::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Mul &x) { return x.dump(os); } }
std::ostream &Intr::Mul::dump(std::ostream &os) const {
  os << "Mul(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Mul::operator==(const Intr::Mul& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Mul::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Mul&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Mul::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Mul>(*this)); }
Intr::Any Intr::Mul::widen() const { return Any(*this); };

Intr::Div::Div(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Div::id() const { return variant_id; };
size_t Intr::Div::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Div &x) { return x.dump(os); } }
std::ostream &Intr::Div::dump(std::ostream &os) const {
  os << "Div(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Div::operator==(const Intr::Div& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Div::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Div&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Div::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Div>(*this)); }
Intr::Any Intr::Div::widen() const { return Any(*this); };

Intr::Rem::Rem(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Rem::id() const { return variant_id; };
size_t Intr::Rem::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Rem &x) { return x.dump(os); } }
std::ostream &Intr::Rem::dump(std::ostream &os) const {
  os << "Rem(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Rem::operator==(const Intr::Rem& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Rem::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Rem&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Rem::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Rem>(*this)); }
Intr::Any Intr::Rem::widen() const { return Any(*this); };

Intr::Min::Min(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Min::id() const { return variant_id; };
size_t Intr::Min::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Min &x) { return x.dump(os); } }
std::ostream &Intr::Min::dump(std::ostream &os) const {
  os << "Min(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Min::operator==(const Intr::Min& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Min::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Min&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Min::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Min>(*this)); }
Intr::Any Intr::Min::widen() const { return Any(*this); };

Intr::Max::Max(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Max::id() const { return variant_id; };
size_t Intr::Max::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::Max &x) { return x.dump(os); } }
std::ostream &Intr::Max::dump(std::ostream &os) const {
  os << "Max(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::Max::operator==(const Intr::Max& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Max::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Max&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Max::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Max>(*this)); }
Intr::Any Intr::Max::widen() const { return Any(*this); };

Intr::BAnd::BAnd(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BAnd::id() const { return variant_id; };
size_t Intr::BAnd::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BAnd &x) { return x.dump(os); } }
std::ostream &Intr::BAnd::dump(std::ostream &os) const {
  os << "BAnd(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BAnd::operator==(const Intr::BAnd& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BAnd::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BAnd&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BAnd::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BAnd>(*this)); }
Intr::Any Intr::BAnd::widen() const { return Any(*this); };

Intr::BOr::BOr(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BOr::id() const { return variant_id; };
size_t Intr::BOr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BOr &x) { return x.dump(os); } }
std::ostream &Intr::BOr::dump(std::ostream &os) const {
  os << "BOr(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BOr::operator==(const Intr::BOr& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BOr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BOr&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BOr::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BOr>(*this)); }
Intr::Any Intr::BOr::widen() const { return Any(*this); };

Intr::BXor::BXor(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BXor::id() const { return variant_id; };
size_t Intr::BXor::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BXor &x) { return x.dump(os); } }
std::ostream &Intr::BXor::dump(std::ostream &os) const {
  os << "BXor(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BXor::operator==(const Intr::BXor& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BXor::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BXor&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BXor::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BXor>(*this)); }
Intr::Any Intr::BXor::widen() const { return Any(*this); };

Intr::BSL::BSL(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSL::id() const { return variant_id; };
size_t Intr::BSL::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BSL &x) { return x.dump(os); } }
std::ostream &Intr::BSL::dump(std::ostream &os) const {
  os << "BSL(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BSL::operator==(const Intr::BSL& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BSL::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BSL&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BSL::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BSL>(*this)); }
Intr::Any Intr::BSL::widen() const { return Any(*this); };

Intr::BSR::BSR(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSR::id() const { return variant_id; };
size_t Intr::BSR::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BSR &x) { return x.dump(os); } }
std::ostream &Intr::BSR::dump(std::ostream &os) const {
  os << "BSR(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BSR::operator==(const Intr::BSR& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BSR::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BSR&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BSR::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BSR>(*this)); }
Intr::Any Intr::BSR::widen() const { return Any(*this); };

Intr::BZSR::BZSR(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BZSR::id() const { return variant_id; };
size_t Intr::BZSR::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::BZSR &x) { return x.dump(os); } }
std::ostream &Intr::BZSR::dump(std::ostream &os) const {
  os << "BZSR(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::BZSR::operator==(const Intr::BZSR& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BZSR::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BZSR&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BZSR::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BZSR>(*this)); }
Intr::Any Intr::BZSR::widen() const { return Any(*this); };

Intr::LogicAnd::LogicAnd(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicAnd::id() const { return variant_id; };
size_t Intr::LogicAnd::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicAnd &x) { return x.dump(os); } }
std::ostream &Intr::LogicAnd::dump(std::ostream &os) const {
  os << "LogicAnd(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicAnd::operator==(const Intr::LogicAnd& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicAnd::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicAnd&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicAnd::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicAnd>(*this)); }
Intr::Any Intr::LogicAnd::widen() const { return Any(*this); };

Intr::LogicOr::LogicOr(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicOr::id() const { return variant_id; };
size_t Intr::LogicOr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicOr &x) { return x.dump(os); } }
std::ostream &Intr::LogicOr::dump(std::ostream &os) const {
  os << "LogicOr(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicOr::operator==(const Intr::LogicOr& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicOr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicOr&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicOr::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicOr>(*this)); }
Intr::Any Intr::LogicOr::widen() const { return Any(*this); };

Intr::LogicEq::LogicEq(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicEq::id() const { return variant_id; };
size_t Intr::LogicEq::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicEq &x) { return x.dump(os); } }
std::ostream &Intr::LogicEq::dump(std::ostream &os) const {
  os << "LogicEq(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicEq::operator==(const Intr::LogicEq& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicEq::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicEq&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicEq::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicEq>(*this)); }
Intr::Any Intr::LogicEq::widen() const { return Any(*this); };

Intr::LogicNeq::LogicNeq(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicNeq::id() const { return variant_id; };
size_t Intr::LogicNeq::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicNeq &x) { return x.dump(os); } }
std::ostream &Intr::LogicNeq::dump(std::ostream &os) const {
  os << "LogicNeq(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicNeq::operator==(const Intr::LogicNeq& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicNeq::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicNeq&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicNeq::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicNeq>(*this)); }
Intr::Any Intr::LogicNeq::widen() const { return Any(*this); };

Intr::LogicLte::LogicLte(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLte::id() const { return variant_id; };
size_t Intr::LogicLte::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicLte &x) { return x.dump(os); } }
std::ostream &Intr::LogicLte::dump(std::ostream &os) const {
  os << "LogicLte(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicLte::operator==(const Intr::LogicLte& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicLte::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicLte&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicLte::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicLte>(*this)); }
Intr::Any Intr::LogicLte::widen() const { return Any(*this); };

Intr::LogicGte::LogicGte(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGte::id() const { return variant_id; };
size_t Intr::LogicGte::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicGte &x) { return x.dump(os); } }
std::ostream &Intr::LogicGte::dump(std::ostream &os) const {
  os << "LogicGte(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicGte::operator==(const Intr::LogicGte& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicGte::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicGte&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicGte::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicGte>(*this)); }
Intr::Any Intr::LogicGte::widen() const { return Any(*this); };

Intr::LogicLt::LogicLt(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLt::id() const { return variant_id; };
size_t Intr::LogicLt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicLt &x) { return x.dump(os); } }
std::ostream &Intr::LogicLt::dump(std::ostream &os) const {
  os << "LogicLt(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicLt::operator==(const Intr::LogicLt& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicLt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicLt&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicLt::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicLt>(*this)); }
Intr::Any Intr::LogicLt::widen() const { return Any(*this); };

Intr::LogicGt::LogicGt(Expr::Any x, Expr::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGt::id() const { return variant_id; };
size_t Intr::LogicGt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Intr { std::ostream &operator<<(std::ostream &os, const Intr::LogicGt &x) { return x.dump(os); } }
std::ostream &Intr::LogicGt::dump(std::ostream &os) const {
  os << "LogicGt(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Intr::LogicGt::operator==(const Intr::LogicGt& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y);
}
POLYREGION_EXPORT bool Intr::LogicGt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicGt&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicGt::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicGt>(*this)); }
Intr::Any Intr::LogicGt::widen() const { return Any(*this); };

Math::Base::Base(std::vector<Overload> overloads, std::vector<Expr::Any> exprs, Type::Any tpe) noexcept : overloads(std::move(overloads)), exprs(std::move(exprs)), tpe(std::move(tpe)) {}
uint32_t Math::Any::id() const { return _v->id(); }
size_t Math::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Math::Any::overloads() const { return _v->overloads; }
std::vector<Expr::Any> Math::Any::exprs() const { return _v->exprs; }
Type::Any Math::Any::tpe() const { return _v->tpe; }
std::ostream &Math::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Math { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Math::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Math::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Math::Abs::Abs(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Abs::id() const { return variant_id; };
size_t Math::Abs::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Abs &x) { return x.dump(os); } }
std::ostream &Math::Abs::dump(std::ostream &os) const {
  os << "Abs(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Abs::operator==(const Math::Abs& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Abs::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Abs&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Abs::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Abs>(*this)); }
Math::Any Math::Abs::widen() const { return Any(*this); };

Math::Sin::Sin(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sin::id() const { return variant_id; };
size_t Math::Sin::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Sin &x) { return x.dump(os); } }
std::ostream &Math::Sin::dump(std::ostream &os) const {
  os << "Sin(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Sin::operator==(const Math::Sin& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Sin::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sin&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sin::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sin>(*this)); }
Math::Any Math::Sin::widen() const { return Any(*this); };

Math::Cos::Cos(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cos::id() const { return variant_id; };
size_t Math::Cos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Cos &x) { return x.dump(os); } }
std::ostream &Math::Cos::dump(std::ostream &os) const {
  os << "Cos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Cos::operator==(const Math::Cos& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Cos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cos&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cos::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cos>(*this)); }
Math::Any Math::Cos::widen() const { return Any(*this); };

Math::Tan::Tan(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tan::id() const { return variant_id; };
size_t Math::Tan::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Tan &x) { return x.dump(os); } }
std::ostream &Math::Tan::dump(std::ostream &os) const {
  os << "Tan(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Tan::operator==(const Math::Tan& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Tan::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Tan&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Tan::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Tan>(*this)); }
Math::Any Math::Tan::widen() const { return Any(*this); };

Math::Asin::Asin(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Asin::id() const { return variant_id; };
size_t Math::Asin::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Asin &x) { return x.dump(os); } }
std::ostream &Math::Asin::dump(std::ostream &os) const {
  os << "Asin(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Asin::operator==(const Math::Asin& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Asin::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Asin&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Asin::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Asin>(*this)); }
Math::Any Math::Asin::widen() const { return Any(*this); };

Math::Acos::Acos(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Acos::id() const { return variant_id; };
size_t Math::Acos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Acos &x) { return x.dump(os); } }
std::ostream &Math::Acos::dump(std::ostream &os) const {
  os << "Acos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Acos::operator==(const Math::Acos& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Acos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Acos&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Acos::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Acos>(*this)); }
Math::Any Math::Acos::widen() const { return Any(*this); };

Math::Atan::Atan(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Atan::id() const { return variant_id; };
size_t Math::Atan::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Atan &x) { return x.dump(os); } }
std::ostream &Math::Atan::dump(std::ostream &os) const {
  os << "Atan(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Atan::operator==(const Math::Atan& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Atan::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Atan&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Atan::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Atan>(*this)); }
Math::Any Math::Atan::widen() const { return Any(*this); };

Math::Sinh::Sinh(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sinh::id() const { return variant_id; };
size_t Math::Sinh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Sinh &x) { return x.dump(os); } }
std::ostream &Math::Sinh::dump(std::ostream &os) const {
  os << "Sinh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Sinh::operator==(const Math::Sinh& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Sinh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sinh&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sinh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sinh>(*this)); }
Math::Any Math::Sinh::widen() const { return Any(*this); };

Math::Cosh::Cosh(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cosh::id() const { return variant_id; };
size_t Math::Cosh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Cosh &x) { return x.dump(os); } }
std::ostream &Math::Cosh::dump(std::ostream &os) const {
  os << "Cosh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Cosh::operator==(const Math::Cosh& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Cosh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cosh&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cosh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cosh>(*this)); }
Math::Any Math::Cosh::widen() const { return Any(*this); };

Math::Tanh::Tanh(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tanh::id() const { return variant_id; };
size_t Math::Tanh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Tanh &x) { return x.dump(os); } }
std::ostream &Math::Tanh::dump(std::ostream &os) const {
  os << "Tanh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Tanh::operator==(const Math::Tanh& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Tanh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Tanh&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Tanh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Tanh>(*this)); }
Math::Any Math::Tanh::widen() const { return Any(*this); };

Math::Signum::Signum(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Signum::id() const { return variant_id; };
size_t Math::Signum::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Signum &x) { return x.dump(os); } }
std::ostream &Math::Signum::dump(std::ostream &os) const {
  os << "Signum(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Signum::operator==(const Math::Signum& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Signum::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Signum&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Signum::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Signum>(*this)); }
Math::Any Math::Signum::widen() const { return Any(*this); };

Math::Round::Round(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Round::id() const { return variant_id; };
size_t Math::Round::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Round &x) { return x.dump(os); } }
std::ostream &Math::Round::dump(std::ostream &os) const {
  os << "Round(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Round::operator==(const Math::Round& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Round::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Round&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Round::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Round>(*this)); }
Math::Any Math::Round::widen() const { return Any(*this); };

Math::Ceil::Ceil(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Ceil::id() const { return variant_id; };
size_t Math::Ceil::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Ceil &x) { return x.dump(os); } }
std::ostream &Math::Ceil::dump(std::ostream &os) const {
  os << "Ceil(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Ceil::operator==(const Math::Ceil& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Ceil::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Ceil&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Ceil::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ceil>(*this)); }
Math::Any Math::Ceil::widen() const { return Any(*this); };

Math::Floor::Floor(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Floor::id() const { return variant_id; };
size_t Math::Floor::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Floor &x) { return x.dump(os); } }
std::ostream &Math::Floor::dump(std::ostream &os) const {
  os << "Floor(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Floor::operator==(const Math::Floor& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Floor::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Floor&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Floor::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Floor>(*this)); }
Math::Any Math::Floor::widen() const { return Any(*this); };

Math::Rint::Rint(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Rint::id() const { return variant_id; };
size_t Math::Rint::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Rint &x) { return x.dump(os); } }
std::ostream &Math::Rint::dump(std::ostream &os) const {
  os << "Rint(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Rint::operator==(const Math::Rint& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Rint::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Rint&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Rint::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Rint>(*this)); }
Math::Any Math::Rint::widen() const { return Any(*this); };

Math::Sqrt::Sqrt(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sqrt::id() const { return variant_id; };
size_t Math::Sqrt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Sqrt &x) { return x.dump(os); } }
std::ostream &Math::Sqrt::dump(std::ostream &os) const {
  os << "Sqrt(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Sqrt::operator==(const Math::Sqrt& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Sqrt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sqrt&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sqrt::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sqrt>(*this)); }
Math::Any Math::Sqrt::widen() const { return Any(*this); };

Math::Cbrt::Cbrt(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cbrt::id() const { return variant_id; };
size_t Math::Cbrt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Cbrt &x) { return x.dump(os); } }
std::ostream &Math::Cbrt::dump(std::ostream &os) const {
  os << "Cbrt(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Cbrt::operator==(const Math::Cbrt& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Cbrt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cbrt&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cbrt::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cbrt>(*this)); }
Math::Any Math::Cbrt::widen() const { return Any(*this); };

Math::Exp::Exp(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Exp::id() const { return variant_id; };
size_t Math::Exp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Exp &x) { return x.dump(os); } }
std::ostream &Math::Exp::dump(std::ostream &os) const {
  os << "Exp(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Exp::operator==(const Math::Exp& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Exp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Exp&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Exp::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Exp>(*this)); }
Math::Any Math::Exp::widen() const { return Any(*this); };

Math::Expm1::Expm1(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Expm1::id() const { return variant_id; };
size_t Math::Expm1::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Expm1 &x) { return x.dump(os); } }
std::ostream &Math::Expm1::dump(std::ostream &os) const {
  os << "Expm1(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Expm1::operator==(const Math::Expm1& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Expm1::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Expm1&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Expm1::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Expm1>(*this)); }
Math::Any Math::Expm1::widen() const { return Any(*this); };

Math::Log::Log(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log::id() const { return variant_id; };
size_t Math::Log::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Log &x) { return x.dump(os); } }
std::ostream &Math::Log::dump(std::ostream &os) const {
  os << "Log(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Log::operator==(const Math::Log& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Log::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log>(*this)); }
Math::Any Math::Log::widen() const { return Any(*this); };

Math::Log1p::Log1p(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log1p::id() const { return variant_id; };
size_t Math::Log1p::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Log1p &x) { return x.dump(os); } }
std::ostream &Math::Log1p::dump(std::ostream &os) const {
  os << "Log1p(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Log1p::operator==(const Math::Log1p& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Log1p::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log1p&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log1p::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log1p>(*this)); }
Math::Any Math::Log1p::widen() const { return Any(*this); };

Math::Log10::Log10(Expr::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log10::id() const { return variant_id; };
size_t Math::Log10::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Log10 &x) { return x.dump(os); } }
std::ostream &Math::Log10::dump(std::ostream &os) const {
  os << "Log10(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Log10::operator==(const Math::Log10& rhs) const {
  return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Log10::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log10&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log10::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log10>(*this)); }
Math::Any Math::Log10::widen() const { return Any(*this); };

Math::Pow::Pow(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Pow::id() const { return variant_id; };
size_t Math::Pow::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Pow &x) { return x.dump(os); } }
std::ostream &Math::Pow::dump(std::ostream &os) const {
  os << "Pow(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Pow::operator==(const Math::Pow& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Pow::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Pow&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Pow::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Pow>(*this)); }
Math::Any Math::Pow::widen() const { return Any(*this); };

Math::Atan2::Atan2(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Atan2::id() const { return variant_id; };
size_t Math::Atan2::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Atan2 &x) { return x.dump(os); } }
std::ostream &Math::Atan2::dump(std::ostream &os) const {
  os << "Atan2(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Atan2::operator==(const Math::Atan2& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Atan2::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Atan2&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Atan2::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Atan2>(*this)); }
Math::Any Math::Atan2::widen() const { return Any(*this); };

Math::Hypot::Hypot(Expr::Any x, Expr::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Hypot::id() const { return variant_id; };
size_t Math::Hypot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Math { std::ostream &operator<<(std::ostream &os, const Math::Hypot &x) { return x.dump(os); } }
std::ostream &Math::Hypot::dump(std::ostream &os) const {
  os << "Hypot(";
  os << x;
  os << ',';
  os << y;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Math::Hypot::operator==(const Math::Hypot& rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Hypot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Hypot&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Hypot::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Hypot>(*this)); }
Math::Any Math::Hypot::widen() const { return Any(*this); };

Stmt::Base::Base() = default;
uint32_t Stmt::Any::id() const { return _v->id(); }
size_t Stmt::Any::hash_code() const { return _v->hash_code(); }
std::ostream &Stmt::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool Stmt::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Stmt::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool Stmt::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

Stmt::Block::Block(std::vector<Stmt::Any> stmts) noexcept : Stmt::Base(), stmts(std::move(stmts)) {}
uint32_t Stmt::Block::id() const { return variant_id; };
size_t Stmt::Block::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(stmts)>()(stmts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Block &x) { return x.dump(os); } }
std::ostream &Stmt::Block::dump(std::ostream &os) const {
  os << "Block(";
  os << '{';
  if (!stmts.empty()) {
    std::for_each(stmts.begin(), std::prev(stmts.end()), [&os](auto &&x) { os << x; os << ','; });
    os << stmts.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Block::operator==(const Stmt::Block& rhs) const {
  return std::equal(this->stmts.begin(), this->stmts.end(), rhs.stmts.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::Block::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Block&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Block::operator<(const Stmt::Block& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Block::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Block::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Block>(*this)); }
Stmt::Any Stmt::Block::widen() const { return Any(*this); };

Stmt::Comment::Comment(std::string value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Comment::id() const { return variant_id; };
size_t Stmt::Comment::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Comment &x) { return x.dump(os); } }
std::ostream &Stmt::Comment::dump(std::ostream &os) const {
  os << "Comment(";
  os << '"' << value << '"';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Comment::operator==(const Stmt::Comment& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Stmt::Comment::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Comment&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Comment::operator<(const Stmt::Comment& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Comment::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Comment::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Comment>(*this)); }
Stmt::Any Stmt::Comment::widen() const { return Any(*this); };

Stmt::Var::Var(Named name, std::optional<Expr::Any> expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
uint32_t Stmt::Var::id() const { return variant_id; };
size_t Stmt::Var::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Var &x) { return x.dump(os); } }
std::ostream &Stmt::Var::dump(std::ostream &os) const {
  os << "Var(";
  os << name;
  os << ',';
  os << '{';
  if (expr) {
    os << (*expr);
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Var::operator==(const Stmt::Var& rhs) const {
  return (this->name == rhs.name) && ( (!this->expr && !rhs.expr) || (this->expr && rhs.expr && *this->expr == *rhs.expr) );
}
POLYREGION_EXPORT bool Stmt::Var::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Var&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Var::operator<(const Stmt::Var& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Var::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Var::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Var>(*this)); }
Stmt::Any Stmt::Var::widen() const { return Any(*this); };

Stmt::Mut::Mut(Expr::Any name, Expr::Any expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
uint32_t Stmt::Mut::id() const { return variant_id; };
size_t Stmt::Mut::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Mut &x) { return x.dump(os); } }
std::ostream &Stmt::Mut::dump(std::ostream &os) const {
  os << "Mut(";
  os << name;
  os << ',';
  os << expr;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Mut::operator==(const Stmt::Mut& rhs) const {
  return (this->name == rhs.name) && (this->expr == rhs.expr);
}
POLYREGION_EXPORT bool Stmt::Mut::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Mut&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Mut::operator<(const Stmt::Mut& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Mut::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Mut::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Mut>(*this)); }
Stmt::Any Stmt::Mut::widen() const { return Any(*this); };

Stmt::Update::Update(Expr::Any lhs, Expr::Any idx, Expr::Any value) noexcept : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
uint32_t Stmt::Update::id() const { return variant_id; };
size_t Stmt::Update::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Update &x) { return x.dump(os); } }
std::ostream &Stmt::Update::dump(std::ostream &os) const {
  os << "Update(";
  os << lhs;
  os << ',';
  os << idx;
  os << ',';
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Update::operator==(const Stmt::Update& rhs) const {
  return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->value == rhs.value);
}
POLYREGION_EXPORT bool Stmt::Update::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Update&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Update::operator<(const Stmt::Update& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Update::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Update::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Update>(*this)); }
Stmt::Any Stmt::Update::widen() const { return Any(*this); };

Stmt::While::While(std::vector<Stmt::Any> tests, Expr::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), tests(std::move(tests)), cond(std::move(cond)), body(std::move(body)) {}
uint32_t Stmt::While::id() const { return variant_id; };
size_t Stmt::While::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(tests)>()(tests) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::While &x) { return x.dump(os); } }
std::ostream &Stmt::While::dump(std::ostream &os) const {
  os << "While(";
  os << '{';
  if (!tests.empty()) {
    std::for_each(tests.begin(), std::prev(tests.end()), [&os](auto &&x) { os << x; os << ','; });
    os << tests.back();
  }
  os << '}';
  os << ',';
  os << cond;
  os << ',';
  os << '{';
  if (!body.empty()) {
    std::for_each(body.begin(), std::prev(body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << body.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::While::operator==(const Stmt::While& rhs) const {
  return std::equal(this->tests.begin(), this->tests.end(), rhs.tests.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->cond == rhs.cond) && std::equal(this->body.begin(), this->body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::While::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::While&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::While::operator<(const Stmt::While& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::While::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::While::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<While>(*this)); }
Stmt::Any Stmt::While::widen() const { return Any(*this); };

Stmt::Break::Break() noexcept : Stmt::Base() {}
uint32_t Stmt::Break::id() const { return variant_id; };
size_t Stmt::Break::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Break &x) { return x.dump(os); } }
std::ostream &Stmt::Break::dump(std::ostream &os) const {
  os << "Break(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Break::operator==(const Stmt::Break& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Stmt::Break::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool Stmt::Break::operator<(const Stmt::Break& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Break::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Break::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Break>(*this)); }
Stmt::Any Stmt::Break::widen() const { return Any(*this); };

Stmt::Cont::Cont() noexcept : Stmt::Base() {}
uint32_t Stmt::Cont::id() const { return variant_id; };
size_t Stmt::Cont::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Cont &x) { return x.dump(os); } }
std::ostream &Stmt::Cont::dump(std::ostream &os) const {
  os << "Cont(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Cont::operator==(const Stmt::Cont& rhs) const {
  return true;
}
POLYREGION_EXPORT bool Stmt::Cont::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool Stmt::Cont::operator<(const Stmt::Cont& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Cont::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Cont::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cont>(*this)); }
Stmt::Any Stmt::Cont::widen() const { return Any(*this); };

Stmt::Cond::Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
uint32_t Stmt::Cond::id() const { return variant_id; };
size_t Stmt::Cond::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(trueBr)>()(trueBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(falseBr)>()(falseBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Cond &x) { return x.dump(os); } }
std::ostream &Stmt::Cond::dump(std::ostream &os) const {
  os << "Cond(";
  os << cond;
  os << ',';
  os << '{';
  if (!trueBr.empty()) {
    std::for_each(trueBr.begin(), std::prev(trueBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << trueBr.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!falseBr.empty()) {
    std::for_each(falseBr.begin(), std::prev(falseBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << falseBr.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Cond::operator==(const Stmt::Cond& rhs) const {
  return (this->cond == rhs.cond) && std::equal(this->trueBr.begin(), this->trueBr.end(), rhs.trueBr.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(this->falseBr.begin(), this->falseBr.end(), rhs.falseBr.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::Cond::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Cond&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Cond::operator<(const Stmt::Cond& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Cond::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Cond::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cond>(*this)); }
Stmt::Any Stmt::Cond::widen() const { return Any(*this); };

Stmt::Return::Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Return::id() const { return variant_id; };
size_t Stmt::Return::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Return &x) { return x.dump(os); } }
std::ostream &Stmt::Return::dump(std::ostream &os) const {
  os << "Return(";
  os << value;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Return::operator==(const Stmt::Return& rhs) const {
  return (this->value == rhs.value);
}
POLYREGION_EXPORT bool Stmt::Return::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Return&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Return::operator<(const Stmt::Return& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Return::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Return::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Return>(*this)); }
Stmt::Any Stmt::Return::widen() const { return Any(*this); };

Stmt::Annotated::Annotated(Stmt::Any expr, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept : Stmt::Base(), expr(std::move(expr)), pos(std::move(pos)), comment(std::move(comment)) {}
uint32_t Stmt::Annotated::id() const { return variant_id; };
size_t Stmt::Annotated::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comment)>()(comment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
namespace Stmt { std::ostream &operator<<(std::ostream &os, const Stmt::Annotated &x) { return x.dump(os); } }
std::ostream &Stmt::Annotated::dump(std::ostream &os) const {
  os << "Annotated(";
  os << expr;
  os << ',';
  os << '{';
  if (pos) {
    os << (*pos);
  }
  os << '}';
  os << ',';
  os << '{';
  if (comment) {
    os << '"' << (*comment) << '"';
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Stmt::Annotated::operator==(const Stmt::Annotated& rhs) const {
  return (this->expr == rhs.expr) && (this->pos == rhs.pos) && (this->comment == rhs.comment);
}
POLYREGION_EXPORT bool Stmt::Annotated::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Annotated&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Annotated::operator<(const Stmt::Annotated& rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Annotated::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
Stmt::Annotated::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Annotated>(*this)); }
Stmt::Any Stmt::Annotated::widen() const { return Any(*this); };

Signature::Signature(std::string name, std::vector<Type::Any> args, Type::Any rtn) noexcept : name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)) {}
size_t Signature::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Signature &x) { return x.dump(os); }
std::ostream &Signature::dump(std::ostream &os) const {
  os << "Signature(";
  os << '"' << name << '"';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Signature::operator==(const Signature& rhs) const {
  return (name == rhs.name) && std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

FunctionAttr::Base::Base() = default;
uint32_t FunctionAttr::Any::id() const { return _v->id(); }
size_t FunctionAttr::Any::hash_code() const { return _v->hash_code(); }
std::ostream &FunctionAttr::Any::dump(std::ostream &os) const { return _v->dump(os); }
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const Any &x) { return x.dump(os); } }
bool FunctionAttr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool FunctionAttr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool FunctionAttr::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

FunctionAttr::Internal::Internal() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::Internal::id() const { return variant_id; };
size_t FunctionAttr::Internal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const FunctionAttr::Internal &x) { return x.dump(os); } }
std::ostream &FunctionAttr::Internal::dump(std::ostream &os) const {
  os << "Internal(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool FunctionAttr::Internal::operator==(const FunctionAttr::Internal& rhs) const {
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Internal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Internal::operator<(const FunctionAttr::Internal& rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAttr::Internal::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
FunctionAttr::Internal::operator FunctionAttr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Internal>(*this)); }
FunctionAttr::Any FunctionAttr::Internal::widen() const { return Any(*this); };

FunctionAttr::Exported::Exported() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::Exported::id() const { return variant_id; };
size_t FunctionAttr::Exported::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const FunctionAttr::Exported &x) { return x.dump(os); } }
std::ostream &FunctionAttr::Exported::dump(std::ostream &os) const {
  os << "Exported(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool FunctionAttr::Exported::operator==(const FunctionAttr::Exported& rhs) const {
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Exported::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Exported::operator<(const FunctionAttr::Exported& rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAttr::Exported::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
FunctionAttr::Exported::operator FunctionAttr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Exported>(*this)); }
FunctionAttr::Any FunctionAttr::Exported::widen() const { return Any(*this); };

FunctionAttr::FPRelaxed::FPRelaxed() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPRelaxed::id() const { return variant_id; };
size_t FunctionAttr::FPRelaxed::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &x) { return x.dump(os); } }
std::ostream &FunctionAttr::FPRelaxed::dump(std::ostream &os) const {
  os << "FPRelaxed(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool FunctionAttr::FPRelaxed::operator==(const FunctionAttr::FPRelaxed& rhs) const {
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::FPRelaxed::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::FPRelaxed::operator<(const FunctionAttr::FPRelaxed& rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAttr::FPRelaxed::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
FunctionAttr::FPRelaxed::operator FunctionAttr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<FPRelaxed>(*this)); }
FunctionAttr::Any FunctionAttr::FPRelaxed::widen() const { return Any(*this); };

FunctionAttr::FPStrict::FPStrict() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPStrict::id() const { return variant_id; };
size_t FunctionAttr::FPStrict::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPStrict &x) { return x.dump(os); } }
std::ostream &FunctionAttr::FPStrict::dump(std::ostream &os) const {
  os << "FPStrict(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool FunctionAttr::FPStrict::operator==(const FunctionAttr::FPStrict& rhs) const {
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::FPStrict::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::FPStrict::operator<(const FunctionAttr::FPStrict& rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAttr::FPStrict::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
FunctionAttr::FPStrict::operator FunctionAttr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<FPStrict>(*this)); }
FunctionAttr::Any FunctionAttr::FPStrict::widen() const { return Any(*this); };

FunctionAttr::Entry::Entry() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::Entry::id() const { return variant_id; };
size_t FunctionAttr::Entry::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
namespace FunctionAttr { std::ostream &operator<<(std::ostream &os, const FunctionAttr::Entry &x) { return x.dump(os); } }
std::ostream &FunctionAttr::Entry::dump(std::ostream &os) const {
  os << "Entry(";
  os << ')';
  return os;
}
POLYREGION_EXPORT bool FunctionAttr::Entry::operator==(const FunctionAttr::Entry& rhs) const {
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Entry::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAttr::Entry::operator<(const FunctionAttr::Entry& rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAttr::Entry::operator<(const Base& rhs_) const { return variant_id < rhs_.id(); }
FunctionAttr::Entry::operator FunctionAttr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Entry>(*this)); }
FunctionAttr::Any FunctionAttr::Entry::widen() const { return Any(*this); };

Arg::Arg(Named named, std::optional<SourcePosition> pos) noexcept : named(std::move(named)), pos(std::move(pos)) {}
size_t Arg::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(named)>()(named) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Arg &x) { return x.dump(os); }
std::ostream &Arg::dump(std::ostream &os) const {
  os << "Arg(";
  os << named;
  os << ',';
  os << '{';
  if (pos) {
    os << (*pos);
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Arg::operator==(const Arg& rhs) const {
  return (named == rhs.named) && (pos == rhs.pos);
}

Function::Function(std::string name, std::vector<Arg> args, Type::Any rtn, std::vector<Stmt::Any> body, std::set<FunctionAttr::Any> attrs) noexcept : name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)), body(std::move(body)), attrs(std::move(attrs)) {}
size_t Function::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(attrs)>()(attrs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Function &x) { return x.dump(os); }
std::ostream &Function::dump(std::ostream &os) const {
  os << "Function(";
  os << '"' << name << '"';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ',';
  os << '{';
  if (!body.empty()) {
    std::for_each(body.begin(), std::prev(body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << body.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!attrs.empty()) {
    std::for_each(attrs.begin(), std::prev(attrs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << *attrs.rbegin();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Function::operator==(const Function& rhs) const {
  return (name == rhs.name) && (args == rhs.args) && (rtn == rhs.rtn) && std::equal(body.begin(), body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(attrs.begin(), attrs.end(), rhs.attrs.begin(), [](auto &&l, auto &&r) { return l == r; });
}

Program::Program(std::vector<StructDef> structs, std::vector<Function> functions) noexcept : structs(std::move(structs)), functions(std::move(functions)) {}
size_t Program::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(structs)>()(structs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(functions)>()(functions) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Program &x) { return x.dump(os); }
std::ostream &Program::dump(std::ostream &os) const {
  os << "Program(";
  os << '{';
  if (!structs.empty()) {
    std::for_each(structs.begin(), std::prev(structs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << structs.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!functions.empty()) {
    std::for_each(functions.begin(), std::prev(functions.end()), [&os](auto &&x) { os << x; os << ','; });
    os << functions.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool Program::operator==(const Program& rhs) const {
  return (structs == rhs.structs) && (functions == rhs.functions);
}

StructDef::StructDef(std::string name, std::vector<Named> members) noexcept : name(std::move(name)), members(std::move(members)) {}
size_t StructDef::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const StructDef &x) { return x.dump(os); }
std::ostream &StructDef::dump(std::ostream &os) const {
  os << "StructDef(";
  os << '"' << name << '"';
  os << ',';
  os << '{';
  if (!members.empty()) {
    std::for_each(members.begin(), std::prev(members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << members.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool StructDef::operator==(const StructDef& rhs) const {
  return (name == rhs.name) && (members == rhs.members);
}

StructLayoutMember::StructLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
size_t StructLayoutMember::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(offsetInBytes)>()(offsetInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const StructLayoutMember &x) { return x.dump(os); }
std::ostream &StructLayoutMember::dump(std::ostream &os) const {
  os << "StructLayoutMember(";
  os << name;
  os << ',';
  os << offsetInBytes;
  os << ',';
  os << sizeInBytes;
  os << ')';
  return os;
}
POLYREGION_EXPORT bool StructLayoutMember::operator==(const StructLayoutMember& rhs) const {
  return (name == rhs.name) && (offsetInBytes == rhs.offsetInBytes) && (sizeInBytes == rhs.sizeInBytes);
}

StructLayout::StructLayout(std::string name, int64_t sizeInBytes, int64_t alignment, std::vector<StructLayoutMember> members) noexcept : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
size_t StructLayout::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(alignment)>()(alignment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const StructLayout &x) { return x.dump(os); }
std::ostream &StructLayout::dump(std::ostream &os) const {
  os << "StructLayout(";
  os << '"' << name << '"';
  os << ',';
  os << sizeInBytes;
  os << ',';
  os << alignment;
  os << ',';
  os << '{';
  if (!members.empty()) {
    std::for_each(members.begin(), std::prev(members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << members.back();
  }
  os << '}';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool StructLayout::operator==(const StructLayout& rhs) const {
  return (name == rhs.name) && (sizeInBytes == rhs.sizeInBytes) && (alignment == rhs.alignment) && (members == rhs.members);
}

CompileEvent::CompileEvent(int64_t epochMillis, int64_t elapsedNanos, std::string name, std::string data) noexcept : epochMillis(epochMillis), elapsedNanos(elapsedNanos), name(std::move(name)), data(std::move(data)) {}
size_t CompileEvent::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(epochMillis)>()(epochMillis) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(elapsedNanos)>()(elapsedNanos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(data)>()(data) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const CompileEvent &x) { return x.dump(os); }
std::ostream &CompileEvent::dump(std::ostream &os) const {
  os << "CompileEvent(";
  os << epochMillis;
  os << ',';
  os << elapsedNanos;
  os << ',';
  os << '"' << name << '"';
  os << ',';
  os << '"' << data << '"';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool CompileEvent::operator==(const CompileEvent& rhs) const {
  return (epochMillis == rhs.epochMillis) && (elapsedNanos == rhs.elapsedNanos) && (name == rhs.name) && (data == rhs.data);
}

CompileResult::CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<StructLayout> layouts, std::string messages) noexcept : binary(std::move(binary)), features(std::move(features)), events(std::move(events)), layouts(std::move(layouts)), messages(std::move(messages)) {}
size_t CompileResult::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(binary)>()(binary) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(features)>()(features) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(events)>()(events) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(layouts)>()(layouts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(messages)>()(messages) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const CompileResult &x) { return x.dump(os); }
std::ostream &CompileResult::dump(std::ostream &os) const {
  os << "CompileResult(";
  os << '{';
  if (binary) {
    os << '{';
  if (!(*binary).empty()) {
    std::for_each((*binary).begin(), std::prev((*binary).end()), [&os](auto &&x) { os << x; os << ','; });
    os << (*binary).back();
  }
  os << '}';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!features.empty()) {
    std::for_each(features.begin(), std::prev(features.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << features.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!events.empty()) {
    std::for_each(events.begin(), std::prev(events.end()), [&os](auto &&x) { os << x; os << ','; });
    os << events.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!layouts.empty()) {
    std::for_each(layouts.begin(), std::prev(layouts.end()), [&os](auto &&x) { os << x; os << ','; });
    os << layouts.back();
  }
  os << '}';
  os << ',';
  os << '"' << messages << '"';
  os << ')';
  return os;
}
POLYREGION_EXPORT bool CompileResult::operator==(const CompileResult& rhs) const {
  return (binary == rhs.binary) && (features == rhs.features) && (events == rhs.events) && (layouts == rhs.layouts) && (messages == rhs.messages);
}

} // namespace polyregion::polyast


std::size_t std::hash<polyregion::polyast::SourcePosition>::operator()(const polyregion::polyast::SourcePosition &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Named>::operator()(const polyregion::polyast::Named &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Any>::operator()(const polyregion::polyast::TypeKind::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::None>::operator()(const polyregion::polyast::TypeKind::None &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Ref>::operator()(const polyregion::polyast::TypeKind::Ref &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Integral>::operator()(const polyregion::polyast::TypeKind::Integral &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Fractional>::operator()(const polyregion::polyast::TypeKind::Fractional &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Any>::operator()(const polyregion::polyast::TypeSpace::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Global>::operator()(const polyregion::polyast::TypeSpace::Global &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Local>::operator()(const polyregion::polyast::TypeSpace::Local &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Any>::operator()(const polyregion::polyast::Type::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Float16>::operator()(const polyregion::polyast::Type::Float16 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Float32>::operator()(const polyregion::polyast::Type::Float32 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Float64>::operator()(const polyregion::polyast::Type::Float64 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntU8>::operator()(const polyregion::polyast::Type::IntU8 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntU16>::operator()(const polyregion::polyast::Type::IntU16 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntU32>::operator()(const polyregion::polyast::Type::IntU32 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntU64>::operator()(const polyregion::polyast::Type::IntU64 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntS8>::operator()(const polyregion::polyast::Type::IntS8 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntS16>::operator()(const polyregion::polyast::Type::IntS16 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntS32>::operator()(const polyregion::polyast::Type::IntS32 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::IntS64>::operator()(const polyregion::polyast::Type::IntS64 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Nothing>::operator()(const polyregion::polyast::Type::Nothing &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Unit0>::operator()(const polyregion::polyast::Type::Unit0 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Bool1>::operator()(const polyregion::polyast::Type::Bool1 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Struct>::operator()(const polyregion::polyast::Type::Struct &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Ptr>::operator()(const polyregion::polyast::Type::Ptr &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Annotated>::operator()(const polyregion::polyast::Type::Annotated &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Any>::operator()(const polyregion::polyast::Expr::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Float16Const>::operator()(const polyregion::polyast::Expr::Float16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Float32Const>::operator()(const polyregion::polyast::Expr::Float32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Float64Const>::operator()(const polyregion::polyast::Expr::Float64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntU8Const>::operator()(const polyregion::polyast::Expr::IntU8Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntU16Const>::operator()(const polyregion::polyast::Expr::IntU16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntU32Const>::operator()(const polyregion::polyast::Expr::IntU32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntU64Const>::operator()(const polyregion::polyast::Expr::IntU64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntS8Const>::operator()(const polyregion::polyast::Expr::IntS8Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntS16Const>::operator()(const polyregion::polyast::Expr::IntS16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntS32Const>::operator()(const polyregion::polyast::Expr::IntS32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntS64Const>::operator()(const polyregion::polyast::Expr::IntS64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Unit0Const>::operator()(const polyregion::polyast::Expr::Unit0Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Bool1Const>::operator()(const polyregion::polyast::Expr::Bool1Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::SpecOp>::operator()(const polyregion::polyast::Expr::SpecOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::MathOp>::operator()(const polyregion::polyast::Expr::MathOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntrOp>::operator()(const polyregion::polyast::Expr::IntrOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Select>::operator()(const polyregion::polyast::Expr::Select &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Poison>::operator()(const polyregion::polyast::Expr::Poison &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Cast>::operator()(const polyregion::polyast::Expr::Cast &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Index>::operator()(const polyregion::polyast::Expr::Index &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::RefTo>::operator()(const polyregion::polyast::Expr::RefTo &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Alloc>::operator()(const polyregion::polyast::Expr::Alloc &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Invoke>::operator()(const polyregion::polyast::Expr::Invoke &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Annotated>::operator()(const polyregion::polyast::Expr::Annotated &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Overload>::operator()(const polyregion::polyast::Overload &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::Any>::operator()(const polyregion::polyast::Spec::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::Assert>::operator()(const polyregion::polyast::Spec::Assert &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierGlobal>::operator()(const polyregion::polyast::Spec::GpuBarrierGlobal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierLocal>::operator()(const polyregion::polyast::Spec::GpuBarrierLocal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierAll>::operator()(const polyregion::polyast::Spec::GpuBarrierAll &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceGlobal>::operator()(const polyregion::polyast::Spec::GpuFenceGlobal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceLocal>::operator()(const polyregion::polyast::Spec::GpuFenceLocal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceAll>::operator()(const polyregion::polyast::Spec::GpuFenceAll &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuGlobalIdx>::operator()(const polyregion::polyast::Spec::GpuGlobalIdx &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuGlobalSize>::operator()(const polyregion::polyast::Spec::GpuGlobalSize &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuGroupIdx>::operator()(const polyregion::polyast::Spec::GpuGroupIdx &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuGroupSize>::operator()(const polyregion::polyast::Spec::GpuGroupSize &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuLocalIdx>::operator()(const polyregion::polyast::Spec::GpuLocalIdx &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Spec::GpuLocalSize>::operator()(const polyregion::polyast::Spec::GpuLocalSize &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Any>::operator()(const polyregion::polyast::Intr::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BNot>::operator()(const polyregion::polyast::Intr::BNot &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicNot>::operator()(const polyregion::polyast::Intr::LogicNot &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Pos>::operator()(const polyregion::polyast::Intr::Pos &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Neg>::operator()(const polyregion::polyast::Intr::Neg &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Add>::operator()(const polyregion::polyast::Intr::Add &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Sub>::operator()(const polyregion::polyast::Intr::Sub &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Mul>::operator()(const polyregion::polyast::Intr::Mul &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Div>::operator()(const polyregion::polyast::Intr::Div &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Rem>::operator()(const polyregion::polyast::Intr::Rem &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Min>::operator()(const polyregion::polyast::Intr::Min &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::Max>::operator()(const polyregion::polyast::Intr::Max &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BAnd>::operator()(const polyregion::polyast::Intr::BAnd &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BOr>::operator()(const polyregion::polyast::Intr::BOr &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BXor>::operator()(const polyregion::polyast::Intr::BXor &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BSL>::operator()(const polyregion::polyast::Intr::BSL &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BSR>::operator()(const polyregion::polyast::Intr::BSR &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::BZSR>::operator()(const polyregion::polyast::Intr::BZSR &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicAnd>::operator()(const polyregion::polyast::Intr::LogicAnd &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicOr>::operator()(const polyregion::polyast::Intr::LogicOr &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicEq>::operator()(const polyregion::polyast::Intr::LogicEq &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicNeq>::operator()(const polyregion::polyast::Intr::LogicNeq &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicLte>::operator()(const polyregion::polyast::Intr::LogicLte &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicGte>::operator()(const polyregion::polyast::Intr::LogicGte &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicLt>::operator()(const polyregion::polyast::Intr::LogicLt &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Intr::LogicGt>::operator()(const polyregion::polyast::Intr::LogicGt &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Any>::operator()(const polyregion::polyast::Math::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Abs>::operator()(const polyregion::polyast::Math::Abs &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Sin>::operator()(const polyregion::polyast::Math::Sin &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Cos>::operator()(const polyregion::polyast::Math::Cos &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Tan>::operator()(const polyregion::polyast::Math::Tan &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Asin>::operator()(const polyregion::polyast::Math::Asin &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Acos>::operator()(const polyregion::polyast::Math::Acos &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Atan>::operator()(const polyregion::polyast::Math::Atan &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Sinh>::operator()(const polyregion::polyast::Math::Sinh &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Cosh>::operator()(const polyregion::polyast::Math::Cosh &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Tanh>::operator()(const polyregion::polyast::Math::Tanh &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Signum>::operator()(const polyregion::polyast::Math::Signum &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Round>::operator()(const polyregion::polyast::Math::Round &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Ceil>::operator()(const polyregion::polyast::Math::Ceil &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Floor>::operator()(const polyregion::polyast::Math::Floor &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Rint>::operator()(const polyregion::polyast::Math::Rint &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Sqrt>::operator()(const polyregion::polyast::Math::Sqrt &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Cbrt>::operator()(const polyregion::polyast::Math::Cbrt &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Exp>::operator()(const polyregion::polyast::Math::Exp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Expm1>::operator()(const polyregion::polyast::Math::Expm1 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Log>::operator()(const polyregion::polyast::Math::Log &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Log1p>::operator()(const polyregion::polyast::Math::Log1p &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Log10>::operator()(const polyregion::polyast::Math::Log10 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Pow>::operator()(const polyregion::polyast::Math::Pow &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Atan2>::operator()(const polyregion::polyast::Math::Atan2 &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Math::Hypot>::operator()(const polyregion::polyast::Math::Hypot &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Any>::operator()(const polyregion::polyast::Stmt::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Block>::operator()(const polyregion::polyast::Stmt::Block &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Comment>::operator()(const polyregion::polyast::Stmt::Comment &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Var>::operator()(const polyregion::polyast::Stmt::Var &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Mut>::operator()(const polyregion::polyast::Stmt::Mut &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Update>::operator()(const polyregion::polyast::Stmt::Update &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::While>::operator()(const polyregion::polyast::Stmt::While &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Break>::operator()(const polyregion::polyast::Stmt::Break &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Cont>::operator()(const polyregion::polyast::Stmt::Cont &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Cond>::operator()(const polyregion::polyast::Stmt::Cond &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Return>::operator()(const polyregion::polyast::Stmt::Return &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Stmt::Annotated>::operator()(const polyregion::polyast::Stmt::Annotated &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Signature>::operator()(const polyregion::polyast::Signature &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::Any>::operator()(const polyregion::polyast::FunctionAttr::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::Internal>::operator()(const polyregion::polyast::FunctionAttr::Internal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::Exported>::operator()(const polyregion::polyast::FunctionAttr::Exported &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPRelaxed>::operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPStrict>::operator()(const polyregion::polyast::FunctionAttr::FPStrict &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::Entry>::operator()(const polyregion::polyast::FunctionAttr::Entry &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Arg>::operator()(const polyregion::polyast::Arg &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Function>::operator()(const polyregion::polyast::Function &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Program>::operator()(const polyregion::polyast::Program &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::StructDef>::operator()(const polyregion::polyast::StructDef &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::StructLayoutMember>::operator()(const polyregion::polyast::StructLayoutMember &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::StructLayout>::operator()(const polyregion::polyast::StructLayout &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileEvent>::operator()(const polyregion::polyast::CompileEvent &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileResult>::operator()(const polyregion::polyast::CompileResult &x) const noexcept { return x.hash_code(); }


