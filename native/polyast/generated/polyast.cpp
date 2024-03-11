#include "polyast.h"

namespace polyregion::polyast {

Sym::Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
size_t Sym::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(fqn)>()(fqn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Sym &x) { return x.dump(os); }
std::ostream &Sym::dump(std::ostream &os) const {
  os << "Sym(";
  os << '{';
  if (!fqn.empty()) {
    std::for_each(fqn.begin(), std::prev(fqn.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << fqn.back() << '"';
  }
  os << '}';
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool Sym::operator==(const Sym& rhs) const {
return (fqn == rhs.fqn);
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
[[nodiscard]] POLYREGION_EXPORT bool Named::operator==(const Named& rhs) const {
return (symbol == rhs.symbol) && (tpe == rhs.tpe);
}

TypeKind::Base::Base() = default;
uint32_t TypeKind::Any::id() const { return _v->id(); }
size_t TypeKind::Any::hash_code() const { return _v->hash_code(); }
std::ostream &TypeKind::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &TypeKind::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool TypeKind::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool TypeKind::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

TypeKind::None::None() noexcept : TypeKind::Base() {}
uint32_t TypeKind::None::id() const { return variant_id; };
size_t TypeKind::None::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::None &x) { return x.dump(os); }
std::ostream &TypeKind::None::dump(std::ostream &os) const {
  os << "None(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeKind::None::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeKind::None::operator TypeKind::Any() const { return static_pointer_cast<Base> (std::make_shared<None>(*this)); }

TypeKind::Ref::Ref() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Ref::id() const { return variant_id; };
size_t TypeKind::Ref::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Ref &x) { return x.dump(os); }
std::ostream &TypeKind::Ref::dump(std::ostream &os) const {
  os << "Ref(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeKind::Ref::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeKind::Ref::operator TypeKind::Any() const { return static_pointer_cast<Base> (std::make_shared<Ref>(*this)); }

TypeKind::Integral::Integral() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Integral::id() const { return variant_id; };
size_t TypeKind::Integral::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Integral &x) { return x.dump(os); }
std::ostream &TypeKind::Integral::dump(std::ostream &os) const {
  os << "Integral(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeKind::Integral::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeKind::Integral::operator TypeKind::Any() const { return static_pointer_cast<Base> (std::make_shared<Integral>(*this)); }

TypeKind::Fractional::Fractional() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Fractional::id() const { return variant_id; };
size_t TypeKind::Fractional::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Fractional &x) { return x.dump(os); }
std::ostream &TypeKind::Fractional::dump(std::ostream &os) const {
  os << "Fractional(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeKind::Fractional::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeKind::Fractional::operator TypeKind::Any() const { return static_pointer_cast<Base> (std::make_shared<Fractional>(*this)); }

Type::Base::Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
uint32_t Type::Any::id() const { return _v->id(); }
size_t Type::Any::hash_code() const { return _v->hash_code(); }
TypeKind::Any Type::Any::kind() const { return _v->kind; }
std::ostream &Type::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Type::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Type::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Type::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Type::Float16::Float16() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float16::id() const { return variant_id; };
size_t Type::Float16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Float16 &x) { return x.dump(os); }
std::ostream &Type::Float16::dump(std::ostream &os) const {
  os << "Float16(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Float16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float16::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Float16>(*this)); }

Type::Float32::Float32() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float32::id() const { return variant_id; };
size_t Type::Float32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Float32 &x) { return x.dump(os); }
std::ostream &Type::Float32::dump(std::ostream &os) const {
  os << "Float32(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Float32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float32::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Float32>(*this)); }

Type::Float64::Float64() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float64::id() const { return variant_id; };
size_t Type::Float64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Float64 &x) { return x.dump(os); }
std::ostream &Type::Float64::dump(std::ostream &os) const {
  os << "Float64(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Float64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Float64::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Float64>(*this)); }

Type::IntU8::IntU8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU8::id() const { return variant_id; };
size_t Type::IntU8::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU8 &x) { return x.dump(os); }
std::ostream &Type::IntU8::dump(std::ostream &os) const {
  os << "IntU8(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntU8::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU8::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU8>(*this)); }

Type::IntU16::IntU16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU16::id() const { return variant_id; };
size_t Type::IntU16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU16 &x) { return x.dump(os); }
std::ostream &Type::IntU16::dump(std::ostream &os) const {
  os << "IntU16(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntU16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU16::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU16>(*this)); }

Type::IntU32::IntU32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU32::id() const { return variant_id; };
size_t Type::IntU32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU32 &x) { return x.dump(os); }
std::ostream &Type::IntU32::dump(std::ostream &os) const {
  os << "IntU32(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntU32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU32::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU32>(*this)); }

Type::IntU64::IntU64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU64::id() const { return variant_id; };
size_t Type::IntU64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU64 &x) { return x.dump(os); }
std::ostream &Type::IntU64::dump(std::ostream &os) const {
  os << "IntU64(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntU64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntU64::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU64>(*this)); }

Type::IntS8::IntS8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS8::id() const { return variant_id; };
size_t Type::IntS8::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS8 &x) { return x.dump(os); }
std::ostream &Type::IntS8::dump(std::ostream &os) const {
  os << "IntS8(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntS8::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS8::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS8>(*this)); }

Type::IntS16::IntS16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS16::id() const { return variant_id; };
size_t Type::IntS16::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS16 &x) { return x.dump(os); }
std::ostream &Type::IntS16::dump(std::ostream &os) const {
  os << "IntS16(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntS16::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS16::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS16>(*this)); }

Type::IntS32::IntS32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS32::id() const { return variant_id; };
size_t Type::IntS32::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS32 &x) { return x.dump(os); }
std::ostream &Type::IntS32::dump(std::ostream &os) const {
  os << "IntS32(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntS32::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS32::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS32>(*this)); }

Type::IntS64::IntS64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS64::id() const { return variant_id; };
size_t Type::IntS64::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS64 &x) { return x.dump(os); }
std::ostream &Type::IntS64::dump(std::ostream &os) const {
  os << "IntS64(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::IntS64::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::IntS64::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS64>(*this)); }

Type::Nothing::Nothing() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Nothing::id() const { return variant_id; };
size_t Type::Nothing::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Nothing &x) { return x.dump(os); }
std::ostream &Type::Nothing::dump(std::ostream &os) const {
  os << "Nothing(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Nothing::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Nothing::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Nothing>(*this)); }

Type::Unit0::Unit0() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Unit0::id() const { return variant_id; };
size_t Type::Unit0::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Unit0 &x) { return x.dump(os); }
std::ostream &Type::Unit0::dump(std::ostream &os) const {
  os << "Unit0(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Unit0::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Unit0::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Unit0>(*this)); }

Type::Bool1::Bool1() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::Bool1::id() const { return variant_id; };
size_t Type::Bool1::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Bool1 &x) { return x.dump(os); }
std::ostream &Type::Bool1::dump(std::ostream &os) const {
  os << "Bool1(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Bool1::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Type::Bool1::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Bool1>(*this)); }

Type::Struct::Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args, std::vector<Sym> parents) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)), tpeVars(std::move(tpeVars)), args(std::move(args)), parents(std::move(parents)) {}
uint32_t Type::Struct::id() const { return variant_id; };
size_t Type::Struct::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(parents)>()(parents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Struct &x) { return x.dump(os); }
std::ostream &Type::Struct::dump(std::ostream &os) const {
  os << "Struct(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!parents.empty()) {
    std::for_each(parents.begin(), std::prev(parents.end()), [&os](auto &&x) { os << x; os << ','; });
    os << parents.back();
  }
  os << '}';
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Struct::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Type::Struct&>(rhs_);
return (this->name == rhs.name) && (this->tpeVars == rhs.tpeVars) && std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->parents == rhs.parents);
}
Type::Struct::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Struct>(*this)); }

Type::Ptr::Ptr(Type::Any component, std::optional<int32_t> length, TypeSpace::Any space) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)), length(std::move(length)), space(std::move(space)) {}
uint32_t Type::Ptr::id() const { return variant_id; };
size_t Type::Ptr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(length)>()(length) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Ptr &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Type::Ptr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Type::Ptr&>(rhs_);
return (this->component == rhs.component) && (this->length == rhs.length) && (this->space == rhs.space);
}
Type::Ptr::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Ptr>(*this)); }

Type::Var::Var(std::string name) noexcept : Type::Base(TypeKind::None()), name(std::move(name)) {}
uint32_t Type::Var::id() const { return variant_id; };
size_t Type::Var::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Var &x) { return x.dump(os); }
std::ostream &Type::Var::dump(std::ostream &os) const {
  os << "Var(";
  os << '"' << name << '"';
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Type::Var::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Type::Var&>(rhs_);
return (this->name == rhs.name);
}
Type::Var::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Var>(*this)); }

Type::Exec::Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept : Type::Base(TypeKind::None()), tpeVars(std::move(tpeVars)), args(std::move(args)), rtn(std::move(rtn)) {}
uint32_t Type::Exec::id() const { return variant_id; };
size_t Type::Exec::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Type::operator<<(std::ostream &os, const Type::Exec &x) { return x.dump(os); }
std::ostream &Type::Exec::dump(std::ostream &os) const {
  os << "Exec(";
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << tpeVars.back() << '"';
  }
  os << '}';
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
[ [nodiscard]] POLYREGION_EXPORT bool Type::Exec::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Type::Exec&>(rhs_);
return (this->tpeVars == rhs.tpeVars) && std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->rtn == rhs.rtn);
}
Type::Exec::operator Type::Any() const { return static_pointer_cast<Base> (std::make_shared<Exec>(*this)); }

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
[[nodiscard]] POLYREGION_EXPORT bool SourcePosition::operator==(const SourcePosition& rhs) const {
return (file == rhs.file) && (line == rhs.line) && (col == rhs.col);
}

Term::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
uint32_t Term::Any::id() const { return _v->id(); }
size_t Term::Any::hash_code() const { return _v->hash_code(); }
Type::Any Term::Any::tpe() const { return _v->tpe; }
std::ostream &Term::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Term::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Term::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Term::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Term::Select::Select(std::vector<Named> init, Named last) noexcept : Term::Base(last.tpe), init(std::move(init)), last(std::move(last)) {}
uint32_t Term::Select::id() const { return variant_id; };
size_t Term::Select::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(init)>()(init) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(last)>()(last) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Select &x) { return x.dump(os); }
std::ostream &Term::Select::dump(std::ostream &os) const {
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
[ [nodiscard]] POLYREGION_EXPORT bool Term::Select::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Select&>(rhs_);
return (this->init == rhs.init) && (this->last == rhs.last);
}
Term::Select::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Select>(*this)); }

Term::Poison::Poison(Type::Any t) noexcept : Term::Base(t), t(std::move(t)) {}
uint32_t Term::Poison::id() const { return variant_id; };
size_t Term::Poison::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(t)>()(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Poison &x) { return x.dump(os); }
std::ostream &Term::Poison::dump(std::ostream &os) const {
  os << "Poison(";
  os << t;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Poison::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Poison&>(rhs_);
return (this->t == rhs.t);
}
Term::Poison::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Poison>(*this)); }

Term::Float16Const::Float16Const(float value) noexcept : Term::Base(Type::Float16()), value(value) {}
uint32_t Term::Float16Const::id() const { return variant_id; };
size_t Term::Float16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Float16Const &x) { return x.dump(os); }
std::ostream &Term::Float16Const::dump(std::ostream &os) const {
  os << "Float16Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Float16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Float16Const&>(rhs_);
return (this->value == rhs.value);
}
Term::Float16Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Float16Const>(*this)); }

Term::Float32Const::Float32Const(float value) noexcept : Term::Base(Type::Float32()), value(value) {}
uint32_t Term::Float32Const::id() const { return variant_id; };
size_t Term::Float32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Float32Const &x) { return x.dump(os); }
std::ostream &Term::Float32Const::dump(std::ostream &os) const {
  os << "Float32Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Float32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Float32Const&>(rhs_);
return (this->value == rhs.value);
}
Term::Float32Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Float32Const>(*this)); }

Term::Float64Const::Float64Const(double value) noexcept : Term::Base(Type::Float64()), value(value) {}
uint32_t Term::Float64Const::id() const { return variant_id; };
size_t Term::Float64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Float64Const &x) { return x.dump(os); }
std::ostream &Term::Float64Const::dump(std::ostream &os) const {
  os << "Float64Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Float64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Float64Const&>(rhs_);
return (this->value == rhs.value);
}
Term::Float64Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Float64Const>(*this)); }

Term::IntU8Const::IntU8Const(int8_t value) noexcept : Term::Base(Type::IntU8()), value(value) {}
uint32_t Term::IntU8Const::id() const { return variant_id; };
size_t Term::IntU8Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU8Const &x) { return x.dump(os); }
std::ostream &Term::IntU8Const::dump(std::ostream &os) const {
  os << "IntU8Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntU8Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntU8Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntU8Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU8Const>(*this)); }

Term::IntU16Const::IntU16Const(uint16_t value) noexcept : Term::Base(Type::IntU16()), value(value) {}
uint32_t Term::IntU16Const::id() const { return variant_id; };
size_t Term::IntU16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU16Const &x) { return x.dump(os); }
std::ostream &Term::IntU16Const::dump(std::ostream &os) const {
  os << "IntU16Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntU16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntU16Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntU16Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU16Const>(*this)); }

Term::IntU32Const::IntU32Const(int32_t value) noexcept : Term::Base(Type::IntU32()), value(value) {}
uint32_t Term::IntU32Const::id() const { return variant_id; };
size_t Term::IntU32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU32Const &x) { return x.dump(os); }
std::ostream &Term::IntU32Const::dump(std::ostream &os) const {
  os << "IntU32Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntU32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntU32Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntU32Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU32Const>(*this)); }

Term::IntU64Const::IntU64Const(int64_t value) noexcept : Term::Base(Type::IntU64()), value(value) {}
uint32_t Term::IntU64Const::id() const { return variant_id; };
size_t Term::IntU64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU64Const &x) { return x.dump(os); }
std::ostream &Term::IntU64Const::dump(std::ostream &os) const {
  os << "IntU64Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntU64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntU64Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntU64Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntU64Const>(*this)); }

Term::IntS8Const::IntS8Const(int8_t value) noexcept : Term::Base(Type::IntS8()), value(value) {}
uint32_t Term::IntS8Const::id() const { return variant_id; };
size_t Term::IntS8Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS8Const &x) { return x.dump(os); }
std::ostream &Term::IntS8Const::dump(std::ostream &os) const {
  os << "IntS8Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntS8Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntS8Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntS8Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS8Const>(*this)); }

Term::IntS16Const::IntS16Const(int16_t value) noexcept : Term::Base(Type::IntS16()), value(value) {}
uint32_t Term::IntS16Const::id() const { return variant_id; };
size_t Term::IntS16Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS16Const &x) { return x.dump(os); }
std::ostream &Term::IntS16Const::dump(std::ostream &os) const {
  os << "IntS16Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntS16Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntS16Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntS16Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS16Const>(*this)); }

Term::IntS32Const::IntS32Const(int32_t value) noexcept : Term::Base(Type::IntS32()), value(value) {}
uint32_t Term::IntS32Const::id() const { return variant_id; };
size_t Term::IntS32Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS32Const &x) { return x.dump(os); }
std::ostream &Term::IntS32Const::dump(std::ostream &os) const {
  os << "IntS32Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntS32Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntS32Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntS32Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS32Const>(*this)); }

Term::IntS64Const::IntS64Const(int64_t value) noexcept : Term::Base(Type::IntS64()), value(value) {}
uint32_t Term::IntS64Const::id() const { return variant_id; };
size_t Term::IntS64Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS64Const &x) { return x.dump(os); }
std::ostream &Term::IntS64Const::dump(std::ostream &os) const {
  os << "IntS64Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::IntS64Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::IntS64Const&>(rhs_);
return (this->value == rhs.value);
}
Term::IntS64Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<IntS64Const>(*this)); }

Term::Unit0Const::Unit0Const() noexcept : Term::Base(Type::Unit0()) {}
uint32_t Term::Unit0Const::id() const { return variant_id; };
size_t Term::Unit0Const::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Unit0Const &x) { return x.dump(os); }
std::ostream &Term::Unit0Const::dump(std::ostream &os) const {
  os << "Unit0Const(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Unit0Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Term::Unit0Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Unit0Const>(*this)); }

Term::Bool1Const::Bool1Const(bool value) noexcept : Term::Base(Type::Bool1()), value(value) {}
uint32_t Term::Bool1Const::id() const { return variant_id; };
size_t Term::Bool1Const::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Term::operator<<(std::ostream &os, const Term::Bool1Const &x) { return x.dump(os); }
std::ostream &Term::Bool1Const::dump(std::ostream &os) const {
  os << "Bool1Const(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Term::Bool1Const::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Term::Bool1Const&>(rhs_);
return (this->value == rhs.value);
}
Term::Bool1Const::operator Term::Any() const { return static_pointer_cast<Base> (std::make_shared<Bool1Const>(*this)); }

TypeSpace::Base::Base() = default;
uint32_t TypeSpace::Any::id() const { return _v->id(); }
size_t TypeSpace::Any::hash_code() const { return _v->hash_code(); }
std::ostream &TypeSpace::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &TypeSpace::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool TypeSpace::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool TypeSpace::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

TypeSpace::Global::Global() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Global::id() const { return variant_id; };
size_t TypeSpace::Global::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeSpace::operator<<(std::ostream &os, const TypeSpace::Global &x) { return x.dump(os); }
std::ostream &TypeSpace::Global::dump(std::ostream &os) const {
  os << "Global(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeSpace::Global::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeSpace::Global::operator TypeSpace::Any() const { return static_pointer_cast<Base> (std::make_shared<Global>(*this)); }

TypeSpace::Local::Local() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Local::id() const { return variant_id; };
size_t TypeSpace::Local::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &TypeSpace::operator<<(std::ostream &os, const TypeSpace::Local &x) { return x.dump(os); }
std::ostream &TypeSpace::Local::dump(std::ostream &os) const {
  os << "Local(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool TypeSpace::Local::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
TypeSpace::Local::operator TypeSpace::Any() const { return static_pointer_cast<Base> (std::make_shared<Local>(*this)); }

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
[[nodiscard]] POLYREGION_EXPORT bool Overload::operator==(const Overload& rhs) const {
return std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

Spec::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Spec::Any::id() const { return _v->id(); }
size_t Spec::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Spec::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Spec::Any::terms() const { return _v->terms; }
Type::Any Spec::Any::tpe() const { return _v->tpe; }
std::ostream &Spec::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Spec::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Spec::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Spec::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Spec::Assert::Assert() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Nothing()) {}
uint32_t Spec::Assert::id() const { return variant_id; };
size_t Spec::Assert::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::Assert &x) { return x.dump(os); }
std::ostream &Spec::Assert::dump(std::ostream &os) const {
  os << "Assert(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::Assert::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::Assert::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<Assert>(*this)); }

Spec::GpuBarrierGlobal::GpuBarrierGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierGlobal::id() const { return variant_id; };
size_t Spec::GpuBarrierGlobal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &x) { return x.dump(os); }
std::ostream &Spec::GpuBarrierGlobal::dump(std::ostream &os) const {
  os << "GpuBarrierGlobal(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuBarrierGlobal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierGlobal::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuBarrierGlobal>(*this)); }

Spec::GpuBarrierLocal::GpuBarrierLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierLocal::id() const { return variant_id; };
size_t Spec::GpuBarrierLocal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierLocal &x) { return x.dump(os); }
std::ostream &Spec::GpuBarrierLocal::dump(std::ostream &os) const {
  os << "GpuBarrierLocal(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuBarrierLocal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierLocal::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuBarrierLocal>(*this)); }

Spec::GpuBarrierAll::GpuBarrierAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierAll::id() const { return variant_id; };
size_t Spec::GpuBarrierAll::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierAll &x) { return x.dump(os); }
std::ostream &Spec::GpuBarrierAll::dump(std::ostream &os) const {
  os << "GpuBarrierAll(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuBarrierAll::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierAll::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuBarrierAll>(*this)); }

Spec::GpuFenceGlobal::GpuFenceGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceGlobal::id() const { return variant_id; };
size_t Spec::GpuFenceGlobal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceGlobal &x) { return x.dump(os); }
std::ostream &Spec::GpuFenceGlobal::dump(std::ostream &os) const {
  os << "GpuFenceGlobal(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuFenceGlobal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceGlobal::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuFenceGlobal>(*this)); }

Spec::GpuFenceLocal::GpuFenceLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceLocal::id() const { return variant_id; };
size_t Spec::GpuFenceLocal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceLocal &x) { return x.dump(os); }
std::ostream &Spec::GpuFenceLocal::dump(std::ostream &os) const {
  os << "GpuFenceLocal(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuFenceLocal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceLocal::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuFenceLocal>(*this)); }

Spec::GpuFenceAll::GpuFenceAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceAll::id() const { return variant_id; };
size_t Spec::GpuFenceAll::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceAll &x) { return x.dump(os); }
std::ostream &Spec::GpuFenceAll::dump(std::ostream &os) const {
  os << "GpuFenceAll(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuFenceAll::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceAll::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuFenceAll>(*this)); }

Spec::GpuGlobalIdx::GpuGlobalIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalIdx::id() const { return variant_id; };
size_t Spec::GpuGlobalIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGlobalIdx &x) { return x.dump(os); }
std::ostream &Spec::GpuGlobalIdx::dump(std::ostream &os) const {
  os << "GpuGlobalIdx(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuGlobalIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuGlobalIdx&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuGlobalIdx::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuGlobalIdx>(*this)); }

Spec::GpuGlobalSize::GpuGlobalSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalSize::id() const { return variant_id; };
size_t Spec::GpuGlobalSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGlobalSize &x) { return x.dump(os); }
std::ostream &Spec::GpuGlobalSize::dump(std::ostream &os) const {
  os << "GpuGlobalSize(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuGlobalSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuGlobalSize&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuGlobalSize::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuGlobalSize>(*this)); }

Spec::GpuGroupIdx::GpuGroupIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupIdx::id() const { return variant_id; };
size_t Spec::GpuGroupIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGroupIdx &x) { return x.dump(os); }
std::ostream &Spec::GpuGroupIdx::dump(std::ostream &os) const {
  os << "GpuGroupIdx(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuGroupIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuGroupIdx&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuGroupIdx::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuGroupIdx>(*this)); }

Spec::GpuGroupSize::GpuGroupSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupSize::id() const { return variant_id; };
size_t Spec::GpuGroupSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGroupSize &x) { return x.dump(os); }
std::ostream &Spec::GpuGroupSize::dump(std::ostream &os) const {
  os << "GpuGroupSize(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuGroupSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuGroupSize&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuGroupSize::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuGroupSize>(*this)); }

Spec::GpuLocalIdx::GpuLocalIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalIdx::id() const { return variant_id; };
size_t Spec::GpuLocalIdx::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuLocalIdx &x) { return x.dump(os); }
std::ostream &Spec::GpuLocalIdx::dump(std::ostream &os) const {
  os << "GpuLocalIdx(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuLocalIdx::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuLocalIdx&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuLocalIdx::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuLocalIdx>(*this)); }

Spec::GpuLocalSize::GpuLocalSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalSize::id() const { return variant_id; };
size_t Spec::GpuLocalSize::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuLocalSize &x) { return x.dump(os); }
std::ostream &Spec::GpuLocalSize::dump(std::ostream &os) const {
  os << "GpuLocalSize(";
  os << dim;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Spec::GpuLocalSize::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Spec::GpuLocalSize&>(rhs_);
return (this->dim == rhs.dim);
}
Spec::GpuLocalSize::operator Spec::Any() const { return static_pointer_cast<Base> (std::make_shared<GpuLocalSize>(*this)); }

Intr::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Intr::Any::id() const { return _v->id(); }
size_t Intr::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Intr::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Intr::Any::terms() const { return _v->terms; }
Type::Any Intr::Any::tpe() const { return _v->tpe; }
std::ostream &Intr::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Intr::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Intr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Intr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Intr::BNot::BNot(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8()},Type::IntU8()),Overload({Type::IntU16()},Type::IntU16()),Overload({Type::IntU32()},Type::IntU32()),Overload({Type::IntU64()},Type::IntU64()),Overload({Type::IntS8()},Type::IntS8()),Overload({Type::IntS16()},Type::IntS16()),Overload({Type::IntS32()},Type::IntS32()),Overload({Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::BNot::id() const { return variant_id; };
size_t Intr::BNot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BNot &x) { return x.dump(os); }
std::ostream &Intr::BNot::dump(std::ostream &os) const {
  os << "BNot(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BNot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BNot&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Intr::BNot::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BNot>(*this)); }

Intr::LogicNot::LogicNot(Term::Any x) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x}, Type::Bool1()), x(std::move(x)) {}
uint32_t Intr::LogicNot::id() const { return variant_id; };
size_t Intr::LogicNot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicNot &x) { return x.dump(os); }
std::ostream &Intr::LogicNot::dump(std::ostream &os) const {
  os << "LogicNot(";
  os << x;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicNot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicNot&>(rhs_);
return (this->x == rhs.x);
}
Intr::LogicNot::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicNot>(*this)); }

Intr::Pos::Pos(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Pos::id() const { return variant_id; };
size_t Intr::Pos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Pos &x) { return x.dump(os); }
std::ostream &Intr::Pos::dump(std::ostream &os) const {
  os << "Pos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Pos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Pos&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Intr::Pos::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Pos>(*this)); }

Intr::Neg::Neg(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Neg::id() const { return variant_id; };
size_t Intr::Neg::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Neg &x) { return x.dump(os); }
std::ostream &Intr::Neg::dump(std::ostream &os) const {
  os << "Neg(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Neg::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Neg&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Intr::Neg::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Neg>(*this)); }

Intr::Add::Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Add::id() const { return variant_id; };
size_t Intr::Add::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Add &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Add::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Add&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Add::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Add>(*this)); }

Intr::Sub::Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Sub::id() const { return variant_id; };
size_t Intr::Sub::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Sub &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Sub::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Sub&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Sub::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Sub>(*this)); }

Intr::Mul::Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Mul::id() const { return variant_id; };
size_t Intr::Mul::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Mul &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Mul::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Mul&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Mul::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Mul>(*this)); }

Intr::Div::Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Div::id() const { return variant_id; };
size_t Intr::Div::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Div &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Div::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Div&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Div::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Div>(*this)); }

Intr::Rem::Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Rem::id() const { return variant_id; };
size_t Intr::Rem::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Rem &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Rem::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Rem&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Rem::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Rem>(*this)); }

Intr::Min::Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Min::id() const { return variant_id; };
size_t Intr::Min::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Min &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Min::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Min&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Min::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Min>(*this)); }

Intr::Max::Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Max::id() const { return variant_id; };
size_t Intr::Max::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Max &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::Max::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::Max&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::Max::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<Max>(*this)); }

Intr::BAnd::BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BAnd::id() const { return variant_id; };
size_t Intr::BAnd::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BAnd &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BAnd::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BAnd&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BAnd::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BAnd>(*this)); }

Intr::BOr::BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BOr::id() const { return variant_id; };
size_t Intr::BOr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BOr &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BOr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BOr&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BOr::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BOr>(*this)); }

Intr::BXor::BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BXor::id() const { return variant_id; };
size_t Intr::BXor::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BXor &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BXor::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BXor&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BXor::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BXor>(*this)); }

Intr::BSL::BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSL::id() const { return variant_id; };
size_t Intr::BSL::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BSL &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BSL::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BSL&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BSL::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BSL>(*this)); }

Intr::BSR::BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSR::id() const { return variant_id; };
size_t Intr::BSR::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BSR &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BSR::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BSR&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BSR::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BSR>(*this)); }

Intr::BZSR::BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BZSR::id() const { return variant_id; };
size_t Intr::BZSR::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BZSR &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Intr::BZSR::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::BZSR&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Intr::BZSR::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<BZSR>(*this)); }

Intr::LogicAnd::LogicAnd(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicAnd::id() const { return variant_id; };
size_t Intr::LogicAnd::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicAnd &x) { return x.dump(os); }
std::ostream &Intr::LogicAnd::dump(std::ostream &os) const {
  os << "LogicAnd(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicAnd::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicAnd&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicAnd::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicAnd>(*this)); }

Intr::LogicOr::LogicOr(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicOr::id() const { return variant_id; };
size_t Intr::LogicOr::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicOr &x) { return x.dump(os); }
std::ostream &Intr::LogicOr::dump(std::ostream &os) const {
  os << "LogicOr(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicOr::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicOr&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicOr::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicOr>(*this)); }

Intr::LogicEq::LogicEq(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicEq::id() const { return variant_id; };
size_t Intr::LogicEq::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicEq &x) { return x.dump(os); }
std::ostream &Intr::LogicEq::dump(std::ostream &os) const {
  os << "LogicEq(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicEq::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicEq&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicEq::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicEq>(*this)); }

Intr::LogicNeq::LogicNeq(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicNeq::id() const { return variant_id; };
size_t Intr::LogicNeq::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicNeq &x) { return x.dump(os); }
std::ostream &Intr::LogicNeq::dump(std::ostream &os) const {
  os << "LogicNeq(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicNeq::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicNeq&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicNeq::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicNeq>(*this)); }

Intr::LogicLte::LogicLte(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLte::id() const { return variant_id; };
size_t Intr::LogicLte::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicLte &x) { return x.dump(os); }
std::ostream &Intr::LogicLte::dump(std::ostream &os) const {
  os << "LogicLte(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicLte::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicLte&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicLte::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicLte>(*this)); }

Intr::LogicGte::LogicGte(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGte::id() const { return variant_id; };
size_t Intr::LogicGte::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicGte &x) { return x.dump(os); }
std::ostream &Intr::LogicGte::dump(std::ostream &os) const {
  os << "LogicGte(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicGte::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicGte&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicGte::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicGte>(*this)); }

Intr::LogicLt::LogicLt(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLt::id() const { return variant_id; };
size_t Intr::LogicLt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicLt &x) { return x.dump(os); }
std::ostream &Intr::LogicLt::dump(std::ostream &os) const {
  os << "LogicLt(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicLt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicLt&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicLt::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicLt>(*this)); }

Intr::LogicGt::LogicGt(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGt::id() const { return variant_id; };
size_t Intr::LogicGt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicGt &x) { return x.dump(os); }
std::ostream &Intr::LogicGt::dump(std::ostream &os) const {
  os << "LogicGt(";
  os << x;
  os << ',';
  os << y;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Intr::LogicGt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Intr::LogicGt&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y);
}
Intr::LogicGt::operator Intr::Any() const { return static_pointer_cast<Base> (std::make_shared<LogicGt>(*this)); }

Math::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Math::Any::id() const { return _v->id(); }
size_t Math::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Math::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Math::Any::terms() const { return _v->terms; }
Type::Any Math::Any::tpe() const { return _v->tpe; }
std::ostream &Math::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Math::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Math::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Math::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Math::Abs::Abs(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Abs::id() const { return variant_id; };
size_t Math::Abs::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Abs &x) { return x.dump(os); }
std::ostream &Math::Abs::dump(std::ostream &os) const {
  os << "Abs(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Abs::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Abs&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Abs::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Abs>(*this)); }

Math::Sin::Sin(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sin::id() const { return variant_id; };
size_t Math::Sin::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Sin &x) { return x.dump(os); }
std::ostream &Math::Sin::dump(std::ostream &os) const {
  os << "Sin(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Sin::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Sin&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Sin::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Sin>(*this)); }

Math::Cos::Cos(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cos::id() const { return variant_id; };
size_t Math::Cos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Cos &x) { return x.dump(os); }
std::ostream &Math::Cos::dump(std::ostream &os) const {
  os << "Cos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Cos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Cos&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Cos::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Cos>(*this)); }

Math::Tan::Tan(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tan::id() const { return variant_id; };
size_t Math::Tan::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Tan &x) { return x.dump(os); }
std::ostream &Math::Tan::dump(std::ostream &os) const {
  os << "Tan(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Tan::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Tan&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Tan::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Tan>(*this)); }

Math::Asin::Asin(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Asin::id() const { return variant_id; };
size_t Math::Asin::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Asin &x) { return x.dump(os); }
std::ostream &Math::Asin::dump(std::ostream &os) const {
  os << "Asin(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Asin::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Asin&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Asin::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Asin>(*this)); }

Math::Acos::Acos(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Acos::id() const { return variant_id; };
size_t Math::Acos::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Acos &x) { return x.dump(os); }
std::ostream &Math::Acos::dump(std::ostream &os) const {
  os << "Acos(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Acos::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Acos&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Acos::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Acos>(*this)); }

Math::Atan::Atan(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Atan::id() const { return variant_id; };
size_t Math::Atan::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Atan &x) { return x.dump(os); }
std::ostream &Math::Atan::dump(std::ostream &os) const {
  os << "Atan(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Atan::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Atan&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Atan::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Atan>(*this)); }

Math::Sinh::Sinh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sinh::id() const { return variant_id; };
size_t Math::Sinh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Sinh &x) { return x.dump(os); }
std::ostream &Math::Sinh::dump(std::ostream &os) const {
  os << "Sinh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Sinh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Sinh&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Sinh::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Sinh>(*this)); }

Math::Cosh::Cosh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cosh::id() const { return variant_id; };
size_t Math::Cosh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Cosh &x) { return x.dump(os); }
std::ostream &Math::Cosh::dump(std::ostream &os) const {
  os << "Cosh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Cosh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Cosh&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Cosh::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Cosh>(*this)); }

Math::Tanh::Tanh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tanh::id() const { return variant_id; };
size_t Math::Tanh::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Tanh &x) { return x.dump(os); }
std::ostream &Math::Tanh::dump(std::ostream &os) const {
  os << "Tanh(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Tanh::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Tanh&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Tanh::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Tanh>(*this)); }

Math::Signum::Signum(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Signum::id() const { return variant_id; };
size_t Math::Signum::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Signum &x) { return x.dump(os); }
std::ostream &Math::Signum::dump(std::ostream &os) const {
  os << "Signum(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Signum::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Signum&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Signum::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Signum>(*this)); }

Math::Round::Round(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Round::id() const { return variant_id; };
size_t Math::Round::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Round &x) { return x.dump(os); }
std::ostream &Math::Round::dump(std::ostream &os) const {
  os << "Round(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Round::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Round&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Round::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Round>(*this)); }

Math::Ceil::Ceil(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Ceil::id() const { return variant_id; };
size_t Math::Ceil::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Ceil &x) { return x.dump(os); }
std::ostream &Math::Ceil::dump(std::ostream &os) const {
  os << "Ceil(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Ceil::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Ceil&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Ceil::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Ceil>(*this)); }

Math::Floor::Floor(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Floor::id() const { return variant_id; };
size_t Math::Floor::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Floor &x) { return x.dump(os); }
std::ostream &Math::Floor::dump(std::ostream &os) const {
  os << "Floor(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Floor::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Floor&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Floor::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Floor>(*this)); }

Math::Rint::Rint(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Rint::id() const { return variant_id; };
size_t Math::Rint::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Rint &x) { return x.dump(os); }
std::ostream &Math::Rint::dump(std::ostream &os) const {
  os << "Rint(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Rint::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Rint&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Rint::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Rint>(*this)); }

Math::Sqrt::Sqrt(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sqrt::id() const { return variant_id; };
size_t Math::Sqrt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Sqrt &x) { return x.dump(os); }
std::ostream &Math::Sqrt::dump(std::ostream &os) const {
  os << "Sqrt(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Sqrt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Sqrt&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Sqrt::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Sqrt>(*this)); }

Math::Cbrt::Cbrt(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cbrt::id() const { return variant_id; };
size_t Math::Cbrt::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Cbrt &x) { return x.dump(os); }
std::ostream &Math::Cbrt::dump(std::ostream &os) const {
  os << "Cbrt(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Cbrt::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Cbrt&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Cbrt::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Cbrt>(*this)); }

Math::Exp::Exp(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Exp::id() const { return variant_id; };
size_t Math::Exp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Exp &x) { return x.dump(os); }
std::ostream &Math::Exp::dump(std::ostream &os) const {
  os << "Exp(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Exp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Exp&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Exp::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Exp>(*this)); }

Math::Expm1::Expm1(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Expm1::id() const { return variant_id; };
size_t Math::Expm1::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Expm1 &x) { return x.dump(os); }
std::ostream &Math::Expm1::dump(std::ostream &os) const {
  os << "Expm1(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Expm1::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Expm1&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Expm1::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Expm1>(*this)); }

Math::Log::Log(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log::id() const { return variant_id; };
size_t Math::Log::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Log &x) { return x.dump(os); }
std::ostream &Math::Log::dump(std::ostream &os) const {
  os << "Log(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Log::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Log&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Log::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Log>(*this)); }

Math::Log1p::Log1p(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log1p::id() const { return variant_id; };
size_t Math::Log1p::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Log1p &x) { return x.dump(os); }
std::ostream &Math::Log1p::dump(std::ostream &os) const {
  os << "Log1p(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Log1p::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Log1p&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Log1p::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Log1p>(*this)); }

Math::Log10::Log10(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log10::id() const { return variant_id; };
size_t Math::Log10::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Log10 &x) { return x.dump(os); }
std::ostream &Math::Log10::dump(std::ostream &os) const {
  os << "Log10(";
  os << x;
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Math::Log10::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Log10&>(rhs_);
return (this->x == rhs.x) && (this->rtn == rhs.rtn);
}
Math::Log10::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Log10>(*this)); }

Math::Pow::Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Pow::id() const { return variant_id; };
size_t Math::Pow::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Pow &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Math::Pow::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Pow&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Math::Pow::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Pow>(*this)); }

Math::Atan2::Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Atan2::id() const { return variant_id; };
size_t Math::Atan2::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Atan2 &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Math::Atan2::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Atan2&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Math::Atan2::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Atan2>(*this)); }

Math::Hypot::Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Hypot::id() const { return variant_id; };
size_t Math::Hypot::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Math::operator<<(std::ostream &os, const Math::Hypot &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Math::Hypot::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Math::Hypot&>(rhs_);
return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
Math::Hypot::operator Math::Any() const { return static_pointer_cast<Base> (std::make_shared<Hypot>(*this)); }

Expr::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
uint32_t Expr::Any::id() const { return _v->id(); }
size_t Expr::Any::hash_code() const { return _v->hash_code(); }
Type::Any Expr::Any::tpe() const { return _v->tpe; }
std::ostream &Expr::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Expr::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Expr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Expr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Expr::SpecOp::SpecOp(Spec::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::SpecOp::id() const { return variant_id; };
size_t Expr::SpecOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::SpecOp &x) { return x.dump(os); }
std::ostream &Expr::SpecOp::dump(std::ostream &os) const {
  os << "SpecOp(";
  os << op;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::SpecOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::SpecOp&>(rhs_);
return (this->op == rhs.op);
}
Expr::SpecOp::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<SpecOp>(*this)); }

Expr::MathOp::MathOp(Math::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::MathOp::id() const { return variant_id; };
size_t Expr::MathOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::MathOp &x) { return x.dump(os); }
std::ostream &Expr::MathOp::dump(std::ostream &os) const {
  os << "MathOp(";
  os << op;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::MathOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::MathOp&>(rhs_);
return (this->op == rhs.op);
}
Expr::MathOp::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<MathOp>(*this)); }

Expr::IntrOp::IntrOp(Intr::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::IntrOp::id() const { return variant_id; };
size_t Expr::IntrOp::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::IntrOp &x) { return x.dump(os); }
std::ostream &Expr::IntrOp::dump(std::ostream &os) const {
  os << "IntrOp(";
  os << op;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::IntrOp::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::IntrOp&>(rhs_);
return (this->op == rhs.op);
}
Expr::IntrOp::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<IntrOp>(*this)); }

Expr::Cast::Cast(Term::Any from, Type::Any as) noexcept : Expr::Base(as), from(std::move(from)), as(std::move(as)) {}
uint32_t Expr::Cast::id() const { return variant_id; };
size_t Expr::Cast::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(from)>()(from) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(as)>()(as) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Cast &x) { return x.dump(os); }
std::ostream &Expr::Cast::dump(std::ostream &os) const {
  os << "Cast(";
  os << from;
  os << ',';
  os << as;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::Cast::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::Cast&>(rhs_);
return (this->from == rhs.from) && (this->as == rhs.as);
}
Expr::Cast::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<Cast>(*this)); }

Expr::Alias::Alias(Term::Any ref) noexcept : Expr::Base(ref.tpe()), ref(std::move(ref)) {}
uint32_t Expr::Alias::id() const { return variant_id; };
size_t Expr::Alias::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(ref)>()(ref) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alias &x) { return x.dump(os); }
std::ostream &Expr::Alias::dump(std::ostream &os) const {
  os << "Alias(";
  os << ref;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::Alias::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::Alias&>(rhs_);
return (this->ref == rhs.ref);
}
Expr::Alias::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<Alias>(*this)); }

Expr::Index::Index(Term::Any lhs, Term::Any idx, Type::Any component) noexcept : Expr::Base(component), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::Index::id() const { return variant_id; };
size_t Expr::Index::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Index &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Expr::Index::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::Index&>(rhs_);
return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->component == rhs.component);
}
Expr::Index::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<Index>(*this)); }

Expr::RefTo::RefTo(Term::Any lhs, std::optional<Term::Any> idx, Type::Any component) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::RefTo::id() const { return variant_id; };
size_t Expr::RefTo::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::RefTo &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Expr::RefTo::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::RefTo&>(rhs_);
return (this->lhs == rhs.lhs) && ( (!this->idx && !rhs.idx) || (this->idx && rhs.idx && *this->idx == *rhs.idx) ) && (this->component == rhs.component);
}
Expr::RefTo::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<RefTo>(*this)); }

Expr::Alloc::Alloc(Type::Any component, Term::Any size) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), component(std::move(component)), size(std::move(size)) {}
uint32_t Expr::Alloc::id() const { return variant_id; };
size_t Expr::Alloc::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(component)>()(component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(size)>()(size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alloc &x) { return x.dump(os); }
std::ostream &Expr::Alloc::dump(std::ostream &os) const {
  os << "Alloc(";
  os << component;
  os << ',';
  os << size;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::Alloc::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::Alloc&>(rhs_);
return (this->component == rhs.component) && (this->size == rhs.size);
}
Expr::Alloc::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<Alloc>(*this)); }

Expr::Invoke::Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, std::vector<Term::Any> captures, Type::Any rtn) noexcept : Expr::Base(rtn), name(std::move(name)), tpeArgs(std::move(tpeArgs)), receiver(std::move(receiver)), args(std::move(args)), captures(std::move(captures)), rtn(std::move(rtn)) {}
uint32_t Expr::Invoke::id() const { return variant_id; };
size_t Expr::Invoke::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeArgs)>()(tpeArgs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(captures)>()(captures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Invoke &x) { return x.dump(os); }
std::ostream &Expr::Invoke::dump(std::ostream &os) const {
  os << "Invoke(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeArgs.empty()) {
    std::for_each(tpeArgs.begin(), std::prev(tpeArgs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << tpeArgs.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (receiver) {
    os << (*receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!captures.empty()) {
    std::for_each(captures.begin(), std::prev(captures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << captures.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Expr::Invoke::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Expr::Invoke&>(rhs_);
return (this->name == rhs.name) && std::equal(this->tpeArgs.begin(), this->tpeArgs.end(), rhs.tpeArgs.begin(), [](auto &&l, auto &&r) { return l == r; }) && ( (!this->receiver && !rhs.receiver) || (this->receiver && rhs.receiver && *this->receiver == *rhs.receiver) ) && std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(this->captures.begin(), this->captures.end(), rhs.captures.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->rtn == rhs.rtn);
}
Expr::Invoke::operator Expr::Any() const { return static_pointer_cast<Base> (std::make_shared<Invoke>(*this)); }

Stmt::Base::Base() = default;
uint32_t Stmt::Any::id() const { return _v->id(); }
size_t Stmt::Any::hash_code() const { return _v->hash_code(); }
std::ostream &Stmt::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &Stmt::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool Stmt::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool Stmt::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

Stmt::Block::Block(std::vector<Stmt::Any> stmts) noexcept : Stmt::Base(), stmts(std::move(stmts)) {}
uint32_t Stmt::Block::id() const { return variant_id; };
size_t Stmt::Block::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(stmts)>()(stmts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Block &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Block::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Block&>(rhs_);
return std::equal(this->stmts.begin(), this->stmts.end(), rhs.stmts.begin(), [](auto &&l, auto &&r) { return l == r; });
}
Stmt::Block::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Block>(*this)); }

Stmt::Comment::Comment(std::string value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Comment::id() const { return variant_id; };
size_t Stmt::Comment::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Comment &x) { return x.dump(os); }
std::ostream &Stmt::Comment::dump(std::ostream &os) const {
  os << "Comment(";
  os << '"' << value << '"';
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Comment::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Comment&>(rhs_);
return (this->value == rhs.value);
}
Stmt::Comment::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Comment>(*this)); }

Stmt::Var::Var(Named name, std::optional<Expr::Any> expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
uint32_t Stmt::Var::id() const { return variant_id; };
size_t Stmt::Var::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Var &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Var::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Var&>(rhs_);
return (this->name == rhs.name) && ( (!this->expr && !rhs.expr) || (this->expr && rhs.expr && *this->expr == *rhs.expr) );
}
Stmt::Var::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Var>(*this)); }

Stmt::Mut::Mut(Term::Any name, Expr::Any expr, bool copy) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)), copy(copy) {}
uint32_t Stmt::Mut::id() const { return variant_id; };
size_t Stmt::Mut::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(copy)>()(copy) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Mut &x) { return x.dump(os); }
std::ostream &Stmt::Mut::dump(std::ostream &os) const {
  os << "Mut(";
  os << name;
  os << ',';
  os << expr;
  os << ',';
  os << copy;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Mut::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Mut&>(rhs_);
return (this->name == rhs.name) && (this->expr == rhs.expr) && (this->copy == rhs.copy);
}
Stmt::Mut::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Mut>(*this)); }

Stmt::Update::Update(Term::Any lhs, Term::Any idx, Term::Any value) noexcept : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
uint32_t Stmt::Update::id() const { return variant_id; };
size_t Stmt::Update::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Update &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Update::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Update&>(rhs_);
return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->value == rhs.value);
}
Stmt::Update::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Update>(*this)); }

Stmt::While::While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), tests(std::move(tests)), cond(std::move(cond)), body(std::move(body)) {}
uint32_t Stmt::While::id() const { return variant_id; };
size_t Stmt::While::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(tests)>()(tests) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::While &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::While::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::While&>(rhs_);
return std::equal(this->tests.begin(), this->tests.end(), rhs.tests.begin(), [](auto &&l, auto &&r) { return l == r; }) && (this->cond == rhs.cond) && std::equal(this->body.begin(), this->body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; });
}
Stmt::While::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<While>(*this)); }

Stmt::Break::Break() noexcept : Stmt::Base() {}
uint32_t Stmt::Break::id() const { return variant_id; };
size_t Stmt::Break::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Break &x) { return x.dump(os); }
std::ostream &Stmt::Break::dump(std::ostream &os) const {
  os << "Break(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Break::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Stmt::Break::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Break>(*this)); }

Stmt::Cont::Cont() noexcept : Stmt::Base() {}
uint32_t Stmt::Cont::id() const { return variant_id; };
size_t Stmt::Cont::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cont &x) { return x.dump(os); }
std::ostream &Stmt::Cont::dump(std::ostream &os) const {
  os << "Cont(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Cont::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
Stmt::Cont::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Cont>(*this)); }

Stmt::Cond::Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
uint32_t Stmt::Cond::id() const { return variant_id; };
size_t Stmt::Cond::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(trueBr)>()(trueBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(falseBr)>()(falseBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cond &x) { return x.dump(os); }
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
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Cond::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Cond&>(rhs_);
return (this->cond == rhs.cond) && std::equal(this->trueBr.begin(), this->trueBr.end(), rhs.trueBr.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(this->falseBr.begin(), this->falseBr.end(), rhs.falseBr.begin(), [](auto &&l, auto &&r) { return l == r; });
}
Stmt::Cond::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Cond>(*this)); }

Stmt::Return::Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Return::id() const { return variant_id; };
size_t Stmt::Return::hash_code() const { 
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Return &x) { return x.dump(os); }
std::ostream &Stmt::Return::dump(std::ostream &os) const {
  os << "Return(";
  os << value;
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool Stmt::Return::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  auto rhs = static_cast<const Stmt::Return&>(rhs_);
return (this->value == rhs.value);
}
Stmt::Return::operator Stmt::Any() const { return static_pointer_cast<Base> (std::make_shared<Return>(*this)); }

StructMember::StructMember(Named named, bool isMutable) noexcept : named(std::move(named)), isMutable(isMutable) {}
size_t StructMember::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(named)>()(named) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(isMutable)>()(isMutable) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const StructMember &x) { return x.dump(os); }
std::ostream &StructMember::dump(std::ostream &os) const {
  os << "StructMember(";
  os << named;
  os << ',';
  os << isMutable;
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool StructMember::operator==(const StructMember& rhs) const {
return (named == rhs.named) && (isMutable == rhs.isMutable);
}

StructDef::StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<StructMember> members, std::vector<Sym> parents) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), members(std::move(members)), parents(std::move(parents)) {}
size_t StructDef::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(parents)>()(parents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const StructDef &x) { return x.dump(os); }
std::ostream &StructDef::dump(std::ostream &os) const {
  os << "StructDef(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!members.empty()) {
    std::for_each(members.begin(), std::prev(members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << members.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!parents.empty()) {
    std::for_each(parents.begin(), std::prev(parents.end()), [&os](auto &&x) { os << x; os << ','; });
    os << parents.back();
  }
  os << '}';
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool StructDef::operator==(const StructDef& rhs) const {
return (name == rhs.name) && (tpeVars == rhs.tpeVars) && (members == rhs.members) && (parents == rhs.parents);
}

Signature::Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)) {}
size_t Signature::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(moduleCaptures)>()(moduleCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(termCaptures)>()(termCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Signature &x) { return x.dump(os); }
std::ostream &Signature::dump(std::ostream &os) const {
  os << "Signature(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (receiver) {
    os << (*receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!moduleCaptures.empty()) {
    std::for_each(moduleCaptures.begin(), std::prev(moduleCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << moduleCaptures.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!termCaptures.empty()) {
    std::for_each(termCaptures.begin(), std::prev(termCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << termCaptures.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool Signature::operator==(const Signature& rhs) const {
return (name == rhs.name) && (tpeVars == rhs.tpeVars) && ( (!receiver && !rhs.receiver) || (receiver && rhs.receiver && *receiver == *rhs.receiver) ) && std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(moduleCaptures.begin(), moduleCaptures.end(), rhs.moduleCaptures.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(termCaptures.begin(), termCaptures.end(), rhs.termCaptures.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

InvokeSignature::InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> captures, Type::Any rtn) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), captures(std::move(captures)), rtn(std::move(rtn)) {}
size_t InvokeSignature::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(captures)>()(captures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const InvokeSignature &x) { return x.dump(os); }
std::ostream &InvokeSignature::dump(std::ostream &os) const {
  os << "InvokeSignature(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << x; os << ','; });
    os << tpeVars.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (receiver) {
    os << (*receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!captures.empty()) {
    std::for_each(captures.begin(), std::prev(captures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << captures.back();
  }
  os << '}';
  os << ',';
  os << rtn;
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool InvokeSignature::operator==(const InvokeSignature& rhs) const {
return (name == rhs.name) && std::equal(tpeVars.begin(), tpeVars.end(), rhs.tpeVars.begin(), [](auto &&l, auto &&r) { return l == r; }) && ( (!receiver && !rhs.receiver) || (receiver && rhs.receiver && *receiver == *rhs.receiver) ) && std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && std::equal(captures.begin(), captures.end(), rhs.captures.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

FunctionKind::Base::Base() = default;
uint32_t FunctionKind::Any::id() const { return _v->id(); }
size_t FunctionKind::Any::hash_code() const { return _v->hash_code(); }
std::ostream &FunctionKind::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &FunctionKind::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool FunctionKind::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool FunctionKind::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

FunctionKind::Internal::Internal() noexcept : FunctionKind::Base() {}
uint32_t FunctionKind::Internal::id() const { return variant_id; };
size_t FunctionKind::Internal::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &FunctionKind::operator<<(std::ostream &os, const FunctionKind::Internal &x) { return x.dump(os); }
std::ostream &FunctionKind::Internal::dump(std::ostream &os) const {
  os << "Internal(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool FunctionKind::Internal::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
FunctionKind::Internal::operator FunctionKind::Any() const { return static_pointer_cast<Base> (std::make_shared<Internal>(*this)); }

FunctionKind::Exported::Exported() noexcept : FunctionKind::Base() {}
uint32_t FunctionKind::Exported::id() const { return variant_id; };
size_t FunctionKind::Exported::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &FunctionKind::operator<<(std::ostream &os, const FunctionKind::Exported &x) { return x.dump(os); }
std::ostream &FunctionKind::Exported::dump(std::ostream &os) const {
  os << "Exported(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool FunctionKind::Exported::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
FunctionKind::Exported::operator FunctionKind::Any() const { return static_pointer_cast<Base> (std::make_shared<Exported>(*this)); }

FunctionAttr::Base::Base() = default;
uint32_t FunctionAttr::Any::id() const { return _v->id(); }
size_t FunctionAttr::Any::hash_code() const { return _v->hash_code(); }
std::ostream &FunctionAttr::Any::dump(std::ostream &os) const { return _v->dump(os); }
std::ostream &FunctionAttr::operator<<(std::ostream &os, const Any &x) { return x.dump(os); }
bool FunctionAttr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v) ; }
bool FunctionAttr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v) ; }

FunctionAttr::FPRelaxed::FPRelaxed() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPRelaxed::id() const { return variant_id; };
size_t FunctionAttr::FPRelaxed::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &FunctionAttr::operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &x) { return x.dump(os); }
std::ostream &FunctionAttr::FPRelaxed::dump(std::ostream &os) const {
  os << "FPRelaxed(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool FunctionAttr::FPRelaxed::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
FunctionAttr::FPRelaxed::operator FunctionAttr::Any() const { return static_pointer_cast<Base> (std::make_shared<FPRelaxed>(*this)); }

FunctionAttr::FPStrict::FPStrict() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPStrict::id() const { return variant_id; };
size_t FunctionAttr::FPStrict::hash_code() const { 
  size_t seed = variant_id;
  return seed;
}
std::ostream &FunctionAttr::operator<<(std::ostream &os, const FunctionAttr::FPStrict &x) { return x.dump(os); }
std::ostream &FunctionAttr::FPStrict::dump(std::ostream &os) const {
  os << "FPStrict(";
  os << ')';
  return os;
}
[ [nodiscard]] POLYREGION_EXPORT bool FunctionAttr::FPStrict::operator==(const Base& rhs_) const {
  if(rhs_.id() != variant_id) return false;
  return true;
}
FunctionAttr::FPStrict::operator FunctionAttr::Any() const { return static_pointer_cast<Base> (std::make_shared<FPStrict>(*this)); }

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
[[nodiscard]] POLYREGION_EXPORT bool Arg::operator==(const Arg& rhs) const {
return (named == rhs.named) && (pos == rhs.pos);
}

Function::Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args, std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body, FunctionKind::Any kind) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)), body(std::move(body)), kind(std::move(kind)) {}
size_t Function::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(moduleCaptures)>()(moduleCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(termCaptures)>()(termCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(kind)>()(kind) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Function &x) { return x.dump(os); }
std::ostream &Function::dump(std::ostream &os) const {
  os << "Function(";
  os << name;
  os << ',';
  os << '{';
  if (!tpeVars.empty()) {
    std::for_each(tpeVars.begin(), std::prev(tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (receiver) {
    os << (*receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!args.empty()) {
    std::for_each(args.begin(), std::prev(args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!moduleCaptures.empty()) {
    std::for_each(moduleCaptures.begin(), std::prev(moduleCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << moduleCaptures.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!termCaptures.empty()) {
    std::for_each(termCaptures.begin(), std::prev(termCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << termCaptures.back();
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
  os << kind;
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool Function::operator==(const Function& rhs) const {
return (name == rhs.name) && (tpeVars == rhs.tpeVars) && (receiver == rhs.receiver) && (args == rhs.args) && (moduleCaptures == rhs.moduleCaptures) && (termCaptures == rhs.termCaptures) && (rtn == rhs.rtn) && std::equal(body.begin(), body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; }) && (kind == rhs.kind);
}

Program::Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept : entry(std::move(entry)), functions(std::move(functions)), defs(std::move(defs)) {}
size_t Program::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(entry)>()(entry) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(functions)>()(functions) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(defs)>()(defs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const Program &x) { return x.dump(os); }
std::ostream &Program::dump(std::ostream &os) const {
  os << "Program(";
  os << entry;
  os << ',';
  os << '{';
  if (!functions.empty()) {
    std::for_each(functions.begin(), std::prev(functions.end()), [&os](auto &&x) { os << x; os << ','; });
    os << functions.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!defs.empty()) {
    std::for_each(defs.begin(), std::prev(defs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << defs.back();
  }
  os << '}';
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool Program::operator==(const Program& rhs) const {
return (entry == rhs.entry) && (functions == rhs.functions) && (defs == rhs.defs);
}

CompileLayoutMember::CompileLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
size_t CompileLayoutMember::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(offsetInBytes)>()(offsetInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const CompileLayoutMember &x) { return x.dump(os); }
std::ostream &CompileLayoutMember::dump(std::ostream &os) const {
  os << "CompileLayoutMember(";
  os << name;
  os << ',';
  os << offsetInBytes;
  os << ',';
  os << sizeInBytes;
  os << ')';
  return os;
}
[[nodiscard]] POLYREGION_EXPORT bool CompileLayoutMember::operator==(const CompileLayoutMember& rhs) const {
return (name == rhs.name) && (offsetInBytes == rhs.offsetInBytes) && (sizeInBytes == rhs.sizeInBytes);
}

CompileLayout::CompileLayout(Sym name, int64_t sizeInBytes, int64_t alignment, std::vector<CompileLayoutMember> members) noexcept : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
size_t CompileLayout::hash_code() const { 
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(alignment)>()(alignment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::ostream &operator<<(std::ostream &os, const CompileLayout &x) { return x.dump(os); }
std::ostream &CompileLayout::dump(std::ostream &os) const {
  os << "CompileLayout(";
  os << name;
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
[[nodiscard]] POLYREGION_EXPORT bool CompileLayout::operator==(const CompileLayout& rhs) const {
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
[[nodiscard]] POLYREGION_EXPORT bool CompileEvent::operator==(const CompileEvent& rhs) const {
return (epochMillis == rhs.epochMillis) && (elapsedNanos == rhs.elapsedNanos) && (name == rhs.name) && (data == rhs.data);
}

CompileResult::CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<CompileLayout> layouts, std::string messages) noexcept : binary(std::move(binary)), features(std::move(features)), events(std::move(events)), layouts(std::move(layouts)), messages(std::move(messages)) {}
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
[[nodiscard]] POLYREGION_EXPORT bool CompileResult::operator==(const CompileResult& rhs) const {
return (binary == rhs.binary) && (features == rhs.features) && (events == rhs.events) && (layouts == rhs.layouts) && (messages == rhs.messages);
}

} // namespace polyregion::polyast


std::size_t std::hash<polyregion::polyast::Sym>::operator()(const polyregion::polyast::Sym &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Named>::operator()(const polyregion::polyast::Named &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Any>::operator()(const polyregion::polyast::TypeKind::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::None>::operator()(const polyregion::polyast::TypeKind::None &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Ref>::operator()(const polyregion::polyast::TypeKind::Ref &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Integral>::operator()(const polyregion::polyast::TypeKind::Integral &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Fractional>::operator()(const polyregion::polyast::TypeKind::Fractional &x) const noexcept { return x.hash_code(); }
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
std::size_t std::hash<polyregion::polyast::Type::Var>::operator()(const polyregion::polyast::Type::Var &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Type::Exec>::operator()(const polyregion::polyast::Type::Exec &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::SourcePosition>::operator()(const polyregion::polyast::SourcePosition &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Any>::operator()(const polyregion::polyast::Term::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Select>::operator()(const polyregion::polyast::Term::Select &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Poison>::operator()(const polyregion::polyast::Term::Poison &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Float16Const>::operator()(const polyregion::polyast::Term::Float16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Float32Const>::operator()(const polyregion::polyast::Term::Float32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Float64Const>::operator()(const polyregion::polyast::Term::Float64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntU8Const>::operator()(const polyregion::polyast::Term::IntU8Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntU16Const>::operator()(const polyregion::polyast::Term::IntU16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntU32Const>::operator()(const polyregion::polyast::Term::IntU32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntU64Const>::operator()(const polyregion::polyast::Term::IntU64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntS8Const>::operator()(const polyregion::polyast::Term::IntS8Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntS16Const>::operator()(const polyregion::polyast::Term::IntS16Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntS32Const>::operator()(const polyregion::polyast::Term::IntS32Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::IntS64Const>::operator()(const polyregion::polyast::Term::IntS64Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Unit0Const>::operator()(const polyregion::polyast::Term::Unit0Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Term::Bool1Const>::operator()(const polyregion::polyast::Term::Bool1Const &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Any>::operator()(const polyregion::polyast::TypeSpace::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Global>::operator()(const polyregion::polyast::TypeSpace::Global &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeSpace::Local>::operator()(const polyregion::polyast::TypeSpace::Local &x) const noexcept { return x.hash_code(); }
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
std::size_t std::hash<polyregion::polyast::Expr::Any>::operator()(const polyregion::polyast::Expr::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::SpecOp>::operator()(const polyregion::polyast::Expr::SpecOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::MathOp>::operator()(const polyregion::polyast::Expr::MathOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::IntrOp>::operator()(const polyregion::polyast::Expr::IntrOp &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Cast>::operator()(const polyregion::polyast::Expr::Cast &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Alias>::operator()(const polyregion::polyast::Expr::Alias &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Index>::operator()(const polyregion::polyast::Expr::Index &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::RefTo>::operator()(const polyregion::polyast::Expr::RefTo &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Alloc>::operator()(const polyregion::polyast::Expr::Alloc &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Expr::Invoke>::operator()(const polyregion::polyast::Expr::Invoke &x) const noexcept { return x.hash_code(); }
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
std::size_t std::hash<polyregion::polyast::StructMember>::operator()(const polyregion::polyast::StructMember &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::StructDef>::operator()(const polyregion::polyast::StructDef &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Signature>::operator()(const polyregion::polyast::Signature &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::InvokeSignature>::operator()(const polyregion::polyast::InvokeSignature &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionKind::Any>::operator()(const polyregion::polyast::FunctionKind::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionKind::Internal>::operator()(const polyregion::polyast::FunctionKind::Internal &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionKind::Exported>::operator()(const polyregion::polyast::FunctionKind::Exported &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::Any>::operator()(const polyregion::polyast::FunctionAttr::Any &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPRelaxed>::operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPStrict>::operator()(const polyregion::polyast::FunctionAttr::FPStrict &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Arg>::operator()(const polyregion::polyast::Arg &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Function>::operator()(const polyregion::polyast::Function &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Program>::operator()(const polyregion::polyast::Program &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileLayoutMember>::operator()(const polyregion::polyast::CompileLayoutMember &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileLayout>::operator()(const polyregion::polyast::CompileLayout &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileEvent>::operator()(const polyregion::polyast::CompileEvent &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::CompileResult>::operator()(const polyregion::polyast::CompileResult &x) const noexcept { return x.hash_code(); }


