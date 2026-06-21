#include "polyast.h"

namespace polyregion::polyast {

Sym::Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
size_t Sym::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(fqn)>()(fqn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Sym Sym::withFqn(const std::vector<std::string> &v_) const { return Sym(v_); }
POLYREGION_EXPORT bool Sym::operator!=(const Sym &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Sym::operator==(const Sym &rhs) const { return (fqn == rhs.fqn); }

SourcePosition::SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept
    : file(std::move(file)), line(line), col(std::move(col)) {}
size_t SourcePosition::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(file)>()(file) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(line)>()(line) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(col)>()(col) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
SourcePosition SourcePosition::withFile(const std::string &v_) const { return SourcePosition(v_, line, col); }
SourcePosition SourcePosition::withLine(const int32_t &v_) const { return SourcePosition(file, v_, col); }
SourcePosition SourcePosition::withCol(const std::optional<int32_t> &v_) const { return SourcePosition(file, line, v_); }
POLYREGION_EXPORT bool SourcePosition::operator!=(const SourcePosition &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool SourcePosition::operator==(const SourcePosition &rhs) const {
  return (file == rhs.file) && (line == rhs.line) && (col == rhs.col);
}

Named::Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
size_t Named::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(symbol)>()(symbol) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpe)>()(tpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Named Named::withSymbol(const std::string &v_) const { return Named(v_, tpe); }
Named Named::withTpe(const Type::Any &v_) const { return Named(symbol, v_); }
POLYREGION_EXPORT bool Named::operator!=(const Named &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Named::operator==(const Named &rhs) const { return (symbol == rhs.symbol) && (tpe == rhs.tpe); }

TypeKind::Base::Base() = default;
uint32_t TypeKind::Any::id() const { return _v->id(); }
size_t TypeKind::Any::hash_code() const { return _v->hash_code(); }
bool TypeKind::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool TypeKind::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool TypeKind::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

TypeKind::None::None() noexcept : TypeKind::Base() {}
uint32_t TypeKind::None::id() const { return variant_id; };
size_t TypeKind::None::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeKind::None::operator==(const TypeKind::None &rhs) const { return true; }
POLYREGION_EXPORT bool TypeKind::None::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::None::operator<(const TypeKind::None &rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::None::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeKind::None::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<None>(*this)); }
TypeKind::Any TypeKind::None::widen() const { return Any(*this); };

TypeKind::Ref::Ref() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Ref::id() const { return variant_id; };
size_t TypeKind::Ref::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeKind::Ref::operator==(const TypeKind::Ref &rhs) const { return true; }
POLYREGION_EXPORT bool TypeKind::Ref::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Ref::operator<(const TypeKind::Ref &rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Ref::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Ref::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ref>(*this)); }
TypeKind::Any TypeKind::Ref::widen() const { return Any(*this); };

TypeKind::Integral::Integral() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Integral::id() const { return variant_id; };
size_t TypeKind::Integral::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeKind::Integral::operator==(const TypeKind::Integral &rhs) const { return true; }
POLYREGION_EXPORT bool TypeKind::Integral::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Integral::operator<(const TypeKind::Integral &rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Integral::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Integral::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Integral>(*this)); }
TypeKind::Any TypeKind::Integral::widen() const { return Any(*this); };

TypeKind::Fractional::Fractional() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Fractional::id() const { return variant_id; };
size_t TypeKind::Fractional::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeKind::Fractional::operator==(const TypeKind::Fractional &rhs) const { return true; }
POLYREGION_EXPORT bool TypeKind::Fractional::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeKind::Fractional::operator<(const TypeKind::Fractional &rhs) const { return false; }
POLYREGION_EXPORT bool TypeKind::Fractional::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeKind::Fractional::operator TypeKind::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Fractional>(*this)); }
TypeKind::Any TypeKind::Fractional::widen() const { return Any(*this); };

TypeSpace::Base::Base() = default;
uint32_t TypeSpace::Any::id() const { return _v->id(); }
size_t TypeSpace::Any::hash_code() const { return _v->hash_code(); }
bool TypeSpace::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool TypeSpace::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool TypeSpace::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

TypeSpace::Global::Global() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Global::id() const { return variant_id; };
size_t TypeSpace::Global::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeSpace::Global::operator==(const TypeSpace::Global &rhs) const { return true; }
POLYREGION_EXPORT bool TypeSpace::Global::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Global::operator<(const TypeSpace::Global &rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Global::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Global::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Global>(*this)); }
TypeSpace::Any TypeSpace::Global::widen() const { return Any(*this); };

TypeSpace::Local::Local() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Local::id() const { return variant_id; };
size_t TypeSpace::Local::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeSpace::Local::operator==(const TypeSpace::Local &rhs) const { return true; }
POLYREGION_EXPORT bool TypeSpace::Local::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Local::operator<(const TypeSpace::Local &rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Local::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Local::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Local>(*this)); }
TypeSpace::Any TypeSpace::Local::widen() const { return Any(*this); };

TypeSpace::Private::Private() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Private::id() const { return variant_id; };
size_t TypeSpace::Private::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeSpace::Private::operator==(const TypeSpace::Private &rhs) const { return true; }
POLYREGION_EXPORT bool TypeSpace::Private::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Private::operator<(const TypeSpace::Private &rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Private::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Private::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Private>(*this)); }
TypeSpace::Any TypeSpace::Private::widen() const { return Any(*this); };

TypeSpace::Constant::Constant() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Constant::id() const { return variant_id; };
size_t TypeSpace::Constant::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool TypeSpace::Constant::operator==(const TypeSpace::Constant &rhs) const { return true; }
POLYREGION_EXPORT bool TypeSpace::Constant::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool TypeSpace::Constant::operator<(const TypeSpace::Constant &rhs) const { return false; }
POLYREGION_EXPORT bool TypeSpace::Constant::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
TypeSpace::Constant::operator TypeSpace::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Constant>(*this)); }
TypeSpace::Any TypeSpace::Constant::widen() const { return Any(*this); };

Type::Base::Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
uint32_t Type::Any::id() const { return _v->id(); }
size_t Type::Any::hash_code() const { return _v->hash_code(); }
TypeKind::Any Type::Any::kind() const { return _v->kind; }
bool Type::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Type::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Type::Float16::Float16() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float16::id() const { return variant_id; };
size_t Type::Float16::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Type::Float16::operator==(const Type::Float16 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Float16::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::Float32::operator==(const Type::Float32 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Float32::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::Float64::operator==(const Type::Float64 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Float64::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntU8::operator==(const Type::IntU8 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntU8::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntU16::operator==(const Type::IntU16 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntU16::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntU32::operator==(const Type::IntU32 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntU32::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntU64::operator==(const Type::IntU64 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntU64::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntS8::operator==(const Type::IntS8 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntS8::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntS16::operator==(const Type::IntS16 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntS16::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntS32::operator==(const Type::IntS32 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntS32::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::IntS64::operator==(const Type::IntS64 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::IntS64::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::Nothing::operator==(const Type::Nothing &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Nothing::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::Unit0::operator==(const Type::Unit0 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Unit0::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
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
POLYREGION_EXPORT bool Type::Bool1::operator==(const Type::Bool1 &rhs) const { return true; }
POLYREGION_EXPORT bool Type::Bool1::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Type::Bool1::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Bool1>(*this)); }
Type::Any Type::Bool1::widen() const { return Any(*this); };

Type::Struct::Struct(Sym name, std::vector<Type::Any> args) noexcept
    : Type::Base(TypeKind::Ref()), name(std::move(name)), args(std::move(args)) {}
uint32_t Type::Struct::id() const { return variant_id; };
size_t Type::Struct::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Type::Struct Type::Struct::withName(const Sym &v_) const { return Type::Struct(v_, args); }
Type::Struct Type::Struct::withArgs(const std::vector<Type::Any> &v_) const { return Type::Struct(name, v_); }
POLYREGION_EXPORT bool Type::Struct::operator==(const Type::Struct &rhs) const {
  return (this->name == rhs.name) &&
         std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Type::Struct::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Struct &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Struct::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Struct>(*this)); }
Type::Any Type::Struct::widen() const { return Any(*this); };

Type::Ptr::Ptr(Type::Any comp, TypeSpace::Any space) noexcept
    : Type::Base(TypeKind::Ref()), comp(std::move(comp)), space(std::move(space)) {}
uint32_t Type::Ptr::id() const { return variant_id; };
size_t Type::Ptr::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Type::Ptr Type::Ptr::withComp(const Type::Any &v_) const { return Type::Ptr(v_, space); }
Type::Ptr Type::Ptr::withSpace(const TypeSpace::Any &v_) const { return Type::Ptr(comp, v_); }
POLYREGION_EXPORT bool Type::Ptr::operator==(const Type::Ptr &rhs) const { return (this->comp == rhs.comp) && (this->space == rhs.space); }
POLYREGION_EXPORT bool Type::Ptr::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Ptr &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Ptr::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ptr>(*this)); }
Type::Any Type::Ptr::widen() const { return Any(*this); };

Type::Arr::Arr(Type::Any comp, int32_t length, TypeSpace::Any space) noexcept
    : Type::Base(TypeKind::Ref()), comp(std::move(comp)), length(length), space(std::move(space)) {}
uint32_t Type::Arr::id() const { return variant_id; };
size_t Type::Arr::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(length)>()(length) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Type::Arr Type::Arr::withComp(const Type::Any &v_) const { return Type::Arr(v_, length, space); }
Type::Arr Type::Arr::withLength(const int32_t &v_) const { return Type::Arr(comp, v_, space); }
Type::Arr Type::Arr::withSpace(const TypeSpace::Any &v_) const { return Type::Arr(comp, length, v_); }
POLYREGION_EXPORT bool Type::Arr::operator==(const Type::Arr &rhs) const {
  return (this->comp == rhs.comp) && (this->length == rhs.length) && (this->space == rhs.space);
}
POLYREGION_EXPORT bool Type::Arr::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Arr &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Arr::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Arr>(*this)); }
Type::Any Type::Arr::widen() const { return Any(*this); };

Type::Var::Var(std::string name) noexcept : Type::Base(TypeKind::None()), name(std::move(name)) {}
uint32_t Type::Var::id() const { return variant_id; };
size_t Type::Var::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Type::Var Type::Var::withName(const std::string &v_) const { return Type::Var(v_); }
POLYREGION_EXPORT bool Type::Var::operator==(const Type::Var &rhs) const { return (this->name == rhs.name); }
POLYREGION_EXPORT bool Type::Var::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Var &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Var::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Var>(*this)); }
Type::Any Type::Var::widen() const { return Any(*this); };

Type::Exec::Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept
    : Type::Base(TypeKind::None()), tpeVars(std::move(tpeVars)), args(std::move(args)), rtn(std::move(rtn)) {}
uint32_t Type::Exec::id() const { return variant_id; };
size_t Type::Exec::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Type::Exec Type::Exec::withTpeVars(const std::vector<std::string> &v_) const { return Type::Exec(v_, args, rtn); }
Type::Exec Type::Exec::withArgs(const std::vector<Type::Any> &v_) const { return Type::Exec(tpeVars, v_, rtn); }
Type::Exec Type::Exec::withRtn(const Type::Any &v_) const { return Type::Exec(tpeVars, args, v_); }
POLYREGION_EXPORT bool Type::Exec::operator==(const Type::Exec &rhs) const {
  return (this->tpeVars == rhs.tpeVars) &&
         std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Type::Exec::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Type::Exec &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Type::Exec::operator Type::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Exec>(*this)); }
Type::Any Type::Exec::widen() const { return Any(*this); };

PathStep::Base::Base() = default;
uint32_t PathStep::Any::id() const { return _v->id(); }
size_t PathStep::Any::hash_code() const { return _v->hash_code(); }
bool PathStep::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool PathStep::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool PathStep::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

PathStep::Field::Field(std::string name) noexcept : PathStep::Base(), name(std::move(name)) {}
uint32_t PathStep::Field::id() const { return variant_id; };
size_t PathStep::Field::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PathStep::Field PathStep::Field::withName(const std::string &v_) const { return PathStep::Field(v_); }
POLYREGION_EXPORT bool PathStep::Field::operator==(const PathStep::Field &rhs) const { return (this->name == rhs.name); }
POLYREGION_EXPORT bool PathStep::Field::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const PathStep::Field &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool PathStep::Field::operator<(const PathStep::Field &rhs) const { return false; }
POLYREGION_EXPORT bool PathStep::Field::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PathStep::Field::operator PathStep::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Field>(*this)); }
PathStep::Any PathStep::Field::widen() const { return Any(*this); };

PathStep::Deref::Deref() noexcept : PathStep::Base() {}
uint32_t PathStep::Deref::id() const { return variant_id; };
size_t PathStep::Deref::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool PathStep::Deref::operator==(const PathStep::Deref &rhs) const { return true; }
POLYREGION_EXPORT bool PathStep::Deref::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool PathStep::Deref::operator<(const PathStep::Deref &rhs) const { return false; }
POLYREGION_EXPORT bool PathStep::Deref::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PathStep::Deref::operator PathStep::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Deref>(*this)); }
PathStep::Any PathStep::Deref::widen() const { return Any(*this); };

PathStep::Index::Index(int32_t idx) noexcept : PathStep::Base(), idx(idx) {}
uint32_t PathStep::Index::id() const { return variant_id; };
size_t PathStep::Index::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PathStep::Index PathStep::Index::withIdx(const int32_t &v_) const { return PathStep::Index(v_); }
POLYREGION_EXPORT bool PathStep::Index::operator==(const PathStep::Index &rhs) const { return (this->idx == rhs.idx); }
POLYREGION_EXPORT bool PathStep::Index::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const PathStep::Index &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool PathStep::Index::operator<(const PathStep::Index &rhs) const { return false; }
POLYREGION_EXPORT bool PathStep::Index::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PathStep::Index::operator PathStep::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Index>(*this)); }
PathStep::Any PathStep::Index::widen() const { return Any(*this); };

PathStep::IndexDyn::IndexDyn(Term::Any idx) noexcept : PathStep::Base(), idx(std::move(idx)) {}
uint32_t PathStep::IndexDyn::id() const { return variant_id; };
size_t PathStep::IndexDyn::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PathStep::IndexDyn PathStep::IndexDyn::withIdx(const Term::Any &v_) const { return PathStep::IndexDyn(v_); }
POLYREGION_EXPORT bool PathStep::IndexDyn::operator==(const PathStep::IndexDyn &rhs) const { return (this->idx == rhs.idx); }
POLYREGION_EXPORT bool PathStep::IndexDyn::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const PathStep::IndexDyn &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool PathStep::IndexDyn::operator<(const PathStep::IndexDyn &rhs) const { return false; }
POLYREGION_EXPORT bool PathStep::IndexDyn::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PathStep::IndexDyn::operator PathStep::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IndexDyn>(*this)); }
PathStep::Any PathStep::IndexDyn::widen() const { return Any(*this); };

Region::Base::Base() = default;
uint32_t Region::Any::id() const { return _v->id(); }
size_t Region::Any::hash_code() const { return _v->hash_code(); }
bool Region::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Region::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool Region::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

Region::Rooted::Rooted(Named root) noexcept : Region::Base(), root(std::move(root)) {}
uint32_t Region::Rooted::id() const { return variant_id; };
size_t Region::Rooted::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(root)>()(root) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Region::Rooted Region::Rooted::withRoot(const Named &v_) const { return Region::Rooted(v_); }
POLYREGION_EXPORT bool Region::Rooted::operator==(const Region::Rooted &rhs) const { return (this->root == rhs.root); }
POLYREGION_EXPORT bool Region::Rooted::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Region::Rooted &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Region::Rooted::operator<(const Region::Rooted &rhs) const { return false; }
POLYREGION_EXPORT bool Region::Rooted::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Region::Rooted::operator Region::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Rooted>(*this)); }
Region::Any Region::Rooted::widen() const { return Any(*this); };

Region::Opaque::Opaque() noexcept : Region::Base() {}
uint32_t Region::Opaque::id() const { return variant_id; };
size_t Region::Opaque::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Region::Opaque::operator==(const Region::Opaque &rhs) const { return true; }
POLYREGION_EXPORT bool Region::Opaque::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool Region::Opaque::operator<(const Region::Opaque &rhs) const { return false; }
POLYREGION_EXPORT bool Region::Opaque::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Region::Opaque::operator Region::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Opaque>(*this)); }
Region::Any Region::Opaque::widen() const { return Any(*this); };

Term::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
uint32_t Term::Any::id() const { return _v->id(); }
size_t Term::Any::hash_code() const { return _v->hash_code(); }
Type::Any Term::Any::tpe() const { return _v->tpe; }
bool Term::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Term::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Term::Float16Const::Float16Const(float value) noexcept : Term::Base(Type::Float16()), value(value) {}
uint32_t Term::Float16Const::id() const { return variant_id; };
size_t Term::Float16Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Float16Const Term::Float16Const::withValue(const float &v_) const { return Term::Float16Const(v_); }
POLYREGION_EXPORT bool Term::Float16Const::operator==(const Term::Float16Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::Float16Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Float16Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Float16Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float16Const>(*this)); }
Term::Any Term::Float16Const::widen() const { return Any(*this); };

Term::Float32Const::Float32Const(float value) noexcept : Term::Base(Type::Float32()), value(value) {}
uint32_t Term::Float32Const::id() const { return variant_id; };
size_t Term::Float32Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Float32Const Term::Float32Const::withValue(const float &v_) const { return Term::Float32Const(v_); }
POLYREGION_EXPORT bool Term::Float32Const::operator==(const Term::Float32Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::Float32Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Float32Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Float32Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float32Const>(*this)); }
Term::Any Term::Float32Const::widen() const { return Any(*this); };

Term::Float64Const::Float64Const(double value) noexcept : Term::Base(Type::Float64()), value(value) {}
uint32_t Term::Float64Const::id() const { return variant_id; };
size_t Term::Float64Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Float64Const Term::Float64Const::withValue(const double &v_) const { return Term::Float64Const(v_); }
POLYREGION_EXPORT bool Term::Float64Const::operator==(const Term::Float64Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::Float64Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Float64Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Float64Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Float64Const>(*this)); }
Term::Any Term::Float64Const::widen() const { return Any(*this); };

Term::IntU8Const::IntU8Const(int8_t value) noexcept : Term::Base(Type::IntU8()), value(value) {}
uint32_t Term::IntU8Const::id() const { return variant_id; };
size_t Term::IntU8Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntU8Const Term::IntU8Const::withValue(const int8_t &v_) const { return Term::IntU8Const(v_); }
POLYREGION_EXPORT bool Term::IntU8Const::operator==(const Term::IntU8Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntU8Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntU8Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntU8Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU8Const>(*this)); }
Term::Any Term::IntU8Const::widen() const { return Any(*this); };

Term::IntU16Const::IntU16Const(uint16_t value) noexcept : Term::Base(Type::IntU16()), value(value) {}
uint32_t Term::IntU16Const::id() const { return variant_id; };
size_t Term::IntU16Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntU16Const Term::IntU16Const::withValue(const uint16_t &v_) const { return Term::IntU16Const(v_); }
POLYREGION_EXPORT bool Term::IntU16Const::operator==(const Term::IntU16Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntU16Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntU16Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntU16Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU16Const>(*this)); }
Term::Any Term::IntU16Const::widen() const { return Any(*this); };

Term::IntU32Const::IntU32Const(int32_t value) noexcept : Term::Base(Type::IntU32()), value(value) {}
uint32_t Term::IntU32Const::id() const { return variant_id; };
size_t Term::IntU32Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntU32Const Term::IntU32Const::withValue(const int32_t &v_) const { return Term::IntU32Const(v_); }
POLYREGION_EXPORT bool Term::IntU32Const::operator==(const Term::IntU32Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntU32Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntU32Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntU32Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU32Const>(*this)); }
Term::Any Term::IntU32Const::widen() const { return Any(*this); };

Term::IntU64Const::IntU64Const(int64_t value) noexcept : Term::Base(Type::IntU64()), value(value) {}
uint32_t Term::IntU64Const::id() const { return variant_id; };
size_t Term::IntU64Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntU64Const Term::IntU64Const::withValue(const int64_t &v_) const { return Term::IntU64Const(v_); }
POLYREGION_EXPORT bool Term::IntU64Const::operator==(const Term::IntU64Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntU64Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntU64Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntU64Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntU64Const>(*this)); }
Term::Any Term::IntU64Const::widen() const { return Any(*this); };

Term::IntS8Const::IntS8Const(int8_t value) noexcept : Term::Base(Type::IntS8()), value(value) {}
uint32_t Term::IntS8Const::id() const { return variant_id; };
size_t Term::IntS8Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntS8Const Term::IntS8Const::withValue(const int8_t &v_) const { return Term::IntS8Const(v_); }
POLYREGION_EXPORT bool Term::IntS8Const::operator==(const Term::IntS8Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntS8Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntS8Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntS8Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS8Const>(*this)); }
Term::Any Term::IntS8Const::widen() const { return Any(*this); };

Term::IntS16Const::IntS16Const(int16_t value) noexcept : Term::Base(Type::IntS16()), value(value) {}
uint32_t Term::IntS16Const::id() const { return variant_id; };
size_t Term::IntS16Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntS16Const Term::IntS16Const::withValue(const int16_t &v_) const { return Term::IntS16Const(v_); }
POLYREGION_EXPORT bool Term::IntS16Const::operator==(const Term::IntS16Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntS16Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntS16Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntS16Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS16Const>(*this)); }
Term::Any Term::IntS16Const::widen() const { return Any(*this); };

Term::IntS32Const::IntS32Const(int32_t value) noexcept : Term::Base(Type::IntS32()), value(value) {}
uint32_t Term::IntS32Const::id() const { return variant_id; };
size_t Term::IntS32Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntS32Const Term::IntS32Const::withValue(const int32_t &v_) const { return Term::IntS32Const(v_); }
POLYREGION_EXPORT bool Term::IntS32Const::operator==(const Term::IntS32Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntS32Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntS32Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntS32Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS32Const>(*this)); }
Term::Any Term::IntS32Const::widen() const { return Any(*this); };

Term::IntS64Const::IntS64Const(int64_t value) noexcept : Term::Base(Type::IntS64()), value(value) {}
uint32_t Term::IntS64Const::id() const { return variant_id; };
size_t Term::IntS64Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::IntS64Const Term::IntS64Const::withValue(const int64_t &v_) const { return Term::IntS64Const(v_); }
POLYREGION_EXPORT bool Term::IntS64Const::operator==(const Term::IntS64Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::IntS64Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::IntS64Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::IntS64Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntS64Const>(*this)); }
Term::Any Term::IntS64Const::widen() const { return Any(*this); };

Term::Unit0Const::Unit0Const() noexcept : Term::Base(Type::Unit0()) {}
uint32_t Term::Unit0Const::id() const { return variant_id; };
size_t Term::Unit0Const::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Term::Unit0Const::operator==(const Term::Unit0Const &rhs) const { return true; }
POLYREGION_EXPORT bool Term::Unit0Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Term::Unit0Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Unit0Const>(*this)); }
Term::Any Term::Unit0Const::widen() const { return Any(*this); };

Term::Bool1Const::Bool1Const(bool value) noexcept : Term::Base(Type::Bool1()), value(value) {}
uint32_t Term::Bool1Const::id() const { return variant_id; };
size_t Term::Bool1Const::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Bool1Const Term::Bool1Const::withValue(const bool &v_) const { return Term::Bool1Const(v_); }
POLYREGION_EXPORT bool Term::Bool1Const::operator==(const Term::Bool1Const &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Term::Bool1Const::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Bool1Const &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Bool1Const::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Bool1Const>(*this)); }
Term::Any Term::Bool1Const::widen() const { return Any(*this); };

Term::NullPtrConst::NullPtrConst(Type::Any comp, TypeSpace::Any space, Region::Any region) noexcept
    : Term::Base(Type::Ptr(comp, space)), comp(std::move(comp)), space(std::move(space)), region(std::move(region)) {}
uint32_t Term::NullPtrConst::id() const { return variant_id; };
size_t Term::NullPtrConst::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(region)>()(region) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::NullPtrConst Term::NullPtrConst::withComp(const Type::Any &v_) const { return Term::NullPtrConst(v_, space, region); }
Term::NullPtrConst Term::NullPtrConst::withSpace(const TypeSpace::Any &v_) const { return Term::NullPtrConst(comp, v_, region); }
Term::NullPtrConst Term::NullPtrConst::withRegion(const Region::Any &v_) const { return Term::NullPtrConst(comp, space, v_); }
POLYREGION_EXPORT bool Term::NullPtrConst::operator==(const Term::NullPtrConst &rhs) const {
  return (this->comp == rhs.comp) && (this->space == rhs.space) && (this->region == rhs.region);
}
POLYREGION_EXPORT bool Term::NullPtrConst::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::NullPtrConst &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::NullPtrConst::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<NullPtrConst>(*this)); }
Term::Any Term::NullPtrConst::widen() const { return Any(*this); };

Term::Poison::Poison(Type::Any t) noexcept : Term::Base(t), t(std::move(t)) {}
uint32_t Term::Poison::id() const { return variant_id; };
size_t Term::Poison::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(t)>()(t) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Poison Term::Poison::withT(const Type::Any &v_) const { return Term::Poison(v_); }
POLYREGION_EXPORT bool Term::Poison::operator==(const Term::Poison &rhs) const { return (this->t == rhs.t); }
POLYREGION_EXPORT bool Term::Poison::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Poison &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Poison::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Poison>(*this)); }
Term::Any Term::Poison::widen() const { return Any(*this); };

Term::Select::Select(Named root, std::vector<PathStep::Any> steps, Type::Any tpe) noexcept
    : Term::Base(tpe), root(std::move(root)), steps(std::move(steps)), tpe(std::move(tpe)) {}
uint32_t Term::Select::id() const { return variant_id; };
size_t Term::Select::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(root)>()(root) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(steps)>()(steps) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpe)>()(tpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Term::Select Term::Select::withRoot(const Named &v_) const { return Term::Select(v_, steps, tpe); }
Term::Select Term::Select::withSteps(const std::vector<PathStep::Any> &v_) const { return Term::Select(root, v_, tpe); }
Term::Select Term::Select::withTpe(const Type::Any &v_) const { return Term::Select(root, steps, v_); }
POLYREGION_EXPORT bool Term::Select::operator==(const Term::Select &rhs) const {
  return (this->root == rhs.root) &&
         std::equal(this->steps.begin(), this->steps.end(), rhs.steps.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (this->tpe == rhs.tpe);
}
POLYREGION_EXPORT bool Term::Select::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Term::Select &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Term::Select::operator Term::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Select>(*this)); }
Term::Any Term::Select::widen() const { return Any(*this); };

Expr::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
uint32_t Expr::Any::id() const { return _v->id(); }
size_t Expr::Any::hash_code() const { return _v->hash_code(); }
Type::Any Expr::Any::tpe() const { return _v->tpe; }
bool Expr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Expr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Expr::Alias::Alias(Term::Any ref) noexcept : Expr::Base(ref.tpe()), ref(std::move(ref)) {}
uint32_t Expr::Alias::id() const { return variant_id; };
size_t Expr::Alias::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(ref)>()(ref) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::Alias Expr::Alias::withRef(const Term::Any &v_) const { return Expr::Alias(v_); }
POLYREGION_EXPORT bool Expr::Alias::operator==(const Expr::Alias &rhs) const { return (this->ref == rhs.ref); }
POLYREGION_EXPORT bool Expr::Alias::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Alias &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Alias::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Alias>(*this)); }
Expr::Any Expr::Alias::widen() const { return Any(*this); };

Expr::SpecOp::SpecOp(Spec::Any op) noexcept : Expr::Base(op.tpe()), op(std::move(op)) {}
uint32_t Expr::SpecOp::id() const { return variant_id; };
size_t Expr::SpecOp::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(op)>()(op) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::SpecOp Expr::SpecOp::withOp(const Spec::Any &v_) const { return Expr::SpecOp(v_); }
POLYREGION_EXPORT bool Expr::SpecOp::operator==(const Expr::SpecOp &rhs) const { return (this->op == rhs.op); }
POLYREGION_EXPORT bool Expr::SpecOp::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::SpecOp &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
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
Expr::MathOp Expr::MathOp::withOp(const Math::Any &v_) const { return Expr::MathOp(v_); }
POLYREGION_EXPORT bool Expr::MathOp::operator==(const Expr::MathOp &rhs) const { return (this->op == rhs.op); }
POLYREGION_EXPORT bool Expr::MathOp::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::MathOp &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
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
Expr::IntrOp Expr::IntrOp::withOp(const Intr::Any &v_) const { return Expr::IntrOp(v_); }
POLYREGION_EXPORT bool Expr::IntrOp::operator==(const Expr::IntrOp &rhs) const { return (this->op == rhs.op); }
POLYREGION_EXPORT bool Expr::IntrOp::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::IntrOp &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::IntrOp::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<IntrOp>(*this)); }
Expr::Any Expr::IntrOp::widen() const { return Any(*this); };

Expr::Cast::Cast(Term::Any from, Type::Any as) noexcept : Expr::Base(as), from(std::move(from)), as(std::move(as)) {}
uint32_t Expr::Cast::id() const { return variant_id; };
size_t Expr::Cast::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(from)>()(from) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(as)>()(as) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::Cast Expr::Cast::withFrom(const Term::Any &v_) const { return Expr::Cast(v_, as); }
Expr::Cast Expr::Cast::withAs(const Type::Any &v_) const { return Expr::Cast(from, v_); }
POLYREGION_EXPORT bool Expr::Cast::operator==(const Expr::Cast &rhs) const { return (this->from == rhs.from) && (this->as == rhs.as); }
POLYREGION_EXPORT bool Expr::Cast::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Cast &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Cast::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cast>(*this)); }
Expr::Any Expr::Cast::widen() const { return Any(*this); };

Expr::Index::Index(Term::Any lhs, Term::Any idx, Type::Any comp) noexcept
    : Expr::Base(comp), lhs(std::move(lhs)), idx(std::move(idx)), comp(std::move(comp)) {}
uint32_t Expr::Index::id() const { return variant_id; };
size_t Expr::Index::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::Index Expr::Index::withLhs(const Term::Any &v_) const { return Expr::Index(v_, idx, comp); }
Expr::Index Expr::Index::withIdx(const Term::Any &v_) const { return Expr::Index(lhs, v_, comp); }
Expr::Index Expr::Index::withComp(const Type::Any &v_) const { return Expr::Index(lhs, idx, v_); }
POLYREGION_EXPORT bool Expr::Index::operator==(const Expr::Index &rhs) const {
  return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->comp == rhs.comp);
}
POLYREGION_EXPORT bool Expr::Index::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Index &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Index::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Index>(*this)); }
Expr::Any Expr::Index::widen() const { return Any(*this); };

Expr::RefTo::RefTo(Term::Any lhs, std::optional<Term::Any> idx, Type::Any comp, TypeSpace::Any space, Region::Any region) noexcept
    : Expr::Base(Type::Ptr(comp, space)), lhs(std::move(lhs)), idx(std::move(idx)), comp(std::move(comp)), space(std::move(space)),
      region(std::move(region)) {}
uint32_t Expr::RefTo::id() const { return variant_id; };
size_t Expr::RefTo::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(region)>()(region) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::RefTo Expr::RefTo::withLhs(const Term::Any &v_) const { return Expr::RefTo(v_, idx, comp, space, region); }
Expr::RefTo Expr::RefTo::withIdx(const std::optional<Term::Any> &v_) const { return Expr::RefTo(lhs, v_, comp, space, region); }
Expr::RefTo Expr::RefTo::withComp(const Type::Any &v_) const { return Expr::RefTo(lhs, idx, v_, space, region); }
Expr::RefTo Expr::RefTo::withSpace(const TypeSpace::Any &v_) const { return Expr::RefTo(lhs, idx, comp, v_, region); }
Expr::RefTo Expr::RefTo::withRegion(const Region::Any &v_) const { return Expr::RefTo(lhs, idx, comp, space, v_); }
POLYREGION_EXPORT bool Expr::RefTo::operator==(const Expr::RefTo &rhs) const {
  return (this->lhs == rhs.lhs) && ((!this->idx && !rhs.idx) || (this->idx && rhs.idx && *this->idx == *rhs.idx)) &&
         (this->comp == rhs.comp) && (this->space == rhs.space) && (this->region == rhs.region);
}
POLYREGION_EXPORT bool Expr::RefTo::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::RefTo &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::RefTo::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<RefTo>(*this)); }
Expr::Any Expr::RefTo::widen() const { return Any(*this); };

Expr::Alloc::Alloc(Type::Any comp, Term::Any size, TypeSpace::Any space, Region::Any region) noexcept
    : Expr::Base(Type::Ptr(comp, space)), comp(std::move(comp)), size(std::move(size)), space(std::move(space)), region(std::move(region)) {
}
uint32_t Expr::Alloc::id() const { return variant_id; };
size_t Expr::Alloc::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(comp)>()(comp) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(size)>()(size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(space)>()(space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(region)>()(region) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::Alloc Expr::Alloc::withComp(const Type::Any &v_) const { return Expr::Alloc(v_, size, space, region); }
Expr::Alloc Expr::Alloc::withSize(const Term::Any &v_) const { return Expr::Alloc(comp, v_, space, region); }
Expr::Alloc Expr::Alloc::withSpace(const TypeSpace::Any &v_) const { return Expr::Alloc(comp, size, v_, region); }
Expr::Alloc Expr::Alloc::withRegion(const Region::Any &v_) const { return Expr::Alloc(comp, size, space, v_); }
POLYREGION_EXPORT bool Expr::Alloc::operator==(const Expr::Alloc &rhs) const {
  return (this->comp == rhs.comp) && (this->size == rhs.size) && (this->space == rhs.space) && (this->region == rhs.region);
}
POLYREGION_EXPORT bool Expr::Alloc::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Alloc &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Alloc::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Alloc>(*this)); }
Expr::Any Expr::Alloc::widen() const { return Any(*this); };

Expr::Invoke::Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args,
                     Type::Any rtn) noexcept
    : Expr::Base(rtn), name(std::move(name)), tpeArgs(std::move(tpeArgs)), receiver(std::move(receiver)), args(std::move(args)),
      rtn(std::move(rtn)) {}
uint32_t Expr::Invoke::id() const { return variant_id; };
size_t Expr::Invoke::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeArgs)>()(tpeArgs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::Invoke Expr::Invoke::withName(const Sym &v_) const { return Expr::Invoke(v_, tpeArgs, receiver, args, rtn); }
Expr::Invoke Expr::Invoke::withTpeArgs(const std::vector<Type::Any> &v_) const { return Expr::Invoke(name, v_, receiver, args, rtn); }
Expr::Invoke Expr::Invoke::withReceiver(const std::optional<Term::Any> &v_) const { return Expr::Invoke(name, tpeArgs, v_, args, rtn); }
Expr::Invoke Expr::Invoke::withArgs(const std::vector<Term::Any> &v_) const { return Expr::Invoke(name, tpeArgs, receiver, v_, rtn); }
Expr::Invoke Expr::Invoke::withRtn(const Type::Any &v_) const { return Expr::Invoke(name, tpeArgs, receiver, args, v_); }
POLYREGION_EXPORT bool Expr::Invoke::operator==(const Expr::Invoke &rhs) const {
  return (this->name == rhs.name) &&
         std::equal(this->tpeArgs.begin(), this->tpeArgs.end(), rhs.tpeArgs.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         ((!this->receiver && !rhs.receiver) || (this->receiver && rhs.receiver && *this->receiver == *rhs.receiver)) &&
         std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Expr::Invoke::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::Invoke &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::Invoke::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Invoke>(*this)); }
Expr::Any Expr::Invoke::widen() const { return Any(*this); };

Expr::ForeignCall::ForeignCall(std::string name, std::vector<Term::Any> args, Type::Any rtn) noexcept
    : Expr::Base(rtn), name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)) {}
uint32_t Expr::ForeignCall::id() const { return variant_id; };
size_t Expr::ForeignCall::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::ForeignCall Expr::ForeignCall::withName(const std::string &v_) const { return Expr::ForeignCall(v_, args, rtn); }
Expr::ForeignCall Expr::ForeignCall::withArgs(const std::vector<Term::Any> &v_) const { return Expr::ForeignCall(name, v_, rtn); }
Expr::ForeignCall Expr::ForeignCall::withRtn(const Type::Any &v_) const { return Expr::ForeignCall(name, args, v_); }
POLYREGION_EXPORT bool Expr::ForeignCall::operator==(const Expr::ForeignCall &rhs) const {
  return (this->name == rhs.name) &&
         std::equal(this->args.begin(), this->args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Expr::ForeignCall::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::ForeignCall &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::ForeignCall::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<ForeignCall>(*this)); }
Expr::Any Expr::ForeignCall::widen() const { return Any(*this); };

Expr::OffsetOf::OffsetOf(Type::Any structTpe, std::string field) noexcept
    : Expr::Base(Type::IntU64()), structTpe(std::move(structTpe)), field(std::move(field)) {}
uint32_t Expr::OffsetOf::id() const { return variant_id; };
size_t Expr::OffsetOf::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(structTpe)>()(structTpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(field)>()(field) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::OffsetOf Expr::OffsetOf::withStructTpe(const Type::Any &v_) const { return Expr::OffsetOf(v_, field); }
Expr::OffsetOf Expr::OffsetOf::withField(const std::string &v_) const { return Expr::OffsetOf(structTpe, v_); }
POLYREGION_EXPORT bool Expr::OffsetOf::operator==(const Expr::OffsetOf &rhs) const {
  return (this->structTpe == rhs.structTpe) && (this->field == rhs.field);
}
POLYREGION_EXPORT bool Expr::OffsetOf::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::OffsetOf &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::OffsetOf::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<OffsetOf>(*this)); }
Expr::Any Expr::OffsetOf::widen() const { return Any(*this); };

Expr::SizeOf::SizeOf(Type::Any forTpe) noexcept : Expr::Base(Type::IntU64()), forTpe(std::move(forTpe)) {}
uint32_t Expr::SizeOf::id() const { return variant_id; };
size_t Expr::SizeOf::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(forTpe)>()(forTpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Expr::SizeOf Expr::SizeOf::withForTpe(const Type::Any &v_) const { return Expr::SizeOf(v_); }
POLYREGION_EXPORT bool Expr::SizeOf::operator==(const Expr::SizeOf &rhs) const { return (this->forTpe == rhs.forTpe); }
POLYREGION_EXPORT bool Expr::SizeOf::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Expr::SizeOf &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Expr::SizeOf::operator Expr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<SizeOf>(*this)); }
Expr::Any Expr::SizeOf::widen() const { return Any(*this); };

Overload::Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept : args(std::move(args)), rtn(std::move(rtn)) {}
size_t Overload::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Overload Overload::withArgs(const std::vector<Type::Any> &v_) const { return Overload(v_, rtn); }
Overload Overload::withRtn(const Type::Any &v_) const { return Overload(args, v_); }
POLYREGION_EXPORT bool Overload::operator!=(const Overload &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Overload::operator==(const Overload &rhs) const {
  return std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

Spec::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept
    : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Spec::Any::id() const { return _v->id(); }
size_t Spec::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Spec::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Spec::Any::terms() const { return _v->terms; }
Type::Any Spec::Any::tpe() const { return _v->tpe; }
bool Spec::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Spec::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Spec::Assert::Assert() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Nothing()) {}
uint32_t Spec::Assert::id() const { return variant_id; };
size_t Spec::Assert::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::Assert::operator==(const Spec::Assert &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::Assert::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::Assert::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Assert>(*this)); }
Spec::Any Spec::Assert::widen() const { return Any(*this); };

Spec::GpuBarrierGlobal::GpuBarrierGlobal() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierGlobal::id() const { return variant_id; };
size_t Spec::GpuBarrierGlobal::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuBarrierGlobal::operator==(const Spec::GpuBarrierGlobal &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuBarrierGlobal::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierGlobal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierGlobal>(*this)); }
Spec::Any Spec::GpuBarrierGlobal::widen() const { return Any(*this); };

Spec::GpuBarrierLocal::GpuBarrierLocal() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierLocal::id() const { return variant_id; };
size_t Spec::GpuBarrierLocal::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuBarrierLocal::operator==(const Spec::GpuBarrierLocal &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuBarrierLocal::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierLocal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierLocal>(*this)); }
Spec::Any Spec::GpuBarrierLocal::widen() const { return Any(*this); };

Spec::GpuBarrierAll::GpuBarrierAll() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierAll::id() const { return variant_id; };
size_t Spec::GpuBarrierAll::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuBarrierAll::operator==(const Spec::GpuBarrierAll &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuBarrierAll::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuBarrierAll::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuBarrierAll>(*this)); }
Spec::Any Spec::GpuBarrierAll::widen() const { return Any(*this); };

Spec::GpuFenceGlobal::GpuFenceGlobal() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceGlobal::id() const { return variant_id; };
size_t Spec::GpuFenceGlobal::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuFenceGlobal::operator==(const Spec::GpuFenceGlobal &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuFenceGlobal::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceGlobal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceGlobal>(*this)); }
Spec::Any Spec::GpuFenceGlobal::widen() const { return Any(*this); };

Spec::GpuFenceLocal::GpuFenceLocal() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceLocal::id() const { return variant_id; };
size_t Spec::GpuFenceLocal::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuFenceLocal::operator==(const Spec::GpuFenceLocal &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuFenceLocal::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceLocal::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceLocal>(*this)); }
Spec::Any Spec::GpuFenceLocal::widen() const { return Any(*this); };

Spec::GpuFenceAll::GpuFenceAll() noexcept : Spec::Base({Overload({}, Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceAll::id() const { return variant_id; };
size_t Spec::GpuFenceAll::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Spec::GpuFenceAll::operator==(const Spec::GpuFenceAll &rhs) const { return true; }
POLYREGION_EXPORT bool Spec::GpuFenceAll::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
Spec::GpuFenceAll::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuFenceAll>(*this)); }
Spec::Any Spec::GpuFenceAll::widen() const { return Any(*this); };

Spec::GpuGlobalIdx::GpuGlobalIdx(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalIdx::id() const { return variant_id; };
size_t Spec::GpuGlobalIdx::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuGlobalIdx Spec::GpuGlobalIdx::withDim(const Term::Any &v_) const { return Spec::GpuGlobalIdx(v_); }
POLYREGION_EXPORT bool Spec::GpuGlobalIdx::operator==(const Spec::GpuGlobalIdx &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuGlobalIdx::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGlobalIdx &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGlobalIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGlobalIdx>(*this)); }
Spec::Any Spec::GpuGlobalIdx::widen() const { return Any(*this); };

Spec::GpuGlobalSize::GpuGlobalSize(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalSize::id() const { return variant_id; };
size_t Spec::GpuGlobalSize::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuGlobalSize Spec::GpuGlobalSize::withDim(const Term::Any &v_) const { return Spec::GpuGlobalSize(v_); }
POLYREGION_EXPORT bool Spec::GpuGlobalSize::operator==(const Spec::GpuGlobalSize &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuGlobalSize::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGlobalSize &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGlobalSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGlobalSize>(*this)); }
Spec::Any Spec::GpuGlobalSize::widen() const { return Any(*this); };

Spec::GpuGroupIdx::GpuGroupIdx(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupIdx::id() const { return variant_id; };
size_t Spec::GpuGroupIdx::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuGroupIdx Spec::GpuGroupIdx::withDim(const Term::Any &v_) const { return Spec::GpuGroupIdx(v_); }
POLYREGION_EXPORT bool Spec::GpuGroupIdx::operator==(const Spec::GpuGroupIdx &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuGroupIdx::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGroupIdx &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGroupIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGroupIdx>(*this)); }
Spec::Any Spec::GpuGroupIdx::widen() const { return Any(*this); };

Spec::GpuGroupSize::GpuGroupSize(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupSize::id() const { return variant_id; };
size_t Spec::GpuGroupSize::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuGroupSize Spec::GpuGroupSize::withDim(const Term::Any &v_) const { return Spec::GpuGroupSize(v_); }
POLYREGION_EXPORT bool Spec::GpuGroupSize::operator==(const Spec::GpuGroupSize &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuGroupSize::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuGroupSize &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuGroupSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuGroupSize>(*this)); }
Spec::Any Spec::GpuGroupSize::widen() const { return Any(*this); };

Spec::GpuLocalIdx::GpuLocalIdx(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalIdx::id() const { return variant_id; };
size_t Spec::GpuLocalIdx::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuLocalIdx Spec::GpuLocalIdx::withDim(const Term::Any &v_) const { return Spec::GpuLocalIdx(v_); }
POLYREGION_EXPORT bool Spec::GpuLocalIdx::operator==(const Spec::GpuLocalIdx &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuLocalIdx::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuLocalIdx &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuLocalIdx::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuLocalIdx>(*this)); }
Spec::Any Spec::GpuLocalIdx::widen() const { return Any(*this); };

Spec::GpuLocalSize::GpuLocalSize(Term::Any dim) noexcept
    : Spec::Base({Overload({Type::IntU32()}, Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalSize::id() const { return variant_id; };
size_t Spec::GpuLocalSize::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(dim)>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Spec::GpuLocalSize Spec::GpuLocalSize::withDim(const Term::Any &v_) const { return Spec::GpuLocalSize(v_); }
POLYREGION_EXPORT bool Spec::GpuLocalSize::operator==(const Spec::GpuLocalSize &rhs) const { return (this->dim == rhs.dim); }
POLYREGION_EXPORT bool Spec::GpuLocalSize::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Spec::GpuLocalSize &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Spec::GpuLocalSize::operator Spec::Any() const { return std::static_pointer_cast<Base>(std::make_shared<GpuLocalSize>(*this)); }
Spec::Any Spec::GpuLocalSize::widen() const { return Any(*this); };

Intr::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept
    : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Intr::Any::id() const { return _v->id(); }
size_t Intr::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Intr::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Intr::Any::terms() const { return _v->terms; }
Type::Any Intr::Any::tpe() const { return _v->tpe; }
bool Intr::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Intr::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Intr::BNot::BNot(Term::Any x, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64()}, Type::IntS64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::BNot::id() const { return variant_id; };
size_t Intr::BNot::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BNot Intr::BNot::withX(const Term::Any &v_) const { return Intr::BNot(v_, rtn); }
Intr::BNot Intr::BNot::withRtn(const Type::Any &v_) const { return Intr::BNot(x, v_); }
POLYREGION_EXPORT bool Intr::BNot::operator==(const Intr::BNot &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Intr::BNot::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BNot &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BNot::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BNot>(*this)); }
Intr::Any Intr::BNot::widen() const { return Any(*this); };

Intr::LogicNot::LogicNot(Term::Any x) noexcept
    : Intr::Base({Overload({Type::Bool1(), Type::Bool1()}, Type::Bool1())}, {x}, Type::Bool1()), x(std::move(x)) {}
uint32_t Intr::LogicNot::id() const { return variant_id; };
size_t Intr::LogicNot::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicNot Intr::LogicNot::withX(const Term::Any &v_) const { return Intr::LogicNot(v_); }
POLYREGION_EXPORT bool Intr::LogicNot::operator==(const Intr::LogicNot &rhs) const { return (this->x == rhs.x); }
POLYREGION_EXPORT bool Intr::LogicNot::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicNot &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicNot::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicNot>(*this)); }
Intr::Any Intr::LogicNot::widen() const { return Any(*this); };

Intr::Pos::Pos(Term::Any x, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Pos::id() const { return variant_id; };
size_t Intr::Pos::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Pos Intr::Pos::withX(const Term::Any &v_) const { return Intr::Pos(v_, rtn); }
Intr::Pos Intr::Pos::withRtn(const Type::Any &v_) const { return Intr::Pos(x, v_); }
POLYREGION_EXPORT bool Intr::Pos::operator==(const Intr::Pos &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Intr::Pos::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Pos &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Pos::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Pos>(*this)); }
Intr::Any Intr::Pos::widen() const { return Any(*this); };

Intr::Neg::Neg(Term::Any x, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Neg::id() const { return variant_id; };
size_t Intr::Neg::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Neg Intr::Neg::withX(const Term::Any &v_) const { return Intr::Neg(v_, rtn); }
Intr::Neg Intr::Neg::withRtn(const Type::Any &v_) const { return Intr::Neg(x, v_); }
POLYREGION_EXPORT bool Intr::Neg::operator==(const Intr::Neg &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Intr::Neg::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Neg &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Neg::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Neg>(*this)); }
Intr::Any Intr::Neg::widen() const { return Any(*this); };

Intr::Add::Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Add::id() const { return variant_id; };
size_t Intr::Add::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Add Intr::Add::withX(const Term::Any &v_) const { return Intr::Add(v_, y, rtn); }
Intr::Add Intr::Add::withY(const Term::Any &v_) const { return Intr::Add(x, v_, rtn); }
Intr::Add Intr::Add::withRtn(const Type::Any &v_) const { return Intr::Add(x, y, v_); }
POLYREGION_EXPORT bool Intr::Add::operator==(const Intr::Add &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Add::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Add &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Add::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Add>(*this)); }
Intr::Any Intr::Add::widen() const { return Any(*this); };

Intr::Sub::Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Sub::id() const { return variant_id; };
size_t Intr::Sub::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Sub Intr::Sub::withX(const Term::Any &v_) const { return Intr::Sub(v_, y, rtn); }
Intr::Sub Intr::Sub::withY(const Term::Any &v_) const { return Intr::Sub(x, v_, rtn); }
Intr::Sub Intr::Sub::withRtn(const Type::Any &v_) const { return Intr::Sub(x, y, v_); }
POLYREGION_EXPORT bool Intr::Sub::operator==(const Intr::Sub &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Sub::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Sub &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Sub::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sub>(*this)); }
Intr::Any Intr::Sub::widen() const { return Any(*this); };

Intr::Mul::Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Mul::id() const { return variant_id; };
size_t Intr::Mul::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Mul Intr::Mul::withX(const Term::Any &v_) const { return Intr::Mul(v_, y, rtn); }
Intr::Mul Intr::Mul::withY(const Term::Any &v_) const { return Intr::Mul(x, v_, rtn); }
Intr::Mul Intr::Mul::withRtn(const Type::Any &v_) const { return Intr::Mul(x, y, v_); }
POLYREGION_EXPORT bool Intr::Mul::operator==(const Intr::Mul &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Mul::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Mul &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Mul::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Mul>(*this)); }
Intr::Any Intr::Mul::widen() const { return Any(*this); };

Intr::Div::Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Div::id() const { return variant_id; };
size_t Intr::Div::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Div Intr::Div::withX(const Term::Any &v_) const { return Intr::Div(v_, y, rtn); }
Intr::Div Intr::Div::withY(const Term::Any &v_) const { return Intr::Div(x, v_, rtn); }
Intr::Div Intr::Div::withRtn(const Type::Any &v_) const { return Intr::Div(x, y, v_); }
POLYREGION_EXPORT bool Intr::Div::operator==(const Intr::Div &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Div::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Div &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Div::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Div>(*this)); }
Intr::Any Intr::Div::widen() const { return Any(*this); };

Intr::Rem::Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Rem::id() const { return variant_id; };
size_t Intr::Rem::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Rem Intr::Rem::withX(const Term::Any &v_) const { return Intr::Rem(v_, y, rtn); }
Intr::Rem Intr::Rem::withY(const Term::Any &v_) const { return Intr::Rem(x, v_, rtn); }
Intr::Rem Intr::Rem::withRtn(const Type::Any &v_) const { return Intr::Rem(x, y, v_); }
POLYREGION_EXPORT bool Intr::Rem::operator==(const Intr::Rem &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Rem::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Rem &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Rem::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Rem>(*this)); }
Intr::Any Intr::Rem::widen() const { return Any(*this); };

Intr::Min::Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Min::id() const { return variant_id; };
size_t Intr::Min::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Min Intr::Min::withX(const Term::Any &v_) const { return Intr::Min(v_, y, rtn); }
Intr::Min Intr::Min::withY(const Term::Any &v_) const { return Intr::Min(x, v_, rtn); }
Intr::Min Intr::Min::withRtn(const Type::Any &v_) const { return Intr::Min(x, y, v_); }
POLYREGION_EXPORT bool Intr::Min::operator==(const Intr::Min &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Min::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Min &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Min::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Min>(*this)); }
Intr::Any Intr::Min::widen() const { return Any(*this); };

Intr::Max::Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Max::id() const { return variant_id; };
size_t Intr::Max::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::Max Intr::Max::withX(const Term::Any &v_) const { return Intr::Max(v_, y, rtn); }
Intr::Max Intr::Max::withY(const Term::Any &v_) const { return Intr::Max(x, v_, rtn); }
Intr::Max Intr::Max::withRtn(const Type::Any &v_) const { return Intr::Max(x, y, v_); }
POLYREGION_EXPORT bool Intr::Max::operator==(const Intr::Max &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::Max::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::Max &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::Max::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Max>(*this)); }
Intr::Any Intr::Max::widen() const { return Any(*this); };

Intr::BAnd::BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BAnd::id() const { return variant_id; };
size_t Intr::BAnd::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BAnd Intr::BAnd::withX(const Term::Any &v_) const { return Intr::BAnd(v_, y, rtn); }
Intr::BAnd Intr::BAnd::withY(const Term::Any &v_) const { return Intr::BAnd(x, v_, rtn); }
Intr::BAnd Intr::BAnd::withRtn(const Type::Any &v_) const { return Intr::BAnd(x, y, v_); }
POLYREGION_EXPORT bool Intr::BAnd::operator==(const Intr::BAnd &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BAnd::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BAnd &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BAnd::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BAnd>(*this)); }
Intr::Any Intr::BAnd::widen() const { return Any(*this); };

Intr::BOr::BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BOr::id() const { return variant_id; };
size_t Intr::BOr::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BOr Intr::BOr::withX(const Term::Any &v_) const { return Intr::BOr(v_, y, rtn); }
Intr::BOr Intr::BOr::withY(const Term::Any &v_) const { return Intr::BOr(x, v_, rtn); }
Intr::BOr Intr::BOr::withRtn(const Type::Any &v_) const { return Intr::BOr(x, y, v_); }
POLYREGION_EXPORT bool Intr::BOr::operator==(const Intr::BOr &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BOr::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BOr &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BOr::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BOr>(*this)); }
Intr::Any Intr::BOr::widen() const { return Any(*this); };

Intr::BXor::BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BXor::id() const { return variant_id; };
size_t Intr::BXor::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BXor Intr::BXor::withX(const Term::Any &v_) const { return Intr::BXor(v_, y, rtn); }
Intr::BXor Intr::BXor::withY(const Term::Any &v_) const { return Intr::BXor(x, v_, rtn); }
Intr::BXor Intr::BXor::withRtn(const Type::Any &v_) const { return Intr::BXor(x, y, v_); }
POLYREGION_EXPORT bool Intr::BXor::operator==(const Intr::BXor &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BXor::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BXor &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BXor::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BXor>(*this)); }
Intr::Any Intr::BXor::widen() const { return Any(*this); };

Intr::BSL::BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSL::id() const { return variant_id; };
size_t Intr::BSL::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BSL Intr::BSL::withX(const Term::Any &v_) const { return Intr::BSL(v_, y, rtn); }
Intr::BSL Intr::BSL::withY(const Term::Any &v_) const { return Intr::BSL(x, v_, rtn); }
Intr::BSL Intr::BSL::withRtn(const Type::Any &v_) const { return Intr::BSL(x, y, v_); }
POLYREGION_EXPORT bool Intr::BSL::operator==(const Intr::BSL &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BSL::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BSL &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BSL::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BSL>(*this)); }
Intr::Any Intr::BSL::widen() const { return Any(*this); };

Intr::BSR::BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSR::id() const { return variant_id; };
size_t Intr::BSR::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BSR Intr::BSR::withX(const Term::Any &v_) const { return Intr::BSR(v_, y, rtn); }
Intr::BSR Intr::BSR::withY(const Term::Any &v_) const { return Intr::BSR(x, v_, rtn); }
Intr::BSR Intr::BSR::withRtn(const Type::Any &v_) const { return Intr::BSR(x, y, v_); }
POLYREGION_EXPORT bool Intr::BSR::operator==(const Intr::BSR &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BSR::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BSR &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BSR::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BSR>(*this)); }
Intr::Any Intr::BSR::widen() const { return Any(*this); };

Intr::BZSR::BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Intr::Base({Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()), Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()),
                  Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()), Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()),
                  Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()), Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()),
                  Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()), Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BZSR::id() const { return variant_id; };
size_t Intr::BZSR::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::BZSR Intr::BZSR::withX(const Term::Any &v_) const { return Intr::BZSR(v_, y, rtn); }
Intr::BZSR Intr::BZSR::withY(const Term::Any &v_) const { return Intr::BZSR(x, v_, rtn); }
Intr::BZSR Intr::BZSR::withRtn(const Type::Any &v_) const { return Intr::BZSR(x, y, v_); }
POLYREGION_EXPORT bool Intr::BZSR::operator==(const Intr::BZSR &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Intr::BZSR::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::BZSR &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::BZSR::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<BZSR>(*this)); }
Intr::Any Intr::BZSR::widen() const { return Any(*this); };

Intr::LogicAnd::LogicAnd(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Bool1(), Type::Bool1()}, Type::Bool1())}, {x, y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicAnd::id() const { return variant_id; };
size_t Intr::LogicAnd::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicAnd Intr::LogicAnd::withX(const Term::Any &v_) const { return Intr::LogicAnd(v_, y); }
Intr::LogicAnd Intr::LogicAnd::withY(const Term::Any &v_) const { return Intr::LogicAnd(x, v_); }
POLYREGION_EXPORT bool Intr::LogicAnd::operator==(const Intr::LogicAnd &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicAnd::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicAnd &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicAnd::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicAnd>(*this)); }
Intr::Any Intr::LogicAnd::widen() const { return Any(*this); };

Intr::LogicOr::LogicOr(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Bool1(), Type::Bool1()}, Type::Bool1())}, {x, y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicOr::id() const { return variant_id; };
size_t Intr::LogicOr::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicOr Intr::LogicOr::withX(const Term::Any &v_) const { return Intr::LogicOr(v_, y); }
Intr::LogicOr Intr::LogicOr::withY(const Term::Any &v_) const { return Intr::LogicOr(x, v_); }
POLYREGION_EXPORT bool Intr::LogicOr::operator==(const Intr::LogicOr &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicOr::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicOr &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicOr::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicOr>(*this)); }
Intr::Any Intr::LogicOr::widen() const { return Any(*this); };

Intr::LogicEq::LogicEq(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicEq::id() const { return variant_id; };
size_t Intr::LogicEq::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicEq Intr::LogicEq::withX(const Term::Any &v_) const { return Intr::LogicEq(v_, y); }
Intr::LogicEq Intr::LogicEq::withY(const Term::Any &v_) const { return Intr::LogicEq(x, v_); }
POLYREGION_EXPORT bool Intr::LogicEq::operator==(const Intr::LogicEq &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicEq::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicEq &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicEq::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicEq>(*this)); }
Intr::Any Intr::LogicEq::widen() const { return Any(*this); };

Intr::LogicNeq::LogicNeq(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicNeq::id() const { return variant_id; };
size_t Intr::LogicNeq::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicNeq Intr::LogicNeq::withX(const Term::Any &v_) const { return Intr::LogicNeq(v_, y); }
Intr::LogicNeq Intr::LogicNeq::withY(const Term::Any &v_) const { return Intr::LogicNeq(x, v_); }
POLYREGION_EXPORT bool Intr::LogicNeq::operator==(const Intr::LogicNeq &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicNeq::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicNeq &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicNeq::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicNeq>(*this)); }
Intr::Any Intr::LogicNeq::widen() const { return Any(*this); };

Intr::LogicLte::LogicLte(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLte::id() const { return variant_id; };
size_t Intr::LogicLte::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicLte Intr::LogicLte::withX(const Term::Any &v_) const { return Intr::LogicLte(v_, y); }
Intr::LogicLte Intr::LogicLte::withY(const Term::Any &v_) const { return Intr::LogicLte(x, v_); }
POLYREGION_EXPORT bool Intr::LogicLte::operator==(const Intr::LogicLte &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicLte::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicLte &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicLte::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicLte>(*this)); }
Intr::Any Intr::LogicLte::widen() const { return Any(*this); };

Intr::LogicGte::LogicGte(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGte::id() const { return variant_id; };
size_t Intr::LogicGte::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicGte Intr::LogicGte::withX(const Term::Any &v_) const { return Intr::LogicGte(v_, y); }
Intr::LogicGte Intr::LogicGte::withY(const Term::Any &v_) const { return Intr::LogicGte(x, v_); }
POLYREGION_EXPORT bool Intr::LogicGte::operator==(const Intr::LogicGte &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicGte::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicGte &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicGte::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicGte>(*this)); }
Intr::Any Intr::LogicGte::widen() const { return Any(*this); };

Intr::LogicLt::LogicLt(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLt::id() const { return variant_id; };
size_t Intr::LogicLt::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicLt Intr::LogicLt::withX(const Term::Any &v_) const { return Intr::LogicLt(v_, y); }
Intr::LogicLt Intr::LogicLt::withY(const Term::Any &v_) const { return Intr::LogicLt(x, v_); }
POLYREGION_EXPORT bool Intr::LogicLt::operator==(const Intr::LogicLt &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicLt::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicLt &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicLt::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicLt>(*this)); }
Intr::Any Intr::LogicLt::widen() const { return Any(*this); };

Intr::LogicGt::LogicGt(Term::Any x, Term::Any y) noexcept
    : Intr::Base({Overload({Type::Float16(), Type::Float16()}, Type::Bool1()), Overload({Type::Float32(), Type::Float32()}, Type::Bool1()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Bool1()), Overload({Type::IntU8(), Type::IntU8()}, Type::Bool1()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::Bool1()), Overload({Type::IntU32(), Type::IntU32()}, Type::Bool1()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::Bool1()), Overload({Type::IntS8(), Type::IntS8()}, Type::Bool1()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::Bool1()), Overload({Type::IntS32(), Type::IntS32()}, Type::Bool1()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::Bool1())},
                 {x, y}, Type::Bool1()),
      x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGt::id() const { return variant_id; };
size_t Intr::LogicGt::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Intr::LogicGt Intr::LogicGt::withX(const Term::Any &v_) const { return Intr::LogicGt(v_, y); }
Intr::LogicGt Intr::LogicGt::withY(const Term::Any &v_) const { return Intr::LogicGt(x, v_); }
POLYREGION_EXPORT bool Intr::LogicGt::operator==(const Intr::LogicGt &rhs) const { return (this->x == rhs.x) && (this->y == rhs.y); }
POLYREGION_EXPORT bool Intr::LogicGt::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Intr::LogicGt &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Intr::LogicGt::operator Intr::Any() const { return std::static_pointer_cast<Base>(std::make_shared<LogicGt>(*this)); }
Intr::Any Intr::LogicGt::widen() const { return Any(*this); };

Math::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept
    : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
uint32_t Math::Any::id() const { return _v->id(); }
size_t Math::Any::hash_code() const { return _v->hash_code(); }
std::vector<Overload> Math::Any::overloads() const { return _v->overloads; }
std::vector<Term::Any> Math::Any::terms() const { return _v->terms; }
Type::Any Math::Any::tpe() const { return _v->tpe; }
bool Math::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Math::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }

Math::Abs::Abs(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64()), Overload({Type::IntU8(), Type::IntU8()}, Type::IntU8()),
                  Overload({Type::IntU16(), Type::IntU16()}, Type::IntU16()), Overload({Type::IntU32(), Type::IntU32()}, Type::IntU32()),
                  Overload({Type::IntU64(), Type::IntU64()}, Type::IntU64()), Overload({Type::IntS8(), Type::IntS8()}, Type::IntS8()),
                  Overload({Type::IntS16(), Type::IntS16()}, Type::IntS16()), Overload({Type::IntS32(), Type::IntS32()}, Type::IntS32()),
                  Overload({Type::IntS64(), Type::IntS64()}, Type::IntS64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Abs::id() const { return variant_id; };
size_t Math::Abs::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Abs Math::Abs::withX(const Term::Any &v_) const { return Math::Abs(v_, rtn); }
Math::Abs Math::Abs::withRtn(const Type::Any &v_) const { return Math::Abs(x, v_); }
POLYREGION_EXPORT bool Math::Abs::operator==(const Math::Abs &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Abs::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Abs &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Abs::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Abs>(*this)); }
Math::Any Math::Abs::widen() const { return Any(*this); };

Math::Sin::Sin(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sin::id() const { return variant_id; };
size_t Math::Sin::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Sin Math::Sin::withX(const Term::Any &v_) const { return Math::Sin(v_, rtn); }
Math::Sin Math::Sin::withRtn(const Type::Any &v_) const { return Math::Sin(x, v_); }
POLYREGION_EXPORT bool Math::Sin::operator==(const Math::Sin &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Sin::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sin &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sin::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sin>(*this)); }
Math::Any Math::Sin::widen() const { return Any(*this); };

Math::Cos::Cos(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cos::id() const { return variant_id; };
size_t Math::Cos::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Cos Math::Cos::withX(const Term::Any &v_) const { return Math::Cos(v_, rtn); }
Math::Cos Math::Cos::withRtn(const Type::Any &v_) const { return Math::Cos(x, v_); }
POLYREGION_EXPORT bool Math::Cos::operator==(const Math::Cos &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Cos::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cos &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cos::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cos>(*this)); }
Math::Any Math::Cos::widen() const { return Any(*this); };

Math::Tan::Tan(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tan::id() const { return variant_id; };
size_t Math::Tan::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Tan Math::Tan::withX(const Term::Any &v_) const { return Math::Tan(v_, rtn); }
Math::Tan Math::Tan::withRtn(const Type::Any &v_) const { return Math::Tan(x, v_); }
POLYREGION_EXPORT bool Math::Tan::operator==(const Math::Tan &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Tan::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Tan &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Tan::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Tan>(*this)); }
Math::Any Math::Tan::widen() const { return Any(*this); };

Math::Asin::Asin(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Asin::id() const { return variant_id; };
size_t Math::Asin::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Asin Math::Asin::withX(const Term::Any &v_) const { return Math::Asin(v_, rtn); }
Math::Asin Math::Asin::withRtn(const Type::Any &v_) const { return Math::Asin(x, v_); }
POLYREGION_EXPORT bool Math::Asin::operator==(const Math::Asin &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Asin::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Asin &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Asin::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Asin>(*this)); }
Math::Any Math::Asin::widen() const { return Any(*this); };

Math::Acos::Acos(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Acos::id() const { return variant_id; };
size_t Math::Acos::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Acos Math::Acos::withX(const Term::Any &v_) const { return Math::Acos(v_, rtn); }
Math::Acos Math::Acos::withRtn(const Type::Any &v_) const { return Math::Acos(x, v_); }
POLYREGION_EXPORT bool Math::Acos::operator==(const Math::Acos &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Acos::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Acos &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Acos::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Acos>(*this)); }
Math::Any Math::Acos::widen() const { return Any(*this); };

Math::Atan::Atan(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Atan::id() const { return variant_id; };
size_t Math::Atan::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Atan Math::Atan::withX(const Term::Any &v_) const { return Math::Atan(v_, rtn); }
Math::Atan Math::Atan::withRtn(const Type::Any &v_) const { return Math::Atan(x, v_); }
POLYREGION_EXPORT bool Math::Atan::operator==(const Math::Atan &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Atan::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Atan &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Atan::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Atan>(*this)); }
Math::Any Math::Atan::widen() const { return Any(*this); };

Math::Sinh::Sinh(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sinh::id() const { return variant_id; };
size_t Math::Sinh::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Sinh Math::Sinh::withX(const Term::Any &v_) const { return Math::Sinh(v_, rtn); }
Math::Sinh Math::Sinh::withRtn(const Type::Any &v_) const { return Math::Sinh(x, v_); }
POLYREGION_EXPORT bool Math::Sinh::operator==(const Math::Sinh &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Sinh::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sinh &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sinh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sinh>(*this)); }
Math::Any Math::Sinh::widen() const { return Any(*this); };

Math::Cosh::Cosh(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cosh::id() const { return variant_id; };
size_t Math::Cosh::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Cosh Math::Cosh::withX(const Term::Any &v_) const { return Math::Cosh(v_, rtn); }
Math::Cosh Math::Cosh::withRtn(const Type::Any &v_) const { return Math::Cosh(x, v_); }
POLYREGION_EXPORT bool Math::Cosh::operator==(const Math::Cosh &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Cosh::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cosh &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cosh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cosh>(*this)); }
Math::Any Math::Cosh::widen() const { return Any(*this); };

Math::Tanh::Tanh(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tanh::id() const { return variant_id; };
size_t Math::Tanh::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Tanh Math::Tanh::withX(const Term::Any &v_) const { return Math::Tanh(v_, rtn); }
Math::Tanh Math::Tanh::withRtn(const Type::Any &v_) const { return Math::Tanh(x, v_); }
POLYREGION_EXPORT bool Math::Tanh::operator==(const Math::Tanh &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Tanh::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Tanh &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Tanh::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Tanh>(*this)); }
Math::Any Math::Tanh::widen() const { return Any(*this); };

Math::Signum::Signum(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Signum::id() const { return variant_id; };
size_t Math::Signum::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Signum Math::Signum::withX(const Term::Any &v_) const { return Math::Signum(v_, rtn); }
Math::Signum Math::Signum::withRtn(const Type::Any &v_) const { return Math::Signum(x, v_); }
POLYREGION_EXPORT bool Math::Signum::operator==(const Math::Signum &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Signum::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Signum &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Signum::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Signum>(*this)); }
Math::Any Math::Signum::widen() const { return Any(*this); };

Math::Round::Round(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Round::id() const { return variant_id; };
size_t Math::Round::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Round Math::Round::withX(const Term::Any &v_) const { return Math::Round(v_, rtn); }
Math::Round Math::Round::withRtn(const Type::Any &v_) const { return Math::Round(x, v_); }
POLYREGION_EXPORT bool Math::Round::operator==(const Math::Round &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Round::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Round &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Round::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Round>(*this)); }
Math::Any Math::Round::widen() const { return Any(*this); };

Math::Ceil::Ceil(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Ceil::id() const { return variant_id; };
size_t Math::Ceil::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Ceil Math::Ceil::withX(const Term::Any &v_) const { return Math::Ceil(v_, rtn); }
Math::Ceil Math::Ceil::withRtn(const Type::Any &v_) const { return Math::Ceil(x, v_); }
POLYREGION_EXPORT bool Math::Ceil::operator==(const Math::Ceil &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Ceil::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Ceil &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Ceil::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Ceil>(*this)); }
Math::Any Math::Ceil::widen() const { return Any(*this); };

Math::Floor::Floor(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Floor::id() const { return variant_id; };
size_t Math::Floor::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Floor Math::Floor::withX(const Term::Any &v_) const { return Math::Floor(v_, rtn); }
Math::Floor Math::Floor::withRtn(const Type::Any &v_) const { return Math::Floor(x, v_); }
POLYREGION_EXPORT bool Math::Floor::operator==(const Math::Floor &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Floor::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Floor &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Floor::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Floor>(*this)); }
Math::Any Math::Floor::widen() const { return Any(*this); };

Math::Rint::Rint(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Rint::id() const { return variant_id; };
size_t Math::Rint::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Rint Math::Rint::withX(const Term::Any &v_) const { return Math::Rint(v_, rtn); }
Math::Rint Math::Rint::withRtn(const Type::Any &v_) const { return Math::Rint(x, v_); }
POLYREGION_EXPORT bool Math::Rint::operator==(const Math::Rint &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Rint::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Rint &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Rint::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Rint>(*this)); }
Math::Any Math::Rint::widen() const { return Any(*this); };

Math::Sqrt::Sqrt(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sqrt::id() const { return variant_id; };
size_t Math::Sqrt::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Sqrt Math::Sqrt::withX(const Term::Any &v_) const { return Math::Sqrt(v_, rtn); }
Math::Sqrt Math::Sqrt::withRtn(const Type::Any &v_) const { return Math::Sqrt(x, v_); }
POLYREGION_EXPORT bool Math::Sqrt::operator==(const Math::Sqrt &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Sqrt::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Sqrt &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Sqrt::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Sqrt>(*this)); }
Math::Any Math::Sqrt::widen() const { return Any(*this); };

Math::Cbrt::Cbrt(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cbrt::id() const { return variant_id; };
size_t Math::Cbrt::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Cbrt Math::Cbrt::withX(const Term::Any &v_) const { return Math::Cbrt(v_, rtn); }
Math::Cbrt Math::Cbrt::withRtn(const Type::Any &v_) const { return Math::Cbrt(x, v_); }
POLYREGION_EXPORT bool Math::Cbrt::operator==(const Math::Cbrt &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Cbrt::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Cbrt &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Cbrt::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cbrt>(*this)); }
Math::Any Math::Cbrt::widen() const { return Any(*this); };

Math::Exp::Exp(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Exp::id() const { return variant_id; };
size_t Math::Exp::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Exp Math::Exp::withX(const Term::Any &v_) const { return Math::Exp(v_, rtn); }
Math::Exp Math::Exp::withRtn(const Type::Any &v_) const { return Math::Exp(x, v_); }
POLYREGION_EXPORT bool Math::Exp::operator==(const Math::Exp &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Exp::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Exp &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Exp::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Exp>(*this)); }
Math::Any Math::Exp::widen() const { return Any(*this); };

Math::Expm1::Expm1(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Expm1::id() const { return variant_id; };
size_t Math::Expm1::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Expm1 Math::Expm1::withX(const Term::Any &v_) const { return Math::Expm1(v_, rtn); }
Math::Expm1 Math::Expm1::withRtn(const Type::Any &v_) const { return Math::Expm1(x, v_); }
POLYREGION_EXPORT bool Math::Expm1::operator==(const Math::Expm1 &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Expm1::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Expm1 &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Expm1::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Expm1>(*this)); }
Math::Any Math::Expm1::widen() const { return Any(*this); };

Math::Log::Log(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log::id() const { return variant_id; };
size_t Math::Log::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Log Math::Log::withX(const Term::Any &v_) const { return Math::Log(v_, rtn); }
Math::Log Math::Log::withRtn(const Type::Any &v_) const { return Math::Log(x, v_); }
POLYREGION_EXPORT bool Math::Log::operator==(const Math::Log &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Log::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log>(*this)); }
Math::Any Math::Log::widen() const { return Any(*this); };

Math::Log1p::Log1p(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log1p::id() const { return variant_id; };
size_t Math::Log1p::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Log1p Math::Log1p::withX(const Term::Any &v_) const { return Math::Log1p(v_, rtn); }
Math::Log1p Math::Log1p::withRtn(const Type::Any &v_) const { return Math::Log1p(x, v_); }
POLYREGION_EXPORT bool Math::Log1p::operator==(const Math::Log1p &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Log1p::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log1p &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log1p::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log1p>(*this)); }
Math::Any Math::Log1p::widen() const { return Any(*this); };

Math::Log10::Log10(Term::Any x, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16()}, Type::Float16()), Overload({Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64()}, Type::Float64())},
                 {x}, rtn),
      x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log10::id() const { return variant_id; };
size_t Math::Log10::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Log10 Math::Log10::withX(const Term::Any &v_) const { return Math::Log10(v_, rtn); }
Math::Log10 Math::Log10::withRtn(const Type::Any &v_) const { return Math::Log10(x, v_); }
POLYREGION_EXPORT bool Math::Log10::operator==(const Math::Log10 &rhs) const { return (this->x == rhs.x) && (this->rtn == rhs.rtn); }
POLYREGION_EXPORT bool Math::Log10::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Log10 &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Log10::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Log10>(*this)); }
Math::Any Math::Log10::widen() const { return Any(*this); };

Math::Pow::Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Pow::id() const { return variant_id; };
size_t Math::Pow::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Pow Math::Pow::withX(const Term::Any &v_) const { return Math::Pow(v_, y, rtn); }
Math::Pow Math::Pow::withY(const Term::Any &v_) const { return Math::Pow(x, v_, rtn); }
Math::Pow Math::Pow::withRtn(const Type::Any &v_) const { return Math::Pow(x, y, v_); }
POLYREGION_EXPORT bool Math::Pow::operator==(const Math::Pow &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Pow::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Pow &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Pow::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Pow>(*this)); }
Math::Any Math::Pow::widen() const { return Any(*this); };

Math::Atan2::Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Atan2::id() const { return variant_id; };
size_t Math::Atan2::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Atan2 Math::Atan2::withX(const Term::Any &v_) const { return Math::Atan2(v_, y, rtn); }
Math::Atan2 Math::Atan2::withY(const Term::Any &v_) const { return Math::Atan2(x, v_, rtn); }
Math::Atan2 Math::Atan2::withRtn(const Type::Any &v_) const { return Math::Atan2(x, y, v_); }
POLYREGION_EXPORT bool Math::Atan2::operator==(const Math::Atan2 &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Atan2::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Atan2 &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Atan2::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Atan2>(*this)); }
Math::Any Math::Atan2::widen() const { return Any(*this); };

Math::Hypot::Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept
    : Math::Base({Overload({Type::Float16(), Type::Float16()}, Type::Float16()),
                  Overload({Type::Float32(), Type::Float32()}, Type::Float32()),
                  Overload({Type::Float64(), Type::Float64()}, Type::Float64())},
                 {x, y}, rtn),
      x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Hypot::id() const { return variant_id; };
size_t Math::Hypot::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(x)>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(y)>()(y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Math::Hypot Math::Hypot::withX(const Term::Any &v_) const { return Math::Hypot(v_, y, rtn); }
Math::Hypot Math::Hypot::withY(const Term::Any &v_) const { return Math::Hypot(x, v_, rtn); }
Math::Hypot Math::Hypot::withRtn(const Type::Any &v_) const { return Math::Hypot(x, y, v_); }
POLYREGION_EXPORT bool Math::Hypot::operator==(const Math::Hypot &rhs) const {
  return (this->x == rhs.x) && (this->y == rhs.y) && (this->rtn == rhs.rtn);
}
POLYREGION_EXPORT bool Math::Hypot::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Math::Hypot &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
Math::Hypot::operator Math::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Hypot>(*this)); }
Math::Any Math::Hypot::widen() const { return Any(*this); };

Stmt::Base::Base() = default;
uint32_t Stmt::Any::id() const { return _v->id(); }
size_t Stmt::Any::hash_code() const { return _v->hash_code(); }
bool Stmt::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool Stmt::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool Stmt::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

Stmt::Var::Var(Named name, std::optional<Expr::Any> expr, bool isMutable) noexcept
    : Stmt::Base(), name(std::move(name)), expr(std::move(expr)), isMutable(isMutable) {}
uint32_t Stmt::Var::id() const { return variant_id; };
size_t Stmt::Var::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(isMutable)>()(isMutable) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Var Stmt::Var::withName(const Named &v_) const { return Stmt::Var(v_, expr, isMutable); }
Stmt::Var Stmt::Var::withExpr(const std::optional<Expr::Any> &v_) const { return Stmt::Var(name, v_, isMutable); }
Stmt::Var Stmt::Var::withIsMutable(const bool &v_) const { return Stmt::Var(name, expr, v_); }
POLYREGION_EXPORT bool Stmt::Var::operator==(const Stmt::Var &rhs) const {
  return (this->name == rhs.name) && ((!this->expr && !rhs.expr) || (this->expr && rhs.expr && *this->expr == *rhs.expr)) &&
         (this->isMutable == rhs.isMutable);
}
POLYREGION_EXPORT bool Stmt::Var::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Var &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Var::operator<(const Stmt::Var &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Var::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Var::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Var>(*this)); }
Stmt::Any Stmt::Var::widen() const { return Any(*this); };

Stmt::Mut::Mut(Term::Select name, Expr::Any expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
uint32_t Stmt::Mut::id() const { return variant_id; };
size_t Stmt::Mut::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(expr)>()(expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Mut Stmt::Mut::withName(const Term::Select &v_) const { return Stmt::Mut(v_, expr); }
Stmt::Mut Stmt::Mut::withExpr(const Expr::Any &v_) const { return Stmt::Mut(name, v_); }
POLYREGION_EXPORT bool Stmt::Mut::operator==(const Stmt::Mut &rhs) const { return (this->name == rhs.name) && (this->expr == rhs.expr); }
POLYREGION_EXPORT bool Stmt::Mut::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Mut &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Mut::operator<(const Stmt::Mut &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Mut::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Mut::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Mut>(*this)); }
Stmt::Any Stmt::Mut::widen() const { return Any(*this); };

Stmt::Update::Update(Term::Select lhs, Term::Any idx, Term::Any value) noexcept
    : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
uint32_t Stmt::Update::id() const { return variant_id; };
size_t Stmt::Update::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(lhs)>()(lhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(idx)>()(idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Update Stmt::Update::withLhs(const Term::Select &v_) const { return Stmt::Update(v_, idx, value); }
Stmt::Update Stmt::Update::withIdx(const Term::Any &v_) const { return Stmt::Update(lhs, v_, value); }
Stmt::Update Stmt::Update::withValue(const Term::Any &v_) const { return Stmt::Update(lhs, idx, v_); }
POLYREGION_EXPORT bool Stmt::Update::operator==(const Stmt::Update &rhs) const {
  return (this->lhs == rhs.lhs) && (this->idx == rhs.idx) && (this->value == rhs.value);
}
POLYREGION_EXPORT bool Stmt::Update::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Update &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Update::operator<(const Stmt::Update &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Update::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Update::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Update>(*this)); }
Stmt::Any Stmt::Update::widen() const { return Any(*this); };

Stmt::While::While(Term::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), cond(std::move(cond)), body(std::move(body)) {}
uint32_t Stmt::While::id() const { return variant_id; };
size_t Stmt::While::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::While Stmt::While::withCond(const Term::Any &v_) const { return Stmt::While(v_, body); }
Stmt::While Stmt::While::withBody(const std::vector<Stmt::Any> &v_) const { return Stmt::While(cond, v_); }
POLYREGION_EXPORT bool Stmt::While::operator==(const Stmt::While &rhs) const {
  return (this->cond == rhs.cond) &&
         std::equal(this->body.begin(), this->body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::While::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::While &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::While::operator<(const Stmt::While &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::While::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::While::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<While>(*this)); }
Stmt::Any Stmt::While::widen() const { return Any(*this); };

Stmt::ForRange::ForRange(Named induction, Term::Any lbIncl, Term::Any ubExcl, Term::Any step, std::vector<Stmt::Any> body) noexcept
    : Stmt::Base(), induction(std::move(induction)), lbIncl(std::move(lbIncl)), ubExcl(std::move(ubExcl)), step(std::move(step)),
      body(std::move(body)) {}
uint32_t Stmt::ForRange::id() const { return variant_id; };
size_t Stmt::ForRange::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(induction)>()(induction) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(lbIncl)>()(lbIncl) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(ubExcl)>()(ubExcl) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(step)>()(step) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(body)>()(body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::ForRange Stmt::ForRange::withInduction(const Named &v_) const { return Stmt::ForRange(v_, lbIncl, ubExcl, step, body); }
Stmt::ForRange Stmt::ForRange::withLbIncl(const Term::Any &v_) const { return Stmt::ForRange(induction, v_, ubExcl, step, body); }
Stmt::ForRange Stmt::ForRange::withUbExcl(const Term::Any &v_) const { return Stmt::ForRange(induction, lbIncl, v_, step, body); }
Stmt::ForRange Stmt::ForRange::withStep(const Term::Any &v_) const { return Stmt::ForRange(induction, lbIncl, ubExcl, v_, body); }
Stmt::ForRange Stmt::ForRange::withBody(const std::vector<Stmt::Any> &v_) const {
  return Stmt::ForRange(induction, lbIncl, ubExcl, step, v_);
}
POLYREGION_EXPORT bool Stmt::ForRange::operator==(const Stmt::ForRange &rhs) const {
  return (this->induction == rhs.induction) && (this->lbIncl == rhs.lbIncl) && (this->ubExcl == rhs.ubExcl) && (this->step == rhs.step) &&
         std::equal(this->body.begin(), this->body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::ForRange::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::ForRange &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::ForRange::operator<(const Stmt::ForRange &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::ForRange::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::ForRange::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<ForRange>(*this)); }
Stmt::Any Stmt::ForRange::widen() const { return Any(*this); };

Stmt::Break::Break() noexcept : Stmt::Base() {}
uint32_t Stmt::Break::id() const { return variant_id; };
size_t Stmt::Break::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Stmt::Break::operator==(const Stmt::Break &rhs) const { return true; }
POLYREGION_EXPORT bool Stmt::Break::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool Stmt::Break::operator<(const Stmt::Break &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Break::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Break::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Break>(*this)); }
Stmt::Any Stmt::Break::widen() const { return Any(*this); };

Stmt::Cont::Cont() noexcept : Stmt::Base() {}
uint32_t Stmt::Cont::id() const { return variant_id; };
size_t Stmt::Cont::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool Stmt::Cont::operator==(const Stmt::Cont &rhs) const { return true; }
POLYREGION_EXPORT bool Stmt::Cont::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool Stmt::Cont::operator<(const Stmt::Cont &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Cont::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Cont::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cont>(*this)); }
Stmt::Any Stmt::Cont::widen() const { return Any(*this); };

Stmt::Cond::Cond(Term::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept
    : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
uint32_t Stmt::Cond::id() const { return variant_id; };
size_t Stmt::Cond::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(cond)>()(cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(trueBr)>()(trueBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(falseBr)>()(falseBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Cond Stmt::Cond::withCond(const Term::Any &v_) const { return Stmt::Cond(v_, trueBr, falseBr); }
Stmt::Cond Stmt::Cond::withTrueBr(const std::vector<Stmt::Any> &v_) const { return Stmt::Cond(cond, v_, falseBr); }
Stmt::Cond Stmt::Cond::withFalseBr(const std::vector<Stmt::Any> &v_) const { return Stmt::Cond(cond, trueBr, v_); }
POLYREGION_EXPORT bool Stmt::Cond::operator==(const Stmt::Cond &rhs) const {
  return (this->cond == rhs.cond) &&
         std::equal(this->trueBr.begin(), this->trueBr.end(), rhs.trueBr.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         std::equal(this->falseBr.begin(), this->falseBr.end(), rhs.falseBr.begin(), [](auto &&l, auto &&r) { return l == r; });
}
POLYREGION_EXPORT bool Stmt::Cond::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Cond &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Cond::operator<(const Stmt::Cond &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Cond::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Cond::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Cond>(*this)); }
Stmt::Any Stmt::Cond::widen() const { return Any(*this); };

Stmt::Return::Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Return::id() const { return variant_id; };
size_t Stmt::Return::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Return Stmt::Return::withValue(const Expr::Any &v_) const { return Stmt::Return(v_); }
POLYREGION_EXPORT bool Stmt::Return::operator==(const Stmt::Return &rhs) const { return (this->value == rhs.value); }
POLYREGION_EXPORT bool Stmt::Return::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Return &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Return::operator<(const Stmt::Return &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Return::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Return::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Return>(*this)); }
Stmt::Any Stmt::Return::widen() const { return Any(*this); };

Stmt::Annotated::Annotated(Stmt::Any inner, std::optional<SourcePosition> pos, std::optional<std::string> comment) noexcept
    : Stmt::Base(), inner(std::move(inner)), pos(std::move(pos)), comment(std::move(comment)) {}
uint32_t Stmt::Annotated::id() const { return variant_id; };
size_t Stmt::Annotated::hash_code() const {
  size_t seed = variant_id;
  seed ^= std::hash<decltype(inner)>()(inner) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(comment)>()(comment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Stmt::Annotated Stmt::Annotated::withInner(const Stmt::Any &v_) const { return Stmt::Annotated(v_, pos, comment); }
Stmt::Annotated Stmt::Annotated::withPos(const std::optional<SourcePosition> &v_) const { return Stmt::Annotated(inner, v_, comment); }
Stmt::Annotated Stmt::Annotated::withComment(const std::optional<std::string> &v_) const { return Stmt::Annotated(inner, pos, v_); }
POLYREGION_EXPORT bool Stmt::Annotated::operator==(const Stmt::Annotated &rhs) const {
  return (this->inner == rhs.inner) && (this->pos == rhs.pos) && (this->comment == rhs.comment);
}
POLYREGION_EXPORT bool Stmt::Annotated::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return this->operator==(static_cast<const Stmt::Annotated &>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)
}
POLYREGION_EXPORT bool Stmt::Annotated::operator<(const Stmt::Annotated &rhs) const { return false; }
POLYREGION_EXPORT bool Stmt::Annotated::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
Stmt::Annotated::operator Stmt::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Annotated>(*this)); }
Stmt::Any Stmt::Annotated::widen() const { return Any(*this); };

Signature::Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args,
                     std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept
    : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)),
      moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)) {}
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
Signature Signature::withName(const Sym &v_) const { return Signature(v_, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn); }
Signature Signature::withTpeVars(const std::vector<std::string> &v_) const {
  return Signature(name, v_, receiver, args, moduleCaptures, termCaptures, rtn);
}
Signature Signature::withReceiver(const std::optional<Type::Any> &v_) const {
  return Signature(name, tpeVars, v_, args, moduleCaptures, termCaptures, rtn);
}
Signature Signature::withArgs(const std::vector<Type::Any> &v_) const {
  return Signature(name, tpeVars, receiver, v_, moduleCaptures, termCaptures, rtn);
}
Signature Signature::withModuleCaptures(const std::vector<Type::Any> &v_) const {
  return Signature(name, tpeVars, receiver, args, v_, termCaptures, rtn);
}
Signature Signature::withTermCaptures(const std::vector<Type::Any> &v_) const {
  return Signature(name, tpeVars, receiver, args, moduleCaptures, v_, rtn);
}
Signature Signature::withRtn(const Type::Any &v_) const {
  return Signature(name, tpeVars, receiver, args, moduleCaptures, termCaptures, v_);
}
POLYREGION_EXPORT bool Signature::operator!=(const Signature &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Signature::operator==(const Signature &rhs) const {
  return (name == rhs.name) && (tpeVars == rhs.tpeVars) &&
         ((!receiver && !rhs.receiver) || (receiver && rhs.receiver && *receiver == *rhs.receiver)) &&
         std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         std::equal(moduleCaptures.begin(), moduleCaptures.end(), rhs.moduleCaptures.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         std::equal(termCaptures.begin(), termCaptures.end(), rhs.termCaptures.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (rtn == rhs.rtn);
}

InvokeSignature::InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args,
                                 Type::Any rtn) noexcept
    : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), rtn(std::move(rtn)) {}
size_t InvokeSignature::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(receiver)>()(receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(rtn)>()(rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
InvokeSignature InvokeSignature::withName(const Sym &v_) const { return InvokeSignature(v_, tpeVars, receiver, args, rtn); }
InvokeSignature InvokeSignature::withTpeVars(const std::vector<Type::Any> &v_) const {
  return InvokeSignature(name, v_, receiver, args, rtn);
}
InvokeSignature InvokeSignature::withReceiver(const std::optional<Type::Any> &v_) const {
  return InvokeSignature(name, tpeVars, v_, args, rtn);
}
InvokeSignature InvokeSignature::withArgs(const std::vector<Type::Any> &v_) const {
  return InvokeSignature(name, tpeVars, receiver, v_, rtn);
}
InvokeSignature InvokeSignature::withRtn(const Type::Any &v_) const { return InvokeSignature(name, tpeVars, receiver, args, v_); }
POLYREGION_EXPORT bool InvokeSignature::operator!=(const InvokeSignature &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool InvokeSignature::operator==(const InvokeSignature &rhs) const {
  return (name == rhs.name) && std::equal(tpeVars.begin(), tpeVars.end(), rhs.tpeVars.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         ((!receiver && !rhs.receiver) || (receiver && rhs.receiver && *receiver == *rhs.receiver)) &&
         std::equal(args.begin(), args.end(), rhs.args.begin(), [](auto &&l, auto &&r) { return l == r; }) && (rtn == rhs.rtn);
}

FunctionVisibility::Base::Base() = default;
uint32_t FunctionVisibility::Any::id() const { return _v->id(); }
size_t FunctionVisibility::Any::hash_code() const { return _v->hash_code(); }
bool FunctionVisibility::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool FunctionVisibility::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool FunctionVisibility::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

FunctionVisibility::Internal::Internal() noexcept : FunctionVisibility::Base() {}
uint32_t FunctionVisibility::Internal::id() const { return variant_id; };
size_t FunctionVisibility::Internal::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionVisibility::Internal::operator==(const FunctionVisibility::Internal &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionVisibility::Internal::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionVisibility::Internal::operator<(const FunctionVisibility::Internal &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionVisibility::Internal::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionVisibility::Internal::operator FunctionVisibility::Any() const {
  return std::static_pointer_cast<Base>(std::make_shared<Internal>(*this));
}
FunctionVisibility::Any FunctionVisibility::Internal::widen() const { return Any(*this); };

FunctionVisibility::Exported::Exported() noexcept : FunctionVisibility::Base() {}
uint32_t FunctionVisibility::Exported::id() const { return variant_id; };
size_t FunctionVisibility::Exported::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionVisibility::Exported::operator==(const FunctionVisibility::Exported &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionVisibility::Exported::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionVisibility::Exported::operator<(const FunctionVisibility::Exported &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionVisibility::Exported::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionVisibility::Exported::operator FunctionVisibility::Any() const {
  return std::static_pointer_cast<Base>(std::make_shared<Exported>(*this));
}
FunctionVisibility::Any FunctionVisibility::Exported::widen() const { return Any(*this); };

FunctionFpMode::Base::Base() = default;
uint32_t FunctionFpMode::Any::id() const { return _v->id(); }
size_t FunctionFpMode::Any::hash_code() const { return _v->hash_code(); }
bool FunctionFpMode::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool FunctionFpMode::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool FunctionFpMode::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

FunctionFpMode::Relaxed::Relaxed() noexcept : FunctionFpMode::Base() {}
uint32_t FunctionFpMode::Relaxed::id() const { return variant_id; };
size_t FunctionFpMode::Relaxed::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionFpMode::Relaxed::operator==(const FunctionFpMode::Relaxed &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionFpMode::Relaxed::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionFpMode::Relaxed::operator<(const FunctionFpMode::Relaxed &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionFpMode::Relaxed::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionFpMode::Relaxed::operator FunctionFpMode::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Relaxed>(*this)); }
FunctionFpMode::Any FunctionFpMode::Relaxed::widen() const { return Any(*this); };

FunctionFpMode::Strict::Strict() noexcept : FunctionFpMode::Base() {}
uint32_t FunctionFpMode::Strict::id() const { return variant_id; };
size_t FunctionFpMode::Strict::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionFpMode::Strict::operator==(const FunctionFpMode::Strict &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionFpMode::Strict::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionFpMode::Strict::operator<(const FunctionFpMode::Strict &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionFpMode::Strict::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionFpMode::Strict::operator FunctionFpMode::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Strict>(*this)); }
FunctionFpMode::Any FunctionFpMode::Strict::widen() const { return Any(*this); };

FunctionAffinity::Base::Base() = default;
uint32_t FunctionAffinity::Any::id() const { return _v->id(); }
size_t FunctionAffinity::Any::hash_code() const { return _v->hash_code(); }
bool FunctionAffinity::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool FunctionAffinity::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool FunctionAffinity::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

FunctionAffinity::Offload::Offload() noexcept : FunctionAffinity::Base() {}
uint32_t FunctionAffinity::Offload::id() const { return variant_id; };
size_t FunctionAffinity::Offload::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionAffinity::Offload::operator==(const FunctionAffinity::Offload &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionAffinity::Offload::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAffinity::Offload::operator<(const FunctionAffinity::Offload &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAffinity::Offload::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionAffinity::Offload::operator FunctionAffinity::Any() const {
  return std::static_pointer_cast<Base>(std::make_shared<Offload>(*this));
}
FunctionAffinity::Any FunctionAffinity::Offload::widen() const { return Any(*this); };

FunctionAffinity::Host::Host() noexcept : FunctionAffinity::Base() {}
uint32_t FunctionAffinity::Host::id() const { return variant_id; };
size_t FunctionAffinity::Host::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool FunctionAffinity::Host::operator==(const FunctionAffinity::Host &rhs) const { return true; }
POLYREGION_EXPORT bool FunctionAffinity::Host::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool FunctionAffinity::Host::operator<(const FunctionAffinity::Host &rhs) const { return false; }
POLYREGION_EXPORT bool FunctionAffinity::Host::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
FunctionAffinity::Host::operator FunctionAffinity::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Host>(*this)); }
FunctionAffinity::Any FunctionAffinity::Host::widen() const { return Any(*this); };

Arg::Arg(Named named, std::optional<SourcePosition> pos) noexcept : named(std::move(named)), pos(std::move(pos)) {}
size_t Arg::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(named)>()(named) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(pos)>()(pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Arg Arg::withNamed(const Named &v_) const { return Arg(v_, pos); }
Arg Arg::withPos(const std::optional<SourcePosition> &v_) const { return Arg(named, v_); }
POLYREGION_EXPORT bool Arg::operator!=(const Arg &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Arg::operator==(const Arg &rhs) const { return (named == rhs.named) && (pos == rhs.pos); }

Function::Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args,
                   std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body,
                   FunctionVisibility::Any visibility, FunctionFpMode::Any fpMode, bool isEntry, FunctionAffinity::Any affinity) noexcept
    : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)),
      moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)), body(std::move(body)),
      visibility(std::move(visibility)), fpMode(std::move(fpMode)), isEntry(isEntry), affinity(std::move(affinity)) {}
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
  seed ^= std::hash<decltype(visibility)>()(visibility) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(fpMode)>()(fpMode) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(isEntry)>()(isEntry) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(affinity)>()(affinity) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Function Function::withName(const Sym &v_) const {
  return Function(v_, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withTpeVars(const std::vector<std::string> &v_) const {
  return Function(name, v_, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withReceiver(const std::optional<Arg> &v_) const {
  return Function(name, tpeVars, v_, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withArgs(const std::vector<Arg> &v_) const {
  return Function(name, tpeVars, receiver, v_, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withModuleCaptures(const std::vector<Arg> &v_) const {
  return Function(name, tpeVars, receiver, args, v_, termCaptures, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withTermCaptures(const std::vector<Arg> &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, v_, rtn, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withRtn(const Type::Any &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, v_, body, visibility, fpMode, isEntry, affinity);
}
Function Function::withBody(const std::vector<Stmt::Any> &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, v_, visibility, fpMode, isEntry, affinity);
}
Function Function::withVisibility(const FunctionVisibility::Any &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, v_, fpMode, isEntry, affinity);
}
Function Function::withFpMode(const FunctionFpMode::Any &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, v_, isEntry, affinity);
}
Function Function::withIsEntry(const bool &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, v_, affinity);
}
Function Function::withAffinity(const FunctionAffinity::Any &v_) const {
  return Function(name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry, v_);
}
POLYREGION_EXPORT bool Function::operator!=(const Function &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Function::operator==(const Function &rhs) const {
  return (name == rhs.name) && (tpeVars == rhs.tpeVars) && (receiver == rhs.receiver) && (args == rhs.args) &&
         (moduleCaptures == rhs.moduleCaptures) && (termCaptures == rhs.termCaptures) && (rtn == rhs.rtn) &&
         std::equal(body.begin(), body.end(), rhs.body.begin(), [](auto &&l, auto &&r) { return l == r; }) &&
         (visibility == rhs.visibility) && (fpMode == rhs.fpMode) && (isEntry == rhs.isEntry) && (affinity == rhs.affinity);
}

StructDef::StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<Named> members, std::vector<Type::Struct> parents) noexcept
    : name(std::move(name)), tpeVars(std::move(tpeVars)), members(std::move(members)), parents(std::move(parents)) {}
size_t StructDef::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(tpeVars)>()(tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(parents)>()(parents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
StructDef StructDef::withName(const Sym &v_) const { return StructDef(v_, tpeVars, members, parents); }
StructDef StructDef::withTpeVars(const std::vector<std::string> &v_) const { return StructDef(name, v_, members, parents); }
StructDef StructDef::withMembers(const std::vector<Named> &v_) const { return StructDef(name, tpeVars, v_, parents); }
StructDef StructDef::withParents(const std::vector<Type::Struct> &v_) const { return StructDef(name, tpeVars, members, v_); }
POLYREGION_EXPORT bool StructDef::operator!=(const StructDef &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool StructDef::operator==(const StructDef &rhs) const {
  return (name == rhs.name) && (tpeVars == rhs.tpeVars) && (members == rhs.members) && (parents == rhs.parents);
}

Mirror::Mirror(Sym source, std::vector<Sym> sourceParents, StructDef structDef, std::vector<Function> functions,
               std::vector<StructDef> dependencies) noexcept
    : source(std::move(source)), sourceParents(std::move(sourceParents)), structDef(std::move(structDef)), functions(std::move(functions)),
      dependencies(std::move(dependencies)) {}
size_t Mirror::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(source)>()(source) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sourceParents)>()(sourceParents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(structDef)>()(structDef) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(functions)>()(functions) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(dependencies)>()(dependencies) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Mirror Mirror::withSource(const Sym &v_) const { return Mirror(v_, sourceParents, structDef, functions, dependencies); }
Mirror Mirror::withSourceParents(const std::vector<Sym> &v_) const { return Mirror(source, v_, structDef, functions, dependencies); }
Mirror Mirror::withStructDef(const StructDef &v_) const { return Mirror(source, sourceParents, v_, functions, dependencies); }
Mirror Mirror::withFunctions(const std::vector<Function> &v_) const { return Mirror(source, sourceParents, structDef, v_, dependencies); }
Mirror Mirror::withDependencies(const std::vector<StructDef> &v_) const { return Mirror(source, sourceParents, structDef, functions, v_); }
POLYREGION_EXPORT bool Mirror::operator!=(const Mirror &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Mirror::operator==(const Mirror &rhs) const {
  return (source == rhs.source) && (sourceParents == rhs.sourceParents) && (structDef == rhs.structDef) && (functions == rhs.functions) &&
         (dependencies == rhs.dependencies);
}

PassPhase::Base::Base() = default;
uint32_t PassPhase::Any::id() const { return _v->id(); }
size_t PassPhase::Any::hash_code() const { return _v->hash_code(); }
bool PassPhase::Any::operator==(const Any &rhs) const { return _v->operator==(*rhs._v); }
bool PassPhase::Any::operator!=(const Any &rhs) const { return !_v->operator==(*rhs._v); }
bool PassPhase::Any::operator<(const Any &rhs) const { return _v->operator<(*rhs._v); };

PassPhase::Initial::Initial() noexcept : PassPhase::Base() {}
uint32_t PassPhase::Initial::id() const { return variant_id; };
size_t PassPhase::Initial::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool PassPhase::Initial::operator==(const PassPhase::Initial &rhs) const { return true; }
POLYREGION_EXPORT bool PassPhase::Initial::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool PassPhase::Initial::operator<(const PassPhase::Initial &rhs) const { return false; }
POLYREGION_EXPORT bool PassPhase::Initial::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PassPhase::Initial::operator PassPhase::Any() const { return std::static_pointer_cast<Base>(std::make_shared<Initial>(*this)); }
PassPhase::Any PassPhase::Initial::widen() const { return Any(*this); };

PassPhase::PostMono::PostMono() noexcept : PassPhase::Base() {}
uint32_t PassPhase::PostMono::id() const { return variant_id; };
size_t PassPhase::PostMono::hash_code() const {
  size_t seed = variant_id;
  return seed;
}
POLYREGION_EXPORT bool PassPhase::PostMono::operator==(const PassPhase::PostMono &rhs) const { return true; }
POLYREGION_EXPORT bool PassPhase::PostMono::operator==(const Base &rhs_) const {
  if (rhs_.id() != variant_id) return false;
  return true;
}
POLYREGION_EXPORT bool PassPhase::PostMono::operator<(const PassPhase::PostMono &rhs) const { return false; }
POLYREGION_EXPORT bool PassPhase::PostMono::operator<(const Base &rhs_) const { return variant_id < rhs_.id(); }
PassPhase::PostMono::operator PassPhase::Any() const { return std::static_pointer_cast<Base>(std::make_shared<PostMono>(*this)); }
PassPhase::Any PassPhase::PostMono::widen() const { return Any(*this); };

MetaEntry::MetaEntry(std::string key, std::string value) noexcept : key(std::move(key)), value(std::move(value)) {}
size_t MetaEntry::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(key)>()(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
MetaEntry MetaEntry::withKey(const std::string &v_) const { return MetaEntry(v_, value); }
MetaEntry MetaEntry::withValue(const std::string &v_) const { return MetaEntry(key, v_); }
POLYREGION_EXPORT bool MetaEntry::operator!=(const MetaEntry &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool MetaEntry::operator==(const MetaEntry &rhs) const { return (key == rhs.key) && (value == rhs.value); }

Program::Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs, PassPhase::Any phase,
                 std::vector<MetaEntry> metadata) noexcept
    : entry(std::move(entry)), functions(std::move(functions)), defs(std::move(defs)), phase(std::move(phase)),
      metadata(std::move(metadata)) {}
size_t Program::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(entry)>()(entry) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(functions)>()(functions) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(defs)>()(defs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(phase)>()(phase) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(metadata)>()(metadata) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
Program Program::withEntry(const Function &v_) const { return Program(v_, functions, defs, phase, metadata); }
Program Program::withFunctions(const std::vector<Function> &v_) const { return Program(entry, v_, defs, phase, metadata); }
Program Program::withDefs(const std::vector<StructDef> &v_) const { return Program(entry, functions, v_, phase, metadata); }
Program Program::withPhase(const PassPhase::Any &v_) const { return Program(entry, functions, defs, v_, metadata); }
Program Program::withMetadata(const std::vector<MetaEntry> &v_) const { return Program(entry, functions, defs, phase, v_); }
POLYREGION_EXPORT bool Program::operator!=(const Program &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool Program::operator==(const Program &rhs) const {
  return (entry == rhs.entry) && (functions == rhs.functions) && (defs == rhs.defs) && (phase == rhs.phase) && (metadata == rhs.metadata);
}

StructLayoutMember::StructLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept
    : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
size_t StructLayoutMember::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(offsetInBytes)>()(offsetInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
StructLayoutMember StructLayoutMember::withName(const Named &v_) const { return StructLayoutMember(v_, offsetInBytes, sizeInBytes); }
StructLayoutMember StructLayoutMember::withOffsetInBytes(const int64_t &v_) const { return StructLayoutMember(name, v_, sizeInBytes); }
StructLayoutMember StructLayoutMember::withSizeInBytes(const int64_t &v_) const { return StructLayoutMember(name, offsetInBytes, v_); }
POLYREGION_EXPORT bool StructLayoutMember::operator!=(const StructLayoutMember &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool StructLayoutMember::operator==(const StructLayoutMember &rhs) const {
  return (name == rhs.name) && (offsetInBytes == rhs.offsetInBytes) && (sizeInBytes == rhs.sizeInBytes);
}

StructLayout::StructLayout(std::string name, int64_t sizeInBytes, int64_t alignment, std::vector<StructLayoutMember> members) noexcept
    : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
size_t StructLayout::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(sizeInBytes)>()(sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(alignment)>()(alignment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(members)>()(members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
StructLayout StructLayout::withName(const std::string &v_) const { return StructLayout(v_, sizeInBytes, alignment, members); }
StructLayout StructLayout::withSizeInBytes(const int64_t &v_) const { return StructLayout(name, v_, alignment, members); }
StructLayout StructLayout::withAlignment(const int64_t &v_) const { return StructLayout(name, sizeInBytes, v_, members); }
StructLayout StructLayout::withMembers(const std::vector<StructLayoutMember> &v_) const {
  return StructLayout(name, sizeInBytes, alignment, v_);
}
POLYREGION_EXPORT bool StructLayout::operator!=(const StructLayout &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool StructLayout::operator==(const StructLayout &rhs) const {
  return (name == rhs.name) && (sizeInBytes == rhs.sizeInBytes) && (alignment == rhs.alignment) && (members == rhs.members);
}

CompileEvent::CompileEvent(int64_t epochMillis, int64_t elapsedNanos, std::string name, std::string data,
                           std::vector<CompileEvent> items) noexcept
    : epochMillis(epochMillis), elapsedNanos(elapsedNanos), name(std::move(name)), data(std::move(data)), items(std::move(items)) {}
size_t CompileEvent::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(epochMillis)>()(epochMillis) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(elapsedNanos)>()(elapsedNanos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(data)>()(data) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(items)>()(items) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
CompileEvent CompileEvent::withEpochMillis(const int64_t &v_) const { return CompileEvent(v_, elapsedNanos, name, data, items); }
CompileEvent CompileEvent::withElapsedNanos(const int64_t &v_) const { return CompileEvent(epochMillis, v_, name, data, items); }
CompileEvent CompileEvent::withName(const std::string &v_) const { return CompileEvent(epochMillis, elapsedNanos, v_, data, items); }
CompileEvent CompileEvent::withData(const std::string &v_) const { return CompileEvent(epochMillis, elapsedNanos, name, v_, items); }
CompileEvent CompileEvent::withItems(const std::vector<CompileEvent> &v_) const {
  return CompileEvent(epochMillis, elapsedNanos, name, data, v_);
}
POLYREGION_EXPORT bool CompileEvent::operator!=(const CompileEvent &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool CompileEvent::operator==(const CompileEvent &rhs) const {
  return (epochMillis == rhs.epochMillis) && (elapsedNanos == rhs.elapsedNanos) && (name == rhs.name) && (data == rhs.data) &&
         (items == rhs.items);
}

PassArg::PassArg(std::string name, std::string value) noexcept : name(std::move(name)), value(std::move(value)) {}
size_t PassArg::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(value)>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PassArg PassArg::withName(const std::string &v_) const { return PassArg(v_, value); }
PassArg PassArg::withValue(const std::string &v_) const { return PassArg(name, v_); }
POLYREGION_EXPORT bool PassArg::operator!=(const PassArg &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool PassArg::operator==(const PassArg &rhs) const { return (name == rhs.name) && (value == rhs.value); }

PassSpec::PassSpec(std::string name, std::vector<PassArg> args) noexcept : name(std::move(name)), args(std::move(args)) {}
size_t PassSpec::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(name)>()(name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(args)>()(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PassSpec PassSpec::withName(const std::string &v_) const { return PassSpec(v_, args); }
PassSpec PassSpec::withArgs(const std::vector<PassArg> &v_) const { return PassSpec(name, v_); }
POLYREGION_EXPORT bool PassSpec::operator!=(const PassSpec &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool PassSpec::operator==(const PassSpec &rhs) const { return (name == rhs.name) && (args == rhs.args); }

PassPipeline::PassPipeline(std::vector<PassSpec> steps) noexcept : steps(std::move(steps)) {}
size_t PassPipeline::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(steps)>()(steps) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PassPipeline PassPipeline::withSteps(const std::vector<PassSpec> &v_) const { return PassPipeline(v_); }
POLYREGION_EXPORT bool PassPipeline::operator!=(const PassPipeline &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool PassPipeline::operator==(const PassPipeline &rhs) const { return (steps == rhs.steps); }

PassRunResult::PassRunResult(Program program, CompileEvent event) noexcept : program(std::move(program)), event(std::move(event)) {}
size_t PassRunResult::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(program)>()(program) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(event)>()(event) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
PassRunResult PassRunResult::withProgram(const Program &v_) const { return PassRunResult(v_, event); }
PassRunResult PassRunResult::withEvent(const CompileEvent &v_) const { return PassRunResult(program, v_); }
POLYREGION_EXPORT bool PassRunResult::operator!=(const PassRunResult &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool PassRunResult::operator==(const PassRunResult &rhs) const {
  return (program == rhs.program) && (event == rhs.event);
}

CompileResult::CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events,
                             std::vector<StructLayout> layouts, std::string messages) noexcept
    : binary(std::move(binary)), features(std::move(features)), events(std::move(events)), layouts(std::move(layouts)),
      messages(std::move(messages)) {}
size_t CompileResult::hash_code() const {
  size_t seed = 0;
  seed ^= std::hash<decltype(binary)>()(binary) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(features)>()(features) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(events)>()(events) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(layouts)>()(layouts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(messages)>()(messages) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
CompileResult CompileResult::withBinary(const std::optional<std::vector<int8_t>> &v_) const {
  return CompileResult(v_, features, events, layouts, messages);
}
CompileResult CompileResult::withFeatures(const std::vector<std::string> &v_) const {
  return CompileResult(binary, v_, events, layouts, messages);
}
CompileResult CompileResult::withEvents(const std::vector<CompileEvent> &v_) const {
  return CompileResult(binary, features, v_, layouts, messages);
}
CompileResult CompileResult::withLayouts(const std::vector<StructLayout> &v_) const {
  return CompileResult(binary, features, events, v_, messages);
}
CompileResult CompileResult::withMessages(const std::string &v_) const { return CompileResult(binary, features, events, layouts, v_); }
POLYREGION_EXPORT bool CompileResult::operator!=(const CompileResult &rhs) const { return !(*this == rhs); }
POLYREGION_EXPORT bool CompileResult::operator==(const CompileResult &rhs) const {
  return (binary == rhs.binary) && (features == rhs.features) && (events == rhs.events) && (layouts == rhs.layouts) &&
         (messages == rhs.messages);
}

} // namespace polyregion::polyast

std::size_t std::hash<polyregion::polyast::Sym>::operator()(const polyregion::polyast::Sym &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::SourcePosition>::operator()(const polyregion::polyast::SourcePosition &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Named>::operator()(const polyregion::polyast::Named &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::TypeKind::Any>::operator()(const polyregion::polyast::TypeKind::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::TypeKind::None>::operator()(const polyregion::polyast::TypeKind::None &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::TypeKind::Ref>::operator()(const polyregion::polyast::TypeKind::Ref &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::TypeKind::Integral>::operator()(const polyregion::polyast::TypeKind::Integral &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::TypeKind::Fractional>::operator()(const polyregion::polyast::TypeKind::Fractional &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::TypeSpace::Any>::operator()(const polyregion::polyast::TypeSpace::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::TypeSpace::Global>::operator()(const polyregion::polyast::TypeSpace::Global &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::TypeSpace::Local>::operator()(const polyregion::polyast::TypeSpace::Local &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::TypeSpace::Private>::operator()(const polyregion::polyast::TypeSpace::Private &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::TypeSpace::Constant>::operator()(const polyregion::polyast::TypeSpace::Constant &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Any>::operator()(const polyregion::polyast::Type::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Float16>::operator()(const polyregion::polyast::Type::Float16 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Float32>::operator()(const polyregion::polyast::Type::Float32 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Float64>::operator()(const polyregion::polyast::Type::Float64 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntU8>::operator()(const polyregion::polyast::Type::IntU8 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntU16>::operator()(const polyregion::polyast::Type::IntU16 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntU32>::operator()(const polyregion::polyast::Type::IntU32 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntU64>::operator()(const polyregion::polyast::Type::IntU64 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntS8>::operator()(const polyregion::polyast::Type::IntS8 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntS16>::operator()(const polyregion::polyast::Type::IntS16 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntS32>::operator()(const polyregion::polyast::Type::IntS32 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::IntS64>::operator()(const polyregion::polyast::Type::IntS64 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Nothing>::operator()(const polyregion::polyast::Type::Nothing &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Unit0>::operator()(const polyregion::polyast::Type::Unit0 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Bool1>::operator()(const polyregion::polyast::Type::Bool1 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Struct>::operator()(const polyregion::polyast::Type::Struct &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Ptr>::operator()(const polyregion::polyast::Type::Ptr &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Arr>::operator()(const polyregion::polyast::Type::Arr &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Var>::operator()(const polyregion::polyast::Type::Var &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Type::Exec>::operator()(const polyregion::polyast::Type::Exec &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PathStep::Any>::operator()(const polyregion::polyast::PathStep::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PathStep::Field>::operator()(const polyregion::polyast::PathStep::Field &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PathStep::Deref>::operator()(const polyregion::polyast::PathStep::Deref &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PathStep::Index>::operator()(const polyregion::polyast::PathStep::Index &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::PathStep::IndexDyn>::operator()(const polyregion::polyast::PathStep::IndexDyn &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Region::Any>::operator()(const polyregion::polyast::Region::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Region::Rooted>::operator()(const polyregion::polyast::Region::Rooted &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Region::Opaque>::operator()(const polyregion::polyast::Region::Opaque &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::Any>::operator()(const polyregion::polyast::Term::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Term::Float16Const>::operator()(const polyregion::polyast::Term::Float16Const &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Term::Float32Const>::operator()(const polyregion::polyast::Term::Float32Const &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Term::Float64Const>::operator()(const polyregion::polyast::Term::Float64Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntU8Const>::operator()(const polyregion::polyast::Term::IntU8Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntU16Const>::operator()(const polyregion::polyast::Term::IntU16Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntU32Const>::operator()(const polyregion::polyast::Term::IntU32Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntU64Const>::operator()(const polyregion::polyast::Term::IntU64Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntS8Const>::operator()(const polyregion::polyast::Term::IntS8Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntS16Const>::operator()(const polyregion::polyast::Term::IntS16Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntS32Const>::operator()(const polyregion::polyast::Term::IntS32Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::IntS64Const>::operator()(const polyregion::polyast::Term::IntS64Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::Unit0Const>::operator()(const polyregion::polyast::Term::Unit0Const &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::Bool1Const>::operator()(const polyregion::polyast::Term::Bool1Const &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Term::NullPtrConst>::operator()(const polyregion::polyast::Term::NullPtrConst &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::Poison>::operator()(const polyregion::polyast::Term::Poison &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Term::Select>::operator()(const polyregion::polyast::Term::Select &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Any>::operator()(const polyregion::polyast::Expr::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Alias>::operator()(const polyregion::polyast::Expr::Alias &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::SpecOp>::operator()(const polyregion::polyast::Expr::SpecOp &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::MathOp>::operator()(const polyregion::polyast::Expr::MathOp &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::IntrOp>::operator()(const polyregion::polyast::Expr::IntrOp &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Cast>::operator()(const polyregion::polyast::Expr::Cast &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Index>::operator()(const polyregion::polyast::Expr::Index &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::RefTo>::operator()(const polyregion::polyast::Expr::RefTo &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Alloc>::operator()(const polyregion::polyast::Expr::Alloc &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::Invoke>::operator()(const polyregion::polyast::Expr::Invoke &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::ForeignCall>::operator()(const polyregion::polyast::Expr::ForeignCall &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::OffsetOf>::operator()(const polyregion::polyast::Expr::OffsetOf &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Expr::SizeOf>::operator()(const polyregion::polyast::Expr::SizeOf &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Overload>::operator()(const polyregion::polyast::Overload &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Spec::Any>::operator()(const polyregion::polyast::Spec::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Spec::Assert>::operator()(const polyregion::polyast::Spec::Assert &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuBarrierGlobal>::operator()(const polyregion::polyast::Spec::GpuBarrierGlobal &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuBarrierLocal>::operator()(const polyregion::polyast::Spec::GpuBarrierLocal &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuBarrierAll>::operator()(const polyregion::polyast::Spec::GpuBarrierAll &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuFenceGlobal>::operator()(const polyregion::polyast::Spec::GpuFenceGlobal &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuFenceLocal>::operator()(const polyregion::polyast::Spec::GpuFenceLocal &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceAll>::operator()(const polyregion::polyast::Spec::GpuFenceAll &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuGlobalIdx>::operator()(const polyregion::polyast::Spec::GpuGlobalIdx &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuGlobalSize>::operator()(const polyregion::polyast::Spec::GpuGlobalSize &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Spec::GpuGroupIdx>::operator()(const polyregion::polyast::Spec::GpuGroupIdx &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuGroupSize>::operator()(const polyregion::polyast::Spec::GpuGroupSize &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Spec::GpuLocalIdx>::operator()(const polyregion::polyast::Spec::GpuLocalIdx &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::Spec::GpuLocalSize>::operator()(const polyregion::polyast::Spec::GpuLocalSize &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Any>::operator()(const polyregion::polyast::Intr::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BNot>::operator()(const polyregion::polyast::Intr::BNot &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicNot>::operator()(const polyregion::polyast::Intr::LogicNot &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Pos>::operator()(const polyregion::polyast::Intr::Pos &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Neg>::operator()(const polyregion::polyast::Intr::Neg &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Add>::operator()(const polyregion::polyast::Intr::Add &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Sub>::operator()(const polyregion::polyast::Intr::Sub &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Mul>::operator()(const polyregion::polyast::Intr::Mul &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Div>::operator()(const polyregion::polyast::Intr::Div &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Rem>::operator()(const polyregion::polyast::Intr::Rem &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Min>::operator()(const polyregion::polyast::Intr::Min &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::Max>::operator()(const polyregion::polyast::Intr::Max &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BAnd>::operator()(const polyregion::polyast::Intr::BAnd &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BOr>::operator()(const polyregion::polyast::Intr::BOr &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BXor>::operator()(const polyregion::polyast::Intr::BXor &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BSL>::operator()(const polyregion::polyast::Intr::BSL &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BSR>::operator()(const polyregion::polyast::Intr::BSR &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::BZSR>::operator()(const polyregion::polyast::Intr::BZSR &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicAnd>::operator()(const polyregion::polyast::Intr::LogicAnd &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicOr>::operator()(const polyregion::polyast::Intr::LogicOr &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicEq>::operator()(const polyregion::polyast::Intr::LogicEq &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicNeq>::operator()(const polyregion::polyast::Intr::LogicNeq &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicLte>::operator()(const polyregion::polyast::Intr::LogicLte &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicGte>::operator()(const polyregion::polyast::Intr::LogicGte &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicLt>::operator()(const polyregion::polyast::Intr::LogicLt &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Intr::LogicGt>::operator()(const polyregion::polyast::Intr::LogicGt &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Any>::operator()(const polyregion::polyast::Math::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Abs>::operator()(const polyregion::polyast::Math::Abs &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Sin>::operator()(const polyregion::polyast::Math::Sin &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Cos>::operator()(const polyregion::polyast::Math::Cos &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Tan>::operator()(const polyregion::polyast::Math::Tan &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Asin>::operator()(const polyregion::polyast::Math::Asin &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Acos>::operator()(const polyregion::polyast::Math::Acos &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Atan>::operator()(const polyregion::polyast::Math::Atan &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Sinh>::operator()(const polyregion::polyast::Math::Sinh &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Cosh>::operator()(const polyregion::polyast::Math::Cosh &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Tanh>::operator()(const polyregion::polyast::Math::Tanh &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Signum>::operator()(const polyregion::polyast::Math::Signum &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Round>::operator()(const polyregion::polyast::Math::Round &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Ceil>::operator()(const polyregion::polyast::Math::Ceil &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Floor>::operator()(const polyregion::polyast::Math::Floor &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Rint>::operator()(const polyregion::polyast::Math::Rint &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Sqrt>::operator()(const polyregion::polyast::Math::Sqrt &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Cbrt>::operator()(const polyregion::polyast::Math::Cbrt &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Exp>::operator()(const polyregion::polyast::Math::Exp &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Expm1>::operator()(const polyregion::polyast::Math::Expm1 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Log>::operator()(const polyregion::polyast::Math::Log &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Log1p>::operator()(const polyregion::polyast::Math::Log1p &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Log10>::operator()(const polyregion::polyast::Math::Log10 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Pow>::operator()(const polyregion::polyast::Math::Pow &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Atan2>::operator()(const polyregion::polyast::Math::Atan2 &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Math::Hypot>::operator()(const polyregion::polyast::Math::Hypot &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Any>::operator()(const polyregion::polyast::Stmt::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Var>::operator()(const polyregion::polyast::Stmt::Var &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Mut>::operator()(const polyregion::polyast::Stmt::Mut &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Update>::operator()(const polyregion::polyast::Stmt::Update &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::While>::operator()(const polyregion::polyast::Stmt::While &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::ForRange>::operator()(const polyregion::polyast::Stmt::ForRange &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Break>::operator()(const polyregion::polyast::Stmt::Break &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Cont>::operator()(const polyregion::polyast::Stmt::Cont &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Cond>::operator()(const polyregion::polyast::Stmt::Cond &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Return>::operator()(const polyregion::polyast::Stmt::Return &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Stmt::Annotated>::operator()(const polyregion::polyast::Stmt::Annotated &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Signature>::operator()(const polyregion::polyast::Signature &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::InvokeSignature>::operator()(const polyregion::polyast::InvokeSignature &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionVisibility::Any>::operator()(const polyregion::polyast::FunctionVisibility::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::FunctionVisibility::Internal>::operator()(
    const polyregion::polyast::FunctionVisibility::Internal &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::FunctionVisibility::Exported>::operator()(
    const polyregion::polyast::FunctionVisibility::Exported &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionFpMode::Any>::operator()(const polyregion::polyast::FunctionFpMode::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionFpMode::Relaxed>::operator()(const polyregion::polyast::FunctionFpMode::Relaxed &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionFpMode::Strict>::operator()(const polyregion::polyast::FunctionFpMode::Strict &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionAffinity::Any>::operator()(const polyregion::polyast::FunctionAffinity::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::FunctionAffinity::Offload>::operator()(
    const polyregion::polyast::FunctionAffinity::Offload &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::FunctionAffinity::Host>::operator()(const polyregion::polyast::FunctionAffinity::Host &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Arg>::operator()(const polyregion::polyast::Arg &x) const noexcept { return x.hash_code(); }
std::size_t std::hash<polyregion::polyast::Function>::operator()(const polyregion::polyast::Function &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::StructDef>::operator()(const polyregion::polyast::StructDef &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Mirror>::operator()(const polyregion::polyast::Mirror &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PassPhase::Any>::operator()(const polyregion::polyast::PassPhase::Any &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::PassPhase::Initial>::operator()(const polyregion::polyast::PassPhase::Initial &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::PassPhase::PostMono>::operator()(const polyregion::polyast::PassPhase::PostMono &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::MetaEntry>::operator()(const polyregion::polyast::MetaEntry &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::Program>::operator()(const polyregion::polyast::Program &x) const noexcept {
  return x.hash_code();
}
std::size_t
std::hash<polyregion::polyast::StructLayoutMember>::operator()(const polyregion::polyast::StructLayoutMember &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::StructLayout>::operator()(const polyregion::polyast::StructLayout &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::CompileEvent>::operator()(const polyregion::polyast::CompileEvent &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PassArg>::operator()(const polyregion::polyast::PassArg &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PassSpec>::operator()(const polyregion::polyast::PassSpec &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PassPipeline>::operator()(const polyregion::polyast::PassPipeline &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::PassRunResult>::operator()(const polyregion::polyast::PassRunResult &x) const noexcept {
  return x.hash_code();
}
std::size_t std::hash<polyregion::polyast::CompileResult>::operator()(const polyregion::polyast::CompileResult &x) const noexcept {
  return x.hash_code();
}
