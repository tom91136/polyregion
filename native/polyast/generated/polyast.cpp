#include "polyast.h"

namespace polyregion::polyast {

Sym::Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
std::ostream &operator<<(std::ostream &os, const Sym &x) {
  os << "Sym(";
  os << '{';
  if (!x.fqn.empty()) {
    std::for_each(x.fqn.begin(), std::prev(x.fqn.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.fqn.back() << '"';
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const Sym &l, const Sym &r) { 
  return l.fqn == r.fqn;
}

Named::Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
std::ostream &operator<<(std::ostream &os, const Named &x) {
  os << "Named(";
  os << '"' << x.symbol << '"';
  os << ',';
  os << x.tpe;
  os << ')';
  return os;
}
bool operator==(const Named &l, const Named &r) { 
  return l.symbol == r.symbol && *l.tpe == *r.tpe;
}

TypeKind::Base::Base() = default;
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool TypeKind::operator==(const TypeKind::Base &, const TypeKind::Base &) { return true; }

TypeKind::None::None() noexcept : TypeKind::Base() {}
uint32_t TypeKind::None::id() const { return 0; };
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::None &x) {
  os << "None(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::None &, const TypeKind::None &) { return true; }
TypeKind::None::operator TypeKind::Any() const { return std::make_shared<None>(*this); }

TypeKind::Ref::Ref() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Ref::id() const { return 1; };
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Ref &x) {
  os << "Ref(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Ref &, const TypeKind::Ref &) { return true; }
TypeKind::Ref::operator TypeKind::Any() const { return std::make_shared<Ref>(*this); }

TypeKind::Integral::Integral() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Integral::id() const { return 2; };
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Integral &x) {
  os << "Integral(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Integral &, const TypeKind::Integral &) { return true; }
TypeKind::Integral::operator TypeKind::Any() const { return std::make_shared<Integral>(*this); }

TypeKind::Fractional::Fractional() noexcept : TypeKind::Base() {}
uint32_t TypeKind::Fractional::id() const { return 3; };
std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Fractional &x) {
  os << "Fractional(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Fractional &, const TypeKind::Fractional &) { return true; }
TypeKind::Fractional::operator TypeKind::Any() const { return std::make_shared<Fractional>(*this); }

Type::Base::Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
std::ostream &Type::operator<<(std::ostream &os, const Type::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Type::operator==(const Type::Base &l, const Type::Base &r) { 
  return *l.kind == *r.kind;
}
TypeKind::Any Type::kind(const Type::Any& x){ return select<&Type::Base::kind>(x); }

Type::Float16::Float16() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float16::id() const { return 0; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Float16 &x) {
  os << "Float16(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Float16 &, const Type::Float16 &) { return true; }
Type::Float16::operator Type::Any() const { return std::make_shared<Float16>(*this); }

Type::Float32::Float32() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float32::id() const { return 1; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Float32 &x) {
  os << "Float32(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Float32 &, const Type::Float32 &) { return true; }
Type::Float32::operator Type::Any() const { return std::make_shared<Float32>(*this); }

Type::Float64::Float64() noexcept : Type::Base(TypeKind::Fractional()) {}
uint32_t Type::Float64::id() const { return 2; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Float64 &x) {
  os << "Float64(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Float64 &, const Type::Float64 &) { return true; }
Type::Float64::operator Type::Any() const { return std::make_shared<Float64>(*this); }

Type::IntU8::IntU8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU8::id() const { return 3; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU8 &x) {
  os << "IntU8(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntU8 &, const Type::IntU8 &) { return true; }
Type::IntU8::operator Type::Any() const { return std::make_shared<IntU8>(*this); }

Type::IntU16::IntU16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU16::id() const { return 4; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU16 &x) {
  os << "IntU16(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntU16 &, const Type::IntU16 &) { return true; }
Type::IntU16::operator Type::Any() const { return std::make_shared<IntU16>(*this); }

Type::IntU32::IntU32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU32::id() const { return 5; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU32 &x) {
  os << "IntU32(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntU32 &, const Type::IntU32 &) { return true; }
Type::IntU32::operator Type::Any() const { return std::make_shared<IntU32>(*this); }

Type::IntU64::IntU64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntU64::id() const { return 6; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntU64 &x) {
  os << "IntU64(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntU64 &, const Type::IntU64 &) { return true; }
Type::IntU64::operator Type::Any() const { return std::make_shared<IntU64>(*this); }

Type::IntS8::IntS8() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS8::id() const { return 7; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS8 &x) {
  os << "IntS8(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntS8 &, const Type::IntS8 &) { return true; }
Type::IntS8::operator Type::Any() const { return std::make_shared<IntS8>(*this); }

Type::IntS16::IntS16() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS16::id() const { return 8; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS16 &x) {
  os << "IntS16(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntS16 &, const Type::IntS16 &) { return true; }
Type::IntS16::operator Type::Any() const { return std::make_shared<IntS16>(*this); }

Type::IntS32::IntS32() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS32::id() const { return 9; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS32 &x) {
  os << "IntS32(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntS32 &, const Type::IntS32 &) { return true; }
Type::IntS32::operator Type::Any() const { return std::make_shared<IntS32>(*this); }

Type::IntS64::IntS64() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::IntS64::id() const { return 10; };
std::ostream &Type::operator<<(std::ostream &os, const Type::IntS64 &x) {
  os << "IntS64(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::IntS64 &, const Type::IntS64 &) { return true; }
Type::IntS64::operator Type::Any() const { return std::make_shared<IntS64>(*this); }

Type::Nothing::Nothing() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Nothing::id() const { return 11; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Nothing &x) {
  os << "Nothing(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Nothing &, const Type::Nothing &) { return true; }
Type::Nothing::operator Type::Any() const { return std::make_shared<Nothing>(*this); }

Type::Unit0::Unit0() noexcept : Type::Base(TypeKind::None()) {}
uint32_t Type::Unit0::id() const { return 12; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Unit0 &x) {
  os << "Unit0(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Unit0 &, const Type::Unit0 &) { return true; }
Type::Unit0::operator Type::Any() const { return std::make_shared<Unit0>(*this); }

Type::Bool1::Bool1() noexcept : Type::Base(TypeKind::Integral()) {}
uint32_t Type::Bool1::id() const { return 13; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Bool1 &x) {
  os << "Bool1(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Bool1 &, const Type::Bool1 &) { return true; }
Type::Bool1::operator Type::Any() const { return std::make_shared<Bool1>(*this); }

Type::Struct::Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args, std::vector<Sym> parents) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)), tpeVars(std::move(tpeVars)), args(std::move(args)), parents(std::move(parents)) {}
uint32_t Type::Struct::id() const { return 14; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Struct &x) {
  os << "Struct(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.parents.empty()) {
    std::for_each(x.parents.begin(), std::prev(x.parents.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.parents.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool Type::operator==(const Type::Struct &l, const Type::Struct &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && l.parents == r.parents;
}
Type::Struct::operator Type::Any() const { return std::make_shared<Struct>(*this); }

Type::Ptr::Ptr(Type::Any component, std::optional<int32_t> length, TypeSpace::Any space) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)), length(std::move(length)), space(std::move(space)) {}
uint32_t Type::Ptr::id() const { return 15; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Ptr &x) {
  os << "Ptr(";
  os << x.component;
  os << ',';
  os << '{';
  if (x.length) {
    os << (*x.length);
  }
  os << '}';
  os << ',';
  os << x.space;
  os << ')';
  return os;
}
bool Type::operator==(const Type::Ptr &l, const Type::Ptr &r) { 
  return *l.component == *r.component && l.length == r.length && *l.space == *r.space;
}
Type::Ptr::operator Type::Any() const { return std::make_shared<Ptr>(*this); }

Type::Var::Var(std::string name) noexcept : Type::Base(TypeKind::None()), name(std::move(name)) {}
uint32_t Type::Var::id() const { return 16; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Var &x) {
  os << "Var(";
  os << '"' << x.name << '"';
  os << ')';
  return os;
}
bool Type::operator==(const Type::Var &l, const Type::Var &r) { 
  return l.name == r.name;
}
Type::Var::operator Type::Any() const { return std::make_shared<Var>(*this); }

Type::Exec::Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept : Type::Base(TypeKind::None()), tpeVars(std::move(tpeVars)), args(std::move(args)), rtn(std::move(rtn)) {}
uint32_t Type::Exec::id() const { return 17; };
std::ostream &Type::operator<<(std::ostream &os, const Type::Exec &x) {
  os << "Exec(";
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Type::operator==(const Type::Exec &l, const Type::Exec &r) { 
  return l.tpeVars == r.tpeVars && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}
Type::Exec::operator Type::Any() const { return std::make_shared<Exec>(*this); }

SourcePosition::SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept : file(std::move(file)), line(line), col(std::move(col)) {}
std::ostream &operator<<(std::ostream &os, const SourcePosition &x) {
  os << "SourcePosition(";
  os << '"' << x.file << '"';
  os << ',';
  os << x.line;
  os << ',';
  os << '{';
  if (x.col) {
    os << (*x.col);
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const SourcePosition &l, const SourcePosition &r) { 
  return l.file == r.file && l.line == r.line && l.col == r.col;
}

Term::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
std::ostream &Term::operator<<(std::ostream &os, const Term::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Term::operator==(const Term::Base &l, const Term::Base &r) { 
  return *l.tpe == *r.tpe;
}
Type::Any Term::tpe(const Term::Any& x){ return select<&Term::Base::tpe>(x); }

Term::Select::Select(std::vector<Named> init, Named last) noexcept : Term::Base(last.tpe), init(std::move(init)), last(std::move(last)) {}
uint32_t Term::Select::id() const { return 0; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Select &x) {
  os << "Select(";
  os << '{';
  if (!x.init.empty()) {
    std::for_each(x.init.begin(), std::prev(x.init.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.init.back();
  }
  os << '}';
  os << ',';
  os << x.last;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Select &l, const Term::Select &r) { 
  return l.init == r.init && l.last == r.last;
}
Term::Select::operator Term::Any() const { return std::make_shared<Select>(*this); }

Term::Poison::Poison(Type::Any t) noexcept : Term::Base(t), t(std::move(t)) {}
uint32_t Term::Poison::id() const { return 1; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Poison &x) {
  os << "Poison(";
  os << x.t;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Poison &l, const Term::Poison &r) { 
  return *l.t == *r.t;
}
Term::Poison::operator Term::Any() const { return std::make_shared<Poison>(*this); }

Term::Float16Const::Float16Const(float value) noexcept : Term::Base(Type::Float16()), value(value) {}
uint32_t Term::Float16Const::id() const { return 2; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Float16Const &x) {
  os << "Float16Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Float16Const &l, const Term::Float16Const &r) { 
  return l.value == r.value;
}
Term::Float16Const::operator Term::Any() const { return std::make_shared<Float16Const>(*this); }

Term::Float32Const::Float32Const(float value) noexcept : Term::Base(Type::Float32()), value(value) {}
uint32_t Term::Float32Const::id() const { return 3; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Float32Const &x) {
  os << "Float32Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Float32Const &l, const Term::Float32Const &r) { 
  return l.value == r.value;
}
Term::Float32Const::operator Term::Any() const { return std::make_shared<Float32Const>(*this); }

Term::Float64Const::Float64Const(double value) noexcept : Term::Base(Type::Float64()), value(value) {}
uint32_t Term::Float64Const::id() const { return 4; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Float64Const &x) {
  os << "Float64Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Float64Const &l, const Term::Float64Const &r) { 
  return l.value == r.value;
}
Term::Float64Const::operator Term::Any() const { return std::make_shared<Float64Const>(*this); }

Term::IntU8Const::IntU8Const(int8_t value) noexcept : Term::Base(Type::IntU8()), value(value) {}
uint32_t Term::IntU8Const::id() const { return 5; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU8Const &x) {
  os << "IntU8Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntU8Const &l, const Term::IntU8Const &r) { 
  return l.value == r.value;
}
Term::IntU8Const::operator Term::Any() const { return std::make_shared<IntU8Const>(*this); }

Term::IntU16Const::IntU16Const(uint16_t value) noexcept : Term::Base(Type::IntU16()), value(value) {}
uint32_t Term::IntU16Const::id() const { return 6; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU16Const &x) {
  os << "IntU16Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntU16Const &l, const Term::IntU16Const &r) { 
  return l.value == r.value;
}
Term::IntU16Const::operator Term::Any() const { return std::make_shared<IntU16Const>(*this); }

Term::IntU32Const::IntU32Const(int32_t value) noexcept : Term::Base(Type::IntU32()), value(value) {}
uint32_t Term::IntU32Const::id() const { return 7; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU32Const &x) {
  os << "IntU32Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntU32Const &l, const Term::IntU32Const &r) { 
  return l.value == r.value;
}
Term::IntU32Const::operator Term::Any() const { return std::make_shared<IntU32Const>(*this); }

Term::IntU64Const::IntU64Const(int64_t value) noexcept : Term::Base(Type::IntU64()), value(value) {}
uint32_t Term::IntU64Const::id() const { return 8; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntU64Const &x) {
  os << "IntU64Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntU64Const &l, const Term::IntU64Const &r) { 
  return l.value == r.value;
}
Term::IntU64Const::operator Term::Any() const { return std::make_shared<IntU64Const>(*this); }

Term::IntS8Const::IntS8Const(int8_t value) noexcept : Term::Base(Type::IntS8()), value(value) {}
uint32_t Term::IntS8Const::id() const { return 9; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS8Const &x) {
  os << "IntS8Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntS8Const &l, const Term::IntS8Const &r) { 
  return l.value == r.value;
}
Term::IntS8Const::operator Term::Any() const { return std::make_shared<IntS8Const>(*this); }

Term::IntS16Const::IntS16Const(int16_t value) noexcept : Term::Base(Type::IntS16()), value(value) {}
uint32_t Term::IntS16Const::id() const { return 10; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS16Const &x) {
  os << "IntS16Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntS16Const &l, const Term::IntS16Const &r) { 
  return l.value == r.value;
}
Term::IntS16Const::operator Term::Any() const { return std::make_shared<IntS16Const>(*this); }

Term::IntS32Const::IntS32Const(int32_t value) noexcept : Term::Base(Type::IntS32()), value(value) {}
uint32_t Term::IntS32Const::id() const { return 11; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS32Const &x) {
  os << "IntS32Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntS32Const &l, const Term::IntS32Const &r) { 
  return l.value == r.value;
}
Term::IntS32Const::operator Term::Any() const { return std::make_shared<IntS32Const>(*this); }

Term::IntS64Const::IntS64Const(int64_t value) noexcept : Term::Base(Type::IntS64()), value(value) {}
uint32_t Term::IntS64Const::id() const { return 12; };
std::ostream &Term::operator<<(std::ostream &os, const Term::IntS64Const &x) {
  os << "IntS64Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntS64Const &l, const Term::IntS64Const &r) { 
  return l.value == r.value;
}
Term::IntS64Const::operator Term::Any() const { return std::make_shared<IntS64Const>(*this); }

Term::Unit0Const::Unit0Const() noexcept : Term::Base(Type::Unit0()) {}
uint32_t Term::Unit0Const::id() const { return 13; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Unit0Const &x) {
  os << "Unit0Const(";
  os << ')';
  return os;
}
bool Term::operator==(const Term::Unit0Const &, const Term::Unit0Const &) { return true; }
Term::Unit0Const::operator Term::Any() const { return std::make_shared<Unit0Const>(*this); }

Term::Bool1Const::Bool1Const(bool value) noexcept : Term::Base(Type::Bool1()), value(value) {}
uint32_t Term::Bool1Const::id() const { return 14; };
std::ostream &Term::operator<<(std::ostream &os, const Term::Bool1Const &x) {
  os << "Bool1Const(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Bool1Const &l, const Term::Bool1Const &r) { 
  return l.value == r.value;
}
Term::Bool1Const::operator Term::Any() const { return std::make_shared<Bool1Const>(*this); }

TypeSpace::Base::Base() = default;
std::ostream &TypeSpace::operator<<(std::ostream &os, const TypeSpace::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool TypeSpace::operator==(const TypeSpace::Base &, const TypeSpace::Base &) { return true; }

TypeSpace::Global::Global() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Global::id() const { return 0; };
std::ostream &TypeSpace::operator<<(std::ostream &os, const TypeSpace::Global &x) {
  os << "Global(";
  os << ')';
  return os;
}
bool TypeSpace::operator==(const TypeSpace::Global &, const TypeSpace::Global &) { return true; }
TypeSpace::Global::operator TypeSpace::Any() const { return std::make_shared<Global>(*this); }

TypeSpace::Local::Local() noexcept : TypeSpace::Base() {}
uint32_t TypeSpace::Local::id() const { return 1; };
std::ostream &TypeSpace::operator<<(std::ostream &os, const TypeSpace::Local &x) {
  os << "Local(";
  os << ')';
  return os;
}
bool TypeSpace::operator==(const TypeSpace::Local &, const TypeSpace::Local &) { return true; }
TypeSpace::Local::operator TypeSpace::Any() const { return std::make_shared<Local>(*this); }

Overload::Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept : args(std::move(args)), rtn(std::move(rtn)) {}
std::ostream &operator<<(std::ostream &os, const Overload &x) {
  os << "Overload(";
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool operator==(const Overload &l, const Overload &r) { 
  return std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}

Spec::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
std::ostream &Spec::operator<<(std::ostream &os, const Spec::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Spec::operator==(const Spec::Base &l, const Spec::Base &r) { 
  return l.overloads == r.overloads && std::equal(l.terms.begin(), l.terms.end(), r.terms.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.tpe == *r.tpe;
}
std::vector<Overload> Spec::overloads(const Spec::Any& x){ return select<&Spec::Base::overloads>(x); }
std::vector<Term::Any> Spec::terms(const Spec::Any& x){ return select<&Spec::Base::terms>(x); }
Type::Any Spec::tpe(const Spec::Any& x){ return select<&Spec::Base::tpe>(x); }

Spec::Assert::Assert() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Nothing()) {}
uint32_t Spec::Assert::id() const { return 0; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::Assert &x) {
  os << "Assert(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::Assert &, const Spec::Assert &) { return true; }
Spec::Assert::operator Spec::Any() const { return std::make_shared<Assert>(*this); }

Spec::GpuBarrierGlobal::GpuBarrierGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierGlobal::id() const { return 1; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &x) {
  os << "GpuBarrierGlobal(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuBarrierGlobal &, const Spec::GpuBarrierGlobal &) { return true; }
Spec::GpuBarrierGlobal::operator Spec::Any() const { return std::make_shared<GpuBarrierGlobal>(*this); }

Spec::GpuBarrierLocal::GpuBarrierLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierLocal::id() const { return 2; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierLocal &x) {
  os << "GpuBarrierLocal(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuBarrierLocal &, const Spec::GpuBarrierLocal &) { return true; }
Spec::GpuBarrierLocal::operator Spec::Any() const { return std::make_shared<GpuBarrierLocal>(*this); }

Spec::GpuBarrierAll::GpuBarrierAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuBarrierAll::id() const { return 3; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuBarrierAll &x) {
  os << "GpuBarrierAll(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuBarrierAll &, const Spec::GpuBarrierAll &) { return true; }
Spec::GpuBarrierAll::operator Spec::Any() const { return std::make_shared<GpuBarrierAll>(*this); }

Spec::GpuFenceGlobal::GpuFenceGlobal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceGlobal::id() const { return 4; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceGlobal &x) {
  os << "GpuFenceGlobal(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuFenceGlobal &, const Spec::GpuFenceGlobal &) { return true; }
Spec::GpuFenceGlobal::operator Spec::Any() const { return std::make_shared<GpuFenceGlobal>(*this); }

Spec::GpuFenceLocal::GpuFenceLocal() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceLocal::id() const { return 5; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceLocal &x) {
  os << "GpuFenceLocal(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuFenceLocal &, const Spec::GpuFenceLocal &) { return true; }
Spec::GpuFenceLocal::operator Spec::Any() const { return std::make_shared<GpuFenceLocal>(*this); }

Spec::GpuFenceAll::GpuFenceAll() noexcept : Spec::Base({Overload({},Type::Unit0())}, {}, Type::Unit0()) {}
uint32_t Spec::GpuFenceAll::id() const { return 6; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuFenceAll &x) {
  os << "GpuFenceAll(";
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuFenceAll &, const Spec::GpuFenceAll &) { return true; }
Spec::GpuFenceAll::operator Spec::Any() const { return std::make_shared<GpuFenceAll>(*this); }

Spec::GpuGlobalIdx::GpuGlobalIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalIdx::id() const { return 7; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGlobalIdx &x) {
  os << "GpuGlobalIdx(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuGlobalIdx &l, const Spec::GpuGlobalIdx &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuGlobalIdx::operator Spec::Any() const { return std::make_shared<GpuGlobalIdx>(*this); }

Spec::GpuGlobalSize::GpuGlobalSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGlobalSize::id() const { return 8; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGlobalSize &x) {
  os << "GpuGlobalSize(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuGlobalSize &l, const Spec::GpuGlobalSize &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuGlobalSize::operator Spec::Any() const { return std::make_shared<GpuGlobalSize>(*this); }

Spec::GpuGroupIdx::GpuGroupIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupIdx::id() const { return 9; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGroupIdx &x) {
  os << "GpuGroupIdx(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuGroupIdx &l, const Spec::GpuGroupIdx &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuGroupIdx::operator Spec::Any() const { return std::make_shared<GpuGroupIdx>(*this); }

Spec::GpuGroupSize::GpuGroupSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuGroupSize::id() const { return 10; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuGroupSize &x) {
  os << "GpuGroupSize(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuGroupSize &l, const Spec::GpuGroupSize &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuGroupSize::operator Spec::Any() const { return std::make_shared<GpuGroupSize>(*this); }

Spec::GpuLocalIdx::GpuLocalIdx(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalIdx::id() const { return 11; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuLocalIdx &x) {
  os << "GpuLocalIdx(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuLocalIdx &l, const Spec::GpuLocalIdx &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuLocalIdx::operator Spec::Any() const { return std::make_shared<GpuLocalIdx>(*this); }

Spec::GpuLocalSize::GpuLocalSize(Term::Any dim) noexcept : Spec::Base({Overload({Type::IntU32()},Type::IntU32())}, {dim}, Type::IntU32()), dim(std::move(dim)) {}
uint32_t Spec::GpuLocalSize::id() const { return 12; };
std::ostream &Spec::operator<<(std::ostream &os, const Spec::GpuLocalSize &x) {
  os << "GpuLocalSize(";
  os << x.dim;
  os << ')';
  return os;
}
bool Spec::operator==(const Spec::GpuLocalSize &l, const Spec::GpuLocalSize &r) { 
  return *l.dim == *r.dim;
}
Spec::GpuLocalSize::operator Spec::Any() const { return std::make_shared<GpuLocalSize>(*this); }

Intr::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Intr::operator==(const Intr::Base &l, const Intr::Base &r) { 
  return l.overloads == r.overloads && std::equal(l.terms.begin(), l.terms.end(), r.terms.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.tpe == *r.tpe;
}
std::vector<Overload> Intr::overloads(const Intr::Any& x){ return select<&Intr::Base::overloads>(x); }
std::vector<Term::Any> Intr::terms(const Intr::Any& x){ return select<&Intr::Base::terms>(x); }
Type::Any Intr::tpe(const Intr::Any& x){ return select<&Intr::Base::tpe>(x); }

Intr::BNot::BNot(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8()},Type::IntU8()),Overload({Type::IntU16()},Type::IntU16()),Overload({Type::IntU32()},Type::IntU32()),Overload({Type::IntU64()},Type::IntU64()),Overload({Type::IntS8()},Type::IntS8()),Overload({Type::IntS16()},Type::IntS16()),Overload({Type::IntS32()},Type::IntS32()),Overload({Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::BNot::id() const { return 0; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BNot &x) {
  os << "BNot(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BNot &l, const Intr::BNot &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Intr::BNot::operator Intr::Any() const { return std::make_shared<BNot>(*this); }

Intr::LogicNot::LogicNot(Term::Any x) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x}, Type::Bool1()), x(std::move(x)) {}
uint32_t Intr::LogicNot::id() const { return 1; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicNot &x) {
  os << "LogicNot(";
  os << x.x;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicNot &l, const Intr::LogicNot &r) { 
  return *l.x == *r.x;
}
Intr::LogicNot::operator Intr::Any() const { return std::make_shared<LogicNot>(*this); }

Intr::Pos::Pos(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Pos::id() const { return 2; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Pos &x) {
  os << "Pos(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Pos &l, const Intr::Pos &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Intr::Pos::operator Intr::Any() const { return std::make_shared<Pos>(*this); }

Intr::Neg::Neg(Term::Any x, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Intr::Neg::id() const { return 3; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Neg &x) {
  os << "Neg(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Neg &l, const Intr::Neg &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Intr::Neg::operator Intr::Any() const { return std::make_shared<Neg>(*this); }

Intr::Add::Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Add::id() const { return 4; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Add &x) {
  os << "Add(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Add &l, const Intr::Add &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Add::operator Intr::Any() const { return std::make_shared<Add>(*this); }

Intr::Sub::Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Sub::id() const { return 5; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Sub &x) {
  os << "Sub(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Sub &l, const Intr::Sub &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Sub::operator Intr::Any() const { return std::make_shared<Sub>(*this); }

Intr::Mul::Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Mul::id() const { return 6; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Mul &x) {
  os << "Mul(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Mul &l, const Intr::Mul &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Mul::operator Intr::Any() const { return std::make_shared<Mul>(*this); }

Intr::Div::Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Div::id() const { return 7; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Div &x) {
  os << "Div(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Div &l, const Intr::Div &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Div::operator Intr::Any() const { return std::make_shared<Div>(*this); }

Intr::Rem::Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Rem::id() const { return 8; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Rem &x) {
  os << "Rem(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Rem &l, const Intr::Rem &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Rem::operator Intr::Any() const { return std::make_shared<Rem>(*this); }

Intr::Min::Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Min::id() const { return 9; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Min &x) {
  os << "Min(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Min &l, const Intr::Min &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Min::operator Intr::Any() const { return std::make_shared<Min>(*this); }

Intr::Max::Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::Max::id() const { return 10; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::Max &x) {
  os << "Max(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::Max &l, const Intr::Max &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::Max::operator Intr::Any() const { return std::make_shared<Max>(*this); }

Intr::BAnd::BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BAnd::id() const { return 11; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BAnd &x) {
  os << "BAnd(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BAnd &l, const Intr::BAnd &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BAnd::operator Intr::Any() const { return std::make_shared<BAnd>(*this); }

Intr::BOr::BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BOr::id() const { return 12; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BOr &x) {
  os << "BOr(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BOr &l, const Intr::BOr &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BOr::operator Intr::Any() const { return std::make_shared<BOr>(*this); }

Intr::BXor::BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BXor::id() const { return 13; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BXor &x) {
  os << "BXor(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BXor &l, const Intr::BXor &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BXor::operator Intr::Any() const { return std::make_shared<BXor>(*this); }

Intr::BSL::BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSL::id() const { return 14; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BSL &x) {
  os << "BSL(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BSL &l, const Intr::BSL &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BSL::operator Intr::Any() const { return std::make_shared<BSL>(*this); }

Intr::BSR::BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BSR::id() const { return 15; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BSR &x) {
  os << "BSR(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BSR &l, const Intr::BSR &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BSR::operator Intr::Any() const { return std::make_shared<BSR>(*this); }

Intr::BZSR::BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Intr::Base({Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Intr::BZSR::id() const { return 16; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::BZSR &x) {
  os << "BZSR(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::BZSR &l, const Intr::BZSR &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Intr::BZSR::operator Intr::Any() const { return std::make_shared<BZSR>(*this); }

Intr::LogicAnd::LogicAnd(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicAnd::id() const { return 17; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicAnd &x) {
  os << "LogicAnd(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicAnd &l, const Intr::LogicAnd &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicAnd::operator Intr::Any() const { return std::make_shared<LogicAnd>(*this); }

Intr::LogicOr::LogicOr(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Bool1(),Type::Bool1()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicOr::id() const { return 18; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicOr &x) {
  os << "LogicOr(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicOr &l, const Intr::LogicOr &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicOr::operator Intr::Any() const { return std::make_shared<LogicOr>(*this); }

Intr::LogicEq::LogicEq(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicEq::id() const { return 19; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicEq &x) {
  os << "LogicEq(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicEq &l, const Intr::LogicEq &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicEq::operator Intr::Any() const { return std::make_shared<LogicEq>(*this); }

Intr::LogicNeq::LogicNeq(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicNeq::id() const { return 20; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicNeq &x) {
  os << "LogicNeq(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicNeq &l, const Intr::LogicNeq &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicNeq::operator Intr::Any() const { return std::make_shared<LogicNeq>(*this); }

Intr::LogicLte::LogicLte(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLte::id() const { return 21; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicLte &x) {
  os << "LogicLte(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicLte &l, const Intr::LogicLte &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicLte::operator Intr::Any() const { return std::make_shared<LogicLte>(*this); }

Intr::LogicGte::LogicGte(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGte::id() const { return 22; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicGte &x) {
  os << "LogicGte(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicGte &l, const Intr::LogicGte &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicGte::operator Intr::Any() const { return std::make_shared<LogicGte>(*this); }

Intr::LogicLt::LogicLt(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicLt::id() const { return 23; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicLt &x) {
  os << "LogicLt(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicLt &l, const Intr::LogicLt &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicLt::operator Intr::Any() const { return std::make_shared<LogicLt>(*this); }

Intr::LogicGt::LogicGt(Term::Any x, Term::Any y) noexcept : Intr::Base({Overload({Type::Float16(),Type::Float16()},Type::Bool1()),Overload({Type::Float32(),Type::Float32()},Type::Bool1()),Overload({Type::Float64(),Type::Float64()},Type::Bool1()),Overload({Type::IntU8(),Type::IntU8()},Type::Bool1()),Overload({Type::IntU16(),Type::IntU16()},Type::Bool1()),Overload({Type::IntU32(),Type::IntU32()},Type::Bool1()),Overload({Type::IntU64(),Type::IntU64()},Type::Bool1()),Overload({Type::IntS8(),Type::IntS8()},Type::Bool1()),Overload({Type::IntS16(),Type::IntS16()},Type::Bool1()),Overload({Type::IntS32(),Type::IntS32()},Type::Bool1()),Overload({Type::IntS64(),Type::IntS64()},Type::Bool1())}, {x,y}, Type::Bool1()), x(std::move(x)), y(std::move(y)) {}
uint32_t Intr::LogicGt::id() const { return 24; };
std::ostream &Intr::operator<<(std::ostream &os, const Intr::LogicGt &x) {
  os << "LogicGt(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ')';
  return os;
}
bool Intr::operator==(const Intr::LogicGt &l, const Intr::LogicGt &r) { 
  return *l.x == *r.x && *l.y == *r.y;
}
Intr::LogicGt::operator Intr::Any() const { return std::make_shared<LogicGt>(*this); }

Math::Base::Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept : overloads(std::move(overloads)), terms(std::move(terms)), tpe(std::move(tpe)) {}
std::ostream &Math::operator<<(std::ostream &os, const Math::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Math::operator==(const Math::Base &l, const Math::Base &r) { 
  return l.overloads == r.overloads && std::equal(l.terms.begin(), l.terms.end(), r.terms.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.tpe == *r.tpe;
}
std::vector<Overload> Math::overloads(const Math::Any& x){ return select<&Math::Base::overloads>(x); }
std::vector<Term::Any> Math::terms(const Math::Any& x){ return select<&Math::Base::terms>(x); }
Type::Any Math::tpe(const Math::Any& x){ return select<&Math::Base::tpe>(x); }

Math::Abs::Abs(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64()),Overload({Type::IntU8(),Type::IntU8()},Type::IntU8()),Overload({Type::IntU16(),Type::IntU16()},Type::IntU16()),Overload({Type::IntU32(),Type::IntU32()},Type::IntU32()),Overload({Type::IntU64(),Type::IntU64()},Type::IntU64()),Overload({Type::IntS8(),Type::IntS8()},Type::IntS8()),Overload({Type::IntS16(),Type::IntS16()},Type::IntS16()),Overload({Type::IntS32(),Type::IntS32()},Type::IntS32()),Overload({Type::IntS64(),Type::IntS64()},Type::IntS64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Abs::id() const { return 0; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Abs &x) {
  os << "Abs(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Abs &l, const Math::Abs &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Abs::operator Math::Any() const { return std::make_shared<Abs>(*this); }

Math::Sin::Sin(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sin::id() const { return 1; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Sin &x) {
  os << "Sin(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Sin &l, const Math::Sin &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Sin::operator Math::Any() const { return std::make_shared<Sin>(*this); }

Math::Cos::Cos(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cos::id() const { return 2; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Cos &x) {
  os << "Cos(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Cos &l, const Math::Cos &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Cos::operator Math::Any() const { return std::make_shared<Cos>(*this); }

Math::Tan::Tan(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tan::id() const { return 3; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Tan &x) {
  os << "Tan(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Tan &l, const Math::Tan &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Tan::operator Math::Any() const { return std::make_shared<Tan>(*this); }

Math::Asin::Asin(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Asin::id() const { return 4; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Asin &x) {
  os << "Asin(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Asin &l, const Math::Asin &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Asin::operator Math::Any() const { return std::make_shared<Asin>(*this); }

Math::Acos::Acos(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Acos::id() const { return 5; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Acos &x) {
  os << "Acos(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Acos &l, const Math::Acos &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Acos::operator Math::Any() const { return std::make_shared<Acos>(*this); }

Math::Atan::Atan(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Atan::id() const { return 6; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Atan &x) {
  os << "Atan(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Atan &l, const Math::Atan &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Atan::operator Math::Any() const { return std::make_shared<Atan>(*this); }

Math::Sinh::Sinh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sinh::id() const { return 7; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Sinh &x) {
  os << "Sinh(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Sinh &l, const Math::Sinh &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Sinh::operator Math::Any() const { return std::make_shared<Sinh>(*this); }

Math::Cosh::Cosh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cosh::id() const { return 8; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Cosh &x) {
  os << "Cosh(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Cosh &l, const Math::Cosh &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Cosh::operator Math::Any() const { return std::make_shared<Cosh>(*this); }

Math::Tanh::Tanh(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Tanh::id() const { return 9; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Tanh &x) {
  os << "Tanh(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Tanh &l, const Math::Tanh &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Tanh::operator Math::Any() const { return std::make_shared<Tanh>(*this); }

Math::Signum::Signum(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Signum::id() const { return 10; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Signum &x) {
  os << "Signum(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Signum &l, const Math::Signum &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Signum::operator Math::Any() const { return std::make_shared<Signum>(*this); }

Math::Round::Round(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Round::id() const { return 11; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Round &x) {
  os << "Round(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Round &l, const Math::Round &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Round::operator Math::Any() const { return std::make_shared<Round>(*this); }

Math::Ceil::Ceil(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Ceil::id() const { return 12; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Ceil &x) {
  os << "Ceil(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Ceil &l, const Math::Ceil &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Ceil::operator Math::Any() const { return std::make_shared<Ceil>(*this); }

Math::Floor::Floor(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Floor::id() const { return 13; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Floor &x) {
  os << "Floor(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Floor &l, const Math::Floor &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Floor::operator Math::Any() const { return std::make_shared<Floor>(*this); }

Math::Rint::Rint(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Rint::id() const { return 14; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Rint &x) {
  os << "Rint(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Rint &l, const Math::Rint &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Rint::operator Math::Any() const { return std::make_shared<Rint>(*this); }

Math::Sqrt::Sqrt(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Sqrt::id() const { return 15; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Sqrt &x) {
  os << "Sqrt(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Sqrt &l, const Math::Sqrt &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Sqrt::operator Math::Any() const { return std::make_shared<Sqrt>(*this); }

Math::Cbrt::Cbrt(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Cbrt::id() const { return 16; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Cbrt &x) {
  os << "Cbrt(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Cbrt &l, const Math::Cbrt &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Cbrt::operator Math::Any() const { return std::make_shared<Cbrt>(*this); }

Math::Exp::Exp(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Exp::id() const { return 17; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Exp &x) {
  os << "Exp(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Exp &l, const Math::Exp &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Exp::operator Math::Any() const { return std::make_shared<Exp>(*this); }

Math::Expm1::Expm1(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Expm1::id() const { return 18; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Expm1 &x) {
  os << "Expm1(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Expm1 &l, const Math::Expm1 &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Expm1::operator Math::Any() const { return std::make_shared<Expm1>(*this); }

Math::Log::Log(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log::id() const { return 19; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Log &x) {
  os << "Log(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Log &l, const Math::Log &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Log::operator Math::Any() const { return std::make_shared<Log>(*this); }

Math::Log1p::Log1p(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log1p::id() const { return 20; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Log1p &x) {
  os << "Log1p(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Log1p &l, const Math::Log1p &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Log1p::operator Math::Any() const { return std::make_shared<Log1p>(*this); }

Math::Log10::Log10(Term::Any x, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16()},Type::Float16()),Overload({Type::Float32()},Type::Float32()),Overload({Type::Float64()},Type::Float64())}, {x}, rtn), x(std::move(x)), rtn(std::move(rtn)) {}
uint32_t Math::Log10::id() const { return 21; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Log10 &x) {
  os << "Log10(";
  os << x.x;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Log10 &l, const Math::Log10 &r) { 
  return *l.x == *r.x && *l.rtn == *r.rtn;
}
Math::Log10::operator Math::Any() const { return std::make_shared<Log10>(*this); }

Math::Pow::Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Pow::id() const { return 22; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Pow &x) {
  os << "Pow(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Pow &l, const Math::Pow &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Math::Pow::operator Math::Any() const { return std::make_shared<Pow>(*this); }

Math::Atan2::Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Atan2::id() const { return 23; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Atan2 &x) {
  os << "Atan2(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Atan2 &l, const Math::Atan2 &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Math::Atan2::operator Math::Any() const { return std::make_shared<Atan2>(*this); }

Math::Hypot::Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept : Math::Base({Overload({Type::Float16(),Type::Float16()},Type::Float16()),Overload({Type::Float32(),Type::Float32()},Type::Float32()),Overload({Type::Float64(),Type::Float64()},Type::Float64())}, {x,y}, rtn), x(std::move(x)), y(std::move(y)), rtn(std::move(rtn)) {}
uint32_t Math::Hypot::id() const { return 24; };
std::ostream &Math::operator<<(std::ostream &os, const Math::Hypot &x) {
  os << "Hypot(";
  os << x.x;
  os << ',';
  os << x.y;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Math::operator==(const Math::Hypot &l, const Math::Hypot &r) { 
  return *l.x == *r.x && *l.y == *r.y && *l.rtn == *r.rtn;
}
Math::Hypot::operator Math::Any() const { return std::make_shared<Hypot>(*this); }

Expr::Base::Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Expr::operator==(const Expr::Base &l, const Expr::Base &r) { 
  return *l.tpe == *r.tpe;
}
Type::Any Expr::tpe(const Expr::Any& x){ return select<&Expr::Base::tpe>(x); }

Expr::SpecOp::SpecOp(Spec::Any op) noexcept : Expr::Base(Spec::tpe(op)), op(std::move(op)) {}
uint32_t Expr::SpecOp::id() const { return 0; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::SpecOp &x) {
  os << "SpecOp(";
  os << x.op;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::SpecOp &l, const Expr::SpecOp &r) { 
  return *l.op == *r.op;
}
Expr::SpecOp::operator Expr::Any() const { return std::make_shared<SpecOp>(*this); }

Expr::MathOp::MathOp(Math::Any op) noexcept : Expr::Base(Math::tpe(op)), op(std::move(op)) {}
uint32_t Expr::MathOp::id() const { return 1; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::MathOp &x) {
  os << "MathOp(";
  os << x.op;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::MathOp &l, const Expr::MathOp &r) { 
  return *l.op == *r.op;
}
Expr::MathOp::operator Expr::Any() const { return std::make_shared<MathOp>(*this); }

Expr::IntrOp::IntrOp(Intr::Any op) noexcept : Expr::Base(Intr::tpe(op)), op(std::move(op)) {}
uint32_t Expr::IntrOp::id() const { return 2; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::IntrOp &x) {
  os << "IntrOp(";
  os << x.op;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::IntrOp &l, const Expr::IntrOp &r) { 
  return *l.op == *r.op;
}
Expr::IntrOp::operator Expr::Any() const { return std::make_shared<IntrOp>(*this); }

Expr::Cast::Cast(Term::Any from, Type::Any as) noexcept : Expr::Base(as), from(std::move(from)), as(std::move(as)) {}
uint32_t Expr::Cast::id() const { return 3; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Cast &x) {
  os << "Cast(";
  os << x.from;
  os << ',';
  os << x.as;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Cast &l, const Expr::Cast &r) { 
  return *l.from == *r.from && *l.as == *r.as;
}
Expr::Cast::operator Expr::Any() const { return std::make_shared<Cast>(*this); }

Expr::Alias::Alias(Term::Any ref) noexcept : Expr::Base(Term::tpe(ref)), ref(std::move(ref)) {}
uint32_t Expr::Alias::id() const { return 4; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alias &x) {
  os << "Alias(";
  os << x.ref;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Alias &l, const Expr::Alias &r) { 
  return *l.ref == *r.ref;
}
Expr::Alias::operator Expr::Any() const { return std::make_shared<Alias>(*this); }

Expr::Index::Index(Term::Any lhs, Term::Any idx, Type::Any component) noexcept : Expr::Base(component), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::Index::id() const { return 5; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Index &x) {
  os << "Index(";
  os << x.lhs;
  os << ',';
  os << x.idx;
  os << ',';
  os << x.component;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Index &l, const Expr::Index &r) { 
  return *l.lhs == *r.lhs && *l.idx == *r.idx && *l.component == *r.component;
}
Expr::Index::operator Expr::Any() const { return std::make_shared<Index>(*this); }

Expr::RefTo::RefTo(Term::Any lhs, std::optional<Term::Any> idx, Type::Any component) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
uint32_t Expr::RefTo::id() const { return 6; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::RefTo &x) {
  os << "RefTo(";
  os << x.lhs;
  os << ',';
  os << '{';
  if (x.idx) {
    os << (*x.idx);
  }
  os << '}';
  os << ',';
  os << x.component;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::RefTo &l, const Expr::RefTo &r) { 
  return *l.lhs == *r.lhs && ( (!l.idx && !r.idx) || (l.idx && r.idx && **l.idx == **r.idx) ) && *l.component == *r.component;
}
Expr::RefTo::operator Expr::Any() const { return std::make_shared<RefTo>(*this); }

Expr::Alloc::Alloc(Type::Any component, Term::Any size) noexcept : Expr::Base(Type::Ptr(component,{},TypeSpace::Global())), component(std::move(component)), size(std::move(size)) {}
uint32_t Expr::Alloc::id() const { return 7; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alloc &x) {
  os << "Alloc(";
  os << x.component;
  os << ',';
  os << x.size;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Alloc &l, const Expr::Alloc &r) { 
  return *l.component == *r.component && *l.size == *r.size;
}
Expr::Alloc::operator Expr::Any() const { return std::make_shared<Alloc>(*this); }

Expr::Invoke::Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, std::vector<Term::Any> captures, Type::Any rtn) noexcept : Expr::Base(rtn), name(std::move(name)), tpeArgs(std::move(tpeArgs)), receiver(std::move(receiver)), args(std::move(args)), captures(std::move(captures)), rtn(std::move(rtn)) {}
uint32_t Expr::Invoke::id() const { return 8; };
std::ostream &Expr::operator<<(std::ostream &os, const Expr::Invoke &x) {
  os << "Invoke(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeArgs.empty()) {
    std::for_each(x.tpeArgs.begin(), std::prev(x.tpeArgs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.tpeArgs.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (x.receiver) {
    os << (*x.receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.captures.empty()) {
    std::for_each(x.captures.begin(), std::prev(x.captures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.captures.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Invoke &l, const Expr::Invoke &r) { 
  return l.name == r.name && std::equal(l.tpeArgs.begin(), l.tpeArgs.end(), r.tpeArgs.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && ( (!l.receiver && !r.receiver) || (l.receiver && r.receiver && **l.receiver == **r.receiver) ) && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.captures.begin(), l.captures.end(), r.captures.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}
Expr::Invoke::operator Expr::Any() const { return std::make_shared<Invoke>(*this); }

Stmt::Base::Base() = default;
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Stmt::operator==(const Stmt::Base &, const Stmt::Base &) { return true; }

Stmt::Block::Block(std::vector<Stmt::Any> stmts) noexcept : Stmt::Base(), stmts(std::move(stmts)) {}
uint32_t Stmt::Block::id() const { return 0; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Block &x) {
  os << "Block(";
  os << '{';
  if (!x.stmts.empty()) {
    std::for_each(x.stmts.begin(), std::prev(x.stmts.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.stmts.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Block &l, const Stmt::Block &r) { 
  return std::equal(l.stmts.begin(), l.stmts.end(), r.stmts.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}
Stmt::Block::operator Stmt::Any() const { return std::make_shared<Block>(*this); }

Stmt::Comment::Comment(std::string value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Comment::id() const { return 1; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Comment &x) {
  os << "Comment(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Comment &l, const Stmt::Comment &r) { 
  return l.value == r.value;
}
Stmt::Comment::operator Stmt::Any() const { return std::make_shared<Comment>(*this); }

Stmt::Var::Var(Named name, std::optional<Expr::Any> expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
uint32_t Stmt::Var::id() const { return 2; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Var &x) {
  os << "Var(";
  os << x.name;
  os << ',';
  os << '{';
  if (x.expr) {
    os << (*x.expr);
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Var &l, const Stmt::Var &r) { 
  return l.name == r.name && ( (!l.expr && !r.expr) || (l.expr && r.expr && **l.expr == **r.expr) );
}
Stmt::Var::operator Stmt::Any() const { return std::make_shared<Var>(*this); }

Stmt::Mut::Mut(Term::Any name, Expr::Any expr, bool copy) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)), copy(copy) {}
uint32_t Stmt::Mut::id() const { return 3; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Mut &x) {
  os << "Mut(";
  os << x.name;
  os << ',';
  os << x.expr;
  os << ',';
  os << x.copy;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Mut &l, const Stmt::Mut &r) { 
  return *l.name == *r.name && *l.expr == *r.expr && l.copy == r.copy;
}
Stmt::Mut::operator Stmt::Any() const { return std::make_shared<Mut>(*this); }

Stmt::Update::Update(Term::Any lhs, Term::Any idx, Term::Any value) noexcept : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
uint32_t Stmt::Update::id() const { return 4; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Update &x) {
  os << "Update(";
  os << x.lhs;
  os << ',';
  os << x.idx;
  os << ',';
  os << x.value;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Update &l, const Stmt::Update &r) { 
  return *l.lhs == *r.lhs && *l.idx == *r.idx && *l.value == *r.value;
}
Stmt::Update::operator Stmt::Any() const { return std::make_shared<Update>(*this); }

Stmt::While::While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), tests(std::move(tests)), cond(std::move(cond)), body(std::move(body)) {}
uint32_t Stmt::While::id() const { return 5; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::While &x) {
  os << "While(";
  os << '{';
  if (!x.tests.empty()) {
    std::for_each(x.tests.begin(), std::prev(x.tests.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.tests.back();
  }
  os << '}';
  os << ',';
  os << x.cond;
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.body.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::While &l, const Stmt::While &r) { 
  return std::equal(l.tests.begin(), l.tests.end(), r.tests.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.cond == *r.cond && std::equal(l.body.begin(), l.body.end(), r.body.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}
Stmt::While::operator Stmt::Any() const { return std::make_shared<While>(*this); }

Stmt::Break::Break() noexcept : Stmt::Base() {}
uint32_t Stmt::Break::id() const { return 6; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Break &x) {
  os << "Break(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Break &, const Stmt::Break &) { return true; }
Stmt::Break::operator Stmt::Any() const { return std::make_shared<Break>(*this); }

Stmt::Cont::Cont() noexcept : Stmt::Base() {}
uint32_t Stmt::Cont::id() const { return 7; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cont &x) {
  os << "Cont(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Cont &, const Stmt::Cont &) { return true; }
Stmt::Cont::operator Stmt::Any() const { return std::make_shared<Cont>(*this); }

Stmt::Cond::Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
uint32_t Stmt::Cond::id() const { return 8; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cond &x) {
  os << "Cond(";
  os << x.cond;
  os << ',';
  os << '{';
  if (!x.trueBr.empty()) {
    std::for_each(x.trueBr.begin(), std::prev(x.trueBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.trueBr.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.falseBr.empty()) {
    std::for_each(x.falseBr.begin(), std::prev(x.falseBr.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.falseBr.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Cond &l, const Stmt::Cond &r) { 
  return *l.cond == *r.cond && std::equal(l.trueBr.begin(), l.trueBr.end(), r.trueBr.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.falseBr.begin(), l.falseBr.end(), r.falseBr.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}
Stmt::Cond::operator Stmt::Any() const { return std::make_shared<Cond>(*this); }

Stmt::Return::Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
uint32_t Stmt::Return::id() const { return 9; };
std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Return &x) {
  os << "Return(";
  os << x.value;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Return &l, const Stmt::Return &r) { 
  return *l.value == *r.value;
}
Stmt::Return::operator Stmt::Any() const { return std::make_shared<Return>(*this); }

StructMember::StructMember(Named named, bool isMutable) noexcept : named(std::move(named)), isMutable(isMutable) {}
std::ostream &operator<<(std::ostream &os, const StructMember &x) {
  os << "StructMember(";
  os << x.named;
  os << ',';
  os << x.isMutable;
  os << ')';
  return os;
}
bool operator==(const StructMember &l, const StructMember &r) { 
  return l.named == r.named && l.isMutable == r.isMutable;
}

StructDef::StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<StructMember> members, std::vector<Sym> parents) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), members(std::move(members)), parents(std::move(parents)) {}
std::ostream &operator<<(std::ostream &os, const StructDef &x) {
  os << "StructDef(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.members.empty()) {
    std::for_each(x.members.begin(), std::prev(x.members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.members.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.parents.empty()) {
    std::for_each(x.parents.begin(), std::prev(x.parents.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.parents.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const StructDef &l, const StructDef &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && l.members == r.members && l.parents == r.parents;
}

Signature::Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)) {}
std::ostream &operator<<(std::ostream &os, const Signature &x) {
  os << "Signature(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (x.receiver) {
    os << (*x.receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.moduleCaptures.empty()) {
    std::for_each(x.moduleCaptures.begin(), std::prev(x.moduleCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.moduleCaptures.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.termCaptures.empty()) {
    std::for_each(x.termCaptures.begin(), std::prev(x.termCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.termCaptures.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool operator==(const Signature &l, const Signature &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && ( (!l.receiver && !r.receiver) || (l.receiver && r.receiver && **l.receiver == **r.receiver) ) && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.moduleCaptures.begin(), l.moduleCaptures.end(), r.moduleCaptures.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.termCaptures.begin(), l.termCaptures.end(), r.termCaptures.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}

InvokeSignature::InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> captures, Type::Any rtn) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), captures(std::move(captures)), rtn(std::move(rtn)) {}
std::ostream &operator<<(std::ostream &os, const InvokeSignature &x) {
  os << "InvokeSignature(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.tpeVars.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (x.receiver) {
    os << (*x.receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.captures.empty()) {
    std::for_each(x.captures.begin(), std::prev(x.captures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.captures.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool operator==(const InvokeSignature &l, const InvokeSignature &r) { 
  return l.name == r.name && std::equal(l.tpeVars.begin(), l.tpeVars.end(), r.tpeVars.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && ( (!l.receiver && !r.receiver) || (l.receiver && r.receiver && **l.receiver == **r.receiver) ) && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && std::equal(l.captures.begin(), l.captures.end(), r.captures.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}

FunctionKind::Base::Base() = default;
std::ostream &FunctionKind::operator<<(std::ostream &os, const FunctionKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool FunctionKind::operator==(const FunctionKind::Base &, const FunctionKind::Base &) { return true; }

FunctionKind::Internal::Internal() noexcept : FunctionKind::Base() {}
uint32_t FunctionKind::Internal::id() const { return 0; };
std::ostream &FunctionKind::operator<<(std::ostream &os, const FunctionKind::Internal &x) {
  os << "Internal(";
  os << ')';
  return os;
}
bool FunctionKind::operator==(const FunctionKind::Internal &, const FunctionKind::Internal &) { return true; }
FunctionKind::Internal::operator FunctionKind::Any() const { return std::make_shared<Internal>(*this); }

FunctionKind::Exported::Exported() noexcept : FunctionKind::Base() {}
uint32_t FunctionKind::Exported::id() const { return 1; };
std::ostream &FunctionKind::operator<<(std::ostream &os, const FunctionKind::Exported &x) {
  os << "Exported(";
  os << ')';
  return os;
}
bool FunctionKind::operator==(const FunctionKind::Exported &, const FunctionKind::Exported &) { return true; }
FunctionKind::Exported::operator FunctionKind::Any() const { return std::make_shared<Exported>(*this); }

FunctionAttr::Base::Base() = default;
std::ostream &FunctionAttr::operator<<(std::ostream &os, const FunctionAttr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool FunctionAttr::operator==(const FunctionAttr::Base &, const FunctionAttr::Base &) { return true; }

FunctionAttr::FPRelaxed::FPRelaxed() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPRelaxed::id() const { return 0; };
std::ostream &FunctionAttr::operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &x) {
  os << "FPRelaxed(";
  os << ')';
  return os;
}
bool FunctionAttr::operator==(const FunctionAttr::FPRelaxed &, const FunctionAttr::FPRelaxed &) { return true; }
FunctionAttr::FPRelaxed::operator FunctionAttr::Any() const { return std::make_shared<FPRelaxed>(*this); }

FunctionAttr::FPStrict::FPStrict() noexcept : FunctionAttr::Base() {}
uint32_t FunctionAttr::FPStrict::id() const { return 1; };
std::ostream &FunctionAttr::operator<<(std::ostream &os, const FunctionAttr::FPStrict &x) {
  os << "FPStrict(";
  os << ')';
  return os;
}
bool FunctionAttr::operator==(const FunctionAttr::FPStrict &, const FunctionAttr::FPStrict &) { return true; }
FunctionAttr::FPStrict::operator FunctionAttr::Any() const { return std::make_shared<FPStrict>(*this); }

Arg::Arg(Named named, std::optional<SourcePosition> pos) noexcept : named(std::move(named)), pos(std::move(pos)) {}
std::ostream &operator<<(std::ostream &os, const Arg &x) {
  os << "Arg(";
  os << x.named;
  os << ',';
  os << '{';
  if (x.pos) {
    os << (*x.pos);
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const Arg &l, const Arg &r) { 
  return l.named == r.named && l.pos == r.pos;
}

Function::Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args, std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), moduleCaptures(std::move(moduleCaptures)), termCaptures(std::move(termCaptures)), rtn(std::move(rtn)), body(std::move(body)) {}
std::ostream &operator<<(std::ostream &os, const Function &x) {
  os << "Function(";
  os << x.name;
  os << ',';
  os << '{';
  if (!x.tpeVars.empty()) {
    std::for_each(x.tpeVars.begin(), std::prev(x.tpeVars.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.tpeVars.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (x.receiver) {
    os << (*x.receiver);
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.moduleCaptures.empty()) {
    std::for_each(x.moduleCaptures.begin(), std::prev(x.moduleCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.moduleCaptures.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.termCaptures.empty()) {
    std::for_each(x.termCaptures.begin(), std::prev(x.termCaptures.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.termCaptures.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ',';
  os << '{';
  if (!x.body.empty()) {
    std::for_each(x.body.begin(), std::prev(x.body.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.body.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const Function &l, const Function &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && l.receiver == r.receiver && l.args == r.args && l.moduleCaptures == r.moduleCaptures && l.termCaptures == r.termCaptures && *l.rtn == *r.rtn && std::equal(l.body.begin(), l.body.end(), r.body.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

Program::Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept : entry(std::move(entry)), functions(std::move(functions)), defs(std::move(defs)) {}
std::ostream &operator<<(std::ostream &os, const Program &x) {
  os << "Program(";
  os << x.entry;
  os << ',';
  os << '{';
  if (!x.functions.empty()) {
    std::for_each(x.functions.begin(), std::prev(x.functions.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.functions.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.defs.empty()) {
    std::for_each(x.defs.begin(), std::prev(x.defs.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.defs.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const Program &l, const Program &r) { 
  return l.entry == r.entry && l.functions == r.functions && l.defs == r.defs;
}

CompileLayoutMember::CompileLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept : name(std::move(name)), offsetInBytes(offsetInBytes), sizeInBytes(sizeInBytes) {}
std::ostream &operator<<(std::ostream &os, const CompileLayoutMember &x) {
  os << "CompileLayoutMember(";
  os << x.name;
  os << ',';
  os << x.offsetInBytes;
  os << ',';
  os << x.sizeInBytes;
  os << ')';
  return os;
}
bool operator==(const CompileLayoutMember &l, const CompileLayoutMember &r) { 
  return l.name == r.name && l.offsetInBytes == r.offsetInBytes && l.sizeInBytes == r.sizeInBytes;
}

CompileLayout::CompileLayout(Sym name, int64_t sizeInBytes, int64_t alignment, std::vector<CompileLayoutMember> members) noexcept : name(std::move(name)), sizeInBytes(sizeInBytes), alignment(alignment), members(std::move(members)) {}
std::ostream &operator<<(std::ostream &os, const CompileLayout &x) {
  os << "CompileLayout(";
  os << x.name;
  os << ',';
  os << x.sizeInBytes;
  os << ',';
  os << x.alignment;
  os << ',';
  os << '{';
  if (!x.members.empty()) {
    std::for_each(x.members.begin(), std::prev(x.members.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.members.back();
  }
  os << '}';
  os << ')';
  return os;
}
bool operator==(const CompileLayout &l, const CompileLayout &r) { 
  return l.name == r.name && l.sizeInBytes == r.sizeInBytes && l.alignment == r.alignment && l.members == r.members;
}

CompileEvent::CompileEvent(int64_t epochMillis, int64_t elapsedNanos, std::string name, std::string data) noexcept : epochMillis(epochMillis), elapsedNanos(elapsedNanos), name(std::move(name)), data(std::move(data)) {}
std::ostream &operator<<(std::ostream &os, const CompileEvent &x) {
  os << "CompileEvent(";
  os << x.epochMillis;
  os << ',';
  os << x.elapsedNanos;
  os << ',';
  os << '"' << x.name << '"';
  os << ',';
  os << '"' << x.data << '"';
  os << ')';
  return os;
}
bool operator==(const CompileEvent &l, const CompileEvent &r) { 
  return l.epochMillis == r.epochMillis && l.elapsedNanos == r.elapsedNanos && l.name == r.name && l.data == r.data;
}

CompileResult::CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<CompileLayout> layouts, std::string messages) noexcept : binary(std::move(binary)), features(std::move(features)), events(std::move(events)), layouts(std::move(layouts)), messages(std::move(messages)) {}
std::ostream &operator<<(std::ostream &os, const CompileResult &x) {
  os << "CompileResult(";
  os << '{';
  if (x.binary) {
    os << '{';
  if (!(*x.binary).empty()) {
    std::for_each((*x.binary).begin(), std::prev((*x.binary).end()), [&os](auto &&x) { os << x; os << ','; });
    os << (*x.binary).back();
  }
  os << '}';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.features.empty()) {
    std::for_each(x.features.begin(), std::prev(x.features.end()), [&os](auto &&x) { os << '"' << x << '"'; os << ','; });
    os << '"' << x.features.back() << '"';
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.events.empty()) {
    std::for_each(x.events.begin(), std::prev(x.events.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.events.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.layouts.empty()) {
    std::for_each(x.layouts.begin(), std::prev(x.layouts.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.layouts.back();
  }
  os << '}';
  os << ',';
  os << '"' << x.messages << '"';
  os << ')';
  return os;
}
bool operator==(const CompileResult &l, const CompileResult &r) { 
  return l.binary == r.binary && l.features == r.features && l.events == r.events && l.layouts == r.layouts && l.messages == r.messages;
}

} // namespace polyregion::polyast


std::size_t std::hash<polyregion::polyast::Sym>::operator()(const polyregion::polyast::Sym &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.fqn)>()(x.fqn);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Named>::operator()(const polyregion::polyast::Named &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.symbol)>()(x.symbol);
  seed ^= std::hash<decltype(x.tpe)>()(x.tpe) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeKind::None>::operator()(const polyregion::polyast::TypeKind::None &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeKind::None");
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeKind::Ref>::operator()(const polyregion::polyast::TypeKind::Ref &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeKind::Ref");
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeKind::Integral>::operator()(const polyregion::polyast::TypeKind::Integral &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeKind::Integral");
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeKind::Fractional>::operator()(const polyregion::polyast::TypeKind::Fractional &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeKind::Fractional");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Float16>::operator()(const polyregion::polyast::Type::Float16 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Float16");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Float32>::operator()(const polyregion::polyast::Type::Float32 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Float32");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Float64>::operator()(const polyregion::polyast::Type::Float64 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Float64");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntU8>::operator()(const polyregion::polyast::Type::IntU8 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntU8");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntU16>::operator()(const polyregion::polyast::Type::IntU16 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntU16");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntU32>::operator()(const polyregion::polyast::Type::IntU32 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntU32");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntU64>::operator()(const polyregion::polyast::Type::IntU64 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntU64");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntS8>::operator()(const polyregion::polyast::Type::IntS8 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntS8");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntS16>::operator()(const polyregion::polyast::Type::IntS16 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntS16");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntS32>::operator()(const polyregion::polyast::Type::IntS32 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntS32");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::IntS64>::operator()(const polyregion::polyast::Type::IntS64 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::IntS64");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Nothing>::operator()(const polyregion::polyast::Type::Nothing &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Nothing");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Unit0>::operator()(const polyregion::polyast::Type::Unit0 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Unit0");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Bool1>::operator()(const polyregion::polyast::Type::Bool1 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Bool1");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Struct>::operator()(const polyregion::polyast::Type::Struct &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.parents)>()(x.parents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Ptr>::operator()(const polyregion::polyast::Type::Ptr &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.component)>()(x.component);
  seed ^= std::hash<decltype(x.length)>()(x.length) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.space)>()(x.space) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Var>::operator()(const polyregion::polyast::Type::Var &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Exec>::operator()(const polyregion::polyast::Type::Exec &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.tpeVars)>()(x.tpeVars);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::SourcePosition>::operator()(const polyregion::polyast::SourcePosition &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.file)>()(x.file);
  seed ^= std::hash<decltype(x.line)>()(x.line) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.col)>()(x.col) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Select>::operator()(const polyregion::polyast::Term::Select &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.init)>()(x.init);
  seed ^= std::hash<decltype(x.last)>()(x.last) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Poison>::operator()(const polyregion::polyast::Term::Poison &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.t)>()(x.t);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Float16Const>::operator()(const polyregion::polyast::Term::Float16Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Float32Const>::operator()(const polyregion::polyast::Term::Float32Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Float64Const>::operator()(const polyregion::polyast::Term::Float64Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntU8Const>::operator()(const polyregion::polyast::Term::IntU8Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntU16Const>::operator()(const polyregion::polyast::Term::IntU16Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntU32Const>::operator()(const polyregion::polyast::Term::IntU32Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntU64Const>::operator()(const polyregion::polyast::Term::IntU64Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntS8Const>::operator()(const polyregion::polyast::Term::IntS8Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntS16Const>::operator()(const polyregion::polyast::Term::IntS16Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntS32Const>::operator()(const polyregion::polyast::Term::IntS32Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntS64Const>::operator()(const polyregion::polyast::Term::IntS64Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Unit0Const>::operator()(const polyregion::polyast::Term::Unit0Const &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Term::Unit0Const");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::Bool1Const>::operator()(const polyregion::polyast::Term::Bool1Const &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeSpace::Global>::operator()(const polyregion::polyast::TypeSpace::Global &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeSpace::Global");
  return seed;
}
std::size_t std::hash<polyregion::polyast::TypeSpace::Local>::operator()(const polyregion::polyast::TypeSpace::Local &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::TypeSpace::Local");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Overload>::operator()(const polyregion::polyast::Overload &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.args)>()(x.args);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::Assert>::operator()(const polyregion::polyast::Spec::Assert &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::Assert");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierGlobal>::operator()(const polyregion::polyast::Spec::GpuBarrierGlobal &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuBarrierGlobal");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierLocal>::operator()(const polyregion::polyast::Spec::GpuBarrierLocal &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuBarrierLocal");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuBarrierAll>::operator()(const polyregion::polyast::Spec::GpuBarrierAll &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuBarrierAll");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceGlobal>::operator()(const polyregion::polyast::Spec::GpuFenceGlobal &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuFenceGlobal");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceLocal>::operator()(const polyregion::polyast::Spec::GpuFenceLocal &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuFenceLocal");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuFenceAll>::operator()(const polyregion::polyast::Spec::GpuFenceAll &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Spec::GpuFenceAll");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuGlobalIdx>::operator()(const polyregion::polyast::Spec::GpuGlobalIdx &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuGlobalSize>::operator()(const polyregion::polyast::Spec::GpuGlobalSize &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuGroupIdx>::operator()(const polyregion::polyast::Spec::GpuGroupIdx &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuGroupSize>::operator()(const polyregion::polyast::Spec::GpuGroupSize &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuLocalIdx>::operator()(const polyregion::polyast::Spec::GpuLocalIdx &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Spec::GpuLocalSize>::operator()(const polyregion::polyast::Spec::GpuLocalSize &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.dim)>()(x.dim);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BNot>::operator()(const polyregion::polyast::Intr::BNot &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicNot>::operator()(const polyregion::polyast::Intr::LogicNot &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Pos>::operator()(const polyregion::polyast::Intr::Pos &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Neg>::operator()(const polyregion::polyast::Intr::Neg &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Add>::operator()(const polyregion::polyast::Intr::Add &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Sub>::operator()(const polyregion::polyast::Intr::Sub &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Mul>::operator()(const polyregion::polyast::Intr::Mul &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Div>::operator()(const polyregion::polyast::Intr::Div &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Rem>::operator()(const polyregion::polyast::Intr::Rem &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Min>::operator()(const polyregion::polyast::Intr::Min &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::Max>::operator()(const polyregion::polyast::Intr::Max &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BAnd>::operator()(const polyregion::polyast::Intr::BAnd &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BOr>::operator()(const polyregion::polyast::Intr::BOr &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BXor>::operator()(const polyregion::polyast::Intr::BXor &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BSL>::operator()(const polyregion::polyast::Intr::BSL &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BSR>::operator()(const polyregion::polyast::Intr::BSR &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::BZSR>::operator()(const polyregion::polyast::Intr::BZSR &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicAnd>::operator()(const polyregion::polyast::Intr::LogicAnd &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicOr>::operator()(const polyregion::polyast::Intr::LogicOr &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicEq>::operator()(const polyregion::polyast::Intr::LogicEq &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicNeq>::operator()(const polyregion::polyast::Intr::LogicNeq &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicLte>::operator()(const polyregion::polyast::Intr::LogicLte &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicGte>::operator()(const polyregion::polyast::Intr::LogicGte &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicLt>::operator()(const polyregion::polyast::Intr::LogicLt &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Intr::LogicGt>::operator()(const polyregion::polyast::Intr::LogicGt &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Abs>::operator()(const polyregion::polyast::Math::Abs &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Sin>::operator()(const polyregion::polyast::Math::Sin &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Cos>::operator()(const polyregion::polyast::Math::Cos &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Tan>::operator()(const polyregion::polyast::Math::Tan &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Asin>::operator()(const polyregion::polyast::Math::Asin &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Acos>::operator()(const polyregion::polyast::Math::Acos &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Atan>::operator()(const polyregion::polyast::Math::Atan &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Sinh>::operator()(const polyregion::polyast::Math::Sinh &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Cosh>::operator()(const polyregion::polyast::Math::Cosh &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Tanh>::operator()(const polyregion::polyast::Math::Tanh &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Signum>::operator()(const polyregion::polyast::Math::Signum &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Round>::operator()(const polyregion::polyast::Math::Round &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Ceil>::operator()(const polyregion::polyast::Math::Ceil &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Floor>::operator()(const polyregion::polyast::Math::Floor &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Rint>::operator()(const polyregion::polyast::Math::Rint &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Sqrt>::operator()(const polyregion::polyast::Math::Sqrt &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Cbrt>::operator()(const polyregion::polyast::Math::Cbrt &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Exp>::operator()(const polyregion::polyast::Math::Exp &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Expm1>::operator()(const polyregion::polyast::Math::Expm1 &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Log>::operator()(const polyregion::polyast::Math::Log &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Log1p>::operator()(const polyregion::polyast::Math::Log1p &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Log10>::operator()(const polyregion::polyast::Math::Log10 &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Pow>::operator()(const polyregion::polyast::Math::Pow &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Atan2>::operator()(const polyregion::polyast::Math::Atan2 &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Math::Hypot>::operator()(const polyregion::polyast::Math::Hypot &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.x)>()(x.x);
  seed ^= std::hash<decltype(x.y)>()(x.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::SpecOp>::operator()(const polyregion::polyast::Expr::SpecOp &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.op)>()(x.op);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::MathOp>::operator()(const polyregion::polyast::Expr::MathOp &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.op)>()(x.op);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::IntrOp>::operator()(const polyregion::polyast::Expr::IntrOp &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.op)>()(x.op);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Cast>::operator()(const polyregion::polyast::Expr::Cast &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.from)>()(x.from);
  seed ^= std::hash<decltype(x.as)>()(x.as) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Alias>::operator()(const polyregion::polyast::Expr::Alias &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.ref)>()(x.ref);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Index>::operator()(const polyregion::polyast::Expr::Index &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.idx)>()(x.idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.component)>()(x.component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::RefTo>::operator()(const polyregion::polyast::Expr::RefTo &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.idx)>()(x.idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.component)>()(x.component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Alloc>::operator()(const polyregion::polyast::Expr::Alloc &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.component)>()(x.component);
  seed ^= std::hash<decltype(x.size)>()(x.size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Invoke>::operator()(const polyregion::polyast::Expr::Invoke &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeArgs)>()(x.tpeArgs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.captures)>()(x.captures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Block>::operator()(const polyregion::polyast::Stmt::Block &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.stmts)>()(x.stmts);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Comment>::operator()(const polyregion::polyast::Stmt::Comment &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Var>::operator()(const polyregion::polyast::Stmt::Var &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.expr)>()(x.expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Mut>::operator()(const polyregion::polyast::Stmt::Mut &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.expr)>()(x.expr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.copy)>()(x.copy) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Update>::operator()(const polyregion::polyast::Stmt::Update &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.idx)>()(x.idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.value)>()(x.value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::While>::operator()(const polyregion::polyast::Stmt::While &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.tests)>()(x.tests);
  seed ^= std::hash<decltype(x.cond)>()(x.cond) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.body)>()(x.body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Break>::operator()(const polyregion::polyast::Stmt::Break &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Stmt::Break");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Cont>::operator()(const polyregion::polyast::Stmt::Cont &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Stmt::Cont");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Cond>::operator()(const polyregion::polyast::Stmt::Cond &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.cond)>()(x.cond);
  seed ^= std::hash<decltype(x.trueBr)>()(x.trueBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.falseBr)>()(x.falseBr) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Stmt::Return>::operator()(const polyregion::polyast::Stmt::Return &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::StructMember>::operator()(const polyregion::polyast::StructMember &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.named)>()(x.named);
  seed ^= std::hash<decltype(x.isMutable)>()(x.isMutable) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::StructDef>::operator()(const polyregion::polyast::StructDef &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.members)>()(x.members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.parents)>()(x.parents) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Signature>::operator()(const polyregion::polyast::Signature &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.moduleCaptures)>()(x.moduleCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.termCaptures)>()(x.termCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::InvokeSignature>::operator()(const polyregion::polyast::InvokeSignature &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.captures)>()(x.captures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::FunctionKind::Internal>::operator()(const polyregion::polyast::FunctionKind::Internal &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::FunctionKind::Internal");
  return seed;
}
std::size_t std::hash<polyregion::polyast::FunctionKind::Exported>::operator()(const polyregion::polyast::FunctionKind::Exported &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::FunctionKind::Exported");
  return seed;
}
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPRelaxed>::operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::FunctionAttr::FPRelaxed");
  return seed;
}
std::size_t std::hash<polyregion::polyast::FunctionAttr::FPStrict>::operator()(const polyregion::polyast::FunctionAttr::FPStrict &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::FunctionAttr::FPStrict");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Arg>::operator()(const polyregion::polyast::Arg &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.named)>()(x.named);
  seed ^= std::hash<decltype(x.pos)>()(x.pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Function>::operator()(const polyregion::polyast::Function &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.moduleCaptures)>()(x.moduleCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.termCaptures)>()(x.termCaptures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.body)>()(x.body) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Program>::operator()(const polyregion::polyast::Program &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.entry)>()(x.entry);
  seed ^= std::hash<decltype(x.functions)>()(x.functions) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.defs)>()(x.defs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::CompileLayoutMember>::operator()(const polyregion::polyast::CompileLayoutMember &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.offsetInBytes)>()(x.offsetInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.sizeInBytes)>()(x.sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::CompileLayout>::operator()(const polyregion::polyast::CompileLayout &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.sizeInBytes)>()(x.sizeInBytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.alignment)>()(x.alignment) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.members)>()(x.members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::CompileEvent>::operator()(const polyregion::polyast::CompileEvent &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.epochMillis)>()(x.epochMillis);
  seed ^= std::hash<decltype(x.elapsedNanos)>()(x.elapsedNanos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.name)>()(x.name) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.data)>()(x.data) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::CompileResult>::operator()(const polyregion::polyast::CompileResult &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.binary)>()(x.binary);
  seed ^= std::hash<decltype(x.features)>()(x.features) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.events)>()(x.events) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.layouts)>()(x.layouts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.messages)>()(x.messages) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}


