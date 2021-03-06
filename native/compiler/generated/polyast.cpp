#include "polyast.h"

namespace polyregion::polyast {

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

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool TypeKind::operator==(const TypeKind::Base &, const TypeKind::Base &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::None &x) {
  os << "None(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::None &, const TypeKind::None &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Ref &x) {
  os << "Ref(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Ref &, const TypeKind::Ref &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Integral &x) {
  os << "Integral(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Integral &, const TypeKind::Integral &) { return true; }

std::ostream &TypeKind::operator<<(std::ostream &os, const TypeKind::Fractional &x) {
  os << "Fractional(";
  os << ')';
  return os;
}
bool TypeKind::operator==(const TypeKind::Fractional &, const TypeKind::Fractional &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Type::operator==(const Type::Base &l, const Type::Base &r) { 
  return *l.kind == *r.kind;
}
TypeKind::Any Type::kind(const Type::Any& x){ return select<&Type::Base::kind>(x); }

std::ostream &Type::operator<<(std::ostream &os, const Type::Float &x) {
  os << "Float(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Float &, const Type::Float &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Double &x) {
  os << "Double(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Double &, const Type::Double &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Bool &x) {
  os << "Bool(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Bool &, const Type::Bool &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Byte &x) {
  os << "Byte(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Byte &, const Type::Byte &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Char &x) {
  os << "Char(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Char &, const Type::Char &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Short &x) {
  os << "Short(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Short &, const Type::Short &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Int &x) {
  os << "Int(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Int &, const Type::Int &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Long &x) {
  os << "Long(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Long &, const Type::Long &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Unit &x) {
  os << "Unit(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Unit &, const Type::Unit &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::Nothing &x) {
  os << "Nothing(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::Nothing &, const Type::Nothing &) { return true; }

std::ostream &Type::operator<<(std::ostream &os, const Type::String &x) {
  os << "String(";
  os << ')';
  return os;
}
bool Type::operator==(const Type::String &, const Type::String &) { return true; }

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
  os << ')';
  return os;
}
bool Type::operator==(const Type::Struct &l, const Type::Struct &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Array &x) {
  os << "Array(";
  os << x.component;
  os << ')';
  return os;
}
bool Type::operator==(const Type::Array &l, const Type::Array &r) { 
  return *l.component == *r.component;
}

std::ostream &Type::operator<<(std::ostream &os, const Type::Var &x) {
  os << "Var(";
  os << '"' << x.name << '"';
  os << ')';
  return os;
}
bool Type::operator==(const Type::Var &l, const Type::Var &r) { 
  return l.name == r.name;
}

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

std::ostream &operator<<(std::ostream &os, const Position &x) {
  os << "Position(";
  os << '"' << x.file << '"';
  os << ',';
  os << x.line;
  os << ',';
  os << x.col;
  os << ')';
  return os;
}
bool operator==(const Position &l, const Position &r) { 
  return l.file == r.file && l.line == r.line && l.col == r.col;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Term::operator==(const Term::Base &l, const Term::Base &r) { 
  return *l.tpe == *r.tpe;
}
Type::Any Term::tpe(const Term::Any& x){ return select<&Term::Base::tpe>(x); }

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

std::ostream &Term::operator<<(std::ostream &os, const Term::Poison &x) {
  os << "Poison(";
  os << x.t;
  os << ')';
  return os;
}
bool Term::operator==(const Term::Poison &l, const Term::Poison &r) { 
  return *l.t == *r.t;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::UnitConst &x) {
  os << "UnitConst(";
  os << ')';
  return os;
}
bool Term::operator==(const Term::UnitConst &, const Term::UnitConst &) { return true; }

std::ostream &Term::operator<<(std::ostream &os, const Term::BoolConst &x) {
  os << "BoolConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::BoolConst &l, const Term::BoolConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ByteConst &x) {
  os << "ByteConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::ByteConst &l, const Term::ByteConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::CharConst &x) {
  os << "CharConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::CharConst &l, const Term::CharConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::ShortConst &x) {
  os << "ShortConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::ShortConst &l, const Term::ShortConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::IntConst &x) {
  os << "IntConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::IntConst &l, const Term::IntConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::LongConst &x) {
  os << "LongConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::LongConst &l, const Term::LongConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::FloatConst &x) {
  os << "FloatConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::FloatConst &l, const Term::FloatConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::DoubleConst &x) {
  os << "DoubleConst(";
  os << x.value;
  os << ')';
  return os;
}
bool Term::operator==(const Term::DoubleConst &l, const Term::DoubleConst &r) { 
  return l.value == r.value;
}

std::ostream &Term::operator<<(std::ostream &os, const Term::StringConst &x) {
  os << "StringConst(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}
bool Term::operator==(const Term::StringConst &l, const Term::StringConst &r) { 
  return l.value == r.value;
}

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::Base &, const NullaryIntrinsicKind::Base &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxX &x) {
  os << "GpuGlobalIdxX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalIdxX &, const NullaryIntrinsicKind::GpuGlobalIdxX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxY &x) {
  os << "GpuGlobalIdxY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalIdxY &, const NullaryIntrinsicKind::GpuGlobalIdxY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxZ &x) {
  os << "GpuGlobalIdxZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalIdxZ &, const NullaryIntrinsicKind::GpuGlobalIdxZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeX &x) {
  os << "GpuGlobalSizeX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalSizeX &, const NullaryIntrinsicKind::GpuGlobalSizeX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeY &x) {
  os << "GpuGlobalSizeY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalSizeY &, const NullaryIntrinsicKind::GpuGlobalSizeY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeZ &x) {
  os << "GpuGlobalSizeZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGlobalSizeZ &, const NullaryIntrinsicKind::GpuGlobalSizeZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxX &x) {
  os << "GpuGroupIdxX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupIdxX &, const NullaryIntrinsicKind::GpuGroupIdxX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxY &x) {
  os << "GpuGroupIdxY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupIdxY &, const NullaryIntrinsicKind::GpuGroupIdxY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxZ &x) {
  os << "GpuGroupIdxZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupIdxZ &, const NullaryIntrinsicKind::GpuGroupIdxZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeX &x) {
  os << "GpuGroupSizeX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupSizeX &, const NullaryIntrinsicKind::GpuGroupSizeX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeY &x) {
  os << "GpuGroupSizeY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupSizeY &, const NullaryIntrinsicKind::GpuGroupSizeY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeZ &x) {
  os << "GpuGroupSizeZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupSizeZ &, const NullaryIntrinsicKind::GpuGroupSizeZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxX &x) {
  os << "GpuLocalIdxX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalIdxX &, const NullaryIntrinsicKind::GpuLocalIdxX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxY &x) {
  os << "GpuLocalIdxY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalIdxY &, const NullaryIntrinsicKind::GpuLocalIdxY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxZ &x) {
  os << "GpuLocalIdxZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalIdxZ &, const NullaryIntrinsicKind::GpuLocalIdxZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeX &x) {
  os << "GpuLocalSizeX(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalSizeX &, const NullaryIntrinsicKind::GpuLocalSizeX &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeY &x) {
  os << "GpuLocalSizeY(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalSizeY &, const NullaryIntrinsicKind::GpuLocalSizeY &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeZ &x) {
  os << "GpuLocalSizeZ(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuLocalSizeZ &, const NullaryIntrinsicKind::GpuLocalSizeZ &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupBarrier &x) {
  os << "GpuGroupBarrier(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupBarrier &, const NullaryIntrinsicKind::GpuGroupBarrier &) { return true; }

std::ostream &NullaryIntrinsicKind::operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupFence &x) {
  os << "GpuGroupFence(";
  os << ')';
  return os;
}
bool NullaryIntrinsicKind::operator==(const NullaryIntrinsicKind::GpuGroupFence &, const NullaryIntrinsicKind::GpuGroupFence &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Base &, const UnaryIntrinsicKind::Base &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Sin &x) {
  os << "Sin(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Sin &, const UnaryIntrinsicKind::Sin &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Cos &x) {
  os << "Cos(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Cos &, const UnaryIntrinsicKind::Cos &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Tan &x) {
  os << "Tan(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Tan &, const UnaryIntrinsicKind::Tan &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Asin &x) {
  os << "Asin(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Asin &, const UnaryIntrinsicKind::Asin &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Acos &x) {
  os << "Acos(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Acos &, const UnaryIntrinsicKind::Acos &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Atan &x) {
  os << "Atan(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Atan &, const UnaryIntrinsicKind::Atan &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Sinh &x) {
  os << "Sinh(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Sinh &, const UnaryIntrinsicKind::Sinh &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Cosh &x) {
  os << "Cosh(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Cosh &, const UnaryIntrinsicKind::Cosh &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Tanh &x) {
  os << "Tanh(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Tanh &, const UnaryIntrinsicKind::Tanh &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Signum &x) {
  os << "Signum(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Signum &, const UnaryIntrinsicKind::Signum &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Abs &x) {
  os << "Abs(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Abs &, const UnaryIntrinsicKind::Abs &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Round &x) {
  os << "Round(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Round &, const UnaryIntrinsicKind::Round &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Ceil &x) {
  os << "Ceil(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Ceil &, const UnaryIntrinsicKind::Ceil &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Floor &x) {
  os << "Floor(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Floor &, const UnaryIntrinsicKind::Floor &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Rint &x) {
  os << "Rint(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Rint &, const UnaryIntrinsicKind::Rint &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Sqrt &x) {
  os << "Sqrt(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Sqrt &, const UnaryIntrinsicKind::Sqrt &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Cbrt &x) {
  os << "Cbrt(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Cbrt &, const UnaryIntrinsicKind::Cbrt &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Exp &x) {
  os << "Exp(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Exp &, const UnaryIntrinsicKind::Exp &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Expm1 &x) {
  os << "Expm1(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Expm1 &, const UnaryIntrinsicKind::Expm1 &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Log &x) {
  os << "Log(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Log &, const UnaryIntrinsicKind::Log &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Log1p &x) {
  os << "Log1p(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Log1p &, const UnaryIntrinsicKind::Log1p &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Log10 &x) {
  os << "Log10(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Log10 &, const UnaryIntrinsicKind::Log10 &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::BNot &x) {
  os << "BNot(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::BNot &, const UnaryIntrinsicKind::BNot &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Pos &x) {
  os << "Pos(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Pos &, const UnaryIntrinsicKind::Pos &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::Neg &x) {
  os << "Neg(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::Neg &, const UnaryIntrinsicKind::Neg &) { return true; }

std::ostream &UnaryIntrinsicKind::operator<<(std::ostream &os, const UnaryIntrinsicKind::LogicNot &x) {
  os << "LogicNot(";
  os << ')';
  return os;
}
bool UnaryIntrinsicKind::operator==(const UnaryIntrinsicKind::LogicNot &, const UnaryIntrinsicKind::LogicNot &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Base &, const BinaryIntrinsicKind::Base &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Add &x) {
  os << "Add(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Add &, const BinaryIntrinsicKind::Add &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Sub &x) {
  os << "Sub(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Sub &, const BinaryIntrinsicKind::Sub &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Mul &x) {
  os << "Mul(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Mul &, const BinaryIntrinsicKind::Mul &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Div &x) {
  os << "Div(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Div &, const BinaryIntrinsicKind::Div &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Rem &x) {
  os << "Rem(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Rem &, const BinaryIntrinsicKind::Rem &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Pow &x) {
  os << "Pow(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Pow &, const BinaryIntrinsicKind::Pow &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Min &x) {
  os << "Min(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Min &, const BinaryIntrinsicKind::Min &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Max &x) {
  os << "Max(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Max &, const BinaryIntrinsicKind::Max &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Atan2 &x) {
  os << "Atan2(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Atan2 &, const BinaryIntrinsicKind::Atan2 &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::Hypot &x) {
  os << "Hypot(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::Hypot &, const BinaryIntrinsicKind::Hypot &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BAnd &x) {
  os << "BAnd(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BAnd &, const BinaryIntrinsicKind::BAnd &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BOr &x) {
  os << "BOr(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BOr &, const BinaryIntrinsicKind::BOr &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BXor &x) {
  os << "BXor(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BXor &, const BinaryIntrinsicKind::BXor &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BSL &x) {
  os << "BSL(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BSL &, const BinaryIntrinsicKind::BSL &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BSR &x) {
  os << "BSR(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BSR &, const BinaryIntrinsicKind::BSR &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::BZSR &x) {
  os << "BZSR(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::BZSR &, const BinaryIntrinsicKind::BZSR &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicEq &x) {
  os << "LogicEq(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicEq &, const BinaryIntrinsicKind::LogicEq &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicNeq &x) {
  os << "LogicNeq(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicNeq &, const BinaryIntrinsicKind::LogicNeq &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicAnd &x) {
  os << "LogicAnd(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicAnd &, const BinaryIntrinsicKind::LogicAnd &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicOr &x) {
  os << "LogicOr(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicOr &, const BinaryIntrinsicKind::LogicOr &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicLte &x) {
  os << "LogicLte(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicLte &, const BinaryIntrinsicKind::LogicLte &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicGte &x) {
  os << "LogicGte(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicGte &, const BinaryIntrinsicKind::LogicGte &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicLt &x) {
  os << "LogicLt(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicLt &, const BinaryIntrinsicKind::LogicLt &) { return true; }

std::ostream &BinaryIntrinsicKind::operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicGt &x) {
  os << "LogicGt(";
  os << ')';
  return os;
}
bool BinaryIntrinsicKind::operator==(const BinaryIntrinsicKind::LogicGt &, const BinaryIntrinsicKind::LogicGt &) { return true; }

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Expr::operator==(const Expr::Base &l, const Expr::Base &r) { 
  return *l.tpe == *r.tpe;
}
Type::Any Expr::tpe(const Expr::Any& x){ return select<&Expr::Base::tpe>(x); }

std::ostream &Expr::operator<<(std::ostream &os, const Expr::NullaryIntrinsic &x) {
  os << "NullaryIntrinsic(";
  os << x.kind;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::NullaryIntrinsic &l, const Expr::NullaryIntrinsic &r) { 
  return *l.kind == *r.kind && *l.rtn == *r.rtn;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::UnaryIntrinsic &x) {
  os << "UnaryIntrinsic(";
  os << x.lhs;
  os << ',';
  os << x.kind;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::UnaryIntrinsic &l, const Expr::UnaryIntrinsic &r) { 
  return *l.lhs == *r.lhs && *l.kind == *r.kind && *l.rtn == *r.rtn;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::BinaryIntrinsic &x) {
  os << "BinaryIntrinsic(";
  os << x.lhs;
  os << ',';
  os << x.rhs;
  os << ',';
  os << x.kind;
  os << ',';
  os << x.rtn;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::BinaryIntrinsic &l, const Expr::BinaryIntrinsic &r) { 
  return *l.lhs == *r.lhs && *l.rhs == *r.rhs && *l.kind == *r.kind && *l.rtn == *r.rtn;
}

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

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alias &x) {
  os << "Alias(";
  os << x.ref;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Alias &l, const Expr::Alias &r) { 
  return *l.ref == *r.ref;
}

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
    os << *x.receiver;
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
bool Expr::operator==(const Expr::Invoke &l, const Expr::Invoke &r) { 
  return l.name == r.name && std::equal(l.tpeArgs.begin(), l.tpeArgs.end(), r.tpeArgs.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && l.receiver == r.receiver && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}

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
  return l.lhs == r.lhs && *l.idx == *r.idx && *l.component == *r.component;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Alloc &x) {
  os << "Alloc(";
  os << x.witness;
  os << ',';
  os << x.size;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Alloc &l, const Expr::Alloc &r) { 
  return l.witness == r.witness && *l.size == *r.size;
}

std::ostream &Expr::operator<<(std::ostream &os, const Expr::Suspend &x) {
  os << "Suspend(";
  os << '{';
  if (!x.args.empty()) {
    std::for_each(x.args.begin(), std::prev(x.args.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.args.back();
  }
  os << '}';
  os << ',';
  os << '{';
  if (!x.stmts.empty()) {
    std::for_each(x.stmts.begin(), std::prev(x.stmts.end()), [&os](auto &&x) { os << x; os << ','; });
    os << x.stmts.back();
  }
  os << '}';
  os << ',';
  os << x.rtn;
  os << ',';
  os << x.shape;
  os << ')';
  return os;
}
bool Expr::operator==(const Expr::Suspend &l, const Expr::Suspend &r) { 
  return l.args == r.args && std::equal(l.stmts.begin(), l.stmts.end(), r.stmts.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn && l.shape == r.shape;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Any &x) {
  std::visit([&os](auto &&arg) { os << *arg; }, x);
  return os;
}
bool Stmt::operator==(const Stmt::Base &, const Stmt::Base &) { return true; }

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Comment &x) {
  os << "Comment(";
  os << '"' << x.value << '"';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Comment &l, const Stmt::Comment &r) { 
  return l.value == r.value;
}

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Var &x) {
  os << "Var(";
  os << x.name;
  os << ',';
  os << '{';
  if (x.expr) {
    os << *x.expr;
  }
  os << '}';
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Var &l, const Stmt::Var &r) { 
  return l.name == r.name && l.expr == r.expr;
}

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
  return l.name == r.name && *l.expr == *r.expr && l.copy == r.copy;
}

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
  return l.lhs == r.lhs && *l.idx == *r.idx && *l.value == *r.value;
}

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

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Break &x) {
  os << "Break(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Break &, const Stmt::Break &) { return true; }

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Cont &x) {
  os << "Cont(";
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Cont &, const Stmt::Cont &) { return true; }

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

std::ostream &Stmt::operator<<(std::ostream &os, const Stmt::Return &x) {
  os << "Return(";
  os << x.value;
  os << ')';
  return os;
}
bool Stmt::operator==(const Stmt::Return &l, const Stmt::Return &r) { 
  return *l.value == *r.value;
}

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
  os << ')';
  return os;
}
bool operator==(const StructDef &l, const StructDef &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && l.members == r.members;
}

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
    os << *x.receiver;
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
bool operator==(const Signature &l, const Signature &r) { 
  return l.name == r.name && l.tpeVars == r.tpeVars && l.receiver == r.receiver && std::equal(l.args.begin(), l.args.end(), r.args.begin(), [](auto &&l, auto &&r) { return *l == *r; }) && *l.rtn == *r.rtn;
}

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
    os << *x.receiver;
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
  return l.name == r.name && l.tpeVars == r.tpeVars && l.receiver == r.receiver && l.args == r.args && l.captures == r.captures && *l.rtn == *r.rtn && std::equal(l.body.begin(), l.body.end(), r.body.begin(), [](auto &&l, auto &&r) { return *l == *r; });
}

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
std::size_t std::hash<polyregion::polyast::Type::Float>::operator()(const polyregion::polyast::Type::Float &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Float");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Double>::operator()(const polyregion::polyast::Type::Double &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Double");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Bool>::operator()(const polyregion::polyast::Type::Bool &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Bool");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Byte>::operator()(const polyregion::polyast::Type::Byte &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Byte");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Char>::operator()(const polyregion::polyast::Type::Char &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Char");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Short>::operator()(const polyregion::polyast::Type::Short &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Short");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Int>::operator()(const polyregion::polyast::Type::Int &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Int");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Long>::operator()(const polyregion::polyast::Type::Long &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Long");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Unit>::operator()(const polyregion::polyast::Type::Unit &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Unit");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Nothing>::operator()(const polyregion::polyast::Type::Nothing &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::Nothing");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::String>::operator()(const polyregion::polyast::Type::String &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Type::String");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Struct>::operator()(const polyregion::polyast::Type::Struct &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Type::Array>::operator()(const polyregion::polyast::Type::Array &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.component)>()(x.component);
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
std::size_t std::hash<polyregion::polyast::Position>::operator()(const polyregion::polyast::Position &x) const noexcept {
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
std::size_t std::hash<polyregion::polyast::Term::UnitConst>::operator()(const polyregion::polyast::Term::UnitConst &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::Term::UnitConst");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::BoolConst>::operator()(const polyregion::polyast::Term::BoolConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::ByteConst>::operator()(const polyregion::polyast::Term::ByteConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::CharConst>::operator()(const polyregion::polyast::Term::CharConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::ShortConst>::operator()(const polyregion::polyast::Term::ShortConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::IntConst>::operator()(const polyregion::polyast::Term::IntConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::LongConst>::operator()(const polyregion::polyast::Term::LongConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::FloatConst>::operator()(const polyregion::polyast::Term::FloatConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::DoubleConst>::operator()(const polyregion::polyast::Term::DoubleConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Term::StringConst>::operator()(const polyregion::polyast::Term::StringConst &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.value)>()(x.value);
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeX>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeX &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeX");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeY>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeY &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeY");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeZ>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeZ &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeZ");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupBarrier>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupBarrier &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupBarrier");
  return seed;
}
std::size_t std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupFence>::operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupFence &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::NullaryIntrinsicKind::GpuGroupFence");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Sin>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Sin &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Sin");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Cos>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Cos &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Cos");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Tan>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Tan &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Tan");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Asin>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Asin &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Asin");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Acos>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Acos &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Acos");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Atan>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Atan &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Atan");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Sinh>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Sinh &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Sinh");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Cosh>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Cosh &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Cosh");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Tanh>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Tanh &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Tanh");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Signum>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Signum &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Signum");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Abs>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Abs &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Abs");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Round>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Round &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Round");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Ceil>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Ceil &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Ceil");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Floor>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Floor &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Floor");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Rint>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Rint &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Rint");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Sqrt>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Sqrt &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Sqrt");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Cbrt>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Cbrt &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Cbrt");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Exp>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Exp &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Exp");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Expm1>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Expm1 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Expm1");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Log>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Log &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Log");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Log1p>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Log1p &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Log1p");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Log10>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Log10 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Log10");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::BNot>::operator()(const polyregion::polyast::UnaryIntrinsicKind::BNot &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::BNot");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Pos>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Pos &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Pos");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::Neg>::operator()(const polyregion::polyast::UnaryIntrinsicKind::Neg &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::Neg");
  return seed;
}
std::size_t std::hash<polyregion::polyast::UnaryIntrinsicKind::LogicNot>::operator()(const polyregion::polyast::UnaryIntrinsicKind::LogicNot &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::UnaryIntrinsicKind::LogicNot");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Add>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Add &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Add");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Sub>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Sub &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Sub");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Mul>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Mul &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Mul");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Div>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Div &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Div");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Rem>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Rem &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Rem");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Pow>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Pow &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Pow");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Min>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Min &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Min");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Max>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Max &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Max");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Atan2>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Atan2 &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Atan2");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::Hypot>::operator()(const polyregion::polyast::BinaryIntrinsicKind::Hypot &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::Hypot");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BAnd>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BAnd &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BAnd");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BOr>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BOr &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BOr");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BXor>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BXor &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BXor");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BSL>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BSL &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BSL");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BSR>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BSR &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BSR");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::BZSR>::operator()(const polyregion::polyast::BinaryIntrinsicKind::BZSR &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::BZSR");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicEq>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicEq &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicEq");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicNeq>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicNeq &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicNeq");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicAnd>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicAnd &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicAnd");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicOr>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicOr &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicOr");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicLte>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicLte &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicLte");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicGte>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicGte &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicGte");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicLt>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicLt &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicLt");
  return seed;
}
std::size_t std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicGt>::operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicGt &x) const noexcept {
  std::size_t seed = std::hash<std::string>()("polyregion::polyast::BinaryIntrinsicKind::LogicGt");
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::NullaryIntrinsic>::operator()(const polyregion::polyast::Expr::NullaryIntrinsic &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.kind)>()(x.kind);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::UnaryIntrinsic>::operator()(const polyregion::polyast::Expr::UnaryIntrinsic &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.kind)>()(x.kind) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::BinaryIntrinsic>::operator()(const polyregion::polyast::Expr::BinaryIntrinsic &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.rhs)>()(x.rhs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.kind)>()(x.kind) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
std::size_t std::hash<polyregion::polyast::Expr::Invoke>::operator()(const polyregion::polyast::Expr::Invoke &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeArgs)>()(x.tpeArgs) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Index>::operator()(const polyregion::polyast::Expr::Index &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.lhs)>()(x.lhs);
  seed ^= std::hash<decltype(x.idx)>()(x.idx) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.component)>()(x.component) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Alloc>::operator()(const polyregion::polyast::Expr::Alloc &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.witness)>()(x.witness);
  seed ^= std::hash<decltype(x.size)>()(x.size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Expr::Suspend>::operator()(const polyregion::polyast::Expr::Suspend &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.args)>()(x.args);
  seed ^= std::hash<decltype(x.stmts)>()(x.stmts) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.shape)>()(x.shape) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
std::size_t std::hash<polyregion::polyast::StructDef>::operator()(const polyregion::polyast::StructDef &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.members)>()(x.members) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Signature>::operator()(const polyregion::polyast::Signature &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.rtn)>()(x.rtn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}
std::size_t std::hash<polyregion::polyast::Function>::operator()(const polyregion::polyast::Function &x) const noexcept {
  std::size_t seed = std::hash<decltype(x.name)>()(x.name);
  seed ^= std::hash<decltype(x.tpeVars)>()(x.tpeVars) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.receiver)>()(x.receiver) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.args)>()(x.args) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= std::hash<decltype(x.captures)>()(x.captures) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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


