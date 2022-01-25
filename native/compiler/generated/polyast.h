#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>
#include "export.h"

namespace polyregion::polyast {

template <typename... T> //
using Alternative = std::variant<std::shared_ptr<T>...>;

template <typename... T> //
constexpr std::variant<T...> unwrap(const Alternative<T...> &a) {
  return std::visit([](auto &&arg) { return std::variant<T...>(*arg); }, a);
}

template <typename... T> //
constexpr std::variant<T...> operator*(const Alternative<T...> &a) {
  return unwrap(a);
}

template <auto member, class... T> //
constexpr auto select(const Alternative<T...> &a) {
  return std::visit([](auto &&arg) { return *(arg).*member; }, a);
}

template <typename T> //
std::string to_string(const T& x) {
  std::ostringstream ss;
  ss << x;
  return ss.str();
}

template <typename T, typename... Ts> //
constexpr std::optional<T> get_opt(const Alternative<Ts...> &a) {
  if (const std::shared_ptr<T> *v = std::get_if<std::shared_ptr<T>>(&a)) return {**v};
  else
    return {};
}
#pragma clang diagnostic push
#pragma ide diagnostic ignored "google-explicit-constructor"



struct EXPORT Sym {
  std::vector<std::string> fqn;
  explicit Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Sym &);
  EXPORT friend bool operator==(const Sym &, const Sym &);
};

namespace TypeKind { 

struct Ref;
struct Integral;
struct Fractional;
using Any = Alternative<Ref, Integral, Fractional>;
struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Any &);
  EXPORT friend bool operator==(const TypeKind::Base &, const TypeKind::Base &);
};

struct EXPORT Ref : TypeKind::Base {
  Ref() noexcept : TypeKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Ref>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &);
  EXPORT friend bool operator==(const TypeKind::Ref &, const TypeKind::Ref &);
};

struct EXPORT Integral : TypeKind::Base {
  Integral() noexcept : TypeKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Integral>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &);
  EXPORT friend bool operator==(const TypeKind::Integral &, const TypeKind::Integral &);
};

struct EXPORT Fractional : TypeKind::Base {
  Fractional() noexcept : TypeKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Fractional>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
  EXPORT friend bool operator==(const TypeKind::Fractional &, const TypeKind::Fractional &);
};
} // namespace TypeKind
namespace Type { 

struct Float;
struct Double;
struct Bool;
struct Byte;
struct Char;
struct Short;
struct Int;
struct Long;
struct String;
struct Unit;
struct Struct;
struct Array;
using Any = Alternative<Float, Double, Bool, Byte, Char, Short, Int, Long, String, Unit, Struct, Array>;
struct EXPORT Base {
  TypeKind::Any kind;
  protected:
  explicit Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Any &);
  EXPORT friend bool operator==(const Type::Base &, const Type::Base &);
};
EXPORT TypeKind::Any kind(const Type::Any&);

struct EXPORT Float : Type::Base {
  Float() noexcept : Type::Base(TypeKind::Fractional()) {}
  EXPORT operator Any() const { return std::make_shared<Float>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float &);
  EXPORT friend bool operator==(const Type::Float &, const Type::Float &);
};

struct EXPORT Double : Type::Base {
  Double() noexcept : Type::Base(TypeKind::Fractional()) {}
  EXPORT operator Any() const { return std::make_shared<Double>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Double &);
  EXPORT friend bool operator==(const Type::Double &, const Type::Double &);
};

struct EXPORT Bool : Type::Base {
  Bool() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Bool>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Bool &);
  EXPORT friend bool operator==(const Type::Bool &, const Type::Bool &);
};

struct EXPORT Byte : Type::Base {
  Byte() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Byte>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Byte &);
  EXPORT friend bool operator==(const Type::Byte &, const Type::Byte &);
};

struct EXPORT Char : Type::Base {
  Char() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Char>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Char &);
  EXPORT friend bool operator==(const Type::Char &, const Type::Char &);
};

struct EXPORT Short : Type::Base {
  Short() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Short>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Short &);
  EXPORT friend bool operator==(const Type::Short &, const Type::Short &);
};

struct EXPORT Int : Type::Base {
  Int() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Int>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Int &);
  EXPORT friend bool operator==(const Type::Int &, const Type::Int &);
};

struct EXPORT Long : Type::Base {
  Long() noexcept : Type::Base(TypeKind::Integral()) {}
  EXPORT operator Any() const { return std::make_shared<Long>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Long &);
  EXPORT friend bool operator==(const Type::Long &, const Type::Long &);
};

struct EXPORT String : Type::Base {
  String() noexcept : Type::Base(TypeKind::Ref()) {}
  EXPORT operator Any() const { return std::make_shared<String>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::String &);
  EXPORT friend bool operator==(const Type::String &, const Type::String &);
};

struct EXPORT Unit : Type::Base {
  Unit() noexcept : Type::Base(TypeKind::Ref()) {}
  EXPORT operator Any() const { return std::make_shared<Unit>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Unit &);
  EXPORT friend bool operator==(const Type::Unit &, const Type::Unit &);
};

struct EXPORT Struct : Type::Base {
  Sym name;
  explicit Struct(Sym name) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)) {}
  EXPORT operator Any() const { return std::make_shared<Struct>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
  EXPORT friend bool operator==(const Type::Struct &, const Type::Struct &);
};

struct EXPORT Array : Type::Base {
  Type::Any component;
  std::optional<int32_t> length;
  Array(Type::Any component, std::optional<int32_t> length) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)), length(std::move(length)) {}
  EXPORT operator Any() const { return std::make_shared<Array>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Array &);
  EXPORT friend bool operator==(const Type::Array &, const Type::Array &);
};
} // namespace Type


struct EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Named &);
  EXPORT friend bool operator==(const Named &, const Named &);
};

struct EXPORT Position {
  std::string file;
  int32_t line;
  int32_t col;
  Position(std::string file, int32_t line, int32_t col) noexcept : file(std::move(file)), line(line), col(col) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Position &);
  EXPORT friend bool operator==(const Position &, const Position &);
};

namespace Term { 

struct Select;
struct UnitConst;
struct BoolConst;
struct ByteConst;
struct CharConst;
struct ShortConst;
struct IntConst;
struct LongConst;
struct FloatConst;
struct DoubleConst;
struct StringConst;
using Any = Alternative<Select, UnitConst, BoolConst, ByteConst, CharConst, ShortConst, IntConst, LongConst, FloatConst, DoubleConst, StringConst>;
struct EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Any &);
  EXPORT friend bool operator==(const Term::Base &, const Term::Base &);
};
EXPORT Type::Any tpe(const Term::Any&);

struct EXPORT Select : Term::Base {
  std::vector<Named> init;
  Named last;
  Select(std::vector<Named> init, Named last) noexcept : Term::Base(last.tpe), init(std::move(init)), last(std::move(last)) {}
  EXPORT operator Any() const { return std::make_shared<Select>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Select &);
  EXPORT friend bool operator==(const Term::Select &, const Term::Select &);
};

struct EXPORT UnitConst : Term::Base {
  UnitConst() noexcept : Term::Base(Type::Unit()) {}
  EXPORT operator Any() const { return std::make_shared<UnitConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::UnitConst &);
  EXPORT friend bool operator==(const Term::UnitConst &, const Term::UnitConst &);
};

struct EXPORT BoolConst : Term::Base {
  bool value;
  explicit BoolConst(bool value) noexcept : Term::Base(Type::Bool()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<BoolConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::BoolConst &);
  EXPORT friend bool operator==(const Term::BoolConst &, const Term::BoolConst &);
};

struct EXPORT ByteConst : Term::Base {
  int8_t value;
  explicit ByteConst(int8_t value) noexcept : Term::Base(Type::Byte()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<ByteConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::ByteConst &);
  EXPORT friend bool operator==(const Term::ByteConst &, const Term::ByteConst &);
};

struct EXPORT CharConst : Term::Base {
  uint16_t value;
  explicit CharConst(uint16_t value) noexcept : Term::Base(Type::Char()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<CharConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::CharConst &);
  EXPORT friend bool operator==(const Term::CharConst &, const Term::CharConst &);
};

struct EXPORT ShortConst : Term::Base {
  int16_t value;
  explicit ShortConst(int16_t value) noexcept : Term::Base(Type::Short()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<ShortConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::ShortConst &);
  EXPORT friend bool operator==(const Term::ShortConst &, const Term::ShortConst &);
};

struct EXPORT IntConst : Term::Base {
  int32_t value;
  explicit IntConst(int32_t value) noexcept : Term::Base(Type::Int()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<IntConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntConst &);
  EXPORT friend bool operator==(const Term::IntConst &, const Term::IntConst &);
};

struct EXPORT LongConst : Term::Base {
  int64_t value;
  explicit LongConst(int64_t value) noexcept : Term::Base(Type::Long()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<LongConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::LongConst &);
  EXPORT friend bool operator==(const Term::LongConst &, const Term::LongConst &);
};

struct EXPORT FloatConst : Term::Base {
  float value;
  explicit FloatConst(float value) noexcept : Term::Base(Type::Float()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<FloatConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::FloatConst &);
  EXPORT friend bool operator==(const Term::FloatConst &, const Term::FloatConst &);
};

struct EXPORT DoubleConst : Term::Base {
  double value;
  explicit DoubleConst(double value) noexcept : Term::Base(Type::Double()), value(value) {}
  EXPORT operator Any() const { return std::make_shared<DoubleConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::DoubleConst &);
  EXPORT friend bool operator==(const Term::DoubleConst &, const Term::DoubleConst &);
};

struct EXPORT StringConst : Term::Base {
  std::string value;
  explicit StringConst(std::string value) noexcept : Term::Base(Type::String()), value(std::move(value)) {}
  EXPORT operator Any() const { return std::make_shared<StringConst>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::StringConst &);
  EXPORT friend bool operator==(const Term::StringConst &, const Term::StringConst &);
};
} // namespace Term
namespace Expr { 

struct Sin;
struct Cos;
struct Tan;
struct Abs;
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Rem;
struct Pow;
struct BNot;
struct BAnd;
struct BOr;
struct BXor;
struct BSL;
struct BSR;
struct Not;
struct Eq;
struct Neq;
struct And;
struct Or;
struct Lte;
struct Gte;
struct Lt;
struct Gt;
struct Alias;
struct Invoke;
struct Index;
using Any = Alternative<Sin, Cos, Tan, Abs, Add, Sub, Mul, Div, Rem, Pow, BNot, BAnd, BOr, BXor, BSL, BSR, Not, Eq, Neq, And, Or, Lte, Gte, Lt, Gt, Alias, Invoke, Index>;
struct EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Any &);
  EXPORT friend bool operator==(const Expr::Base &, const Expr::Base &);
};
EXPORT Type::Any tpe(const Expr::Any&);

struct EXPORT Sin : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Sin(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Sin>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Sin &);
  EXPORT friend bool operator==(const Expr::Sin &, const Expr::Sin &);
};

struct EXPORT Cos : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Cos(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Cos>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cos &);
  EXPORT friend bool operator==(const Expr::Cos &, const Expr::Cos &);
};

struct EXPORT Tan : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Tan(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Tan>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Tan &);
  EXPORT friend bool operator==(const Expr::Tan &, const Expr::Tan &);
};

struct EXPORT Abs : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Abs(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Abs>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Abs &);
  EXPORT friend bool operator==(const Expr::Abs &, const Expr::Abs &);
};

struct EXPORT Add : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Add(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Add>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Add &);
  EXPORT friend bool operator==(const Expr::Add &, const Expr::Add &);
};

struct EXPORT Sub : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Sub(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Sub>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Sub &);
  EXPORT friend bool operator==(const Expr::Sub &, const Expr::Sub &);
};

struct EXPORT Mul : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Mul(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Mul>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Mul &);
  EXPORT friend bool operator==(const Expr::Mul &, const Expr::Mul &);
};

struct EXPORT Div : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Div(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Div>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Div &);
  EXPORT friend bool operator==(const Expr::Div &, const Expr::Div &);
};

struct EXPORT Rem : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Rem(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Rem>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Rem &);
  EXPORT friend bool operator==(const Expr::Rem &, const Expr::Rem &);
};

struct EXPORT Pow : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Pow(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Pow>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Pow &);
  EXPORT friend bool operator==(const Expr::Pow &, const Expr::Pow &);
};

struct EXPORT BNot : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  BNot(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BNot>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BNot &);
  EXPORT friend bool operator==(const Expr::BNot &, const Expr::BNot &);
};

struct EXPORT BAnd : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  BAnd(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BAnd>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BAnd &);
  EXPORT friend bool operator==(const Expr::BAnd &, const Expr::BAnd &);
};

struct EXPORT BOr : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  BOr(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BOr>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BOr &);
  EXPORT friend bool operator==(const Expr::BOr &, const Expr::BOr &);
};

struct EXPORT BXor : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  BXor(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BXor>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BXor &);
  EXPORT friend bool operator==(const Expr::BXor &, const Expr::BXor &);
};

struct EXPORT BSL : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  BSL(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BSL>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BSL &);
  EXPORT friend bool operator==(const Expr::BSL &, const Expr::BSL &);
};

struct EXPORT BSR : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  BSR(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BSR>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BSR &);
  EXPORT friend bool operator==(const Expr::BSR &, const Expr::BSR &);
};

struct EXPORT Not : Expr::Base {
  Term::Any lhs;
  explicit Not(Term::Any lhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)) {}
  EXPORT operator Any() const { return std::make_shared<Not>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Not &);
  EXPORT friend bool operator==(const Expr::Not &, const Expr::Not &);
};

struct EXPORT Eq : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Eq(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Eq>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Eq &);
  EXPORT friend bool operator==(const Expr::Eq &, const Expr::Eq &);
};

struct EXPORT Neq : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Neq(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Neq>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Neq &);
  EXPORT friend bool operator==(const Expr::Neq &, const Expr::Neq &);
};

struct EXPORT And : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  And(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<And>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::And &);
  EXPORT friend bool operator==(const Expr::And &, const Expr::And &);
};

struct EXPORT Or : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Or(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Or>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Or &);
  EXPORT friend bool operator==(const Expr::Or &, const Expr::Or &);
};

struct EXPORT Lte : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Lte(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Lte>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Lte &);
  EXPORT friend bool operator==(const Expr::Lte &, const Expr::Lte &);
};

struct EXPORT Gte : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Gte(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Gte>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Gte &);
  EXPORT friend bool operator==(const Expr::Gte &, const Expr::Gte &);
};

struct EXPORT Lt : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Lt(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Lt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Lt &);
  EXPORT friend bool operator==(const Expr::Lt &, const Expr::Lt &);
};

struct EXPORT Gt : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Gt(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Gt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Gt &);
  EXPORT friend bool operator==(const Expr::Gt &, const Expr::Gt &);
};

struct EXPORT Alias : Expr::Base {
  Term::Any ref;
  explicit Alias(Term::Any ref) noexcept : Expr::Base(Term::tpe(ref)), ref(std::move(ref)) {}
  EXPORT operator Any() const { return std::make_shared<Alias>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
  EXPORT friend bool operator==(const Expr::Alias &, const Expr::Alias &);
};

struct EXPORT Invoke : Expr::Base {
  Term::Any lhs;
  std::string name;
  std::vector<Term::Any> args;
  Type::Any rtn;
  Invoke(Term::Any lhs, std::string name, std::vector<Term::Any> args, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Invoke>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
  EXPORT friend bool operator==(const Expr::Invoke &, const Expr::Invoke &);
};

struct EXPORT Index : Expr::Base {
  Term::Select lhs;
  Term::Any idx;
  Type::Any component;
  Index(Term::Select lhs, Term::Any idx, Type::Any component) noexcept : Expr::Base(component), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
  EXPORT operator Any() const { return std::make_shared<Index>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
  EXPORT friend bool operator==(const Expr::Index &, const Expr::Index &);
};
} // namespace Expr
namespace Stmt { 

struct Comment;
struct Var;
struct Mut;
struct Update;
struct Effect;
struct While;
struct Break;
struct Cont;
struct Cond;
struct Return;
using Any = Alternative<Comment, Var, Mut, Update, Effect, While, Break, Cont, Cond, Return>;
struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Any &);
  EXPORT friend bool operator==(const Stmt::Base &, const Stmt::Base &);
};

struct EXPORT Comment : Stmt::Base {
  std::string value;
  explicit Comment(std::string value) noexcept : Stmt::Base(), value(std::move(value)) {}
  EXPORT operator Any() const { return std::make_shared<Comment>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Comment &);
  EXPORT friend bool operator==(const Stmt::Comment &, const Stmt::Comment &);
};

struct EXPORT Var : Stmt::Base {
  Named name;
  std::optional<Expr::Any> expr;
  Var(Named name, std::optional<Expr::Any> expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
  EXPORT operator Any() const { return std::make_shared<Var>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
  EXPORT friend bool operator==(const Stmt::Var &, const Stmt::Var &);
};

struct EXPORT Mut : Stmt::Base {
  Term::Select name;
  Expr::Any expr;
  Mut(Term::Select name, Expr::Any expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
  EXPORT operator Any() const { return std::make_shared<Mut>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
  EXPORT friend bool operator==(const Stmt::Mut &, const Stmt::Mut &);
};

struct EXPORT Update : Stmt::Base {
  Term::Select lhs;
  Term::Any idx;
  Term::Any value;
  Update(Term::Select lhs, Term::Any idx, Term::Any value) noexcept : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
  EXPORT operator Any() const { return std::make_shared<Update>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
  EXPORT friend bool operator==(const Stmt::Update &, const Stmt::Update &);
};

struct EXPORT Effect : Stmt::Base {
  Term::Select lhs;
  std::string name;
  std::vector<Term::Any> args;
  Effect(Term::Select lhs, std::string name, std::vector<Term::Any> args) noexcept : Stmt::Base(), lhs(std::move(lhs)), name(std::move(name)), args(std::move(args)) {}
  EXPORT operator Any() const { return std::make_shared<Effect>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Effect &);
  EXPORT friend bool operator==(const Stmt::Effect &, const Stmt::Effect &);
};

struct EXPORT While : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> body;
  While(Expr::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), cond(std::move(cond)), body(std::move(body)) {}
  EXPORT operator Any() const { return std::make_shared<While>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
  EXPORT friend bool operator==(const Stmt::While &, const Stmt::While &);
};

struct EXPORT Break : Stmt::Base {
  Break() noexcept : Stmt::Base() {}
  EXPORT operator Any() const { return std::make_shared<Break>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Break &);
  EXPORT friend bool operator==(const Stmt::Break &, const Stmt::Break &);
};

struct EXPORT Cont : Stmt::Base {
  Cont() noexcept : Stmt::Base() {}
  EXPORT operator Any() const { return std::make_shared<Cont>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cont &);
  EXPORT friend bool operator==(const Stmt::Cont &, const Stmt::Cont &);
};

struct EXPORT Cond : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> trueBr;
  std::vector<Stmt::Any> falseBr;
  Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
  EXPORT operator Any() const { return std::make_shared<Cond>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cond &);
  EXPORT friend bool operator==(const Stmt::Cond &, const Stmt::Cond &);
};

struct EXPORT Return : Stmt::Base {
  Expr::Any value;
  explicit Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
  EXPORT operator Any() const { return std::make_shared<Return>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
  EXPORT friend bool operator==(const Stmt::Return &, const Stmt::Return &);
};
} // namespace Stmt


struct EXPORT StructDef {
  Sym name;
  std::vector<Named> members;
  StructDef(Sym name, std::vector<Named> members) noexcept : name(std::move(name)), members(std::move(members)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
  EXPORT friend bool operator==(const StructDef &, const StructDef &);
};

struct EXPORT Function {
  std::string name;
  std::vector<Named> args;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  std::vector<StructDef> defs;
  Function(std::string name, std::vector<Named> args, Type::Any rtn, std::vector<Stmt::Any> body, std::vector<StructDef> defs) noexcept : name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)), body(std::move(body)), defs(std::move(defs)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
  EXPORT friend bool operator==(const Function &, const Function &);
};

} // namespace polyregion::polyast
#pragma clang diagnostic pop // ide google-explicit-constructor
namespace std {

template <typename T> struct std::hash<std::vector<T>> {
  std::size_t operator()(std::vector<T> const &xs) const noexcept {
    std::size_t seed = xs.size();
    for (auto &x : xs) {
      seed ^= std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <> struct std::hash<polyregion::polyast::Sym> {
  std::size_t operator()(const polyregion::polyast::Sym &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeKind::Base> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Base &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeKind::Ref> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Ref &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeKind::Integral> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Integral &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeKind::Fractional> {
  std::size_t operator()(const polyregion::polyast::TypeKind::Fractional &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Base> {
  std::size_t operator()(const polyregion::polyast::Type::Base &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Float> {
  std::size_t operator()(const polyregion::polyast::Type::Float &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Double> {
  std::size_t operator()(const polyregion::polyast::Type::Double &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Bool> {
  std::size_t operator()(const polyregion::polyast::Type::Bool &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Byte> {
  std::size_t operator()(const polyregion::polyast::Type::Byte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Char> {
  std::size_t operator()(const polyregion::polyast::Type::Char &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Short> {
  std::size_t operator()(const polyregion::polyast::Type::Short &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Int> {
  std::size_t operator()(const polyregion::polyast::Type::Int &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Long> {
  std::size_t operator()(const polyregion::polyast::Type::Long &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::String> {
  std::size_t operator()(const polyregion::polyast::Type::String &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Unit> {
  std::size_t operator()(const polyregion::polyast::Type::Unit &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Struct> {
  std::size_t operator()(const polyregion::polyast::Type::Struct &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Array> {
  std::size_t operator()(const polyregion::polyast::Type::Array &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Named> {
  std::size_t operator()(const polyregion::polyast::Named &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Position> {
  std::size_t operator()(const polyregion::polyast::Position &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Base> {
  std::size_t operator()(const polyregion::polyast::Term::Base &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Select> {
  std::size_t operator()(const polyregion::polyast::Term::Select &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::UnitConst> {
  std::size_t operator()(const polyregion::polyast::Term::UnitConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::BoolConst> {
  std::size_t operator()(const polyregion::polyast::Term::BoolConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::ByteConst> {
  std::size_t operator()(const polyregion::polyast::Term::ByteConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::CharConst> {
  std::size_t operator()(const polyregion::polyast::Term::CharConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::ShortConst> {
  std::size_t operator()(const polyregion::polyast::Term::ShortConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntConst> {
  std::size_t operator()(const polyregion::polyast::Term::IntConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::LongConst> {
  std::size_t operator()(const polyregion::polyast::Term::LongConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::FloatConst> {
  std::size_t operator()(const polyregion::polyast::Term::FloatConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::DoubleConst> {
  std::size_t operator()(const polyregion::polyast::Term::DoubleConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::StringConst> {
  std::size_t operator()(const polyregion::polyast::Term::StringConst &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Base> {
  std::size_t operator()(const polyregion::polyast::Expr::Base &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Sin> {
  std::size_t operator()(const polyregion::polyast::Expr::Sin &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Cos> {
  std::size_t operator()(const polyregion::polyast::Expr::Cos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Tan> {
  std::size_t operator()(const polyregion::polyast::Expr::Tan &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Abs> {
  std::size_t operator()(const polyregion::polyast::Expr::Abs &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Add> {
  std::size_t operator()(const polyregion::polyast::Expr::Add &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Sub> {
  std::size_t operator()(const polyregion::polyast::Expr::Sub &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Mul> {
  std::size_t operator()(const polyregion::polyast::Expr::Mul &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Div> {
  std::size_t operator()(const polyregion::polyast::Expr::Div &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Rem> {
  std::size_t operator()(const polyregion::polyast::Expr::Rem &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Pow> {
  std::size_t operator()(const polyregion::polyast::Expr::Pow &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BNot> {
  std::size_t operator()(const polyregion::polyast::Expr::BNot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BAnd> {
  std::size_t operator()(const polyregion::polyast::Expr::BAnd &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BOr> {
  std::size_t operator()(const polyregion::polyast::Expr::BOr &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BXor> {
  std::size_t operator()(const polyregion::polyast::Expr::BXor &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BSL> {
  std::size_t operator()(const polyregion::polyast::Expr::BSL &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BSR> {
  std::size_t operator()(const polyregion::polyast::Expr::BSR &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Not> {
  std::size_t operator()(const polyregion::polyast::Expr::Not &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Eq> {
  std::size_t operator()(const polyregion::polyast::Expr::Eq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Neq> {
  std::size_t operator()(const polyregion::polyast::Expr::Neq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::And> {
  std::size_t operator()(const polyregion::polyast::Expr::And &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Or> {
  std::size_t operator()(const polyregion::polyast::Expr::Or &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Lte> {
  std::size_t operator()(const polyregion::polyast::Expr::Lte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Gte> {
  std::size_t operator()(const polyregion::polyast::Expr::Gte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Lt> {
  std::size_t operator()(const polyregion::polyast::Expr::Lt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Gt> {
  std::size_t operator()(const polyregion::polyast::Expr::Gt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Alias> {
  std::size_t operator()(const polyregion::polyast::Expr::Alias &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Invoke> {
  std::size_t operator()(const polyregion::polyast::Expr::Invoke &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Index> {
  std::size_t operator()(const polyregion::polyast::Expr::Index &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Base> {
  std::size_t operator()(const polyregion::polyast::Stmt::Base &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Comment> {
  std::size_t operator()(const polyregion::polyast::Stmt::Comment &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Var> {
  std::size_t operator()(const polyregion::polyast::Stmt::Var &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Mut> {
  std::size_t operator()(const polyregion::polyast::Stmt::Mut &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Update> {
  std::size_t operator()(const polyregion::polyast::Stmt::Update &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Effect> {
  std::size_t operator()(const polyregion::polyast::Stmt::Effect &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::While> {
  std::size_t operator()(const polyregion::polyast::Stmt::While &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Break> {
  std::size_t operator()(const polyregion::polyast::Stmt::Break &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Cont> {
  std::size_t operator()(const polyregion::polyast::Stmt::Cont &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Cond> {
  std::size_t operator()(const polyregion::polyast::Stmt::Cond &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Return> {
  std::size_t operator()(const polyregion::polyast::Stmt::Return &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::StructDef> {
  std::size_t operator()(const polyregion::polyast::StructDef &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Function> {
  std::size_t operator()(const polyregion::polyast::Function &) const noexcept;
};

}

#pragma clang diagnostic pop // -Wunknown-pragmas
