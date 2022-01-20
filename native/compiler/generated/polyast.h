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
  std::vector<Type::Any> args;
  Struct(Sym name, std::vector<Type::Any> args) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)), args(std::move(args)) {}
  EXPORT operator Any() const { return std::make_shared<Struct>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
  EXPORT friend bool operator==(const Type::Struct &, const Type::Struct &);
};

struct EXPORT Array : Type::Base {
  Type::Any component;
  explicit Array(Type::Any component) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)) {}
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
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Mod;
struct Pow;
struct Inv;
struct Eq;
struct Lte;
struct Gte;
struct Lt;
struct Gt;
struct Alias;
struct Invoke;
struct Index;
using Any = Alternative<Sin, Cos, Tan, Add, Sub, Mul, Div, Mod, Pow, Inv, Eq, Lte, Gte, Lt, Gt, Alias, Invoke, Index>;
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

struct EXPORT Mod : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Mod(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Mod>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Mod &);
  EXPORT friend bool operator==(const Expr::Mod &, const Expr::Mod &);
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

struct EXPORT Inv : Expr::Base {
  Term::Any lhs;
  explicit Inv(Term::Any lhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)) {}
  EXPORT operator Any() const { return std::make_shared<Inv>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Inv &);
  EXPORT friend bool operator==(const Expr::Inv &, const Expr::Inv &);
};

struct EXPORT Eq : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Eq(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  EXPORT operator Any() const { return std::make_shared<Eq>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Eq &);
  EXPORT friend bool operator==(const Expr::Eq &, const Expr::Eq &);
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


struct EXPORT Function {
  std::string name;
  std::vector<Named> args;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  Function(std::string name, std::vector<Named> args, Type::Any rtn, std::vector<Stmt::Any> body) noexcept : name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)), body(std::move(body)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
  EXPORT friend bool operator==(const Function &, const Function &);
};

struct EXPORT StructDef {
  std::vector<Named> members;
  explicit StructDef(std::vector<Named> members) noexcept : members(std::move(members)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
  EXPORT friend bool operator==(const StructDef &, const StructDef &);
};

} // namespace polyregion::polyast
#pragma clang diagnostic pop

