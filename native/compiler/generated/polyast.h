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

template <typename... T> //
constexpr bool operator==(const Alternative<T...> &l,const Alternative<T...> &r) {
  return unwrap(l) == unwrap(r);
}

template <typename... T> //
constexpr bool operator!=(const Alternative<T...> &l,const Alternative<T...> &r) {
  return unwrap(l) != unwrap(r);
}

template <typename R, typename... T> //
constexpr bool holds(const Alternative<T...> &l ) {
  return std::holds_alternative<R>(unwrap(l));
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



namespace TypeKind { 
struct None;
struct Ref;
struct Integral;
struct Fractional;
using Any = Alternative<None, Ref, Integral, Fractional>;
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
struct Unit;
struct Nothing;
struct String;
struct Struct;
struct Array;
struct Var;
struct Exec;
using Any = Alternative<Float, Double, Bool, Byte, Char, Short, Int, Long, Unit, Nothing, String, Struct, Array, Var, Exec>;
} // namespace Type


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
} // namespace Term
namespace NullaryIntrinsicKind { 
struct GpuGlobalIdxX;
struct GpuGlobalIdxY;
struct GpuGlobalIdxZ;
struct GpuGlobalSizeX;
struct GpuGlobalSizeY;
struct GpuGlobalSizeZ;
struct GpuGroupIdxX;
struct GpuGroupIdxY;
struct GpuGroupIdxZ;
struct GpuGroupSizeX;
struct GpuGroupSizeY;
struct GpuGroupSizeZ;
struct GpuLocalIdxX;
struct GpuLocalIdxY;
struct GpuLocalIdxZ;
struct GpuLocalSizeX;
struct GpuLocalSizeY;
struct GpuLocalSizeZ;
struct GpuGroupBarrier;
struct GpuGroupFence;
using Any = Alternative<GpuGlobalIdxX, GpuGlobalIdxY, GpuGlobalIdxZ, GpuGlobalSizeX, GpuGlobalSizeY, GpuGlobalSizeZ, GpuGroupIdxX, GpuGroupIdxY, GpuGroupIdxZ, GpuGroupSizeX, GpuGroupSizeY, GpuGroupSizeZ, GpuLocalIdxX, GpuLocalIdxY, GpuLocalIdxZ, GpuLocalSizeX, GpuLocalSizeY, GpuLocalSizeZ, GpuGroupBarrier, GpuGroupFence>;
} // namespace NullaryIntrinsicKind
namespace UnaryIntrinsicKind { 
struct Sin;
struct Cos;
struct Tan;
struct Asin;
struct Acos;
struct Atan;
struct Sinh;
struct Cosh;
struct Tanh;
struct Signum;
struct Abs;
struct Round;
struct Ceil;
struct Floor;
struct Rint;
struct Sqrt;
struct Cbrt;
struct Exp;
struct Expm1;
struct Log;
struct Log1p;
struct Log10;
struct BNot;
struct Pos;
struct Neg;
struct LogicNot;
using Any = Alternative<Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Signum, Abs, Round, Ceil, Floor, Rint, Sqrt, Cbrt, Exp, Expm1, Log, Log1p, Log10, BNot, Pos, Neg, LogicNot>;
} // namespace UnaryIntrinsicKind
namespace BinaryIntrinsicKind { 
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Rem;
struct Pow;
struct Min;
struct Max;
struct Atan2;
struct Hypot;
struct BAnd;
struct BOr;
struct BXor;
struct BSL;
struct BSR;
struct BZSR;
struct LogicEq;
struct LogicNeq;
struct LogicAnd;
struct LogicOr;
struct LogicLte;
struct LogicGte;
struct LogicLt;
struct LogicGt;
using Any = Alternative<Add, Sub, Mul, Div, Rem, Pow, Min, Max, Atan2, Hypot, BAnd, BOr, BXor, BSL, BSR, BZSR, LogicEq, LogicNeq, LogicAnd, LogicOr, LogicLte, LogicGte, LogicLt, LogicGt>;
} // namespace BinaryIntrinsicKind
namespace Expr { 
struct NullaryIntrinsic;
struct UnaryIntrinsic;
struct BinaryIntrinsic;
struct Cast;
struct Alias;
struct Invoke;
struct Index;
struct Alloc;
struct Suspend;
using Any = Alternative<NullaryIntrinsic, UnaryIntrinsic, BinaryIntrinsic, Cast, Alias, Invoke, Index, Alloc, Suspend>;
} // namespace Expr
namespace Stmt { 
struct Comment;
struct Var;
struct Mut;
struct Update;
struct While;
struct Break;
struct Cont;
struct Cond;
struct Return;
using Any = Alternative<Comment, Var, Mut, Update, While, Break, Cont, Cond, Return>;
} // namespace Stmt





struct EXPORT Sym {
  std::vector<std::string> fqn;
  explicit Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Sym &);
  EXPORT friend bool operator==(const Sym &, const Sym &);
};

struct EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Named &);
  EXPORT friend bool operator==(const Named &, const Named &);
};

namespace TypeKind { 

struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Any &);
  EXPORT friend bool operator==(const TypeKind::Base &, const TypeKind::Base &);
};

struct EXPORT None : TypeKind::Base {
  None() noexcept : TypeKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<None>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::None &);
  EXPORT friend bool operator==(const TypeKind::None &, const TypeKind::None &);
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

struct EXPORT Unit : Type::Base {
  Unit() noexcept : Type::Base(TypeKind::None()) {}
  EXPORT operator Any() const { return std::make_shared<Unit>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Unit &);
  EXPORT friend bool operator==(const Type::Unit &, const Type::Unit &);
};

struct EXPORT Nothing : Type::Base {
  Nothing() noexcept : Type::Base(TypeKind::None()) {}
  EXPORT operator Any() const { return std::make_shared<Nothing>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Nothing &);
  EXPORT friend bool operator==(const Type::Nothing &, const Type::Nothing &);
};

struct EXPORT String : Type::Base {
  String() noexcept : Type::Base(TypeKind::Ref()) {}
  EXPORT operator Any() const { return std::make_shared<String>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::String &);
  EXPORT friend bool operator==(const Type::String &, const Type::String &);
};

struct EXPORT Struct : Type::Base {
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)), tpeVars(std::move(tpeVars)), args(std::move(args)) {}
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

struct EXPORT Var : Type::Base {
  std::string name;
  explicit Var(std::string name) noexcept : Type::Base(TypeKind::None()), name(std::move(name)) {}
  EXPORT operator Any() const { return std::make_shared<Var>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Var &);
  EXPORT friend bool operator==(const Type::Var &, const Type::Var &);
};

struct EXPORT Exec : Type::Base {
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  Type::Any rtn;
  Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept : Type::Base(TypeKind::None()), tpeVars(std::move(tpeVars)), args(std::move(args)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<Exec>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Exec &);
  EXPORT friend bool operator==(const Type::Exec &, const Type::Exec &);
};
} // namespace Type


struct EXPORT Position {
  std::string file;
  int32_t line;
  int32_t col;
  Position(std::string file, int32_t line, int32_t col) noexcept : file(std::move(file)), line(line), col(col) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Position &);
  EXPORT friend bool operator==(const Position &, const Position &);
};

namespace Term { 

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
namespace NullaryIntrinsicKind { 

struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::Any &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::Base &, const NullaryIntrinsicKind::Base &);
};

struct EXPORT GpuGlobalIdxX : NullaryIntrinsicKind::Base {
  GpuGlobalIdxX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalIdxX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalIdxX &, const NullaryIntrinsicKind::GpuGlobalIdxX &);
};

struct EXPORT GpuGlobalIdxY : NullaryIntrinsicKind::Base {
  GpuGlobalIdxY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalIdxY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalIdxY &, const NullaryIntrinsicKind::GpuGlobalIdxY &);
};

struct EXPORT GpuGlobalIdxZ : NullaryIntrinsicKind::Base {
  GpuGlobalIdxZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalIdxZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalIdxZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalIdxZ &, const NullaryIntrinsicKind::GpuGlobalIdxZ &);
};

struct EXPORT GpuGlobalSizeX : NullaryIntrinsicKind::Base {
  GpuGlobalSizeX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalSizeX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalSizeX &, const NullaryIntrinsicKind::GpuGlobalSizeX &);
};

struct EXPORT GpuGlobalSizeY : NullaryIntrinsicKind::Base {
  GpuGlobalSizeY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalSizeY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalSizeY &, const NullaryIntrinsicKind::GpuGlobalSizeY &);
};

struct EXPORT GpuGlobalSizeZ : NullaryIntrinsicKind::Base {
  GpuGlobalSizeZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGlobalSizeZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGlobalSizeZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGlobalSizeZ &, const NullaryIntrinsicKind::GpuGlobalSizeZ &);
};

struct EXPORT GpuGroupIdxX : NullaryIntrinsicKind::Base {
  GpuGroupIdxX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupIdxX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupIdxX &, const NullaryIntrinsicKind::GpuGroupIdxX &);
};

struct EXPORT GpuGroupIdxY : NullaryIntrinsicKind::Base {
  GpuGroupIdxY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupIdxY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupIdxY &, const NullaryIntrinsicKind::GpuGroupIdxY &);
};

struct EXPORT GpuGroupIdxZ : NullaryIntrinsicKind::Base {
  GpuGroupIdxZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupIdxZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupIdxZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupIdxZ &, const NullaryIntrinsicKind::GpuGroupIdxZ &);
};

struct EXPORT GpuGroupSizeX : NullaryIntrinsicKind::Base {
  GpuGroupSizeX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupSizeX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupSizeX &, const NullaryIntrinsicKind::GpuGroupSizeX &);
};

struct EXPORT GpuGroupSizeY : NullaryIntrinsicKind::Base {
  GpuGroupSizeY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupSizeY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupSizeY &, const NullaryIntrinsicKind::GpuGroupSizeY &);
};

struct EXPORT GpuGroupSizeZ : NullaryIntrinsicKind::Base {
  GpuGroupSizeZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupSizeZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupSizeZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupSizeZ &, const NullaryIntrinsicKind::GpuGroupSizeZ &);
};

struct EXPORT GpuLocalIdxX : NullaryIntrinsicKind::Base {
  GpuLocalIdxX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalIdxX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalIdxX &, const NullaryIntrinsicKind::GpuLocalIdxX &);
};

struct EXPORT GpuLocalIdxY : NullaryIntrinsicKind::Base {
  GpuLocalIdxY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalIdxY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalIdxY &, const NullaryIntrinsicKind::GpuLocalIdxY &);
};

struct EXPORT GpuLocalIdxZ : NullaryIntrinsicKind::Base {
  GpuLocalIdxZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalIdxZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalIdxZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalIdxZ &, const NullaryIntrinsicKind::GpuLocalIdxZ &);
};

struct EXPORT GpuLocalSizeX : NullaryIntrinsicKind::Base {
  GpuLocalSizeX() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalSizeX>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeX &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalSizeX &, const NullaryIntrinsicKind::GpuLocalSizeX &);
};

struct EXPORT GpuLocalSizeY : NullaryIntrinsicKind::Base {
  GpuLocalSizeY() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalSizeY>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeY &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalSizeY &, const NullaryIntrinsicKind::GpuLocalSizeY &);
};

struct EXPORT GpuLocalSizeZ : NullaryIntrinsicKind::Base {
  GpuLocalSizeZ() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuLocalSizeZ>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuLocalSizeZ &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuLocalSizeZ &, const NullaryIntrinsicKind::GpuLocalSizeZ &);
};

struct EXPORT GpuGroupBarrier : NullaryIntrinsicKind::Base {
  GpuGroupBarrier() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupBarrier>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupBarrier &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupBarrier &, const NullaryIntrinsicKind::GpuGroupBarrier &);
};

struct EXPORT GpuGroupFence : NullaryIntrinsicKind::Base {
  GpuGroupFence() noexcept : NullaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<GpuGroupFence>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const NullaryIntrinsicKind::GpuGroupFence &);
  EXPORT friend bool operator==(const NullaryIntrinsicKind::GpuGroupFence &, const NullaryIntrinsicKind::GpuGroupFence &);
};
} // namespace NullaryIntrinsicKind
namespace UnaryIntrinsicKind { 

struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Any &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Base &, const UnaryIntrinsicKind::Base &);
};

struct EXPORT Sin : UnaryIntrinsicKind::Base {
  Sin() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Sin>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Sin &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Sin &, const UnaryIntrinsicKind::Sin &);
};

struct EXPORT Cos : UnaryIntrinsicKind::Base {
  Cos() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Cos>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Cos &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Cos &, const UnaryIntrinsicKind::Cos &);
};

struct EXPORT Tan : UnaryIntrinsicKind::Base {
  Tan() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Tan>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Tan &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Tan &, const UnaryIntrinsicKind::Tan &);
};

struct EXPORT Asin : UnaryIntrinsicKind::Base {
  Asin() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Asin>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Asin &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Asin &, const UnaryIntrinsicKind::Asin &);
};

struct EXPORT Acos : UnaryIntrinsicKind::Base {
  Acos() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Acos>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Acos &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Acos &, const UnaryIntrinsicKind::Acos &);
};

struct EXPORT Atan : UnaryIntrinsicKind::Base {
  Atan() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Atan>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Atan &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Atan &, const UnaryIntrinsicKind::Atan &);
};

struct EXPORT Sinh : UnaryIntrinsicKind::Base {
  Sinh() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Sinh>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Sinh &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Sinh &, const UnaryIntrinsicKind::Sinh &);
};

struct EXPORT Cosh : UnaryIntrinsicKind::Base {
  Cosh() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Cosh>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Cosh &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Cosh &, const UnaryIntrinsicKind::Cosh &);
};

struct EXPORT Tanh : UnaryIntrinsicKind::Base {
  Tanh() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Tanh>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Tanh &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Tanh &, const UnaryIntrinsicKind::Tanh &);
};

struct EXPORT Signum : UnaryIntrinsicKind::Base {
  Signum() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Signum>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Signum &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Signum &, const UnaryIntrinsicKind::Signum &);
};

struct EXPORT Abs : UnaryIntrinsicKind::Base {
  Abs() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Abs>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Abs &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Abs &, const UnaryIntrinsicKind::Abs &);
};

struct EXPORT Round : UnaryIntrinsicKind::Base {
  Round() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Round>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Round &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Round &, const UnaryIntrinsicKind::Round &);
};

struct EXPORT Ceil : UnaryIntrinsicKind::Base {
  Ceil() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Ceil>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Ceil &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Ceil &, const UnaryIntrinsicKind::Ceil &);
};

struct EXPORT Floor : UnaryIntrinsicKind::Base {
  Floor() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Floor>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Floor &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Floor &, const UnaryIntrinsicKind::Floor &);
};

struct EXPORT Rint : UnaryIntrinsicKind::Base {
  Rint() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Rint>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Rint &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Rint &, const UnaryIntrinsicKind::Rint &);
};

struct EXPORT Sqrt : UnaryIntrinsicKind::Base {
  Sqrt() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Sqrt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Sqrt &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Sqrt &, const UnaryIntrinsicKind::Sqrt &);
};

struct EXPORT Cbrt : UnaryIntrinsicKind::Base {
  Cbrt() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Cbrt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Cbrt &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Cbrt &, const UnaryIntrinsicKind::Cbrt &);
};

struct EXPORT Exp : UnaryIntrinsicKind::Base {
  Exp() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Exp>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Exp &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Exp &, const UnaryIntrinsicKind::Exp &);
};

struct EXPORT Expm1 : UnaryIntrinsicKind::Base {
  Expm1() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Expm1>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Expm1 &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Expm1 &, const UnaryIntrinsicKind::Expm1 &);
};

struct EXPORT Log : UnaryIntrinsicKind::Base {
  Log() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Log>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Log &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Log &, const UnaryIntrinsicKind::Log &);
};

struct EXPORT Log1p : UnaryIntrinsicKind::Base {
  Log1p() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Log1p>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Log1p &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Log1p &, const UnaryIntrinsicKind::Log1p &);
};

struct EXPORT Log10 : UnaryIntrinsicKind::Base {
  Log10() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Log10>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Log10 &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Log10 &, const UnaryIntrinsicKind::Log10 &);
};

struct EXPORT BNot : UnaryIntrinsicKind::Base {
  BNot() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BNot>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::BNot &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::BNot &, const UnaryIntrinsicKind::BNot &);
};

struct EXPORT Pos : UnaryIntrinsicKind::Base {
  Pos() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Pos>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Pos &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Pos &, const UnaryIntrinsicKind::Pos &);
};

struct EXPORT Neg : UnaryIntrinsicKind::Base {
  Neg() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Neg>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::Neg &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::Neg &, const UnaryIntrinsicKind::Neg &);
};

struct EXPORT LogicNot : UnaryIntrinsicKind::Base {
  LogicNot() noexcept : UnaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicNot>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const UnaryIntrinsicKind::LogicNot &);
  EXPORT friend bool operator==(const UnaryIntrinsicKind::LogicNot &, const UnaryIntrinsicKind::LogicNot &);
};
} // namespace UnaryIntrinsicKind
namespace BinaryIntrinsicKind { 

struct EXPORT Base {
  protected:
  Base() = default;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Any &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Base &, const BinaryIntrinsicKind::Base &);
};

struct EXPORT Add : BinaryIntrinsicKind::Base {
  Add() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Add>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Add &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Add &, const BinaryIntrinsicKind::Add &);
};

struct EXPORT Sub : BinaryIntrinsicKind::Base {
  Sub() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Sub>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Sub &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Sub &, const BinaryIntrinsicKind::Sub &);
};

struct EXPORT Mul : BinaryIntrinsicKind::Base {
  Mul() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Mul>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Mul &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Mul &, const BinaryIntrinsicKind::Mul &);
};

struct EXPORT Div : BinaryIntrinsicKind::Base {
  Div() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Div>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Div &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Div &, const BinaryIntrinsicKind::Div &);
};

struct EXPORT Rem : BinaryIntrinsicKind::Base {
  Rem() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Rem>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Rem &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Rem &, const BinaryIntrinsicKind::Rem &);
};

struct EXPORT Pow : BinaryIntrinsicKind::Base {
  Pow() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Pow>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Pow &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Pow &, const BinaryIntrinsicKind::Pow &);
};

struct EXPORT Min : BinaryIntrinsicKind::Base {
  Min() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Min>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Min &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Min &, const BinaryIntrinsicKind::Min &);
};

struct EXPORT Max : BinaryIntrinsicKind::Base {
  Max() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Max>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Max &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Max &, const BinaryIntrinsicKind::Max &);
};

struct EXPORT Atan2 : BinaryIntrinsicKind::Base {
  Atan2() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Atan2>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Atan2 &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Atan2 &, const BinaryIntrinsicKind::Atan2 &);
};

struct EXPORT Hypot : BinaryIntrinsicKind::Base {
  Hypot() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<Hypot>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::Hypot &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::Hypot &, const BinaryIntrinsicKind::Hypot &);
};

struct EXPORT BAnd : BinaryIntrinsicKind::Base {
  BAnd() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BAnd>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BAnd &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BAnd &, const BinaryIntrinsicKind::BAnd &);
};

struct EXPORT BOr : BinaryIntrinsicKind::Base {
  BOr() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BOr>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BOr &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BOr &, const BinaryIntrinsicKind::BOr &);
};

struct EXPORT BXor : BinaryIntrinsicKind::Base {
  BXor() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BXor>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BXor &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BXor &, const BinaryIntrinsicKind::BXor &);
};

struct EXPORT BSL : BinaryIntrinsicKind::Base {
  BSL() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BSL>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BSL &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BSL &, const BinaryIntrinsicKind::BSL &);
};

struct EXPORT BSR : BinaryIntrinsicKind::Base {
  BSR() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BSR>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BSR &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BSR &, const BinaryIntrinsicKind::BSR &);
};

struct EXPORT BZSR : BinaryIntrinsicKind::Base {
  BZSR() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<BZSR>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::BZSR &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::BZSR &, const BinaryIntrinsicKind::BZSR &);
};

struct EXPORT LogicEq : BinaryIntrinsicKind::Base {
  LogicEq() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicEq>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicEq &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicEq &, const BinaryIntrinsicKind::LogicEq &);
};

struct EXPORT LogicNeq : BinaryIntrinsicKind::Base {
  LogicNeq() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicNeq>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicNeq &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicNeq &, const BinaryIntrinsicKind::LogicNeq &);
};

struct EXPORT LogicAnd : BinaryIntrinsicKind::Base {
  LogicAnd() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicAnd>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicAnd &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicAnd &, const BinaryIntrinsicKind::LogicAnd &);
};

struct EXPORT LogicOr : BinaryIntrinsicKind::Base {
  LogicOr() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicOr>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicOr &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicOr &, const BinaryIntrinsicKind::LogicOr &);
};

struct EXPORT LogicLte : BinaryIntrinsicKind::Base {
  LogicLte() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicLte>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicLte &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicLte &, const BinaryIntrinsicKind::LogicLte &);
};

struct EXPORT LogicGte : BinaryIntrinsicKind::Base {
  LogicGte() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicGte>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicGte &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicGte &, const BinaryIntrinsicKind::LogicGte &);
};

struct EXPORT LogicLt : BinaryIntrinsicKind::Base {
  LogicLt() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicLt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicLt &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicLt &, const BinaryIntrinsicKind::LogicLt &);
};

struct EXPORT LogicGt : BinaryIntrinsicKind::Base {
  LogicGt() noexcept : BinaryIntrinsicKind::Base() {}
  EXPORT operator Any() const { return std::make_shared<LogicGt>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const BinaryIntrinsicKind::LogicGt &);
  EXPORT friend bool operator==(const BinaryIntrinsicKind::LogicGt &, const BinaryIntrinsicKind::LogicGt &);
};
} // namespace BinaryIntrinsicKind
namespace Expr { 

struct EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Any &);
  EXPORT friend bool operator==(const Expr::Base &, const Expr::Base &);
};
EXPORT Type::Any tpe(const Expr::Any&);

struct EXPORT NullaryIntrinsic : Expr::Base {
  NullaryIntrinsicKind::Any kind;
  Type::Any rtn;
  NullaryIntrinsic(NullaryIntrinsicKind::Any kind, Type::Any rtn) noexcept : Expr::Base(rtn), kind(std::move(kind)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<NullaryIntrinsic>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::NullaryIntrinsic &);
  EXPORT friend bool operator==(const Expr::NullaryIntrinsic &, const Expr::NullaryIntrinsic &);
};

struct EXPORT UnaryIntrinsic : Expr::Base {
  Term::Any lhs;
  UnaryIntrinsicKind::Any kind;
  Type::Any rtn;
  UnaryIntrinsic(Term::Any lhs, UnaryIntrinsicKind::Any kind, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), kind(std::move(kind)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<UnaryIntrinsic>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::UnaryIntrinsic &);
  EXPORT friend bool operator==(const Expr::UnaryIntrinsic &, const Expr::UnaryIntrinsic &);
};

struct EXPORT BinaryIntrinsic : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  BinaryIntrinsicKind::Any kind;
  Type::Any rtn;
  BinaryIntrinsic(Term::Any lhs, Term::Any rhs, BinaryIntrinsicKind::Any kind, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), kind(std::move(kind)), rtn(std::move(rtn)) {}
  EXPORT operator Any() const { return std::make_shared<BinaryIntrinsic>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::BinaryIntrinsic &);
  EXPORT friend bool operator==(const Expr::BinaryIntrinsic &, const Expr::BinaryIntrinsic &);
};

struct EXPORT Cast : Expr::Base {
  Term::Any from;
  Type::Any as;
  Cast(Term::Any from, Type::Any as) noexcept : Expr::Base(as), from(std::move(from)), as(std::move(as)) {}
  EXPORT operator Any() const { return std::make_shared<Cast>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cast &);
  EXPORT friend bool operator==(const Expr::Cast &, const Expr::Cast &);
};

struct EXPORT Alias : Expr::Base {
  Term::Any ref;
  explicit Alias(Term::Any ref) noexcept : Expr::Base(Term::tpe(ref)), ref(std::move(ref)) {}
  EXPORT operator Any() const { return std::make_shared<Alias>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
  EXPORT friend bool operator==(const Expr::Alias &, const Expr::Alias &);
};

struct EXPORT Invoke : Expr::Base {
  Sym name;
  std::vector<Type::Any> tpeArgs;
  std::optional<Term::Any> receiver;
  std::vector<Term::Any> args;
  Type::Any rtn;
  Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, Type::Any rtn) noexcept : Expr::Base(rtn), name(std::move(name)), tpeArgs(std::move(tpeArgs)), receiver(std::move(receiver)), args(std::move(args)), rtn(std::move(rtn)) {}
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

struct EXPORT Alloc : Expr::Base {
  Type::Array witness;
  Term::Any size;
  Alloc(Type::Array witness, Term::Any size) noexcept : Expr::Base(witness), witness(std::move(witness)), size(std::move(size)) {}
  EXPORT operator Any() const { return std::make_shared<Alloc>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
  EXPORT friend bool operator==(const Expr::Alloc &, const Expr::Alloc &);
};

struct EXPORT Suspend : Expr::Base {
  std::vector<Named> args;
  std::vector<Stmt::Any> stmts;
  Type::Any rtn;
  Type::Exec shape;
  Suspend(std::vector<Named> args, std::vector<Stmt::Any> stmts, Type::Any rtn, Type::Exec shape) noexcept : Expr::Base(shape), args(std::move(args)), stmts(std::move(stmts)), rtn(std::move(rtn)), shape(std::move(shape)) {}
  EXPORT operator Any() const { return std::make_shared<Suspend>(*this); };
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Suspend &);
  EXPORT friend bool operator==(const Expr::Suspend &, const Expr::Suspend &);
};
} // namespace Expr
namespace Stmt { 

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
  bool copy;
  Mut(Term::Select name, Expr::Any expr, bool copy) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)), copy(copy) {}
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

struct EXPORT While : Stmt::Base {
  std::vector<Stmt::Any> tests;
  Term::Any cond;
  std::vector<Stmt::Any> body;
  While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), tests(std::move(tests)), cond(std::move(cond)), body(std::move(body)) {}
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
  std::vector<std::string> tpeVars;
  std::vector<Named> members;
  StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<Named> members) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), members(std::move(members)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
  EXPORT friend bool operator==(const StructDef &, const StructDef &);
};

struct EXPORT Signature {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  Type::Any rtn;
  Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, Type::Any rtn) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), rtn(std::move(rtn)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Signature &);
  EXPORT friend bool operator==(const Signature &, const Signature &);
};

struct EXPORT Function {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Named> receiver;
  std::vector<Named> args;
  std::vector<Named> captures;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  Function(Sym name, std::vector<std::string> tpeVars, std::optional<Named> receiver, std::vector<Named> args, std::vector<Named> captures, Type::Any rtn, std::vector<Stmt::Any> body) noexcept : name(std::move(name)), tpeVars(std::move(tpeVars)), receiver(std::move(receiver)), args(std::move(args)), captures(std::move(captures)), rtn(std::move(rtn)), body(std::move(body)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
  EXPORT friend bool operator==(const Function &, const Function &);
};

struct EXPORT Program {
  Function entry;
  std::vector<Function> functions;
  std::vector<StructDef> defs;
  Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept : entry(std::move(entry)), functions(std::move(functions)), defs(std::move(defs)) {}
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
  EXPORT friend bool operator==(const Program &, const Program &);
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

template <typename ...T> struct std::hash<polyregion::polyast::Alternative<T...>> {
  std::size_t operator()(polyregion::polyast::Alternative<T...> const &x) const noexcept {
    return std::hash<std::variant<T...>>()(polyregion::polyast::unwrap(x));
  }
};

template <> struct std::hash<polyregion::polyast::Sym> {
  std::size_t operator()(const polyregion::polyast::Sym &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Named> {
  std::size_t operator()(const polyregion::polyast::Named &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeKind::None> {
  std::size_t operator()(const polyregion::polyast::TypeKind::None &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::Type::Unit> {
  std::size_t operator()(const polyregion::polyast::Type::Unit &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Nothing> {
  std::size_t operator()(const polyregion::polyast::Type::Nothing &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::String> {
  std::size_t operator()(const polyregion::polyast::Type::String &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Struct> {
  std::size_t operator()(const polyregion::polyast::Type::Struct &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Array> {
  std::size_t operator()(const polyregion::polyast::Type::Array &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Var> {
  std::size_t operator()(const polyregion::polyast::Type::Var &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Exec> {
  std::size_t operator()(const polyregion::polyast::Type::Exec &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Position> {
  std::size_t operator()(const polyregion::polyast::Position &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalIdxZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGlobalSizeZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupIdxZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupSizeZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalIdxZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeX> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeX &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeY> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeY &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeZ> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuLocalSizeZ &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupBarrier> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupBarrier &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::NullaryIntrinsicKind::GpuGroupFence> {
  std::size_t operator()(const polyregion::polyast::NullaryIntrinsicKind::GpuGroupFence &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Sin> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Sin &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Cos> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Cos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Tan> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Tan &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Asin> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Asin &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Acos> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Acos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Atan> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Atan &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Sinh> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Sinh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Cosh> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Cosh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Tanh> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Tanh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Signum> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Signum &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Abs> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Abs &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Round> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Round &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Ceil> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Ceil &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Floor> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Floor &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Rint> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Rint &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Sqrt> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Sqrt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Cbrt> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Cbrt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Exp> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Exp &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Expm1> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Expm1 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Log> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Log &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Log1p> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Log1p &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Log10> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Log10 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::BNot> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::BNot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Pos> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Pos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::Neg> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::Neg &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::UnaryIntrinsicKind::LogicNot> {
  std::size_t operator()(const polyregion::polyast::UnaryIntrinsicKind::LogicNot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Add> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Add &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Sub> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Sub &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Mul> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Mul &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Div> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Div &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Rem> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Rem &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Pow> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Pow &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Min> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Min &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Max> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Max &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Atan2> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Atan2 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::Hypot> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::Hypot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BAnd> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BAnd &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BOr> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BOr &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BXor> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BXor &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BSL> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BSL &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BSR> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BSR &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::BZSR> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::BZSR &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicEq> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicEq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicNeq> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicNeq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicAnd> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicAnd &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicOr> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicOr &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicLte> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicLte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicGte> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicGte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicLt> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicLt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::BinaryIntrinsicKind::LogicGt> {
  std::size_t operator()(const polyregion::polyast::BinaryIntrinsicKind::LogicGt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::NullaryIntrinsic> {
  std::size_t operator()(const polyregion::polyast::Expr::NullaryIntrinsic &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::UnaryIntrinsic> {
  std::size_t operator()(const polyregion::polyast::Expr::UnaryIntrinsic &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::BinaryIntrinsic> {
  std::size_t operator()(const polyregion::polyast::Expr::BinaryIntrinsic &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Cast> {
  std::size_t operator()(const polyregion::polyast::Expr::Cast &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::Expr::Alloc> {
  std::size_t operator()(const polyregion::polyast::Expr::Alloc &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Suspend> {
  std::size_t operator()(const polyregion::polyast::Expr::Suspend &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::Signature> {
  std::size_t operator()(const polyregion::polyast::Signature &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Function> {
  std::size_t operator()(const polyregion::polyast::Function &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Program> {
  std::size_t operator()(const polyregion::polyast::Program &) const noexcept;
};

}

#pragma clang diagnostic pop // -Wunknown-pragmas
