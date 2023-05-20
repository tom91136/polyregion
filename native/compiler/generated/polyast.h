#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif

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
#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma ide diagnostic ignored "google-explicit-constructor"
#endif



namespace TypeKind { 
struct None;
struct Ref;
struct Integral;
struct Fractional;
using Any = Alternative<None, Ref, Integral, Fractional>;
} // namespace TypeKind
namespace Type { 
struct Float16;
struct Float32;
struct Float64;
struct IntU8;
struct IntU16;
struct IntU32;
struct IntU64;
struct IntS8;
struct IntS16;
struct IntS32;
struct IntS64;
struct Nothing;
struct Unit0;
struct Bool1;
struct Struct;
struct Array;
struct Var;
struct Exec;
using Any = Alternative<Float16, Float32, Float64, IntU8, IntU16, IntU32, IntU64, IntS8, IntS16, IntS32, IntS64, Nothing, Unit0, Bool1, Struct, Array, Var, Exec>;
} // namespace Type


namespace Term { 
struct Select;
struct Poison;
struct Float16Const;
struct Float32Const;
struct Float64Const;
struct IntU8Const;
struct IntU16Const;
struct IntU32Const;
struct IntU64Const;
struct IntS8Const;
struct IntS16Const;
struct IntS32Const;
struct IntS64Const;
struct Unit0Const;
struct Bool1Const;
using Any = Alternative<Select, Poison, Float16Const, Float32Const, Float64Const, IntU8Const, IntU16Const, IntU32Const, IntU64Const, IntS8Const, IntS16Const, IntS32Const, IntS64Const, Unit0Const, Bool1Const>;
} // namespace Term
namespace TypeSpace { 
struct Global;
struct Local;
using Any = Alternative<Global, Local>;
} // namespace TypeSpace


namespace Spec { 
struct Assert;
struct GpuBarrierGlobal;
struct GpuBarrierLocal;
struct GpuBarrierAll;
struct GpuFenceGlobal;
struct GpuFenceLocal;
struct GpuFenceAll;
struct GpuGlobalIdx;
struct GpuGlobalSize;
struct GpuGroupIdx;
struct GpuGroupSize;
struct GpuLocalIdx;
struct GpuLocalSize;
using Any = Alternative<Assert, GpuBarrierGlobal, GpuBarrierLocal, GpuBarrierAll, GpuFenceGlobal, GpuFenceLocal, GpuFenceAll, GpuGlobalIdx, GpuGlobalSize, GpuGroupIdx, GpuGroupSize, GpuLocalIdx, GpuLocalSize>;
} // namespace Spec
namespace Intr { 
struct BNot;
struct LogicNot;
struct Pos;
struct Neg;
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Rem;
struct Min;
struct Max;
struct BAnd;
struct BOr;
struct BXor;
struct BSL;
struct BSR;
struct BZSR;
struct LogicAnd;
struct LogicOr;
struct LogicEq;
struct LogicNeq;
struct LogicLte;
struct LogicGte;
struct LogicLt;
struct LogicGt;
using Any = Alternative<BNot, LogicNot, Pos, Neg, Add, Sub, Mul, Div, Rem, Min, Max, BAnd, BOr, BXor, BSL, BSR, BZSR, LogicAnd, LogicOr, LogicEq, LogicNeq, LogicLte, LogicGte, LogicLt, LogicGt>;
} // namespace Intr
namespace Math { 
struct Abs;
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
struct Pow;
struct Atan2;
struct Hypot;
using Any = Alternative<Abs, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Signum, Round, Ceil, Floor, Rint, Sqrt, Cbrt, Exp, Expm1, Log, Log1p, Log10, Pow, Atan2, Hypot>;
} // namespace Math
namespace Expr { 
struct SpecOp;
struct MathOp;
struct IntrOp;
struct Cast;
struct Alias;
struct Index;
struct Alloc;
struct Invoke;
using Any = Alternative<SpecOp, MathOp, IntrOp, Cast, Alias, Index, Alloc, Invoke>;
} // namespace Expr
namespace Stmt { 
struct Block;
struct Comment;
struct Var;
struct Mut;
struct Update;
struct While;
struct Break;
struct Cont;
struct Cond;
struct Return;
using Any = Alternative<Block, Comment, Var, Mut, Update, While, Break, Cont, Cond, Return>;
} // namespace Stmt


namespace FunctionKind { 
struct Internal;
struct Exported;
using Any = Alternative<Internal, Exported>;
} // namespace FunctionKind
namespace FunctionAttr { 
struct FPRelaxed;
struct FPStrict;
using Any = Alternative<FPRelaxed, FPStrict>;
} // namespace FunctionAttr





struct EXPORT Sym {
  std::vector<std::string> fqn;
  explicit Sym(std::vector<std::string> fqn) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Sym &);
  EXPORT friend bool operator==(const Sym &, const Sym &);
};

struct EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  Named(std::string symbol, Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Named &);
  EXPORT friend bool operator==(const Named &, const Named &);
};

namespace TypeKind { 

struct EXPORT Base {
  protected:
  Base();
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Any &);
  EXPORT friend bool operator==(const TypeKind::Base &, const TypeKind::Base &);
};

struct EXPORT None : TypeKind::Base {
  None() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::None &);
  EXPORT friend bool operator==(const TypeKind::None &, const TypeKind::None &);
};

struct EXPORT Ref : TypeKind::Base {
  Ref() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &);
  EXPORT friend bool operator==(const TypeKind::Ref &, const TypeKind::Ref &);
};

struct EXPORT Integral : TypeKind::Base {
  Integral() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &);
  EXPORT friend bool operator==(const TypeKind::Integral &, const TypeKind::Integral &);
};

struct EXPORT Fractional : TypeKind::Base {
  Fractional() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
  EXPORT friend bool operator==(const TypeKind::Fractional &, const TypeKind::Fractional &);
};
} // namespace TypeKind
namespace Type { 

struct EXPORT Base {
  TypeKind::Any kind;
  protected:
  explicit Base(TypeKind::Any kind) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Any &);
  EXPORT friend bool operator==(const Type::Base &, const Type::Base &);
};
EXPORT TypeKind::Any kind(const Type::Any&);

struct EXPORT Float16 : Type::Base {
  Float16() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float16 &);
  EXPORT friend bool operator==(const Type::Float16 &, const Type::Float16 &);
};

struct EXPORT Float32 : Type::Base {
  Float32() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float32 &);
  EXPORT friend bool operator==(const Type::Float32 &, const Type::Float32 &);
};

struct EXPORT Float64 : Type::Base {
  Float64() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float64 &);
  EXPORT friend bool operator==(const Type::Float64 &, const Type::Float64 &);
};

struct EXPORT IntU8 : Type::Base {
  IntU8() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU8 &);
  EXPORT friend bool operator==(const Type::IntU8 &, const Type::IntU8 &);
};

struct EXPORT IntU16 : Type::Base {
  IntU16() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU16 &);
  EXPORT friend bool operator==(const Type::IntU16 &, const Type::IntU16 &);
};

struct EXPORT IntU32 : Type::Base {
  IntU32() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU32 &);
  EXPORT friend bool operator==(const Type::IntU32 &, const Type::IntU32 &);
};

struct EXPORT IntU64 : Type::Base {
  IntU64() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU64 &);
  EXPORT friend bool operator==(const Type::IntU64 &, const Type::IntU64 &);
};

struct EXPORT IntS8 : Type::Base {
  IntS8() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS8 &);
  EXPORT friend bool operator==(const Type::IntS8 &, const Type::IntS8 &);
};

struct EXPORT IntS16 : Type::Base {
  IntS16() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS16 &);
  EXPORT friend bool operator==(const Type::IntS16 &, const Type::IntS16 &);
};

struct EXPORT IntS32 : Type::Base {
  IntS32() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS32 &);
  EXPORT friend bool operator==(const Type::IntS32 &, const Type::IntS32 &);
};

struct EXPORT IntS64 : Type::Base {
  IntS64() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS64 &);
  EXPORT friend bool operator==(const Type::IntS64 &, const Type::IntS64 &);
};

struct EXPORT Nothing : Type::Base {
  Nothing() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Nothing &);
  EXPORT friend bool operator==(const Type::Nothing &, const Type::Nothing &);
};

struct EXPORT Unit0 : Type::Base {
  Unit0() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Unit0 &);
  EXPORT friend bool operator==(const Type::Unit0 &, const Type::Unit0 &);
};

struct EXPORT Bool1 : Type::Base {
  Bool1() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Bool1 &);
  EXPORT friend bool operator==(const Type::Bool1 &, const Type::Bool1 &);
};

struct EXPORT Struct : Type::Base {
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  std::vector<Sym> parents;
  Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args, std::vector<Sym> parents) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
  EXPORT friend bool operator==(const Type::Struct &, const Type::Struct &);
};

struct EXPORT Array : Type::Base {
  Type::Any component;
  TypeSpace::Any space;
  Array(Type::Any component, TypeSpace::Any space) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Array &);
  EXPORT friend bool operator==(const Type::Array &, const Type::Array &);
};

struct EXPORT Var : Type::Base {
  std::string name;
  explicit Var(std::string name) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Var &);
  EXPORT friend bool operator==(const Type::Var &, const Type::Var &);
};

struct EXPORT Exec : Type::Base {
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  Type::Any rtn;
  Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Exec &);
  EXPORT friend bool operator==(const Type::Exec &, const Type::Exec &);
};
} // namespace Type


struct EXPORT SourcePosition {
  std::string file;
  int32_t line;
  std::optional<int32_t> col;
  SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const SourcePosition &);
  EXPORT friend bool operator==(const SourcePosition &, const SourcePosition &);
};

namespace Term { 

struct EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Any &);
  EXPORT friend bool operator==(const Term::Base &, const Term::Base &);
};
EXPORT Type::Any tpe(const Term::Any&);

struct EXPORT Select : Term::Base {
  std::vector<Named> init;
  Named last;
  Select(std::vector<Named> init, Named last) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Select &);
  EXPORT friend bool operator==(const Term::Select &, const Term::Select &);
};

struct EXPORT Poison : Term::Base {
  Type::Any t;
  explicit Poison(Type::Any t) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Poison &);
  EXPORT friend bool operator==(const Term::Poison &, const Term::Poison &);
};

struct EXPORT Float16Const : Term::Base {
  float value;
  explicit Float16Const(float value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float16Const &);
  EXPORT friend bool operator==(const Term::Float16Const &, const Term::Float16Const &);
};

struct EXPORT Float32Const : Term::Base {
  float value;
  explicit Float32Const(float value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float32Const &);
  EXPORT friend bool operator==(const Term::Float32Const &, const Term::Float32Const &);
};

struct EXPORT Float64Const : Term::Base {
  double value;
  explicit Float64Const(double value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float64Const &);
  EXPORT friend bool operator==(const Term::Float64Const &, const Term::Float64Const &);
};

struct EXPORT IntU8Const : Term::Base {
  int8_t value;
  explicit IntU8Const(int8_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU8Const &);
  EXPORT friend bool operator==(const Term::IntU8Const &, const Term::IntU8Const &);
};

struct EXPORT IntU16Const : Term::Base {
  uint16_t value;
  explicit IntU16Const(uint16_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU16Const &);
  EXPORT friend bool operator==(const Term::IntU16Const &, const Term::IntU16Const &);
};

struct EXPORT IntU32Const : Term::Base {
  int32_t value;
  explicit IntU32Const(int32_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU32Const &);
  EXPORT friend bool operator==(const Term::IntU32Const &, const Term::IntU32Const &);
};

struct EXPORT IntU64Const : Term::Base {
  int64_t value;
  explicit IntU64Const(int64_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU64Const &);
  EXPORT friend bool operator==(const Term::IntU64Const &, const Term::IntU64Const &);
};

struct EXPORT IntS8Const : Term::Base {
  int8_t value;
  explicit IntS8Const(int8_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS8Const &);
  EXPORT friend bool operator==(const Term::IntS8Const &, const Term::IntS8Const &);
};

struct EXPORT IntS16Const : Term::Base {
  int16_t value;
  explicit IntS16Const(int16_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS16Const &);
  EXPORT friend bool operator==(const Term::IntS16Const &, const Term::IntS16Const &);
};

struct EXPORT IntS32Const : Term::Base {
  int32_t value;
  explicit IntS32Const(int32_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS32Const &);
  EXPORT friend bool operator==(const Term::IntS32Const &, const Term::IntS32Const &);
};

struct EXPORT IntS64Const : Term::Base {
  int64_t value;
  explicit IntS64Const(int64_t value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS64Const &);
  EXPORT friend bool operator==(const Term::IntS64Const &, const Term::IntS64Const &);
};

struct EXPORT Unit0Const : Term::Base {
  Unit0Const() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Unit0Const &);
  EXPORT friend bool operator==(const Term::Unit0Const &, const Term::Unit0Const &);
};

struct EXPORT Bool1Const : Term::Base {
  bool value;
  explicit Bool1Const(bool value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Bool1Const &);
  EXPORT friend bool operator==(const Term::Bool1Const &, const Term::Bool1Const &);
};
} // namespace Term
namespace TypeSpace { 

struct EXPORT Base {
  protected:
  Base();
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Any &);
  EXPORT friend bool operator==(const TypeSpace::Base &, const TypeSpace::Base &);
};

struct EXPORT Global : TypeSpace::Base {
  Global() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Global &);
  EXPORT friend bool operator==(const TypeSpace::Global &, const TypeSpace::Global &);
};

struct EXPORT Local : TypeSpace::Base {
  Local() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &);
  EXPORT friend bool operator==(const TypeSpace::Local &, const TypeSpace::Local &);
};
} // namespace TypeSpace


struct EXPORT Overload {
  std::vector<Type::Any> args;
  Type::Any rtn;
  Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Overload &);
  EXPORT friend bool operator==(const Overload &, const Overload &);
};

namespace Spec { 

struct EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::Any &);
  EXPORT friend bool operator==(const Spec::Base &, const Spec::Base &);
};
EXPORT std::vector<Overload> overloads(const Spec::Any&);
EXPORT std::vector<Term::Any> terms(const Spec::Any&);
EXPORT Type::Any tpe(const Spec::Any&);

struct EXPORT Assert : Spec::Base {
  Assert() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::Assert &);
  EXPORT friend bool operator==(const Spec::Assert &, const Spec::Assert &);
};

struct EXPORT GpuBarrierGlobal : Spec::Base {
  GpuBarrierGlobal() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &);
  EXPORT friend bool operator==(const Spec::GpuBarrierGlobal &, const Spec::GpuBarrierGlobal &);
};

struct EXPORT GpuBarrierLocal : Spec::Base {
  GpuBarrierLocal() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierLocal &);
  EXPORT friend bool operator==(const Spec::GpuBarrierLocal &, const Spec::GpuBarrierLocal &);
};

struct EXPORT GpuBarrierAll : Spec::Base {
  GpuBarrierAll() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierAll &);
  EXPORT friend bool operator==(const Spec::GpuBarrierAll &, const Spec::GpuBarrierAll &);
};

struct EXPORT GpuFenceGlobal : Spec::Base {
  GpuFenceGlobal() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceGlobal &);
  EXPORT friend bool operator==(const Spec::GpuFenceGlobal &, const Spec::GpuFenceGlobal &);
};

struct EXPORT GpuFenceLocal : Spec::Base {
  GpuFenceLocal() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceLocal &);
  EXPORT friend bool operator==(const Spec::GpuFenceLocal &, const Spec::GpuFenceLocal &);
};

struct EXPORT GpuFenceAll : Spec::Base {
  GpuFenceAll() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceAll &);
  EXPORT friend bool operator==(const Spec::GpuFenceAll &, const Spec::GpuFenceAll &);
};

struct EXPORT GpuGlobalIdx : Spec::Base {
  Term::Any dim;
  explicit GpuGlobalIdx(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalIdx &);
  EXPORT friend bool operator==(const Spec::GpuGlobalIdx &, const Spec::GpuGlobalIdx &);
};

struct EXPORT GpuGlobalSize : Spec::Base {
  Term::Any dim;
  explicit GpuGlobalSize(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalSize &);
  EXPORT friend bool operator==(const Spec::GpuGlobalSize &, const Spec::GpuGlobalSize &);
};

struct EXPORT GpuGroupIdx : Spec::Base {
  Term::Any dim;
  explicit GpuGroupIdx(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupIdx &);
  EXPORT friend bool operator==(const Spec::GpuGroupIdx &, const Spec::GpuGroupIdx &);
};

struct EXPORT GpuGroupSize : Spec::Base {
  Term::Any dim;
  explicit GpuGroupSize(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupSize &);
  EXPORT friend bool operator==(const Spec::GpuGroupSize &, const Spec::GpuGroupSize &);
};

struct EXPORT GpuLocalIdx : Spec::Base {
  Term::Any dim;
  explicit GpuLocalIdx(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalIdx &);
  EXPORT friend bool operator==(const Spec::GpuLocalIdx &, const Spec::GpuLocalIdx &);
};

struct EXPORT GpuLocalSize : Spec::Base {
  Term::Any dim;
  explicit GpuLocalSize(Term::Any dim) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalSize &);
  EXPORT friend bool operator==(const Spec::GpuLocalSize &, const Spec::GpuLocalSize &);
};
} // namespace Spec
namespace Intr { 

struct EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Any &);
  EXPORT friend bool operator==(const Intr::Base &, const Intr::Base &);
};
EXPORT std::vector<Overload> overloads(const Intr::Any&);
EXPORT std::vector<Term::Any> terms(const Intr::Any&);
EXPORT Type::Any tpe(const Intr::Any&);

struct EXPORT BNot : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  BNot(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BNot &);
  EXPORT friend bool operator==(const Intr::BNot &, const Intr::BNot &);
};

struct EXPORT LogicNot : Intr::Base {
  Term::Any x;
  explicit LogicNot(Term::Any x) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNot &);
  EXPORT friend bool operator==(const Intr::LogicNot &, const Intr::LogicNot &);
};

struct EXPORT Pos : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  Pos(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Pos &);
  EXPORT friend bool operator==(const Intr::Pos &, const Intr::Pos &);
};

struct EXPORT Neg : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  Neg(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Neg &);
  EXPORT friend bool operator==(const Intr::Neg &, const Intr::Neg &);
};

struct EXPORT Add : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Add &);
  EXPORT friend bool operator==(const Intr::Add &, const Intr::Add &);
};

struct EXPORT Sub : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Sub &);
  EXPORT friend bool operator==(const Intr::Sub &, const Intr::Sub &);
};

struct EXPORT Mul : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Mul &);
  EXPORT friend bool operator==(const Intr::Mul &, const Intr::Mul &);
};

struct EXPORT Div : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Div &);
  EXPORT friend bool operator==(const Intr::Div &, const Intr::Div &);
};

struct EXPORT Rem : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Rem &);
  EXPORT friend bool operator==(const Intr::Rem &, const Intr::Rem &);
};

struct EXPORT Min : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Min &);
  EXPORT friend bool operator==(const Intr::Min &, const Intr::Min &);
};

struct EXPORT Max : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Max &);
  EXPORT friend bool operator==(const Intr::Max &, const Intr::Max &);
};

struct EXPORT BAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BAnd &);
  EXPORT friend bool operator==(const Intr::BAnd &, const Intr::BAnd &);
};

struct EXPORT BOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BOr &);
  EXPORT friend bool operator==(const Intr::BOr &, const Intr::BOr &);
};

struct EXPORT BXor : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BXor &);
  EXPORT friend bool operator==(const Intr::BXor &, const Intr::BXor &);
};

struct EXPORT BSL : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSL &);
  EXPORT friend bool operator==(const Intr::BSL &, const Intr::BSL &);
};

struct EXPORT BSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSR &);
  EXPORT friend bool operator==(const Intr::BSR &, const Intr::BSR &);
};

struct EXPORT BZSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BZSR &);
  EXPORT friend bool operator==(const Intr::BZSR &, const Intr::BZSR &);
};

struct EXPORT LogicAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicAnd(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicAnd &);
  EXPORT friend bool operator==(const Intr::LogicAnd &, const Intr::LogicAnd &);
};

struct EXPORT LogicOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicOr(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicOr &);
  EXPORT friend bool operator==(const Intr::LogicOr &, const Intr::LogicOr &);
};

struct EXPORT LogicEq : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicEq(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicEq &);
  EXPORT friend bool operator==(const Intr::LogicEq &, const Intr::LogicEq &);
};

struct EXPORT LogicNeq : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicNeq(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNeq &);
  EXPORT friend bool operator==(const Intr::LogicNeq &, const Intr::LogicNeq &);
};

struct EXPORT LogicLte : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicLte(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLte &);
  EXPORT friend bool operator==(const Intr::LogicLte &, const Intr::LogicLte &);
};

struct EXPORT LogicGte : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicGte(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGte &);
  EXPORT friend bool operator==(const Intr::LogicGte &, const Intr::LogicGte &);
};

struct EXPORT LogicLt : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicLt(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLt &);
  EXPORT friend bool operator==(const Intr::LogicLt &, const Intr::LogicLt &);
};

struct EXPORT LogicGt : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicGt(Term::Any x, Term::Any y) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGt &);
  EXPORT friend bool operator==(const Intr::LogicGt &, const Intr::LogicGt &);
};
} // namespace Intr
namespace Math { 

struct EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Any &);
  EXPORT friend bool operator==(const Math::Base &, const Math::Base &);
};
EXPORT std::vector<Overload> overloads(const Math::Any&);
EXPORT std::vector<Term::Any> terms(const Math::Any&);
EXPORT Type::Any tpe(const Math::Any&);

struct EXPORT Abs : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Abs(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Abs &);
  EXPORT friend bool operator==(const Math::Abs &, const Math::Abs &);
};

struct EXPORT Sin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sin(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sin &);
  EXPORT friend bool operator==(const Math::Sin &, const Math::Sin &);
};

struct EXPORT Cos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cos(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cos &);
  EXPORT friend bool operator==(const Math::Cos &, const Math::Cos &);
};

struct EXPORT Tan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Tan(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tan &);
  EXPORT friend bool operator==(const Math::Tan &, const Math::Tan &);
};

struct EXPORT Asin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Asin(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Asin &);
  EXPORT friend bool operator==(const Math::Asin &, const Math::Asin &);
};

struct EXPORT Acos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Acos(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Acos &);
  EXPORT friend bool operator==(const Math::Acos &, const Math::Acos &);
};

struct EXPORT Atan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Atan(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan &);
  EXPORT friend bool operator==(const Math::Atan &, const Math::Atan &);
};

struct EXPORT Sinh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sinh(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sinh &);
  EXPORT friend bool operator==(const Math::Sinh &, const Math::Sinh &);
};

struct EXPORT Cosh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cosh(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cosh &);
  EXPORT friend bool operator==(const Math::Cosh &, const Math::Cosh &);
};

struct EXPORT Tanh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Tanh(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tanh &);
  EXPORT friend bool operator==(const Math::Tanh &, const Math::Tanh &);
};

struct EXPORT Signum : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Signum(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Signum &);
  EXPORT friend bool operator==(const Math::Signum &, const Math::Signum &);
};

struct EXPORT Round : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Round(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Round &);
  EXPORT friend bool operator==(const Math::Round &, const Math::Round &);
};

struct EXPORT Ceil : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Ceil(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Ceil &);
  EXPORT friend bool operator==(const Math::Ceil &, const Math::Ceil &);
};

struct EXPORT Floor : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Floor(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Floor &);
  EXPORT friend bool operator==(const Math::Floor &, const Math::Floor &);
};

struct EXPORT Rint : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Rint(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Rint &);
  EXPORT friend bool operator==(const Math::Rint &, const Math::Rint &);
};

struct EXPORT Sqrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sqrt(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sqrt &);
  EXPORT friend bool operator==(const Math::Sqrt &, const Math::Sqrt &);
};

struct EXPORT Cbrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cbrt(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cbrt &);
  EXPORT friend bool operator==(const Math::Cbrt &, const Math::Cbrt &);
};

struct EXPORT Exp : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Exp(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Exp &);
  EXPORT friend bool operator==(const Math::Exp &, const Math::Exp &);
};

struct EXPORT Expm1 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Expm1(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Expm1 &);
  EXPORT friend bool operator==(const Math::Expm1 &, const Math::Expm1 &);
};

struct EXPORT Log : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log &);
  EXPORT friend bool operator==(const Math::Log &, const Math::Log &);
};

struct EXPORT Log1p : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log1p(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log1p &);
  EXPORT friend bool operator==(const Math::Log1p &, const Math::Log1p &);
};

struct EXPORT Log10 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log10(Term::Any x, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log10 &);
  EXPORT friend bool operator==(const Math::Log10 &, const Math::Log10 &);
};

struct EXPORT Pow : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Pow &);
  EXPORT friend bool operator==(const Math::Pow &, const Math::Pow &);
};

struct EXPORT Atan2 : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan2 &);
  EXPORT friend bool operator==(const Math::Atan2 &, const Math::Atan2 &);
};

struct EXPORT Hypot : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Hypot &);
  EXPORT friend bool operator==(const Math::Hypot &, const Math::Hypot &);
};
} // namespace Math
namespace Expr { 

struct EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Any &);
  EXPORT friend bool operator==(const Expr::Base &, const Expr::Base &);
};
EXPORT Type::Any tpe(const Expr::Any&);

struct EXPORT SpecOp : Expr::Base {
  Spec::Any op;
  explicit SpecOp(Spec::Any op) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::SpecOp &);
  EXPORT friend bool operator==(const Expr::SpecOp &, const Expr::SpecOp &);
};

struct EXPORT MathOp : Expr::Base {
  Math::Any op;
  explicit MathOp(Math::Any op) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::MathOp &);
  EXPORT friend bool operator==(const Expr::MathOp &, const Expr::MathOp &);
};

struct EXPORT IntrOp : Expr::Base {
  Intr::Any op;
  explicit IntrOp(Intr::Any op) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntrOp &);
  EXPORT friend bool operator==(const Expr::IntrOp &, const Expr::IntrOp &);
};

struct EXPORT Cast : Expr::Base {
  Term::Any from;
  Type::Any as;
  Cast(Term::Any from, Type::Any as) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cast &);
  EXPORT friend bool operator==(const Expr::Cast &, const Expr::Cast &);
};

struct EXPORT Alias : Expr::Base {
  Term::Any ref;
  explicit Alias(Term::Any ref) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
  EXPORT friend bool operator==(const Expr::Alias &, const Expr::Alias &);
};

struct EXPORT Index : Expr::Base {
  Term::Any lhs;
  Term::Any idx;
  Type::Any component;
  Index(Term::Any lhs, Term::Any idx, Type::Any component) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
  EXPORT friend bool operator==(const Expr::Index &, const Expr::Index &);
};

struct EXPORT Alloc : Expr::Base {
  Type::Any component;
  Term::Any size;
  Alloc(Type::Any component, Term::Any size) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
  EXPORT friend bool operator==(const Expr::Alloc &, const Expr::Alloc &);
};

struct EXPORT Invoke : Expr::Base {
  Sym name;
  std::vector<Type::Any> tpeArgs;
  std::optional<Term::Any> receiver;
  std::vector<Term::Any> args;
  std::vector<Term::Any> captures;
  Type::Any rtn;
  Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, std::vector<Term::Any> captures, Type::Any rtn) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
  EXPORT friend bool operator==(const Expr::Invoke &, const Expr::Invoke &);
};
} // namespace Expr
namespace Stmt { 

struct EXPORT Base {
  protected:
  Base();
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Any &);
  EXPORT friend bool operator==(const Stmt::Base &, const Stmt::Base &);
};

struct EXPORT Block : Stmt::Base {
  std::vector<Stmt::Any> stmts;
  explicit Block(std::vector<Stmt::Any> stmts) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Block &);
  EXPORT friend bool operator==(const Stmt::Block &, const Stmt::Block &);
};

struct EXPORT Comment : Stmt::Base {
  std::string value;
  explicit Comment(std::string value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Comment &);
  EXPORT friend bool operator==(const Stmt::Comment &, const Stmt::Comment &);
};

struct EXPORT Var : Stmt::Base {
  Named name;
  std::optional<Expr::Any> expr;
  Var(Named name, std::optional<Expr::Any> expr) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
  EXPORT friend bool operator==(const Stmt::Var &, const Stmt::Var &);
};

struct EXPORT Mut : Stmt::Base {
  Term::Any name;
  Expr::Any expr;
  bool copy;
  Mut(Term::Any name, Expr::Any expr, bool copy) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
  EXPORT friend bool operator==(const Stmt::Mut &, const Stmt::Mut &);
};

struct EXPORT Update : Stmt::Base {
  Term::Any lhs;
  Term::Any idx;
  Term::Any value;
  Update(Term::Any lhs, Term::Any idx, Term::Any value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
  EXPORT friend bool operator==(const Stmt::Update &, const Stmt::Update &);
};

struct EXPORT While : Stmt::Base {
  std::vector<Stmt::Any> tests;
  Term::Any cond;
  std::vector<Stmt::Any> body;
  While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
  EXPORT friend bool operator==(const Stmt::While &, const Stmt::While &);
};

struct EXPORT Break : Stmt::Base {
  Break() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Break &);
  EXPORT friend bool operator==(const Stmt::Break &, const Stmt::Break &);
};

struct EXPORT Cont : Stmt::Base {
  Cont() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cont &);
  EXPORT friend bool operator==(const Stmt::Cont &, const Stmt::Cont &);
};

struct EXPORT Cond : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> trueBr;
  std::vector<Stmt::Any> falseBr;
  Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cond &);
  EXPORT friend bool operator==(const Stmt::Cond &, const Stmt::Cond &);
};

struct EXPORT Return : Stmt::Base {
  Expr::Any value;
  explicit Return(Expr::Any value) noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
  EXPORT friend bool operator==(const Stmt::Return &, const Stmt::Return &);
};
} // namespace Stmt


struct EXPORT StructMember {
  Named named;
  bool isMutable;
  StructMember(Named named, bool isMutable) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const StructMember &);
  EXPORT friend bool operator==(const StructMember &, const StructMember &);
};

struct EXPORT StructDef {
  Sym name;
  bool isReference;
  std::vector<std::string> tpeVars;
  std::vector<StructMember> members;
  std::vector<Sym> parents;
  StructDef(Sym name, bool isReference, std::vector<std::string> tpeVars, std::vector<StructMember> members, std::vector<Sym> parents) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
  EXPORT friend bool operator==(const StructDef &, const StructDef &);
};

struct EXPORT Signature {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> moduleCaptures;
  std::vector<Type::Any> termCaptures;
  Type::Any rtn;
  Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Signature &);
  EXPORT friend bool operator==(const Signature &, const Signature &);
};

struct EXPORT InvokeSignature {
  Sym name;
  std::vector<Type::Any> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> captures;
  Type::Any rtn;
  InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> captures, Type::Any rtn) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const InvokeSignature &);
  EXPORT friend bool operator==(const InvokeSignature &, const InvokeSignature &);
};

namespace FunctionKind { 

struct EXPORT Base {
  protected:
  Base();
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Any &);
  EXPORT friend bool operator==(const FunctionKind::Base &, const FunctionKind::Base &);
};

struct EXPORT Internal : FunctionKind::Base {
  Internal() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Internal &);
  EXPORT friend bool operator==(const FunctionKind::Internal &, const FunctionKind::Internal &);
};

struct EXPORT Exported : FunctionKind::Base {
  Exported() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Exported &);
  EXPORT friend bool operator==(const FunctionKind::Exported &, const FunctionKind::Exported &);
};
} // namespace FunctionKind
namespace FunctionAttr { 

struct EXPORT Base {
  protected:
  Base();
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::Any &);
  EXPORT friend bool operator==(const FunctionAttr::Base &, const FunctionAttr::Base &);
};

struct EXPORT FPRelaxed : FunctionAttr::Base {
  FPRelaxed() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &);
  EXPORT friend bool operator==(const FunctionAttr::FPRelaxed &, const FunctionAttr::FPRelaxed &);
};

struct EXPORT FPStrict : FunctionAttr::Base {
  FPStrict() noexcept;
  EXPORT operator Any() const;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPStrict &);
  EXPORT friend bool operator==(const FunctionAttr::FPStrict &, const FunctionAttr::FPStrict &);
};
} // namespace FunctionAttr


struct EXPORT Arg {
  Named named;
  std::optional<SourcePosition> pos;
  Arg(Named named, std::optional<SourcePosition> pos) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Arg &);
  EXPORT friend bool operator==(const Arg &, const Arg &);
};

struct EXPORT Function {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Arg> receiver;
  std::vector<Arg> args;
  std::vector<Arg> moduleCaptures;
  std::vector<Arg> termCaptures;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args, std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
  EXPORT friend bool operator==(const Function &, const Function &);
};

struct EXPORT Program {
  Function entry;
  std::vector<Function> functions;
  std::vector<StructDef> defs;
  Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept;
  EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
  EXPORT friend bool operator==(const Program &, const Program &);
};

} // namespace polyregion::polyast
#ifndef _MSC_VER
  #pragma clang diagnostic pop // ide google-explicit-constructor
#endif
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
template <> struct std::hash<polyregion::polyast::Type::Float16> {
  std::size_t operator()(const polyregion::polyast::Type::Float16 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Float32> {
  std::size_t operator()(const polyregion::polyast::Type::Float32 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Float64> {
  std::size_t operator()(const polyregion::polyast::Type::Float64 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntU8> {
  std::size_t operator()(const polyregion::polyast::Type::IntU8 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntU16> {
  std::size_t operator()(const polyregion::polyast::Type::IntU16 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntU32> {
  std::size_t operator()(const polyregion::polyast::Type::IntU32 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntU64> {
  std::size_t operator()(const polyregion::polyast::Type::IntU64 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntS8> {
  std::size_t operator()(const polyregion::polyast::Type::IntS8 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntS16> {
  std::size_t operator()(const polyregion::polyast::Type::IntS16 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntS32> {
  std::size_t operator()(const polyregion::polyast::Type::IntS32 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::IntS64> {
  std::size_t operator()(const polyregion::polyast::Type::IntS64 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Nothing> {
  std::size_t operator()(const polyregion::polyast::Type::Nothing &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Unit0> {
  std::size_t operator()(const polyregion::polyast::Type::Unit0 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Type::Bool1> {
  std::size_t operator()(const polyregion::polyast::Type::Bool1 &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::SourcePosition> {
  std::size_t operator()(const polyregion::polyast::SourcePosition &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Select> {
  std::size_t operator()(const polyregion::polyast::Term::Select &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Poison> {
  std::size_t operator()(const polyregion::polyast::Term::Poison &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Float16Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float16Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Float32Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float32Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Float64Const> {
  std::size_t operator()(const polyregion::polyast::Term::Float64Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntU8Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU8Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntU16Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU16Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntU32Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU32Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntU64Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntU64Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntS8Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS8Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntS16Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS16Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntS32Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS32Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::IntS64Const> {
  std::size_t operator()(const polyregion::polyast::Term::IntS64Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Unit0Const> {
  std::size_t operator()(const polyregion::polyast::Term::Unit0Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Term::Bool1Const> {
  std::size_t operator()(const polyregion::polyast::Term::Bool1Const &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeSpace::Global> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Global &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::TypeSpace::Local> {
  std::size_t operator()(const polyregion::polyast::TypeSpace::Local &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Overload> {
  std::size_t operator()(const polyregion::polyast::Overload &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::Assert> {
  std::size_t operator()(const polyregion::polyast::Spec::Assert &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuBarrierGlobal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierGlobal &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuBarrierLocal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierLocal &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuBarrierAll> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuBarrierAll &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuFenceGlobal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceGlobal &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuFenceLocal> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceLocal &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuFenceAll> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuFenceAll &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuGlobalIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGlobalIdx &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuGlobalSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGlobalSize &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuGroupIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGroupIdx &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuGroupSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuGroupSize &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuLocalIdx> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuLocalIdx &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Spec::GpuLocalSize> {
  std::size_t operator()(const polyregion::polyast::Spec::GpuLocalSize &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BNot> {
  std::size_t operator()(const polyregion::polyast::Intr::BNot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicNot> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicNot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Pos> {
  std::size_t operator()(const polyregion::polyast::Intr::Pos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Neg> {
  std::size_t operator()(const polyregion::polyast::Intr::Neg &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Add> {
  std::size_t operator()(const polyregion::polyast::Intr::Add &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Sub> {
  std::size_t operator()(const polyregion::polyast::Intr::Sub &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Mul> {
  std::size_t operator()(const polyregion::polyast::Intr::Mul &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Div> {
  std::size_t operator()(const polyregion::polyast::Intr::Div &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Rem> {
  std::size_t operator()(const polyregion::polyast::Intr::Rem &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Min> {
  std::size_t operator()(const polyregion::polyast::Intr::Min &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::Max> {
  std::size_t operator()(const polyregion::polyast::Intr::Max &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BAnd> {
  std::size_t operator()(const polyregion::polyast::Intr::BAnd &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BOr> {
  std::size_t operator()(const polyregion::polyast::Intr::BOr &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BXor> {
  std::size_t operator()(const polyregion::polyast::Intr::BXor &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BSL> {
  std::size_t operator()(const polyregion::polyast::Intr::BSL &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BSR> {
  std::size_t operator()(const polyregion::polyast::Intr::BSR &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::BZSR> {
  std::size_t operator()(const polyregion::polyast::Intr::BZSR &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicAnd> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicAnd &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicOr> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicOr &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicEq> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicEq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicNeq> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicNeq &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicLte> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicLte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicGte> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicGte &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicLt> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicLt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Intr::LogicGt> {
  std::size_t operator()(const polyregion::polyast::Intr::LogicGt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Abs> {
  std::size_t operator()(const polyregion::polyast::Math::Abs &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Sin> {
  std::size_t operator()(const polyregion::polyast::Math::Sin &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Cos> {
  std::size_t operator()(const polyregion::polyast::Math::Cos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Tan> {
  std::size_t operator()(const polyregion::polyast::Math::Tan &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Asin> {
  std::size_t operator()(const polyregion::polyast::Math::Asin &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Acos> {
  std::size_t operator()(const polyregion::polyast::Math::Acos &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Atan> {
  std::size_t operator()(const polyregion::polyast::Math::Atan &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Sinh> {
  std::size_t operator()(const polyregion::polyast::Math::Sinh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Cosh> {
  std::size_t operator()(const polyregion::polyast::Math::Cosh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Tanh> {
  std::size_t operator()(const polyregion::polyast::Math::Tanh &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Signum> {
  std::size_t operator()(const polyregion::polyast::Math::Signum &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Round> {
  std::size_t operator()(const polyregion::polyast::Math::Round &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Ceil> {
  std::size_t operator()(const polyregion::polyast::Math::Ceil &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Floor> {
  std::size_t operator()(const polyregion::polyast::Math::Floor &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Rint> {
  std::size_t operator()(const polyregion::polyast::Math::Rint &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Sqrt> {
  std::size_t operator()(const polyregion::polyast::Math::Sqrt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Cbrt> {
  std::size_t operator()(const polyregion::polyast::Math::Cbrt &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Exp> {
  std::size_t operator()(const polyregion::polyast::Math::Exp &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Expm1> {
  std::size_t operator()(const polyregion::polyast::Math::Expm1 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Log> {
  std::size_t operator()(const polyregion::polyast::Math::Log &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Log1p> {
  std::size_t operator()(const polyregion::polyast::Math::Log1p &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Log10> {
  std::size_t operator()(const polyregion::polyast::Math::Log10 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Pow> {
  std::size_t operator()(const polyregion::polyast::Math::Pow &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Atan2> {
  std::size_t operator()(const polyregion::polyast::Math::Atan2 &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Math::Hypot> {
  std::size_t operator()(const polyregion::polyast::Math::Hypot &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::SpecOp> {
  std::size_t operator()(const polyregion::polyast::Expr::SpecOp &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::MathOp> {
  std::size_t operator()(const polyregion::polyast::Expr::MathOp &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::IntrOp> {
  std::size_t operator()(const polyregion::polyast::Expr::IntrOp &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Cast> {
  std::size_t operator()(const polyregion::polyast::Expr::Cast &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Alias> {
  std::size_t operator()(const polyregion::polyast::Expr::Alias &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Index> {
  std::size_t operator()(const polyregion::polyast::Expr::Index &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Alloc> {
  std::size_t operator()(const polyregion::polyast::Expr::Alloc &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Expr::Invoke> {
  std::size_t operator()(const polyregion::polyast::Expr::Invoke &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Stmt::Block> {
  std::size_t operator()(const polyregion::polyast::Stmt::Block &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::StructMember> {
  std::size_t operator()(const polyregion::polyast::StructMember &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::StructDef> {
  std::size_t operator()(const polyregion::polyast::StructDef &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Signature> {
  std::size_t operator()(const polyregion::polyast::Signature &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::InvokeSignature> {
  std::size_t operator()(const polyregion::polyast::InvokeSignature &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::FunctionKind::Internal> {
  std::size_t operator()(const polyregion::polyast::FunctionKind::Internal &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::FunctionKind::Exported> {
  std::size_t operator()(const polyregion::polyast::FunctionKind::Exported &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::FunctionAttr::FPRelaxed> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPRelaxed &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::FunctionAttr::FPStrict> {
  std::size_t operator()(const polyregion::polyast::FunctionAttr::FPStrict &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Arg> {
  std::size_t operator()(const polyregion::polyast::Arg &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Function> {
  std::size_t operator()(const polyregion::polyast::Function &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::Program> {
  std::size_t operator()(const polyregion::polyast::Program &) const noexcept;
};

}

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wunknown-pragmas
#endif
