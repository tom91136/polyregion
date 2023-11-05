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
struct Ptr;
struct Var;
struct Exec;
using Any = Alternative<Float16, Float32, Float64, IntU8, IntU16, IntU32, IntU64, IntS8, IntS16, IntS32, IntS64, Nothing, Unit0, Bool1, Struct, Ptr, Var, Exec>;
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
struct RefTo;
struct Alloc;
struct Invoke;
using Any = Alternative<SpecOp, MathOp, IntrOp, Cast, Alias, Index, RefTo, Alloc, Invoke>;
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





struct POLYREGION_EXPORT Sym {
  std::vector<std::string> fqn;
  explicit Sym(std::vector<std::string> fqn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Sym &);
  POLYREGION_EXPORT friend bool operator==(const Sym &, const Sym &);
};

struct POLYREGION_EXPORT Named {
  std::string symbol;
  Type::Any tpe;
  Named(std::string symbol, Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Named &);
  POLYREGION_EXPORT friend bool operator==(const Named &, const Named &);
};

namespace TypeKind { 

struct POLYREGION_EXPORT Base {
  protected:
  Base();
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Any &);
  POLYREGION_EXPORT friend bool operator==(const TypeKind::Base &, const TypeKind::Base &);
};

struct POLYREGION_EXPORT None : TypeKind::Base {
  None() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::None &);
  POLYREGION_EXPORT friend bool operator==(const TypeKind::None &, const TypeKind::None &);
};

struct POLYREGION_EXPORT Ref : TypeKind::Base {
  Ref() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &);
  POLYREGION_EXPORT friend bool operator==(const TypeKind::Ref &, const TypeKind::Ref &);
};

struct POLYREGION_EXPORT Integral : TypeKind::Base {
  Integral() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &);
  POLYREGION_EXPORT friend bool operator==(const TypeKind::Integral &, const TypeKind::Integral &);
};

struct POLYREGION_EXPORT Fractional : TypeKind::Base {
  Fractional() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
  POLYREGION_EXPORT friend bool operator==(const TypeKind::Fractional &, const TypeKind::Fractional &);
};
} // namespace TypeKind
namespace Type { 

struct POLYREGION_EXPORT Base {
  TypeKind::Any kind;
  protected:
  explicit Base(TypeKind::Any kind) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Any &);
  POLYREGION_EXPORT friend bool operator==(const Type::Base &, const Type::Base &);
};
POLYREGION_EXPORT TypeKind::Any kind(const Type::Any&);

struct POLYREGION_EXPORT Float16 : Type::Base {
  Float16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float16 &);
  POLYREGION_EXPORT friend bool operator==(const Type::Float16 &, const Type::Float16 &);
};

struct POLYREGION_EXPORT Float32 : Type::Base {
  Float32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float32 &);
  POLYREGION_EXPORT friend bool operator==(const Type::Float32 &, const Type::Float32 &);
};

struct POLYREGION_EXPORT Float64 : Type::Base {
  Float64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Float64 &);
  POLYREGION_EXPORT friend bool operator==(const Type::Float64 &, const Type::Float64 &);
};

struct POLYREGION_EXPORT IntU8 : Type::Base {
  IntU8() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU8 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntU8 &, const Type::IntU8 &);
};

struct POLYREGION_EXPORT IntU16 : Type::Base {
  IntU16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU16 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntU16 &, const Type::IntU16 &);
};

struct POLYREGION_EXPORT IntU32 : Type::Base {
  IntU32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU32 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntU32 &, const Type::IntU32 &);
};

struct POLYREGION_EXPORT IntU64 : Type::Base {
  IntU64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntU64 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntU64 &, const Type::IntU64 &);
};

struct POLYREGION_EXPORT IntS8 : Type::Base {
  IntS8() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS8 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntS8 &, const Type::IntS8 &);
};

struct POLYREGION_EXPORT IntS16 : Type::Base {
  IntS16() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS16 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntS16 &, const Type::IntS16 &);
};

struct POLYREGION_EXPORT IntS32 : Type::Base {
  IntS32() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS32 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntS32 &, const Type::IntS32 &);
};

struct POLYREGION_EXPORT IntS64 : Type::Base {
  IntS64() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::IntS64 &);
  POLYREGION_EXPORT friend bool operator==(const Type::IntS64 &, const Type::IntS64 &);
};

struct POLYREGION_EXPORT Nothing : Type::Base {
  Nothing() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Nothing &);
  POLYREGION_EXPORT friend bool operator==(const Type::Nothing &, const Type::Nothing &);
};

struct POLYREGION_EXPORT Unit0 : Type::Base {
  Unit0() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Unit0 &);
  POLYREGION_EXPORT friend bool operator==(const Type::Unit0 &, const Type::Unit0 &);
};

struct POLYREGION_EXPORT Bool1 : Type::Base {
  Bool1() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Bool1 &);
  POLYREGION_EXPORT friend bool operator==(const Type::Bool1 &, const Type::Bool1 &);
};

struct POLYREGION_EXPORT Struct : Type::Base {
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  std::vector<Sym> parents;
  Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args, std::vector<Sym> parents) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
  POLYREGION_EXPORT friend bool operator==(const Type::Struct &, const Type::Struct &);
};

struct POLYREGION_EXPORT Ptr : Type::Base {
  Type::Any component;
  std::optional<int32_t> length;
  TypeSpace::Any space;
  Ptr(Type::Any component, std::optional<int32_t> length, TypeSpace::Any space) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Ptr &);
  POLYREGION_EXPORT friend bool operator==(const Type::Ptr &, const Type::Ptr &);
};

struct POLYREGION_EXPORT Var : Type::Base {
  std::string name;
  explicit Var(std::string name) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Var &);
  POLYREGION_EXPORT friend bool operator==(const Type::Var &, const Type::Var &);
};

struct POLYREGION_EXPORT Exec : Type::Base {
  std::vector<std::string> tpeVars;
  std::vector<Type::Any> args;
  Type::Any rtn;
  Exec(std::vector<std::string> tpeVars, std::vector<Type::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Type::Exec &);
  POLYREGION_EXPORT friend bool operator==(const Type::Exec &, const Type::Exec &);
};
} // namespace Type


struct POLYREGION_EXPORT SourcePosition {
  std::string file;
  int32_t line;
  std::optional<int32_t> col;
  SourcePosition(std::string file, int32_t line, std::optional<int32_t> col) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const SourcePosition &);
  POLYREGION_EXPORT friend bool operator==(const SourcePosition &, const SourcePosition &);
};

namespace Term { 

struct POLYREGION_EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Any &);
  POLYREGION_EXPORT friend bool operator==(const Term::Base &, const Term::Base &);
};
POLYREGION_EXPORT Type::Any tpe(const Term::Any&);

struct POLYREGION_EXPORT Select : Term::Base {
  std::vector<Named> init;
  Named last;
  Select(std::vector<Named> init, Named last) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Select &);
  POLYREGION_EXPORT friend bool operator==(const Term::Select &, const Term::Select &);
};

struct POLYREGION_EXPORT Poison : Term::Base {
  Type::Any t;
  explicit Poison(Type::Any t) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Poison &);
  POLYREGION_EXPORT friend bool operator==(const Term::Poison &, const Term::Poison &);
};

struct POLYREGION_EXPORT Float16Const : Term::Base {
  float value;
  explicit Float16Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float16Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::Float16Const &, const Term::Float16Const &);
};

struct POLYREGION_EXPORT Float32Const : Term::Base {
  float value;
  explicit Float32Const(float value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float32Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::Float32Const &, const Term::Float32Const &);
};

struct POLYREGION_EXPORT Float64Const : Term::Base {
  double value;
  explicit Float64Const(double value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Float64Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::Float64Const &, const Term::Float64Const &);
};

struct POLYREGION_EXPORT IntU8Const : Term::Base {
  int8_t value;
  explicit IntU8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU8Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntU8Const &, const Term::IntU8Const &);
};

struct POLYREGION_EXPORT IntU16Const : Term::Base {
  uint16_t value;
  explicit IntU16Const(uint16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU16Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntU16Const &, const Term::IntU16Const &);
};

struct POLYREGION_EXPORT IntU32Const : Term::Base {
  int32_t value;
  explicit IntU32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU32Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntU32Const &, const Term::IntU32Const &);
};

struct POLYREGION_EXPORT IntU64Const : Term::Base {
  int64_t value;
  explicit IntU64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntU64Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntU64Const &, const Term::IntU64Const &);
};

struct POLYREGION_EXPORT IntS8Const : Term::Base {
  int8_t value;
  explicit IntS8Const(int8_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS8Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntS8Const &, const Term::IntS8Const &);
};

struct POLYREGION_EXPORT IntS16Const : Term::Base {
  int16_t value;
  explicit IntS16Const(int16_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS16Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntS16Const &, const Term::IntS16Const &);
};

struct POLYREGION_EXPORT IntS32Const : Term::Base {
  int32_t value;
  explicit IntS32Const(int32_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS32Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntS32Const &, const Term::IntS32Const &);
};

struct POLYREGION_EXPORT IntS64Const : Term::Base {
  int64_t value;
  explicit IntS64Const(int64_t value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::IntS64Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::IntS64Const &, const Term::IntS64Const &);
};

struct POLYREGION_EXPORT Unit0Const : Term::Base {
  Unit0Const() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Unit0Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::Unit0Const &, const Term::Unit0Const &);
};

struct POLYREGION_EXPORT Bool1Const : Term::Base {
  bool value;
  explicit Bool1Const(bool value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Term::Bool1Const &);
  POLYREGION_EXPORT friend bool operator==(const Term::Bool1Const &, const Term::Bool1Const &);
};
} // namespace Term
namespace TypeSpace { 

struct POLYREGION_EXPORT Base {
  protected:
  Base();
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Any &);
  POLYREGION_EXPORT friend bool operator==(const TypeSpace::Base &, const TypeSpace::Base &);
};

struct POLYREGION_EXPORT Global : TypeSpace::Base {
  Global() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Global &);
  POLYREGION_EXPORT friend bool operator==(const TypeSpace::Global &, const TypeSpace::Global &);
};

struct POLYREGION_EXPORT Local : TypeSpace::Base {
  Local() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const TypeSpace::Local &);
  POLYREGION_EXPORT friend bool operator==(const TypeSpace::Local &, const TypeSpace::Local &);
};
} // namespace TypeSpace


struct POLYREGION_EXPORT Overload {
  std::vector<Type::Any> args;
  Type::Any rtn;
  Overload(std::vector<Type::Any> args, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Overload &);
  POLYREGION_EXPORT friend bool operator==(const Overload &, const Overload &);
};

namespace Spec { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::Any &);
  POLYREGION_EXPORT friend bool operator==(const Spec::Base &, const Spec::Base &);
};
POLYREGION_EXPORT std::vector<Overload> overloads(const Spec::Any&);
POLYREGION_EXPORT std::vector<Term::Any> terms(const Spec::Any&);
POLYREGION_EXPORT Type::Any tpe(const Spec::Any&);

struct POLYREGION_EXPORT Assert : Spec::Base {
  Assert() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::Assert &);
  POLYREGION_EXPORT friend bool operator==(const Spec::Assert &, const Spec::Assert &);
};

struct POLYREGION_EXPORT GpuBarrierGlobal : Spec::Base {
  GpuBarrierGlobal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierGlobal &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuBarrierGlobal &, const Spec::GpuBarrierGlobal &);
};

struct POLYREGION_EXPORT GpuBarrierLocal : Spec::Base {
  GpuBarrierLocal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierLocal &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuBarrierLocal &, const Spec::GpuBarrierLocal &);
};

struct POLYREGION_EXPORT GpuBarrierAll : Spec::Base {
  GpuBarrierAll() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuBarrierAll &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuBarrierAll &, const Spec::GpuBarrierAll &);
};

struct POLYREGION_EXPORT GpuFenceGlobal : Spec::Base {
  GpuFenceGlobal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceGlobal &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuFenceGlobal &, const Spec::GpuFenceGlobal &);
};

struct POLYREGION_EXPORT GpuFenceLocal : Spec::Base {
  GpuFenceLocal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceLocal &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuFenceLocal &, const Spec::GpuFenceLocal &);
};

struct POLYREGION_EXPORT GpuFenceAll : Spec::Base {
  GpuFenceAll() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuFenceAll &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuFenceAll &, const Spec::GpuFenceAll &);
};

struct POLYREGION_EXPORT GpuGlobalIdx : Spec::Base {
  Term::Any dim;
  explicit GpuGlobalIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalIdx &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuGlobalIdx &, const Spec::GpuGlobalIdx &);
};

struct POLYREGION_EXPORT GpuGlobalSize : Spec::Base {
  Term::Any dim;
  explicit GpuGlobalSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGlobalSize &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuGlobalSize &, const Spec::GpuGlobalSize &);
};

struct POLYREGION_EXPORT GpuGroupIdx : Spec::Base {
  Term::Any dim;
  explicit GpuGroupIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupIdx &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuGroupIdx &, const Spec::GpuGroupIdx &);
};

struct POLYREGION_EXPORT GpuGroupSize : Spec::Base {
  Term::Any dim;
  explicit GpuGroupSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuGroupSize &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuGroupSize &, const Spec::GpuGroupSize &);
};

struct POLYREGION_EXPORT GpuLocalIdx : Spec::Base {
  Term::Any dim;
  explicit GpuLocalIdx(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalIdx &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuLocalIdx &, const Spec::GpuLocalIdx &);
};

struct POLYREGION_EXPORT GpuLocalSize : Spec::Base {
  Term::Any dim;
  explicit GpuLocalSize(Term::Any dim) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Spec::GpuLocalSize &);
  POLYREGION_EXPORT friend bool operator==(const Spec::GpuLocalSize &, const Spec::GpuLocalSize &);
};
} // namespace Spec
namespace Intr { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Any &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Base &, const Intr::Base &);
};
POLYREGION_EXPORT std::vector<Overload> overloads(const Intr::Any&);
POLYREGION_EXPORT std::vector<Term::Any> terms(const Intr::Any&);
POLYREGION_EXPORT Type::Any tpe(const Intr::Any&);

struct POLYREGION_EXPORT BNot : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  BNot(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BNot &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BNot &, const Intr::BNot &);
};

struct POLYREGION_EXPORT LogicNot : Intr::Base {
  Term::Any x;
  explicit LogicNot(Term::Any x) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNot &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicNot &, const Intr::LogicNot &);
};

struct POLYREGION_EXPORT Pos : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  Pos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Pos &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Pos &, const Intr::Pos &);
};

struct POLYREGION_EXPORT Neg : Intr::Base {
  Term::Any x;
  Type::Any rtn;
  Neg(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Neg &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Neg &, const Intr::Neg &);
};

struct POLYREGION_EXPORT Add : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Add(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Add &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Add &, const Intr::Add &);
};

struct POLYREGION_EXPORT Sub : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Sub(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Sub &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Sub &, const Intr::Sub &);
};

struct POLYREGION_EXPORT Mul : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Mul(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Mul &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Mul &, const Intr::Mul &);
};

struct POLYREGION_EXPORT Div : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Div(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Div &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Div &, const Intr::Div &);
};

struct POLYREGION_EXPORT Rem : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Rem(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Rem &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Rem &, const Intr::Rem &);
};

struct POLYREGION_EXPORT Min : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Min(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Min &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Min &, const Intr::Min &);
};

struct POLYREGION_EXPORT Max : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Max(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::Max &);
  POLYREGION_EXPORT friend bool operator==(const Intr::Max &, const Intr::Max &);
};

struct POLYREGION_EXPORT BAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BAnd(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BAnd &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BAnd &, const Intr::BAnd &);
};

struct POLYREGION_EXPORT BOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BOr(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BOr &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BOr &, const Intr::BOr &);
};

struct POLYREGION_EXPORT BXor : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BXor(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BXor &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BXor &, const Intr::BXor &);
};

struct POLYREGION_EXPORT BSL : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BSL(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSL &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BSL &, const Intr::BSL &);
};

struct POLYREGION_EXPORT BSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BSR &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BSR &, const Intr::BSR &);
};

struct POLYREGION_EXPORT BZSR : Intr::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  BZSR(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::BZSR &);
  POLYREGION_EXPORT friend bool operator==(const Intr::BZSR &, const Intr::BZSR &);
};

struct POLYREGION_EXPORT LogicAnd : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicAnd(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicAnd &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicAnd &, const Intr::LogicAnd &);
};

struct POLYREGION_EXPORT LogicOr : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicOr(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicOr &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicOr &, const Intr::LogicOr &);
};

struct POLYREGION_EXPORT LogicEq : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicEq(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicEq &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicEq &, const Intr::LogicEq &);
};

struct POLYREGION_EXPORT LogicNeq : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicNeq(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicNeq &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicNeq &, const Intr::LogicNeq &);
};

struct POLYREGION_EXPORT LogicLte : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicLte(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLte &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicLte &, const Intr::LogicLte &);
};

struct POLYREGION_EXPORT LogicGte : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicGte(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGte &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicGte &, const Intr::LogicGte &);
};

struct POLYREGION_EXPORT LogicLt : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicLt(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicLt &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicLt &, const Intr::LogicLt &);
};

struct POLYREGION_EXPORT LogicGt : Intr::Base {
  Term::Any x;
  Term::Any y;
  LogicGt(Term::Any x, Term::Any y) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Intr::LogicGt &);
  POLYREGION_EXPORT friend bool operator==(const Intr::LogicGt &, const Intr::LogicGt &);
};
} // namespace Intr
namespace Math { 

struct POLYREGION_EXPORT Base {
  std::vector<Overload> overloads;
  std::vector<Term::Any> terms;
  Type::Any tpe;
  protected:
  Base(std::vector<Overload> overloads, std::vector<Term::Any> terms, Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Any &);
  POLYREGION_EXPORT friend bool operator==(const Math::Base &, const Math::Base &);
};
POLYREGION_EXPORT std::vector<Overload> overloads(const Math::Any&);
POLYREGION_EXPORT std::vector<Term::Any> terms(const Math::Any&);
POLYREGION_EXPORT Type::Any tpe(const Math::Any&);

struct POLYREGION_EXPORT Abs : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Abs(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Abs &);
  POLYREGION_EXPORT friend bool operator==(const Math::Abs &, const Math::Abs &);
};

struct POLYREGION_EXPORT Sin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sin(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sin &);
  POLYREGION_EXPORT friend bool operator==(const Math::Sin &, const Math::Sin &);
};

struct POLYREGION_EXPORT Cos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cos &);
  POLYREGION_EXPORT friend bool operator==(const Math::Cos &, const Math::Cos &);
};

struct POLYREGION_EXPORT Tan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Tan(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tan &);
  POLYREGION_EXPORT friend bool operator==(const Math::Tan &, const Math::Tan &);
};

struct POLYREGION_EXPORT Asin : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Asin(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Asin &);
  POLYREGION_EXPORT friend bool operator==(const Math::Asin &, const Math::Asin &);
};

struct POLYREGION_EXPORT Acos : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Acos(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Acos &);
  POLYREGION_EXPORT friend bool operator==(const Math::Acos &, const Math::Acos &);
};

struct POLYREGION_EXPORT Atan : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Atan(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan &);
  POLYREGION_EXPORT friend bool operator==(const Math::Atan &, const Math::Atan &);
};

struct POLYREGION_EXPORT Sinh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sinh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sinh &);
  POLYREGION_EXPORT friend bool operator==(const Math::Sinh &, const Math::Sinh &);
};

struct POLYREGION_EXPORT Cosh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cosh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cosh &);
  POLYREGION_EXPORT friend bool operator==(const Math::Cosh &, const Math::Cosh &);
};

struct POLYREGION_EXPORT Tanh : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Tanh(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Tanh &);
  POLYREGION_EXPORT friend bool operator==(const Math::Tanh &, const Math::Tanh &);
};

struct POLYREGION_EXPORT Signum : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Signum(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Signum &);
  POLYREGION_EXPORT friend bool operator==(const Math::Signum &, const Math::Signum &);
};

struct POLYREGION_EXPORT Round : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Round(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Round &);
  POLYREGION_EXPORT friend bool operator==(const Math::Round &, const Math::Round &);
};

struct POLYREGION_EXPORT Ceil : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Ceil(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Ceil &);
  POLYREGION_EXPORT friend bool operator==(const Math::Ceil &, const Math::Ceil &);
};

struct POLYREGION_EXPORT Floor : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Floor(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Floor &);
  POLYREGION_EXPORT friend bool operator==(const Math::Floor &, const Math::Floor &);
};

struct POLYREGION_EXPORT Rint : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Rint(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Rint &);
  POLYREGION_EXPORT friend bool operator==(const Math::Rint &, const Math::Rint &);
};

struct POLYREGION_EXPORT Sqrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Sqrt(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Sqrt &);
  POLYREGION_EXPORT friend bool operator==(const Math::Sqrt &, const Math::Sqrt &);
};

struct POLYREGION_EXPORT Cbrt : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Cbrt(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Cbrt &);
  POLYREGION_EXPORT friend bool operator==(const Math::Cbrt &, const Math::Cbrt &);
};

struct POLYREGION_EXPORT Exp : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Exp(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Exp &);
  POLYREGION_EXPORT friend bool operator==(const Math::Exp &, const Math::Exp &);
};

struct POLYREGION_EXPORT Expm1 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Expm1(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Expm1 &);
  POLYREGION_EXPORT friend bool operator==(const Math::Expm1 &, const Math::Expm1 &);
};

struct POLYREGION_EXPORT Log : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log &);
  POLYREGION_EXPORT friend bool operator==(const Math::Log &, const Math::Log &);
};

struct POLYREGION_EXPORT Log1p : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log1p(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log1p &);
  POLYREGION_EXPORT friend bool operator==(const Math::Log1p &, const Math::Log1p &);
};

struct POLYREGION_EXPORT Log10 : Math::Base {
  Term::Any x;
  Type::Any rtn;
  Log10(Term::Any x, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Log10 &);
  POLYREGION_EXPORT friend bool operator==(const Math::Log10 &, const Math::Log10 &);
};

struct POLYREGION_EXPORT Pow : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Pow(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Pow &);
  POLYREGION_EXPORT friend bool operator==(const Math::Pow &, const Math::Pow &);
};

struct POLYREGION_EXPORT Atan2 : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Atan2(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Atan2 &);
  POLYREGION_EXPORT friend bool operator==(const Math::Atan2 &, const Math::Atan2 &);
};

struct POLYREGION_EXPORT Hypot : Math::Base {
  Term::Any x;
  Term::Any y;
  Type::Any rtn;
  Hypot(Term::Any x, Term::Any y, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Math::Hypot &);
  POLYREGION_EXPORT friend bool operator==(const Math::Hypot &, const Math::Hypot &);
};
} // namespace Math
namespace Expr { 

struct POLYREGION_EXPORT Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Any &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Base &, const Expr::Base &);
};
POLYREGION_EXPORT Type::Any tpe(const Expr::Any&);

struct POLYREGION_EXPORT SpecOp : Expr::Base {
  Spec::Any op;
  explicit SpecOp(Spec::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::SpecOp &);
  POLYREGION_EXPORT friend bool operator==(const Expr::SpecOp &, const Expr::SpecOp &);
};

struct POLYREGION_EXPORT MathOp : Expr::Base {
  Math::Any op;
  explicit MathOp(Math::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::MathOp &);
  POLYREGION_EXPORT friend bool operator==(const Expr::MathOp &, const Expr::MathOp &);
};

struct POLYREGION_EXPORT IntrOp : Expr::Base {
  Intr::Any op;
  explicit IntrOp(Intr::Any op) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::IntrOp &);
  POLYREGION_EXPORT friend bool operator==(const Expr::IntrOp &, const Expr::IntrOp &);
};

struct POLYREGION_EXPORT Cast : Expr::Base {
  Term::Any from;
  Type::Any as;
  Cast(Term::Any from, Type::Any as) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Cast &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Cast &, const Expr::Cast &);
};

struct POLYREGION_EXPORT Alias : Expr::Base {
  Term::Any ref;
  explicit Alias(Term::Any ref) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Alias &, const Expr::Alias &);
};

struct POLYREGION_EXPORT Index : Expr::Base {
  Term::Any lhs;
  Term::Any idx;
  Type::Any component;
  Index(Term::Any lhs, Term::Any idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Index &, const Expr::Index &);
};

struct POLYREGION_EXPORT RefTo : Expr::Base {
  Term::Any lhs;
  std::optional<Term::Any> idx;
  Type::Any component;
  RefTo(Term::Any lhs, std::optional<Term::Any> idx, Type::Any component) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::RefTo &);
  POLYREGION_EXPORT friend bool operator==(const Expr::RefTo &, const Expr::RefTo &);
};

struct POLYREGION_EXPORT Alloc : Expr::Base {
  Type::Any component;
  Term::Any size;
  Alloc(Type::Any component, Term::Any size) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Alloc &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Alloc &, const Expr::Alloc &);
};

struct POLYREGION_EXPORT Invoke : Expr::Base {
  Sym name;
  std::vector<Type::Any> tpeArgs;
  std::optional<Term::Any> receiver;
  std::vector<Term::Any> args;
  std::vector<Term::Any> captures;
  Type::Any rtn;
  Invoke(Sym name, std::vector<Type::Any> tpeArgs, std::optional<Term::Any> receiver, std::vector<Term::Any> args, std::vector<Term::Any> captures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
  POLYREGION_EXPORT friend bool operator==(const Expr::Invoke &, const Expr::Invoke &);
};
} // namespace Expr
namespace Stmt { 

struct POLYREGION_EXPORT Base {
  protected:
  Base();
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Any &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Base &, const Stmt::Base &);
};

struct POLYREGION_EXPORT Block : Stmt::Base {
  std::vector<Stmt::Any> stmts;
  explicit Block(std::vector<Stmt::Any> stmts) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Block &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Block &, const Stmt::Block &);
};

struct POLYREGION_EXPORT Comment : Stmt::Base {
  std::string value;
  explicit Comment(std::string value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Comment &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Comment &, const Stmt::Comment &);
};

struct POLYREGION_EXPORT Var : Stmt::Base {
  Named name;
  std::optional<Expr::Any> expr;
  Var(Named name, std::optional<Expr::Any> expr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Var &, const Stmt::Var &);
};

struct POLYREGION_EXPORT Mut : Stmt::Base {
  Term::Any name;
  Expr::Any expr;
  bool copy;
  Mut(Term::Any name, Expr::Any expr, bool copy) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Mut &, const Stmt::Mut &);
};

struct POLYREGION_EXPORT Update : Stmt::Base {
  Term::Any lhs;
  Term::Any idx;
  Term::Any value;
  Update(Term::Any lhs, Term::Any idx, Term::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Update &, const Stmt::Update &);
};

struct POLYREGION_EXPORT While : Stmt::Base {
  std::vector<Stmt::Any> tests;
  Term::Any cond;
  std::vector<Stmt::Any> body;
  While(std::vector<Stmt::Any> tests, Term::Any cond, std::vector<Stmt::Any> body) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::While &, const Stmt::While &);
};

struct POLYREGION_EXPORT Break : Stmt::Base {
  Break() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Break &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Break &, const Stmt::Break &);
};

struct POLYREGION_EXPORT Cont : Stmt::Base {
  Cont() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cont &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Cont &, const Stmt::Cont &);
};

struct POLYREGION_EXPORT Cond : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> trueBr;
  std::vector<Stmt::Any> falseBr;
  Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Cond &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Cond &, const Stmt::Cond &);
};

struct POLYREGION_EXPORT Return : Stmt::Base {
  Expr::Any value;
  explicit Return(Expr::Any value) noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
  POLYREGION_EXPORT friend bool operator==(const Stmt::Return &, const Stmt::Return &);
};
} // namespace Stmt


struct POLYREGION_EXPORT StructMember {
  Named named;
  bool isMutable;
  StructMember(Named named, bool isMutable) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructMember &);
  POLYREGION_EXPORT friend bool operator==(const StructMember &, const StructMember &);
};

struct POLYREGION_EXPORT StructDef {
  Sym name;
  std::vector<std::string> tpeVars;
  std::vector<StructMember> members;
  std::vector<Sym> parents;
  StructDef(Sym name, std::vector<std::string> tpeVars, std::vector<StructMember> members, std::vector<Sym> parents) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const StructDef &);
  POLYREGION_EXPORT friend bool operator==(const StructDef &, const StructDef &);
};

struct POLYREGION_EXPORT Signature {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> moduleCaptures;
  std::vector<Type::Any> termCaptures;
  Type::Any rtn;
  Signature(Sym name, std::vector<std::string> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> moduleCaptures, std::vector<Type::Any> termCaptures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Signature &);
  POLYREGION_EXPORT friend bool operator==(const Signature &, const Signature &);
};

struct POLYREGION_EXPORT InvokeSignature {
  Sym name;
  std::vector<Type::Any> tpeVars;
  std::optional<Type::Any> receiver;
  std::vector<Type::Any> args;
  std::vector<Type::Any> captures;
  Type::Any rtn;
  InvokeSignature(Sym name, std::vector<Type::Any> tpeVars, std::optional<Type::Any> receiver, std::vector<Type::Any> args, std::vector<Type::Any> captures, Type::Any rtn) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const InvokeSignature &);
  POLYREGION_EXPORT friend bool operator==(const InvokeSignature &, const InvokeSignature &);
};

namespace FunctionKind { 

struct POLYREGION_EXPORT Base {
  protected:
  Base();
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Any &);
  POLYREGION_EXPORT friend bool operator==(const FunctionKind::Base &, const FunctionKind::Base &);
};

struct POLYREGION_EXPORT Internal : FunctionKind::Base {
  Internal() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Internal &);
  POLYREGION_EXPORT friend bool operator==(const FunctionKind::Internal &, const FunctionKind::Internal &);
};

struct POLYREGION_EXPORT Exported : FunctionKind::Base {
  Exported() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionKind::Exported &);
  POLYREGION_EXPORT friend bool operator==(const FunctionKind::Exported &, const FunctionKind::Exported &);
};
} // namespace FunctionKind
namespace FunctionAttr { 

struct POLYREGION_EXPORT Base {
  protected:
  Base();
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::Any &);
  POLYREGION_EXPORT friend bool operator==(const FunctionAttr::Base &, const FunctionAttr::Base &);
};

struct POLYREGION_EXPORT FPRelaxed : FunctionAttr::Base {
  FPRelaxed() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPRelaxed &);
  POLYREGION_EXPORT friend bool operator==(const FunctionAttr::FPRelaxed &, const FunctionAttr::FPRelaxed &);
};

struct POLYREGION_EXPORT FPStrict : FunctionAttr::Base {
  FPStrict() noexcept;
  POLYREGION_EXPORT operator Any() const;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const FunctionAttr::FPStrict &);
  POLYREGION_EXPORT friend bool operator==(const FunctionAttr::FPStrict &, const FunctionAttr::FPStrict &);
};
} // namespace FunctionAttr


struct POLYREGION_EXPORT Arg {
  Named named;
  std::optional<SourcePosition> pos;
  Arg(Named named, std::optional<SourcePosition> pos) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Arg &);
  POLYREGION_EXPORT friend bool operator==(const Arg &, const Arg &);
};

struct POLYREGION_EXPORT Function {
  Sym name;
  std::vector<std::string> tpeVars;
  std::optional<Arg> receiver;
  std::vector<Arg> args;
  std::vector<Arg> moduleCaptures;
  std::vector<Arg> termCaptures;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  Function(Sym name, std::vector<std::string> tpeVars, std::optional<Arg> receiver, std::vector<Arg> args, std::vector<Arg> moduleCaptures, std::vector<Arg> termCaptures, Type::Any rtn, std::vector<Stmt::Any> body) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Function &);
  POLYREGION_EXPORT friend bool operator==(const Function &, const Function &);
};

struct POLYREGION_EXPORT Program {
  Function entry;
  std::vector<Function> functions;
  std::vector<StructDef> defs;
  Program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Program &);
  POLYREGION_EXPORT friend bool operator==(const Program &, const Program &);
};

struct POLYREGION_EXPORT CompileLayoutMember {
  Named name;
  int64_t offsetInBytes;
  int64_t sizeInBytes;
  CompileLayoutMember(Named name, int64_t offsetInBytes, int64_t sizeInBytes) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileLayoutMember &);
  POLYREGION_EXPORT friend bool operator==(const CompileLayoutMember &, const CompileLayoutMember &);
};

struct POLYREGION_EXPORT CompileLayout {
  Sym name;
  int64_t sizeInBytes;
  int64_t alignment;
  std::vector<CompileLayoutMember> members;
  CompileLayout(Sym name, int64_t sizeInBytes, int64_t alignment, std::vector<CompileLayoutMember> members) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileLayout &);
  POLYREGION_EXPORT friend bool operator==(const CompileLayout &, const CompileLayout &);
};

struct POLYREGION_EXPORT CompileEvent {
  int64_t epochMillis;
  int64_t elapsedNanos;
  std::string name;
  std::string data;
  CompileEvent(int64_t epochMillis, int64_t elapsedNanos, std::string name, std::string data) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileEvent &);
  POLYREGION_EXPORT friend bool operator==(const CompileEvent &, const CompileEvent &);
};

struct POLYREGION_EXPORT CompileResult {
  std::optional<std::vector<int8_t>> binary;
  std::vector<std::string> features;
  std::vector<CompileEvent> events;
  std::vector<CompileLayout> layouts;
  std::string messages;
  CompileResult(std::optional<std::vector<int8_t>> binary, std::vector<std::string> features, std::vector<CompileEvent> events, std::vector<CompileLayout> layouts, std::string messages) noexcept;
  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const CompileResult &);
  POLYREGION_EXPORT friend bool operator==(const CompileResult &, const CompileResult &);
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
template <> struct std::hash<polyregion::polyast::Type::Ptr> {
  std::size_t operator()(const polyregion::polyast::Type::Ptr &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::Expr::RefTo> {
  std::size_t operator()(const polyregion::polyast::Expr::RefTo &) const noexcept;
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
template <> struct std::hash<polyregion::polyast::CompileLayoutMember> {
  std::size_t operator()(const polyregion::polyast::CompileLayoutMember &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::CompileLayout> {
  std::size_t operator()(const polyregion::polyast::CompileLayout &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::CompileEvent> {
  std::size_t operator()(const polyregion::polyast::CompileEvent &) const noexcept;
};
template <> struct std::hash<polyregion::polyast::CompileResult> {
  std::size_t operator()(const polyregion::polyast::CompileResult &) const noexcept;
};

}

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wunknown-pragmas
#endif
