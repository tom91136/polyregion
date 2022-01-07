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



struct Sym {
  std::vector<std::string> fqn;
  explicit Sym(std::vector<std::string> fqn) noexcept : fqn(std::move(fqn)) {}
  friend std::ostream &operator<<(std::ostream &os, const Sym &);
};

namespace TypeKind { 

struct Ref;
struct Integral;
struct Fractional;
using Any = Alternative<Ref, Integral, Fractional>;
struct Base {
  protected:
  Base() = default;
  friend std::ostream &operator<<(std::ostream &os, const TypeKind::Any &);
};

struct Ref : TypeKind::Base {
  Ref() noexcept : TypeKind::Base() {}
  operator Any() const { return std::make_shared<Ref>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const TypeKind::Ref &);
};

struct Integral : TypeKind::Base {
  Integral() noexcept : TypeKind::Base() {}
  operator Any() const { return std::make_shared<Integral>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const TypeKind::Integral &);
};

struct Fractional : TypeKind::Base {
  Fractional() noexcept : TypeKind::Base() {}
  operator Any() const { return std::make_shared<Fractional>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const TypeKind::Fractional &);
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
struct Base {
  TypeKind::Any kind;
  protected:
  explicit Base(TypeKind::Any kind) noexcept : kind(std::move(kind)) {}
  friend std::ostream &operator<<(std::ostream &os, const Type::Any &);
};
TypeKind::Any kind(const Type::Any&);

struct Float : Type::Base {
  Float() noexcept : Type::Base(TypeKind::Fractional()) {}
  operator Any() const { return std::make_shared<Float>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Float &);
};

struct Double : Type::Base {
  Double() noexcept : Type::Base(TypeKind::Fractional()) {}
  operator Any() const { return std::make_shared<Double>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Double &);
};

struct Bool : Type::Base {
  Bool() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Bool>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Bool &);
};

struct Byte : Type::Base {
  Byte() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Byte>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Byte &);
};

struct Char : Type::Base {
  Char() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Char>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Char &);
};

struct Short : Type::Base {
  Short() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Short>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Short &);
};

struct Int : Type::Base {
  Int() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Int>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Int &);
};

struct Long : Type::Base {
  Long() noexcept : Type::Base(TypeKind::Integral()) {}
  operator Any() const { return std::make_shared<Long>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Long &);
};

struct String : Type::Base {
  String() noexcept : Type::Base(TypeKind::Ref()) {}
  operator Any() const { return std::make_shared<String>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::String &);
};

struct Unit : Type::Base {
  Unit() noexcept : Type::Base(TypeKind::Ref()) {}
  operator Any() const { return std::make_shared<Unit>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Unit &);
};

struct Struct : Type::Base {
  Sym name;
  std::vector<Type::Any> args;
  Struct(Sym name, std::vector<Type::Any> args) noexcept : Type::Base(TypeKind::Ref()), name(std::move(name)), args(std::move(args)) {}
  operator Any() const { return std::make_shared<Struct>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Struct &);
};

struct Array : Type::Base {
  Type::Any component;
  explicit Array(Type::Any component) noexcept : Type::Base(TypeKind::Ref()), component(std::move(component)) {}
  operator Any() const { return std::make_shared<Array>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Type::Array &);
};
} // namespace Type


struct Named {
  std::string symbol;
  Type::Any tpe;
  Named(std::string symbol, Type::Any tpe) noexcept : symbol(std::move(symbol)), tpe(std::move(tpe)) {}
  friend std::ostream &operator<<(std::ostream &os, const Named &);
};

struct Position {
  std::string file;
  int32_t line;
  int32_t col;
  Position(std::string file, int32_t line, int32_t col) noexcept : file(std::move(file)), line(line), col(col) {}
  friend std::ostream &operator<<(std::ostream &os, const Position &);
};

namespace Term { 

struct Select;
struct BoolConst;
struct ByteConst;
struct CharConst;
struct ShortConst;
struct IntConst;
struct LongConst;
struct FloatConst;
struct DoubleConst;
struct StringConst;
using Any = Alternative<Select, BoolConst, ByteConst, CharConst, ShortConst, IntConst, LongConst, FloatConst, DoubleConst, StringConst>;
struct Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
  friend std::ostream &operator<<(std::ostream &os, const Term::Any &);
};
Type::Any tpe(const Term::Any&);

struct Select : Term::Base {
  std::vector<Named> init;
  Named last;
  Select(std::vector<Named> init, Named last) noexcept : Term::Base(last.tpe), init(std::move(init)), last(std::move(last)) {}
  operator Any() const { return std::make_shared<Select>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::Select &);
};

struct BoolConst : Term::Base {
  bool value;
  explicit BoolConst(bool value) noexcept : Term::Base(Type::Bool()), value(value) {}
  operator Any() const { return std::make_shared<BoolConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::BoolConst &);
};

struct ByteConst : Term::Base {
  int8_t value;
  explicit ByteConst(int8_t value) noexcept : Term::Base(Type::Byte()), value(value) {}
  operator Any() const { return std::make_shared<ByteConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::ByteConst &);
};

struct CharConst : Term::Base {
  uint16_t value;
  explicit CharConst(uint16_t value) noexcept : Term::Base(Type::Char()), value(value) {}
  operator Any() const { return std::make_shared<CharConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::CharConst &);
};

struct ShortConst : Term::Base {
  int16_t value;
  explicit ShortConst(int16_t value) noexcept : Term::Base(Type::Short()), value(value) {}
  operator Any() const { return std::make_shared<ShortConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::ShortConst &);
};

struct IntConst : Term::Base {
  int32_t value;
  explicit IntConst(int32_t value) noexcept : Term::Base(Type::Int()), value(value) {}
  operator Any() const { return std::make_shared<IntConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::IntConst &);
};

struct LongConst : Term::Base {
  int64_t value;
  explicit LongConst(int64_t value) noexcept : Term::Base(Type::Long()), value(value) {}
  operator Any() const { return std::make_shared<LongConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::LongConst &);
};

struct FloatConst : Term::Base {
  float value;
  explicit FloatConst(float value) noexcept : Term::Base(Type::Float()), value(value) {}
  operator Any() const { return std::make_shared<FloatConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::FloatConst &);
};

struct DoubleConst : Term::Base {
  double value;
  explicit DoubleConst(double value) noexcept : Term::Base(Type::Double()), value(value) {}
  operator Any() const { return std::make_shared<DoubleConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::DoubleConst &);
};

struct StringConst : Term::Base {
  std::string value;
  explicit StringConst(std::string value) noexcept : Term::Base(Type::String()), value(std::move(value)) {}
  operator Any() const { return std::make_shared<StringConst>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Term::StringConst &);
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
struct Base {
  Type::Any tpe;
  protected:
  explicit Base(Type::Any tpe) noexcept : tpe(std::move(tpe)) {}
  friend std::ostream &operator<<(std::ostream &os, const Expr::Any &);
};
Type::Any tpe(const Expr::Any&);

struct Sin : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Sin(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Sin>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Sin &);
};

struct Cos : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Cos(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Cos>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Cos &);
};

struct Tan : Expr::Base {
  Term::Any lhs;
  Type::Any rtn;
  Tan(Term::Any lhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Tan>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Tan &);
};

struct Add : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Add(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Add>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Add &);
};

struct Sub : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Sub(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Sub>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Sub &);
};

struct Mul : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Mul(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Mul>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Mul &);
};

struct Div : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Div(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Div>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Div &);
};

struct Mod : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Mod(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Mod>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Mod &);
};

struct Pow : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Type::Any rtn;
  Pow(Term::Any lhs, Term::Any rhs, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), rhs(std::move(rhs)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Pow>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Pow &);
};

struct Inv : Expr::Base {
  Term::Any lhs;
  explicit Inv(Term::Any lhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)) {}
  operator Any() const { return std::make_shared<Inv>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Inv &);
};

struct Eq : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Eq(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  operator Any() const { return std::make_shared<Eq>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Eq &);
};

struct Lte : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Lte(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  operator Any() const { return std::make_shared<Lte>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Lte &);
};

struct Gte : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Gte(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  operator Any() const { return std::make_shared<Gte>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Gte &);
};

struct Lt : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Lt(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  operator Any() const { return std::make_shared<Lt>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Lt &);
};

struct Gt : Expr::Base {
  Term::Any lhs;
  Term::Any rhs;
  Gt(Term::Any lhs, Term::Any rhs) noexcept : Expr::Base(Type::Bool()), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  operator Any() const { return std::make_shared<Gt>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Gt &);
};

struct Alias : Expr::Base {
  Term::Any ref;
  explicit Alias(Term::Any ref) noexcept : Expr::Base(Term::tpe(ref)), ref(std::move(ref)) {}
  operator Any() const { return std::make_shared<Alias>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Alias &);
};

struct Invoke : Expr::Base {
  Term::Any lhs;
  std::string name;
  std::vector<Term::Any> args;
  Type::Any rtn;
  Invoke(Term::Any lhs, std::string name, std::vector<Term::Any> args, Type::Any rtn) noexcept : Expr::Base(rtn), lhs(std::move(lhs)), name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)) {}
  operator Any() const { return std::make_shared<Invoke>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Invoke &);
};

struct Index : Expr::Base {
  Term::Select lhs;
  Term::Any idx;
  Type::Any component;
  Index(Term::Select lhs, Term::Any idx, Type::Any component) noexcept : Expr::Base(component), lhs(std::move(lhs)), idx(std::move(idx)), component(std::move(component)) {}
  operator Any() const { return std::make_shared<Index>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Expr::Index &);
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
struct Base {
  protected:
  Base() = default;
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Any &);
};

struct Comment : Stmt::Base {
  std::string value;
  explicit Comment(std::string value) noexcept : Stmt::Base(), value(std::move(value)) {}
  operator Any() const { return std::make_shared<Comment>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Comment &);
};

struct Var : Stmt::Base {
  Named name;
  Expr::Any expr;
  Var(Named name, Expr::Any expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
  operator Any() const { return std::make_shared<Var>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Var &);
};

struct Mut : Stmt::Base {
  Term::Select name;
  Expr::Any expr;
  Mut(Term::Select name, Expr::Any expr) noexcept : Stmt::Base(), name(std::move(name)), expr(std::move(expr)) {}
  operator Any() const { return std::make_shared<Mut>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Mut &);
};

struct Update : Stmt::Base {
  Term::Select lhs;
  Term::Any idx;
  Term::Any value;
  Update(Term::Select lhs, Term::Any idx, Term::Any value) noexcept : Stmt::Base(), lhs(std::move(lhs)), idx(std::move(idx)), value(std::move(value)) {}
  operator Any() const { return std::make_shared<Update>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Update &);
};

struct Effect : Stmt::Base {
  Term::Select lhs;
  std::string name;
  std::vector<Term::Any> args;
  Effect(Term::Select lhs, std::string name, std::vector<Term::Any> args) noexcept : Stmt::Base(), lhs(std::move(lhs)), name(std::move(name)), args(std::move(args)) {}
  operator Any() const { return std::make_shared<Effect>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Effect &);
};

struct While : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> body;
  While(Expr::Any cond, std::vector<Stmt::Any> body) noexcept : Stmt::Base(), cond(std::move(cond)), body(std::move(body)) {}
  operator Any() const { return std::make_shared<While>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::While &);
};

struct Break : Stmt::Base {
  Break() noexcept : Stmt::Base() {}
  operator Any() const { return std::make_shared<Break>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Break &);
};

struct Cont : Stmt::Base {
  Cont() noexcept : Stmt::Base() {}
  operator Any() const { return std::make_shared<Cont>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Cont &);
};

struct Cond : Stmt::Base {
  Expr::Any cond;
  std::vector<Stmt::Any> trueBr;
  std::vector<Stmt::Any> falseBr;
  Cond(Expr::Any cond, std::vector<Stmt::Any> trueBr, std::vector<Stmt::Any> falseBr) noexcept : Stmt::Base(), cond(std::move(cond)), trueBr(std::move(trueBr)), falseBr(std::move(falseBr)) {}
  operator Any() const { return std::make_shared<Cond>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Cond &);
};

struct Return : Stmt::Base {
  Expr::Any value;
  explicit Return(Expr::Any value) noexcept : Stmt::Base(), value(std::move(value)) {}
  operator Any() const { return std::make_shared<Return>(*this); };
  friend std::ostream &operator<<(std::ostream &os, const Stmt::Return &);
};
} // namespace Stmt


struct Function {
  std::string name;
  std::vector<Named> args;
  Type::Any rtn;
  std::vector<Stmt::Any> body;
  Function(std::string name, std::vector<Named> args, Type::Any rtn, std::vector<Stmt::Any> body) noexcept : name(std::move(name)), args(std::move(args)), rtn(std::move(rtn)), body(std::move(body)) {}
  friend std::ostream &operator<<(std::ostream &os, const Function &);
};

struct StructDef {
  std::vector<Named> members;
  explicit StructDef(std::vector<Named> members) noexcept : members(std::move(members)) {}
  friend std::ostream &operator<<(std::ostream &os, const StructDef &);
};

} // namespace polyregion::polyast
#pragma clang diagnostic pop

