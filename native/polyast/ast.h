#pragma once

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "generated/polyast.h"
#include "generated/polyast_repr.h"
#include "polyregion/compat.h"

namespace polyregion::polyast {

using Bytes = std::vector<char>;
template <typename T> using Vec = std::vector<T>;
template <typename T> using Opt = std::optional<T>;
template <typename T> using Set = std::unordered_set<T>;
template <typename T, typename U> using Pair = std::pair<T, U>;
template <typename T, typename U> using Map = std::unordered_map<T, U>;

const static auto show_repr = [](auto &x) { return repr(x); };

std::string qualified(const Expr::Select &);
Vec<Named> path(const Expr::Select &);
Named head(const Expr::Select &);
Vec<Named> tail(const Expr::Select &);

std::pair<Named, Vec<Named>> uncons(const Expr::Select &);

Expr::Select selectNamed(const Expr::Select &select, const Named &that);

Expr::Select selectNamed(const Vec<Named> &names);

Expr::Select selectNamed(const Named &name);

Expr::Select parent(const Expr::Select &select);

Type::Struct typeOf(const StructDef &def);

std::string repr(const CompileResult &);

Opt<Type::Any> extractComponent(const Type::Any &t);

Opt<size_t> primitiveSize(const Type::Any &t);

std::pair<size_t, Opt<size_t>> countIndirectionsAndComponentSize(const Type::Any &t, const Map<Type::Struct, StructLayout> &table);

namespace dsl {

using namespace Stmt;
using namespace Expr;
namespace Tpe = Type;
// namespace Fn0 = NullaryIntrinsicKind;
// namespace Fn2 = BinaryIntrinsicKind;
// namespace Fn1 = UnaryIntrinsicKind;

const static auto Global = TypeSpace::Global();
const static auto Local = TypeSpace::Local();

const static auto Float = Tpe::Float32();
const static auto Double = Tpe::Float64();
const static auto Bool = Tpe::Bool1();
const static auto Byte = Tpe::IntS8();
const static auto Char = Tpe::IntU16();
const static auto Short = Tpe::IntS16();
const static auto SInt = Tpe::IntS32();
const static auto UInt = Tpe::IntU32();
const static auto Long = Tpe::IntS64();
const static auto Unit = Tpe::Unit0();
const static auto Nothing = Tpe::Nothing();

Tpe::Ptr Ptr(const Tpe::Any &t, Opt<int32_t> l = {}, const TypeSpace::Any &s = TypeSpace::Global());
Tpe::Struct Struct(std::string name, Vec<Type::Any> members);

struct AssignmentBuilder {
  std::string name;
  explicit AssignmentBuilder(const std::string &name);
  Stmt::Any operator=(Expr::Any u) const; // NOLINT(misc-unconventional-assign-operator)
  Stmt::Any operator=(Type::Any) const;   // NOLINT(misc-unconventional-assign-operator)
};

struct IndexBuilder {
  Index index;
  explicit IndexBuilder(const Index &index);
  operator Expr::Any() const; // NOLINT(google-explicit-constructor)
  Update operator=(const Expr::Any &that) const;
};

struct NamedBuilder {
  Named named;
  explicit NamedBuilder(const Named &named);
  operator Expr::Any() const; // NOLINT(google-explicit-constructor)
  operator Expr::Select() const;
  //  operator const Expr::Any() const; // NOLINT(google-explicit-constructor)
  operator Named() const; // NOLINT(google-explicit-constructor)
  Arg operator()() const;
  IndexBuilder operator[](const Expr::Any &idx) const;
  Mut operator=(const Expr::Any &that) const;
};

Expr::Any integral(const Type::Any &tpe, unsigned long long int x);
Expr::Any fractional(const Type::Any &tpe, long double x);

template <typename F> Expr::Any numeric(const Type::Any &tpe, F tagged) {
  auto unsupported = [](auto &&t) -> Expr::Any { throw std::logic_error("Cannot create numeric constant of type " + to_string(t)); };
  return tpe.match_total(                                                                 //
      [&](const Type::Float16 &) -> Expr::Any { return Float16Const(tagged(float{})); },  //
      [&](const Type::Float32 &) -> Expr::Any { return Float32Const(tagged(float{})); },  //
      [&](const Type::Float64 &) -> Expr::Any { return Float64Const(tagged(double{})); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return IntU8Const(tagged(uint8_t{})); },    //
      [&](const Type::IntU16 &) -> Expr::Any { return IntU16Const(tagged(uint16_t{})); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return IntU32Const(tagged(uint32_t{})); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return IntU64Const(tagged(uint64_t{})); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return IntS8Const(tagged(int8_t{})); },    //
      [&](const Type::IntS16 &) -> Expr::Any { return IntS16Const(tagged(int16_t{})); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return IntS32Const(tagged(int32_t{})); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return IntS64Const(tagged(int64_t{})); }, //

      [&](const Type::Nothing &t) -> Expr::Any { return unsupported(t); },          //
      [&](const Type::Unit0 &t) -> Expr::Any { return unsupported(t); },            //
      [&](const Type::Bool1 &) -> Expr::Any { return Bool1Const(tagged(bool{})); }, //

      [&](const Type::Struct &t) -> Expr::Any { return unsupported(t); },   //
      [&](const Type::Ptr &t) -> Expr::Any { return unsupported(t); },      //
      [&](const Type::Annotated &t) -> Expr::Any { return unsupported(t); } //
  );
}

std::function<Expr::Any(Type::Any)> operator"" _(long double x);
std::function<Expr::Any(Type::Any)> operator"" _(unsigned long long int x);
std::function<NamedBuilder(Type::Any)> operator"" _(const char *name, size_t);

Stmt::Any let(const std::string &name, const Type::Any &tpe);
AssignmentBuilder let(const std::string &name);
IntrOp call(const Intr::Any &);
MathOp call(const Math::Any &);
SpecOp call(const Spec::Any &);
std::function<Function(Vec<Stmt::Any>)> function(const std::string &name, const Vec<Arg> &args, const Type::Any &rtn,
                                                 const std::set<FunctionAttr::Any> &attrs = {FunctionAttr::Exported()});
Program program(const Vec<StructDef> &structs = {}, const Vec<Function> &functions = {});
Program program(const Function &function);

Return ret(const Expr::Any &expr = Unit0Const());

} // namespace dsl

} // namespace polyregion::polyast