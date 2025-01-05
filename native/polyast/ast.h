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
                              //  operator const Expr::Any() const; // NOLINT(google-explicit-constructor)
  operator Named() const;     // NOLINT(google-explicit-constructor)
  Arg operator()() const;
  IndexBuilder operator[](const Expr::Any &idx) const;
  Mut operator=(const Expr::Any &that) const;
};

Expr::Any integral(const Type::Any &tpe, unsigned long long int x);
Expr::Any fractional(const Type::Any &tpe, long double x);

std::function<Expr::Any(Type::Any)> operator"" _(long double x);
std::function<Expr::Any(Type::Any)> operator"" _(unsigned long long int x);
std::function<NamedBuilder(Type::Any)> operator"" _(const char *name, size_t);

Stmt::Any let(const std::string &name, const Type::Any &tpe);
AssignmentBuilder let(const std::string &name);
IntrOp invoke(const Intr::Any &);
MathOp invoke(const Math::Any &);
SpecOp invoke(const Spec::Any &);
std::function<Function(Vec<Stmt::Any>)> function(const std::string &name, const Vec<Arg> &args, const Type::Any &rtn,
                                                 const std::set<FunctionAttr::Any> &attrs = {FunctionAttr::Exported()});
Program program(const Vec<StructDef> &structs = {}, const Vec<Function> &functions = {});
Program program(const Function &function);

Return ret(const Expr::Any &expr = Unit0Const());

} // namespace dsl

} // namespace polyregion::polyast