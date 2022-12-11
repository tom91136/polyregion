#pragma once

#include "generated/polyast.h"
#include "variants.hpp"
#include <optional>
#include <ostream>
#include <unordered_map>
#include <utility>

namespace polyregion::polyast {

template <typename T> using Opt = std::optional<T>;
template <typename T, typename U> using Pair = std::pair<T, U>;
template <typename T, typename U> using Map = std::unordered_map<T, U>;

std::string repr(const Sym &);
std::string repr(const Type::Any &);
std::string repr(const Named &);
std::string repr(const Term::Any &);
std::string repr(const Expr::Any &);
std::string repr(const Stmt::Any &);
std::string repr(const Function &);
std::string repr(const StructDef &);
std::string repr(const Program &);

std::string qualified(const Term::Select &);
std::string qualified(const Sym &);
std::vector<Named> path(const Term::Select &);
Named head(const Term::Select &);
std::vector<Named> tail(const Term::Select &);

std::pair<Named, std::vector<Named>> uncons(const Term::Select &);

namespace dsl {

using namespace Stmt;
using namespace Term;
using namespace Expr;
namespace Tpe = Type;
namespace Fn0 = NullaryIntrinsicKind;
namespace Fn2 = BinaryIntrinsicKind;
namespace Fn1 = UnaryIntrinsicKind;

const static Tpe::Float Float = Tpe::Float();
const static Tpe::Double Double = Tpe::Double();
const static Tpe::Bool Bool = Tpe::Bool();
const static Tpe::Byte Byte = Tpe::Byte();
const static Tpe::Char Char = Tpe::Char();
const static Tpe::Short Short = Tpe::Short();
const static Tpe::Int Int = Tpe::Int();
const static Tpe::Long Long = Tpe::Long();
const static Tpe::Unit Unit = Tpe::Unit();
const static Tpe::Nothing Nothing = Tpe::Nothing();

std::string dslRepr(const Function &fn);

Tpe::Array Array(const Tpe::Any &t);
Tpe::Struct Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args);

struct AssignmentBuilder {
  std::string name;
  explicit AssignmentBuilder(const std::string &name);
  Stmt::Any operator=(Expr::Any u) const; // NOLINT(misc-unconventional-assign-operator)
  Stmt::Any operator=(Term::Any u) const; // NOLINT(misc-unconventional-assign-operator)
};

struct IndexBuilder {
  Index index;
  explicit IndexBuilder(const Index &index);
  operator const Expr::Any() const; // NOLINT(google-explicit-constructor)
  Update operator=(const Term::Any &term) const;
};

struct NamedBuilder {
  Named named;
  explicit NamedBuilder(const Named &named);
  operator const Term::Any() const; // NOLINT(google-explicit-constructor)
//  operator const Expr::Any() const; // NOLINT(google-explicit-constructor)
  operator const Named() const;     // NOLINT(google-explicit-constructor)
  IndexBuilder operator[](const Term::Any &idx) const;
};

Term::Any integral(const Type::Any &tpe, unsigned long long int x);
Term::Any fractional(const Type::Any &tpe, long double x);

std::function<Term::Any(Type::Any)> operator"" _(long double x);
std::function<Term::Any(Type::Any)> operator"" _(unsigned long long int x);
std::function<NamedBuilder(Type::Any)> operator"" _(const char *name, size_t);

Stmt::Any let(const std::string &name, const Type::Any &tpe);
AssignmentBuilder let(const std::string &name);
BinaryIntrinsic invoke(const BinaryIntrinsicKind::Any &kind, const Term::Any &lhs, const Term::Any &rhs,
                       const Type::Any &rtn);

UnaryIntrinsic invoke(const UnaryIntrinsicKind::Any &kind, const Term::Any &lhs, const Type::Any &rtn);

NullaryIntrinsic invoke(const NullaryIntrinsicKind::Any &kind, const Type::Any &rtn);

std::function<Function(std::vector<Stmt::Any>)> function(const std::string &name, const std::vector<Named> &args,
                                                         const Type::Any &rtn);
Program program(Function entry, std::vector<StructDef> defs = {}, std::vector<Function> functions = {});

Return ret(const Expr::Any &expr = Alias(UnitConst()));
Return ret(const Term::Any &term);

} // namespace dsl

} // namespace polyregion::polyast