#pragma once

#include <functional>

#include "polyregion/aliases.h"
#include "polyregion/compat.h"

#include "generated/polyast.h"
#include "generated/polyast_repr.h"

namespace polyregion::polyast {

using polyregion::Map;
using polyregion::Opt;
using polyregion::Pair;
using polyregion::Set;
using polyregion::String;
using polyregion::StringView;
using polyregion::Vector;

using Bytes = Vector<char>;

const static auto show_repr = [](auto &x) { return repr(x); };

std::string qualified(const Term::Select &);

Term::Select selectNamed(const Named &name);
Term::Select selectField(const Term::Select &select, const Named &field);

Type::Struct typeOf(const StructDef &def);

std::string repr(const CompileResult &);

Opt<Type::Any> extractComponent(const Type::Any &t);

Opt<size_t> primitiveSize(const Type::Any &t);

std::pair<size_t, Opt<size_t>> countIndirectionsAndComponentSize(const Type::Any &t, const Map<Type::Struct, StructLayout> &table);

bool isSelfOpaque(const Type::Any &tpe);

bool isSelfOpaque(const StructLayout &sl);

bool isOpaque(const StructLayout &sl, const std::unordered_map<Type::Struct, StructLayout> &table);

namespace dsl {

using namespace Stmt;
using namespace Expr;
namespace Tpe = Type;

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

Tpe::Ptr Ptr(const Tpe::Any &t, const TypeSpace::Any &s = TypeSpace::Global());
// Backwards-compat shim: legacy DSL passed `Ptr(t, {}, space)` where {} was an optional length.
// The new schema splits Arr (with length) from Ptr (no length); the middle arg is ignored here so
// existing call sites keep building.
Tpe::Ptr Ptr(const Tpe::Any &t, std::optional<int32_t>, const TypeSpace::Any &s);
Tpe::Arr Arr(const Tpe::Any &t, int32_t length, const TypeSpace::Any &s = TypeSpace::Global());
Tpe::Struct Struct(std::string name, Vector<Type::Any> args);

// Backwards-compat shim for the legacy While(prelude, cond, body) shape. The new Stmt::While takes
// (cond, body) only; we inline the prelude into the start of body so the cond term gets re-bound
// each iteration. Best-effort - callers should migrate to the 2-arg form for clarity.
Stmt::While whileLoop(const std::vector<Stmt::Any> &prelude, const Term::Any &cond, const std::vector<Stmt::Any> &body);

struct AssignmentBuilder {
  std::string name;
  bool isMutable;
  AssignmentBuilder(const std::string &name, bool isMutable);
  Stmt::Any operator=(Term::Any t) const;        // NOLINT(misc-unconventional-assign-operator)
  Stmt::Any operator=(Type::Any) const;          // NOLINT(misc-unconventional-assign-operator)
  Stmt::Any operator=(const Expr::Any &e) const; // NOLINT(misc-unconventional-assign-operator)
};

struct IndexBuilder {
  Index index;
  explicit IndexBuilder(const Index &index);
  operator Expr::Any() const; // NOLINT(google-explicit-constructor)
  // Update lhs requires a Term::Select; convert from the held lhs term if possible.
  Update operator=(const Term::Any &that) const;
};

struct NamedBuilder {
  Named named;
  explicit NamedBuilder(const Named &named);
  operator Term::Any() const; // NOLINT(google-explicit-constructor)
  operator Term::Select() const;
  operator Named() const; // NOLINT(google-explicit-constructor)
  Arg operator()() const;
  IndexBuilder operator[](const Term::Any &idx) const;
  Mut operator=(const Expr::Any &that) const;
};

Term::Any integral(const Type::Any &tpe, unsigned long long int x);
Term::Any fractional(const Type::Any &tpe, long double x);

template <typename F> Term::Any numeric(const Type::Any &tpe, F tagged) {
  auto unsupported = [](auto &&t) -> Term::Any { throw std::logic_error("Cannot create numeric constant of type " + to_string(t)); };
  return tpe.match_total(                                                                       //
      [&](const Type::Float16 &) -> Term::Any { return Term::Float16Const(tagged(float{})); },  //
      [&](const Type::Float32 &) -> Term::Any { return Term::Float32Const(tagged(float{})); },  //
      [&](const Type::Float64 &) -> Term::Any { return Term::Float64Const(tagged(double{})); }, //

      [&](const Type::IntU8 &) -> Term::Any { return Term::IntU8Const(tagged(uint8_t{})); },    //
      [&](const Type::IntU16 &) -> Term::Any { return Term::IntU16Const(tagged(uint16_t{})); }, //
      [&](const Type::IntU32 &) -> Term::Any { return Term::IntU32Const(tagged(uint32_t{})); }, //
      [&](const Type::IntU64 &) -> Term::Any { return Term::IntU64Const(tagged(uint64_t{})); }, //

      [&](const Type::IntS8 &) -> Term::Any { return Term::IntS8Const(tagged(int8_t{})); },    //
      [&](const Type::IntS16 &) -> Term::Any { return Term::IntS16Const(tagged(int16_t{})); }, //
      [&](const Type::IntS32 &) -> Term::Any { return Term::IntS32Const(tagged(int32_t{})); }, //
      [&](const Type::IntS64 &) -> Term::Any { return Term::IntS64Const(tagged(int64_t{})); }, //

      [&](const Type::Nothing &t) -> Term::Any { return unsupported(t); },                //
      [&](const Type::Unit0 &t) -> Term::Any { return unsupported(t); },                  //
      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(tagged(bool{})); }, //

      [&](const Type::Struct &t) -> Term::Any { return unsupported(t); }, //
      [&](const Type::Ptr &t) -> Term::Any { return unsupported(t); },    //
      [&](const Type::Arr &t) -> Term::Any { return unsupported(t); },    //
      [&](const Type::Var &t) -> Term::Any { return unsupported(t); },    //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(t); }    //
  );
}

std::function<Term::Any(Type::Any)> operator"" _(long double x);
std::function<Term::Any(Type::Any)> operator"" _(unsigned long long int x);
std::function<NamedBuilder(Type::Any)> operator"" _(const char *name, size_t);

Stmt::Any let(const std::string &name, const Type::Any &tpe);
AssignmentBuilder let(const std::string &name);
AssignmentBuilder var(const std::string &name);

// DSL shim mirroring the old `Select(init, last)` shape. With the new IR, Select is a Term with a
// root Named + a list of PathStep::Field; preserve the ergonomic `Select({roots...}, last)` form.
Term::Select Select(const Vector<Named> &init, const Named &last);
// Disambiguated NamedBuilder overload: name it differently to avoid ambiguity with the Named
// overload via implicit conversions from braced-init `{NamedBuilder}` lists.
Term::Select selectFromBuilders(const Vector<NamedBuilder> &init, const Named &last);
IntrOp call(const Intr::Any &);
MathOp call(const Math::Any &);
SpecOp call(const Spec::Any &);

std::function<Function(Vector<Stmt::Any>)> function(const std::string &name, const Vector<Arg> &args, const Type::Any &rtn,
                                                    FunctionVisibility::Any visibility = FunctionVisibility::Exported(),
                                                    FunctionFpMode::Any fpMode = FunctionFpMode::Relaxed(), bool isEntry = false);

Program program(const Vector<StructDef> &structs = {}, const Vector<Function> &functions = {});
Program program(const Function &function);

Return ret(const Expr::Any &expr = Alias(Term::Unit0Const()));
Return ret(const Term::Any &term);

} // namespace dsl

} // namespace polyregion::polyast
