#pragma once

#include "generated/polyast.h"
#include "variants.hpp"
#include <optional>
#include <ostream>
#include <unordered_map>
#include <utility>

namespace polyregion::polyast {

using Bytes = std::vector<char>;
template <typename T> using Opt = std::optional<T>;
template <typename T, typename U> using Pair = std::pair<T, U>;
template <typename T, typename U> using Map = std::unordered_map<T, U>;

std::string repr(const Sym &);
std::string repr(const Type::Any &);
std::string repr(const Named &);
std::string repr(const Term::Any &);
std::string repr(const Expr::Any &);
std::string repr(const Stmt::Any &);
std::string repr(const Intr::Any &);
std::string repr(const Spec::Any &);
std::string repr(const Math::Any &);
std::string repr(const Arg &);
std::string repr(const TypeSpace::Any &space);
std::string repr(const Function &);
std::string repr(const StructDef &);
std::string repr(const Program &);

std::string qualified(const Term::Select &);
std::string qualified(const Sym &);
std::vector<Named> path(const Term::Select &);
Named head(const Term::Select &);
std::vector<Named> tail(const Term::Select &);

std::pair<Named, std::vector<Named>> uncons(const Term::Select &);

enum class Target : uint8_t {
  Object_LLVM_HOST = 10,
  Object_LLVM_x86_64,
  Object_LLVM_AArch64,
  Object_LLVM_ARM,

  Object_LLVM_NVPTX64 = 20,
  Object_LLVM_AMDGCN,
  Object_LLVM_SPIRV32,
  Object_LLVM_SPIRV64,

  Source_C_C11 = 30,
  Source_C_OpenCL1_1,
  Source_C_Metal1_0,
};

enum class OptLevel : uint8_t {
  O0 = 10,
  O1,
  O2,
  O3,
  Ofast,
};

std::optional<Target> targetFromOrdinal(std::underlying_type_t<Target> ordinal);

std::optional<OptLevel> optFromOrdinal(std::underlying_type_t<OptLevel> ordinal);

std::string repr(const polyast::CompileResult &);

namespace dsl {

using namespace Stmt;
using namespace Term;
using namespace Expr;
namespace Tpe = Type;
// namespace Fn0 = NullaryIntrinsicKind;
// namespace Fn2 = BinaryIntrinsicKind;
// namespace Fn1 = UnaryIntrinsicKind;

const static TypeSpace::Global Global = TypeSpace::Global();
const static TypeSpace::Local Local = TypeSpace::Local();

const static Tpe::Float32 Float = Tpe::Float32();
const static Tpe::Float64 Double = Tpe::Float64();
const static Tpe::Bool1 Bool = Tpe::Bool1();
const static Tpe::IntS8 Byte = Tpe::IntS8();
const static Tpe::IntU16 Char = Tpe::IntU16();
const static Tpe::IntS16 Short = Tpe::IntS16();
const static Tpe::IntS32 SInt = Tpe::IntS32();
const static Tpe::IntU32 UInt = Tpe::IntU32();
const static Tpe::IntS64 Long = Tpe::IntS64();
const static Tpe::Unit0 Unit = Tpe::Unit0();
const static Tpe::Nothing Nothing = Tpe::Nothing();

Tpe::Array Array(const Tpe::Any &t, const TypeSpace::Any &s = TypeSpace::Global());
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
  operator Expr::Any() const; // NOLINT(google-explicit-constructor)
  Update operator=(const Term::Any &term) const;
};

struct NamedBuilder {
  Named named;
  explicit NamedBuilder(const Named &named);
  operator Term::Any() const; // NOLINT(google-explicit-constructor)
                              //  operator const Expr::Any() const; // NOLINT(google-explicit-constructor)
  operator Named() const;     // NOLINT(google-explicit-constructor)
  Arg operator()() const;
  IndexBuilder operator[](const Term::Any &idx) const;
};

Term::Any integral(const Type::Any &tpe, unsigned long long int x);
Term::Any fractional(const Type::Any &tpe, long double x);

std::function<Term::Any(Type::Any)> operator"" _(long double x);
std::function<Term::Any(Type::Any)> operator"" _(unsigned long long int x);
std::function<NamedBuilder(Type::Any)> operator"" _(const char *name, size_t);

Stmt::Any let(const std::string &name, const Type::Any &tpe);
AssignmentBuilder let(const std::string &name);
IntrOp invoke(const Intr::Any &);
MathOp invoke(const Math::Any &);
SpecOp invoke(const Spec::Any &);
std::function<Function(std::vector<Stmt::Any>)> function(const std::string &name, const std::vector<Arg> &args, const Type::Any &rtn);
Program program(Function entry, std::vector<StructDef> defs = {}, std::vector<Function> functions = {});

Return ret(const Expr::Any &expr = Alias(Unit0Const()));
Return ret(const Term::Any &term);

} // namespace dsl

} // namespace polyregion::polyast