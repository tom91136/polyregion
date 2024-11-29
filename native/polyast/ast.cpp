#include <iomanip>
#include <string>

#include "ast.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

using namespace aspartame;

string polyast::qualified(const Expr::Select &select) {
  return select.init | append(select.last) | mk_string(".", [](auto &x) { return x.symbol; });
}

std::vector<Named> polyast::path(const Expr::Select &select) { return select.init | append(select.last) | to_vector(); }

Named polyast::head(const Expr::Select &select) { return select.init.empty() ? select.last : select.init.front(); }

std::vector<Named> polyast::tail(const Expr::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}

std::pair<Named, std::vector<Named>> polyast::uncons(const Expr::Select &select) {
  if (select.init.empty()) return {{select.last}, {}};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return {select.init.front(), xs};
  }
}

string polyast::repr(const polyast::CompileResult &compilation) {
  std::ostringstream os;
  os << "Compilation {"                                                                                            //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)") //
     << "\n  messages: `" << compilation.messages << "`"                                                           //
     << "\n  features: `" << (compilation.features ^ mk_string(",")) << "`"                                        //
     << "\n  layouts: `" << (compilation.layouts ^ mk_string("\n    ")) << "`"                                     //
     << "\n  events:\n";

  for (auto &e : compilation.events) {
    os << "    [" << e.epochMillis << ", +" << static_cast<double>(e.elapsedNanos) / 1e6 << "ms] " << e.name;
    if (e.data.empty()) continue;
    os << ":\n";
    std::stringstream ss(e.data);
    string l;
    size_t ln = 0;
    while (std::getline(ss, l, '\n')) {
      ln++;
      os << "    " << std::setw(3) << ln << "│" << l << '\n';
    }
    os << "       ╰───\n";
  }
  os << "\n}";
  return os.str();
}

Type::Ptr dsl::Ptr(const Type::Any &t, std::optional<int32_t> l, const ::TypeSpace::Any &s) { return Tpe::Ptr(t, l, s); }
// Type::Struct dsl::Struct(string name,  std::vector<Type::Any> members) { return {name,   args, {}}; }
Expr::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](auto &&t, auto &&v) -> Expr::Any {
    throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value" + std::to_string(v));
  };
  return tpe.match_total(                                                  //
      [&](const Type::Float16 &) -> Expr::Any { return Float16Const(x); }, //
      [&](const Type::Float32 &) -> Expr::Any { return Float32Const(x); }, //
      [&](const Type::Float64 &) -> Expr::Any { return Float64Const(x); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return IntU8Const(x); },   //
      [&](const Type::IntU16 &) -> Expr::Any { return IntU16Const(x); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return IntU32Const(x); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return IntU64Const(x); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return IntS8Const(x); },   //
      [&](const Type::IntS16 &) -> Expr::Any { return IntS16Const(x); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return IntS32Const(x); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return IntS64Const(x); }, //

      [&](const Type::Nothing &t) -> Expr::Any { return unsupported(t, x); }, //
      [&](const Type::Unit0 &t) -> Expr::Any { return unsupported(t, x); },   //
      [&](const Type::Bool1 &) -> Expr::Any { return Bool1Const(x); },        //

      [&](const Type::Struct &t) -> Expr::Any { return unsupported(t, x); },   //
      [&](const Type::Ptr &t) -> Expr::Any { return unsupported(t, x); },      //
      [&](const Type::Annotated &t) -> Expr::Any { return unsupported(t, x); } //
  );
}
Expr::Any dsl::fractional(const Type::Any &tpe, long double x) {
  if (tpe.is<Type::Float64>()) return Float64Const(static_cast<double>(x));
  if (tpe.is<Type::Float32>()) return Float32Const(static_cast<float>(x));
  if (tpe.is<Type::Float16>()) return Float16Const(static_cast<float>(x));
  throw std::logic_error("Cannot create fractional constant of type " + to_string(tpe) + " for value" + std::to_string(x));
}
std::function<Expr::Any(Type::Any)> dsl::operator""_(unsigned long long int x) {
  return [=](const Type::Any &t) { return integral(t, x); };
}
std::function<Expr::Any(Type::Any)> dsl::operator""_(long double x) {
  return [=](const Type::Any &t) { return fractional(t, x); };
}
std::function<dsl::NamedBuilder(Type::Any)> dsl::operator""_(const char *name, size_t) {
  string name_(name);
  return [=](auto &&tpe) { return NamedBuilder{Named(name_, tpe)}; };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Var(Named(name, tpe), {}); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name}; }
Expr::IntrOp dsl::invoke(const Intr::Any &intr) { return IntrOp(intr); }
Expr::MathOp dsl::invoke(const Math::Any &intr) { return MathOp(intr); }
Expr::SpecOp dsl::invoke(const Spec::Any &intr) { return SpecOp(intr); }
std::function<Function(std::vector<Stmt::Any>)> dsl::function(const string &name, const std::vector<Arg> &args, const Type::Any &rtn,
                                                              const std::set<FunctionAttr::Any> &attrs) {
  return [=](auto &&stmts) { return Function(name, args, rtn, stmts, attrs); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Program dsl::program(const std::vector<StructDef> &structs, const std::vector<Function> &functions) { return {structs, functions}; }
Program dsl::program(const Function &function) { return Program({}, {function}); }

dsl::IndexBuilder::IndexBuilder(const Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Expr::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Expr::Any() const { return Select({}, named); }
// dsl::NamedBuilder::operator const Expr::Any() const { return Alias(Select({}, named)); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Expr::Any &idx) const {
  if (auto arr = named.tpe.get<Type::Ptr>()) {
    return IndexBuilder({Select({}, named), idx, arr->comp});
  }
  throw std::logic_error("Cannot index a reference to non-array type" + to_string(named));
}
dsl::Mut dsl::NamedBuilder::operator=(const Expr::Any &that) const { return Mut(Select({}, named), that); }

dsl::AssignmentBuilder::AssignmentBuilder(const string &name) : name(name) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Expr::Any rhs) const { return Var(Named(name, rhs.tpe()), {rhs}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Type::Any tpe) const { return Var(Named(name, tpe), {}); }
