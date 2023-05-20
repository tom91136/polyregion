#include <string>

#include "ast.h"
#include "utils.hpp"
#include "variants.hpp"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

[[nodiscard]] string polyast::repr(const Sym &sym) {
  return mk_string<string>(
      sym.fqn, [](auto &&x) { return x; }, ".");
}

[[nodiscard]] string polyast::repr(const Type::Any &type) {
  return variants::total(
      *type,                                            //
      [](const Type::Float16 &) { return "Float16"s; }, //
      [](const Type::Float32 &) { return "Float32"s; }, //
      [](const Type::Float64 &) { return "Float64"s; }, //

      [](const Type::IntS8 &) { return "IntS8"s; },   //
      [](const Type::IntS16 &) { return "IntS16"s; }, //
      [](const Type::IntS32 &) { return "IntS32"s; }, //
      [](const Type::IntS64 &) { return "IntS64"s; }, //
      [](const Type::IntU8 &) { return "IntU8"s; },   //
      [](const Type::IntU16 &) { return "IntU16"s; }, //
      [](const Type::IntU32 &) { return "IntU32"s; }, //
      [](const Type::IntU64 &) { return "IntU64"s; }, //

      [](const Type::Unit0 &) { return "Unit0"s; },                            //
      [](const Type::Bool1 &) { return "Bool1"s; },                            //
      [](const Type::Nothing &) { return "Nothing"s; },                        //
      [](const Type::Struct &x) { return "Struct[" + repr(x.name) + "]"; },    //
      [](const Type::Array &x) { return "Array[" + repr(x.component) + "]"; }, //
      [](const Type::Var &x) { return "Var[" + x.name + "]"; },                //
      [](const Type::Exec &) { return "Exec[???]"s; }                          //
  );
}

[[nodiscard]] string polyast::repr(const Named &path) { return "(" + path.symbol + ":" + repr(path.tpe) + ")"; }

[[nodiscard]] string polyast::repr(const Term::Any &ref) {
  return variants::total(
      *ref,
      [](const Term::Select &x) {
        return x.init.empty() //
                   ? repr(x.last)
                   : mk_string<Named>(
                         x.init, [&](auto x) { return repr(x); }, ".") +
                         "." + repr(x.last);
      },
      [](const Term::Poison &x) { return "Poison(" + repr(x.tpe) + ")"; },

      [](const Term::Float16Const &x) { return "Float16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::Float32Const &x) { return "Float32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::Float64Const &x) { return "Float64(" + std::to_string(x.value) + ")"; }, //

      [](const Term::IntU8Const &x) { return "IntU8(" + std::to_string(x.value) + ")"; },   //
      [](const Term::IntU16Const &x) { return "IntU16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntU32Const &x) { return "IntU32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntU64Const &x) { return "IntU64(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS8Const &x) { return "IntS8(" + std::to_string(x.value) + ")"; },   //
      [](const Term::IntS16Const &x) { return "IntS16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS32Const &x) { return "IntS32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS64Const &x) { return "IntS64(" + std::to_string(x.value) + ")"; }, //

      [](const Term::Bool1Const &x) { return "Bool1(" + std::to_string(x.value) + ")"; }, //
      [](const Term::Unit0Const &x) { return "Unit0()"s; }                                //

  );
}

[[nodiscard]] string polyast::repr(const Expr::Any &expr) {
  return variants::total(
      *expr, //
      [](const Expr::SpecOp &x) { return to_string(x); }, [](const Expr::MathOp &x) { return to_string(x); },
      [](const Expr::IntrOp &x) { return to_string(x); },
      [](const Expr::Cast &x) { return "(" + repr(x.from) + ".to[" + repr(x.as) + "])"; },
      [](const Expr::Alias &x) { return "(~>" + repr(x.ref) + ")"; },
      [](const Expr::Invoke &x) {
        return (x.receiver ? repr(*x.receiver) : "<module>") + "." + repr(x.name) + "(" +
               mk_string<Term::Any>(
                   x.args, [&](auto x) { return repr(x); }, ",") +
               ")" + ":" + repr(x.tpe);
      },
      [](const Expr::Index &x) { return repr(x.lhs) + "[" + repr(x.idx) + "]"; },
      [](const Expr::Alloc &x) { return "new [" + repr(x.tpe) + "*" + repr(x.size) + "]"; });
}

[[nodiscard]] string polyast::repr(const Stmt::Any &stmt) {
  return variants::total(
      *stmt, //
      [](const Stmt::Block &x) {
        return "{ \n" +
               indent(2, mk_string<Stmt::Any>(
                             x.stmts, [&](auto x) { return repr(x); }, "\n")) +
               "}";
      },
      [](const Stmt::Comment &x) { return "// " + x.value; },
      [](const Stmt::Var &x) { return "var " + repr(x.name) + " = " + (x.expr ? repr(*x.expr) : "_"); },
      [](const Stmt::Mut &x) { return repr(x.name) + " := " + repr(x.expr); },
      [](const Stmt::Update &x) { return repr(x.lhs) + "[" + repr(x.idx) + "] = " + repr(x.value); },
      [](const Stmt::While &x) {
        auto tests = mk_string<Stmt::Any>(
            x.tests, [&](auto x) { return repr(x); }, "\n");

        return "while({" + tests + ";" + repr(x.cond) + "}){\n" +
               indent(2, mk_string<Stmt::Any>(
                             x.body, [&](auto x) { return repr(x); }, "\n")) +
               "\n}";
      },
      [](const Stmt::Break &x) { return "break;"s; }, [](const Stmt::Cont &x) { return "continue;"s; },
      [](const Stmt::Cond &x) {
        auto elseStmts = x.falseBr.empty() //
                             ? "\n}"
                             : "\n} else {\n" +
                                   indent(2, mk_string<Stmt::Any>(
                                                 x.falseBr, [&](auto x) { return repr(x); }, "\n")) +
                                   "\n}";
        return "if(" + repr(x.cond) + ") { \n" +
               indent(2, mk_string<Stmt::Any>(
                             x.trueBr, [&](auto x) { return repr(x); }, "\n")) +
               elseStmts;
      },
      [](const Stmt::Return &x) { return "return " + repr(x.value); });
}

[[nodiscard]] string polyast::repr(const Arg &arg) { return repr(arg.named); }

[[nodiscard]] string polyast::repr(const TypeSpace::Any &space) {
  return variants::total(
      *space, //
      [&](const TypeSpace::Global &x) { return "^Global"; }, [&](const TypeSpace::Local &x) { return "^Local"; });
}

[[nodiscard]] string polyast::repr(const Function &fn) {
  return "def " + repr(fn.name) + "(" +
         mk_string<Arg>(
             fn.args, [&](auto x) { return repr(x); }, ",") +
         ") : " + repr(fn.rtn) + " = {\n" +
         indent(2, mk_string<Stmt::Any>(
                       fn.body, [&](auto x) { return repr(x); }, "\n")) +
         "\n}";
}

[[nodiscard]] string polyast::repr(const StructDef &def) {
  return "struct " + repr(def.name) + " { " +
         mk_string<StructMember>(
             def.members, [](auto &&x) { return (x.isMutable ? "mut " : "") + repr(x.named); }, ",") +
         " }";
}

[[nodiscard]] string polyast::repr(const Program &program) {
  auto defs = mk_string<StructDef>(
      program.defs, [](auto &&x) { return repr(x); }, "\n");
  auto fns = mk_string<Function>(
      program.functions, [](auto &&x) { return repr(x); }, "\n");
  return defs + "\n" + fns + "\n" + repr(program.entry);
}

std::string polyast::qualified(const Term::Select &select) {
  return select.init.empty() //
             ? select.last.symbol
             : polyregion::mk_string<Named>(
                   select.init, [](auto &n) { return n.symbol; }, ".") +
                   "." + select.last.symbol;
}

std::string polyast::qualified(const Sym &sym) {
  return polyregion::mk_string<std::string>(
      sym.fqn, [](auto &n) { return n; }, ".");
}

std::vector<Named> polyast::path(const Term::Select &select) {
  std::vector<Named> xs(select.init);
  xs.push_back(select.last);
  return xs;
}

Named polyast::head(const Term::Select &select) { return select.init.empty() ? select.last : select.init.front(); }

std::vector<Named> polyast::tail(const Term::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}

std::pair<Named, std::vector<Named>> polyast::uncons(const Term::Select &select) {
  if (select.init.empty()) return {{select.last}, {}};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return {select.init.front(), xs};
  }
}

Type::Array dsl::Array(const Type::Any &t, const ::TypeSpace::Any &s) { return Tpe::Array(t, s); }
Type::Struct dsl::Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args) { return {name, tpeVars, args, {}}; }
Term::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](auto &&t, auto &&v) -> Term::Any {
    throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value" + std::to_string(v));
  };
  return variants::total(                                                        //
      *tpe,                                                                      //
      [&](const Type::Float16 &) -> Term::Any { return Term::Float16Const(x); }, //
      [&](const Type::Float32 &) -> Term::Any { return Term::Float32Const(x); }, //
      [&](const Type::Float64 &) -> Term::Any { return Term::Float64Const(x); }, //

      [&](const Type::IntU8 &) -> Term::Any { return Term::IntU8Const(x); },   //
      [&](const Type::IntU16 &) -> Term::Any { return Term::IntU16Const(x); }, //
      [&](const Type::IntU32 &) -> Term::Any { return Term::IntU32Const(x); }, //
      [&](const Type::IntU64 &) -> Term::Any { return Term::IntU64Const(x); }, //
      [&](const Type::IntS8 &) -> Term::Any { return Term::IntS8Const(x); },   //
      [&](const Type::IntS16 &) -> Term::Any { return Term::IntS16Const(x); }, //
      [&](const Type::IntS32 &) -> Term::Any { return Term::IntS32Const(x); }, //
      [&](const Type::IntS64 &) -> Term::Any { return Term::IntS64Const(x); }, //

      [&](const Type::Unit0 &t) -> Term::Any { return unsupported(t, x); },   //
      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(x); },  //
      [&](const Type::Nothing &t) -> Term::Any { return unsupported(t, x); }, //
      [&](const Type::Struct &t) -> Term::Any { return unsupported(t, x); },  //
      [&](const Type::Array &t) -> Term::Any { return unsupported(t, x); },   //
      [&](const Type::Var &t) -> Term::Any { return unsupported(t, x); },     //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(t, x); }     //

  );
}
Term::Any dsl::fractional(const Type::Any &tpe, long double x) {
  if (holds<Type::Float64>(tpe)) return Term::Float64Const(static_cast<double>(x));
  if (holds<Type::Float32>(tpe)) return Term::Float32Const(static_cast<float>(x));
  if (holds<Type::Float16>(tpe)) return Term::Float16Const(static_cast<float>(x));
  throw std::logic_error("Cannot create fractional constant of type " + to_string(tpe) + " for value" + std::to_string(x));
}
std::function<Term::Any(Type::Any)> dsl::operator""_(unsigned long long int x) {
  return [=](const Type::Any &t) { return integral(t, x); };
}
std::function<Term::Any(Type::Any)> dsl::operator""_(long double x) {
  return [=](const Type::Any &t) { return fractional(t, x); };
}
std::function<dsl::NamedBuilder(Type::Any)> dsl::operator""_(const char *name, size_t) {
  std::string name_(name);
  return [=](auto &&tpe) { return NamedBuilder{Named(name_, tpe)}; };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Var(Named(name, tpe), {}); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name}; }
Expr::IntrOp dsl::invoke(const Intr::Any &intr) { return Expr::IntrOp(intr); }
Expr::MathOp dsl::invoke(const Math::Any &intr) { return Expr::MathOp(intr); }
Expr::SpecOp dsl::invoke(const Spec::Any &intr) { return Expr::SpecOp(intr); }
std::function<Function(std::vector<Stmt::Any>)> dsl::function(const string &name, const std::vector<Arg> &args, const Type::Any &rtn) {
  return [=](auto &&stmts) { return Function(Sym({name}), {}, {}, args, {}, {}, rtn, stmts); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Stmt::Return dsl::ret(const Term::Any &term) { return Return(Alias(term)); }
Program dsl::program(Function entry, std::vector<StructDef> defs, std::vector<Function> functions) {
  return Program(entry, functions, defs);
}

dsl::IndexBuilder::IndexBuilder(const Expr::Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Term::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Term::Any() const { return Select({}, named); }
// dsl::NamedBuilder::operator const Expr::Any() const { return Alias(Select({}, named)); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Term::Any &idx) const {
  if (auto arr = get_opt<Type::Array>(named.tpe); arr) {
    return IndexBuilder({Select({}, named), idx, arr->component});
  } else {
    throw std::logic_error("Cannot index a reference to non-array type" + to_string(named));
  }
}

dsl::AssignmentBuilder::AssignmentBuilder(const std::string &name) : name(name) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Expr::Any rhs) const { return Var(Named(name, tpe(rhs)), {rhs}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Term::Any rhs) const { return Var(Named(name, tpe(rhs)), {Alias(rhs)}); }
