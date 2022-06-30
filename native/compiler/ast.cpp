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
      *type,                                                                   //
      [](const Type::Float &x) { return "Float"s; },                           //
      [](const Type::Double &x) { return "Double"s; },                         //
      [](const Type::Bool &x) { return "Bool"s; },                             //
      [](const Type::Byte &x) { return "Byte"s; },                             //
      [](const Type::Char &x) { return "Char"s; },                             //
      [](const Type::Short &x) { return "Short"s; },                           //
      [](const Type::Int &x) { return "Int"s; },                               //
      [](const Type::Long &x) { return "Long"s; },                             //
      [](const Type::String &x) { return "String"s; },                         //
      [](const Type::Unit &x) { return "Unit"s; },                             //
      [](const Type::Nothing &x) { return "Nothing"s; },                       //
      [](const Type::Struct &x) { return "Struct[" + repr(x.name) + "]"; },    //
      [](const Type::Array &x) { return "Array[" + repr(x.component) + "]"; }, //
      [](const Type::Var &x) { return "Var[" + x.name + "]"; },                //
      [](const Type::Exec &x) { return "Exec[???]"s; }                         //
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
      [](const Term::Poison &x) { return "Null(" + repr(x.tpe) + ")"s; },
      [](const Term::UnitConst &x) { return "Unit()"s; },
      [](const Term::BoolConst &x) { return "Bool(" + std::to_string(x.value) + ")"; },
      [](const Term::ByteConst &x) { return "Byte(" + std::to_string(x.value) + ")"; },
      [](const Term::CharConst &x) { return "Char(" + std::to_string(x.value) + ")"; },
      [](const Term::ShortConst &x) { return "Short(" + std::to_string(x.value) + ")"; },
      [](const Term::IntConst &x) { return "Int(" + std::to_string(x.value) + ")"; },
      [](const Term::LongConst &x) { return "Long(" + std::to_string(x.value) + ")"; },
      [](const Term::DoubleConst &x) { return "Double(" + std::to_string(x.value) + ")"; },
      [](const Term::FloatConst &x) { return "Float(" + std::to_string(x.value) + ")"; },
      [](const Term::StringConst &x) { return "String(" + x.value + ")"; });
}

[[nodiscard]] string polyast::repr(const Expr::Any &expr) {
  return variants::total(
      *expr, //
      [](const Expr::NullaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind, //
            [](const NullaryIntrinsicKind::GpuGlobalIdxX &) { return "GpuGlobalIdxX"; },
            [](const NullaryIntrinsicKind::GpuGlobalIdxY &) { return "GpuGlobalIdxY"; },
            [](const NullaryIntrinsicKind::GpuGlobalIdxZ &) { return "GpuGlobalIdxZ"; },
            [](const NullaryIntrinsicKind::GpuGlobalSizeX &) { return "GpuGlobalSizeX"; },
            [](const NullaryIntrinsicKind::GpuGlobalSizeY &) { return "GpuGlobalSizeY"; },
            [](const NullaryIntrinsicKind::GpuGlobalSizeZ &) { return "GpuGlobalSizeZ"; },
            [](const NullaryIntrinsicKind::GpuGroupIdxX &) { return "GpuGroupIdxX"; },
            [](const NullaryIntrinsicKind::GpuGroupIdxY &) { return "GpuGroupIdxY"; },
            [](const NullaryIntrinsicKind::GpuGroupIdxZ &) { return "GpuGroupIdxZ"; },
            [](const NullaryIntrinsicKind::GpuGroupSizeX &) { return "GpuGroupSizeX"; },
            [](const NullaryIntrinsicKind::GpuGroupSizeY &) { return "GpuGroupSizeY"; },
            [](const NullaryIntrinsicKind::GpuGroupSizeZ &) { return "GpuGroupSizeZ"; },
            [](const NullaryIntrinsicKind::GpuLocalIdxX &) { return "GpuLocalIdxX"; },
            [](const NullaryIntrinsicKind::GpuLocalIdxY &) { return "GpuLocalIdxY"; },
            [](const NullaryIntrinsicKind::GpuLocalIdxZ &) { return "GpuLocalIdxZ"; },
            [](const NullaryIntrinsicKind::GpuLocalSizeX &) { return "GpuLocalSizeX"; },
            [](const NullaryIntrinsicKind::GpuLocalSizeY &) { return "GpuLocalSizeY"; },
            [](const NullaryIntrinsicKind::GpuLocalSizeZ &) { return "GpuLocalSizeZ"; },
            [](const NullaryIntrinsicKind::GpuGroupBarrier &) { return "GpuGroupBarrier"; },
            [](const NullaryIntrinsicKind::GpuGroupFence &) { return "GpuGroupFence"; });
        return std::string(op) + "()";
      },
      [](const Expr::UnaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind,                                                 //
            [](const UnaryIntrinsicKind::Sin &) { return "sin"; },   //
            [](const UnaryIntrinsicKind::Cos &) { return "cos"; },   //
            [](const UnaryIntrinsicKind::Tan &) { return "tan"; },   //
            [](const UnaryIntrinsicKind::Asin &) { return "asin"; }, //
            [](const UnaryIntrinsicKind::Acos &) { return "acos"; }, //
            [](const UnaryIntrinsicKind::Atan &) { return "atan"; }, //
            [](const UnaryIntrinsicKind::Sinh &) { return "sinh"; }, //
            [](const UnaryIntrinsicKind::Cosh &) { return "cosh"; }, //
            [](const UnaryIntrinsicKind::Tanh &) { return "tanh"; }, //

            [](const UnaryIntrinsicKind::Signum &) { return "signum"; }, //
            [](const UnaryIntrinsicKind::Abs &) { return "abs"; },       //
            [](const UnaryIntrinsicKind::Round &) { return "round"; },   //
            [](const UnaryIntrinsicKind::Ceil &) { return "ceil"; },     //
            [](const UnaryIntrinsicKind::Floor &) { return "floor"; },   //
            [](const UnaryIntrinsicKind::Rint &) { return "rint"; },     //

            [](const UnaryIntrinsicKind::Sqrt &) { return "sqrt"; },   //
            [](const UnaryIntrinsicKind::Cbrt &) { return "cbrt"; },   //
            [](const UnaryIntrinsicKind::Exp &) { return "exp"; },     //
            [](const UnaryIntrinsicKind::Expm1 &) { return "expm1"; }, //
            [](const UnaryIntrinsicKind::Log &) { return "log"; },     //
            [](const UnaryIntrinsicKind::Log1p &) { return "log1p"; }, //
            [](const UnaryIntrinsicKind::Log10 &) { return "log10"; }, //
            [](const UnaryIntrinsicKind::BNot &) { return "~"; },      //
            [](const UnaryIntrinsicKind::Pos &) { return "+"; },       //
            [](const UnaryIntrinsicKind::Neg &) { return "-"; },       //

            [](const UnaryIntrinsicKind::LogicNot &x) { return "!"; } //
        );
        return std::string(op) + "(" + repr(x.lhs) + ")";
      },
      [](const Expr::BinaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind,                                              //
            [](const BinaryIntrinsicKind::Add &) { return "+"; }, //
            [](const BinaryIntrinsicKind::Sub &) { return "-"; }, //
            [](const BinaryIntrinsicKind::Mul &) { return "*"; }, //
            [](const BinaryIntrinsicKind::Div &) { return "/"; }, //
            [](const BinaryIntrinsicKind::Rem &) { return "%"; }, //

            [](const BinaryIntrinsicKind::Pow &) { return "**"; }, //

            [](const BinaryIntrinsicKind::Min &) { return "min"; }, //
            [](const BinaryIntrinsicKind::Max &) { return "max"; }, //

            [](const BinaryIntrinsicKind::Atan2 &) { return "atan2"; }, //
            [](const BinaryIntrinsicKind::Hypot &) { return "hypot"; }, //

            [](const BinaryIntrinsicKind::BAnd &) { return "&"; },   //
            [](const BinaryIntrinsicKind::BOr &) { return "|"; },    //
            [](const BinaryIntrinsicKind::BXor &) { return "^"; },   //
            [](const BinaryIntrinsicKind::BSL &) { return "<<"; },   //
            [](const BinaryIntrinsicKind::BSR &) { return ">>"; },   //
            [](const BinaryIntrinsicKind::BZSR &) { return ">>>"; }, //

            [](const BinaryIntrinsicKind::LogicEq &) { return "=="; },  //
            [](const BinaryIntrinsicKind::LogicNeq &) { return "!="; }, //
            [](const BinaryIntrinsicKind::LogicAnd &) { return "&&"; }, //
            [](const BinaryIntrinsicKind::LogicOr &) { return "||"; },  //
            [](const BinaryIntrinsicKind::LogicLte &) { return "<="; }, //
            [](const BinaryIntrinsicKind::LogicGte &) { return ">="; }, //
            [](const BinaryIntrinsicKind::LogicLt &) { return "<"; },   //
            [](const BinaryIntrinsicKind::LogicGt &) { return ">"; }    //
        );
        return repr(x.lhs) + " " + std::string(op) + " " + repr(x.rhs);
      },
      [](const Expr::Cast &x) { return "(" + repr(x.from) + ".to[" + repr(x.as) + "])"; },
      [](const Expr::Alias &x) { return "(~>" + repr(x.ref) + ")"; },
      [](const Expr::Invoke &x) {
        return (x.receiver ? repr(*x.receiver) : "<module>") + "." + repr(x.name) + "(" +
               mk_string<Term::Any>(
                   x.args, [&](auto x) { return repr(x); }, ",") +
               ")" + ":" + repr(x.tpe);
      },
      [](const Expr::Index &x) { return repr(x.lhs) + "[" + repr(x.idx) + "]"; },
      [](const Expr::Alloc &x) { return "new [" + repr(x.witness.component) + "*" + repr(x.size) + "]"; },
      [](const Expr::Suspend &x) { return "Suspend(???)"s; } //
  );
}

[[nodiscard]] string polyast::repr(const Stmt::Any &stmt) {
  return variants::total(
      *stmt, //
      [](const Stmt::Comment &x) { return "// " + x.value; },
      [](const Stmt::Var &x) { return "var " + repr(x.name) + " = " + (x.expr ? repr(*x.expr) : "_"); },
      [](const Stmt::Mut &x) { return repr(x.name) + " := " + repr(x.expr); },
      [](const Stmt::Update &x) { return repr(x.lhs) + "[" + repr(x.idx) + "] = " + repr(x.value); },
      [](const Stmt::While &x) {
        auto tests = mk_string<Stmt::Any>(
            x.tests, [&](auto x) { return repr(x); }, "\n");

        return "while({" + tests + ";" + repr(x.cond) + "}){\n" +
               mk_string<Stmt::Any>(
                   x.body, [&](auto x) { return repr(x); }, "\n") +
               "}";
      },
      [](const Stmt::Break &x) { return "break;"s; }, [](const Stmt::Cont &x) { return "continue;"s; },
      [](const Stmt::Cond &x) {
        return "if(" + repr(x.cond) + ") { \n" +
               mk_string<Stmt::Any>(
                   x.trueBr, [&](auto x) { return repr(x); }, "\n") +
               "} else {\n" +
               mk_string<Stmt::Any>(
                   x.falseBr, [&](auto x) { return repr(x); }, "\n") +
               "}";
      },
      [](const Stmt::Return &x) { return "return " + repr(x.value); });
}

[[nodiscard]] string polyast::repr(const Function &fn) {
  return "def " + repr(fn.name) + "(" +
         mk_string<Named>(
             fn.args, [&](auto x) { return repr(x); }, ",") +
         ") : " + repr(fn.rtn) + " = {\n" +
         mk_string<Stmt::Any>(
             fn.body, [&](auto x) { return repr(x); }, "\n") +
         "\n}";
}

[[nodiscard]] string polyast::repr(const StructDef &def) {
  return "struct " + repr(def.name) + " { " +
         mk_string<Named>(
             def.members, [](auto &&x) { return repr(x); }, ",") +
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

std::string dsl::dslRepr(const Function &fn) {
  if (fn.name.fqn.size() != 1) {
    throw std::logic_error("Name fragments is not supported");
  }

  auto nameRepr = [](const Named &n) { return "\"" + n.symbol + "\"_(" + repr(n.tpe) + ")"; };

  auto proto =
      "function(\"" + fn.name.fqn[0] + "\",{" + mk_string<Named>(fn.args, nameRepr, ", ") + "}, " + repr(fn.rtn) + ")";

  auto body = mk_string<Stmt::Any>(
      fn.body,
      [&](const Stmt::Any &stmt) {
        return variants::total(
            *stmt, //
            [](const Stmt::Comment &x) { return to_string(x); },
            [](const Stmt::Var &x) {
              if (x.expr) return "let(\"" + x.name.symbol + "\") = " + to_string(*(x.expr));
              else
                return to_string(x);
            },
            [&](const Stmt::Mut &x) {
              if (x.name.init.empty()) return nameRepr(x.name.last) + " = " + to_string(x.expr);
              else
                return to_string(x);
            },
            [&](const Stmt::Update &x) {
              if (x.lhs.init.empty())
                return nameRepr(x.lhs.last) + "[" + to_string(x.idx) + "] = " + to_string(x.value);
              else
                return to_string(x);
            },
            [](const Stmt::While &x) { return to_string(x); }, [](const Stmt::Break &x) { return to_string(x); },
            [](const Stmt::Cont &x) { return to_string(x); }, [](const Stmt::Cond &x) { return to_string(x); },
            [](const Stmt::Return &x) { return to_string(x); });
      },
      ",\n");
  return proto + "({" + body + "})";
}

Type::Array dsl::Array(Type::Any t) { return Tpe::Array(t); }
Type::Struct dsl::Struct(Sym name, std::vector<std::string> tpeVars, std::vector<Type::Any> args) {
  return {name, tpeVars, args};
}
std::function<dsl::NamedBuilder(Type::Any)> dsl::operator""_(const char *name, size_t) {
  std::string name_(name);
  return [=](auto &&tpe) { return NamedBuilder{Named(name_, tpe)}; };
}
std::function<Term::Any(Type::Any)> dsl::operator""_(unsigned long long int x) {
  return [=](const Type::Any &t) {
    auto unsupported = [](auto &&t, auto &&v) -> Term::Any {
      throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value" +
                             std::to_string(v));
    };
    return variants::total(                                                      //
        *t,                                                                      //
        [&](const Type::Float &) -> Term::Any { return Term::FloatConst(x); },   //
        [&](const Type::Double &) -> Term::Any { return Term::DoubleConst(x); }, //
        [&](const Type::Bool &) -> Term::Any { return Term::BoolConst(x); },     //
        [&](const Type::Byte &) -> Term::Any { return Term::ByteConst(x); },     //
        [&](const Type::Char &) -> Term::Any { return Term::CharConst(x); },     //
        [&](const Type::Short &) -> Term::Any { return Term::ShortConst(x); },   //
        [&](const Type::Int &) -> Term::Any { return Term::IntConst(x); },       //
        [&](const Type::Long &) -> Term::Any { return Term::LongConst(x); },     //
        [&](const Type::String &t) -> Term::Any { return unsupported(t, x); },   //
        [&](const Type::Unit &t) -> Term::Any { return unsupported(t, x); },     //
        [&](const Type::Nothing &t) -> Term::Any { return unsupported(t, x); },  //
        [&](const Type::Struct &t) -> Term::Any { return unsupported(t, x); },   //
        [&](const Type::Array &t) -> Term::Any { return unsupported(t, x); },    //
        [&](const Type::Var &t) -> Term::Any { return unsupported(t, x); },      //
        [&](const Type::Exec &t) -> Term::Any { return unsupported(t, x); }      //

    );
  };
}
std::function<Term::Any(Type::Any)> dsl::operator""_(long double x) {
  return [=](const Type::Any &t) -> Term::Any {
    if (holds<Type::Double>(t)) return Term::DoubleConst(static_cast<double>(x));
    if (holds<Type::Float>(t)) return Term::FloatConst(static_cast<float>(x));
    throw std::logic_error("Cannot create fractional constant of type " + to_string(t) + " for value" +
                           std::to_string(x));
  };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Var(Named(name, tpe), {}); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name}; }
Expr::BinaryIntrinsic dsl::invoke(const BinaryIntrinsicKind::Any &kind, const Term::Any &lhs, const Term::Any &rhs,
                                  const Type::Any &rtn) {
  return {lhs, rhs, kind, rtn};
}
Expr::UnaryIntrinsic dsl::invoke(const UnaryIntrinsicKind::Any &kind, const Term::Any &lhs, const Type::Any &rtn) {
  return {lhs, kind, rtn};
}
Expr::NullaryIntrinsic dsl::invoke(const NullaryIntrinsicKind::Any &kind, const Type::Any &rtn) { return {kind, rtn}; }
std::function<Function(std::vector<Stmt::Any>)> dsl::function(const string &name, const std::vector<Named> &args,
                                                              const Type::Any &rtn) {
  return [=](auto &&stmts) { return Function(Sym({name}), {}, {}, args, {}, rtn, stmts); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Program dsl::program(Function entry, std::vector<Function> functions, std::vector<StructDef> defs) {
  return Program(entry, functions, defs);
}
dsl::IndexBuilder::IndexBuilder(const Expr::Index &index) : index(index) {}
dsl::IndexBuilder::operator const Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Term::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator const Term::Any() const { return Select({}, named); }
dsl::NamedBuilder::operator const Named() const { return named; }
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
