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
      *type,                                                                //
      [](const Type::Float &x) { return "Float"s; },                        //
      [](const Type::Double &x) { return "Double"s; },                      //
      [](const Type::Bool &x) { return "Bool"s; },                          //
      [](const Type::Byte &x) { return "Byte"s; },                          //
      [](const Type::Char &x) { return "Char"s; },                          //
      [](const Type::Short &x) { return "Short"s; },                        //
      [](const Type::Int &x) { return "Int"s; },                            //
      [](const Type::Long &x) { return "Long"s; },                          //
      [](const Type::String &x) { return "String"s; },                      //
      [](const Type::Unit &x) { return "Unit"s; },                          //
      [](const Type::Struct &x) { return "Struct[" + repr(x.name) + "]"; }, //
      [](const Type::Array &x) { return "Array[" + repr(x.component) + "]"; });
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
      [](const Expr::Sin &x) { return "sin(" + repr(x.lhs) + ")"; },
      [](const Expr::Cos &x) { return "cos(" + repr(x.lhs) + ")"; },
      [](const Expr::Tan &x) { return "tan(" + repr(x.lhs) + ")"; },
      [](const Expr::Abs &x) { return "abs(" + repr(x.lhs) + ")"; },

      [](const Expr::Add &x) { return repr(x.lhs) + " + " + repr(x.rhs); },
      [](const Expr::Sub &x) { return repr(x.lhs) + " - " + repr(x.rhs); },
      [](const Expr::Div &x) { return repr(x.lhs) + " / " + repr(x.rhs); },
      [](const Expr::Mul &x) { return repr(x.lhs) + " * " + repr(x.rhs); },
      [](const Expr::Rem &x) { return repr(x.lhs) + " % " + repr(x.rhs); },
      [](const Expr::Pow &x) { return repr(x.lhs) + " ^ " + repr(x.rhs); },

      [](const Expr::BNot &x) { return "^" + repr(x.lhs); },
      [](const Expr::BAnd &x) { return repr(x.lhs) + " & " + repr(x.rhs); },
      [](const Expr::BOr &x) { return repr(x.lhs) + " | " + repr(x.rhs); },
      [](const Expr::BXor &x) { return repr(x.lhs) + " ^ " + repr(x.rhs); },
      [](const Expr::BSL &x) { return repr(x.lhs) + " >> " + repr(x.rhs); },
      [](const Expr::BSR &x) { return repr(x.lhs) + " << " + repr(x.rhs); },

      [](const Expr::Not &x) { return "!(" + repr(x.lhs) + ")"; },
      [](const Expr::Eq &x) { return repr(x.lhs) + " == " + repr(x.rhs); },
      [](const Expr::Neq &x) { return repr(x.lhs) + " != " + repr(x.rhs); },
      [](const Expr::And &x) { return repr(x.lhs) + " && " + repr(x.rhs); },
      [](const Expr::Or &x) { return repr(x.lhs) + " || " + repr(x.rhs); },
      [](const Expr::Lte &x) { return repr(x.lhs) + " <= " + repr(x.rhs); },
      [](const Expr::Gte &x) { return repr(x.lhs) + " >= " + repr(x.rhs); },
      [](const Expr::Lt &x) { return repr(x.lhs) + " < " + repr(x.rhs); },
      [](const Expr::Gt &x) { return repr(x.lhs) + " > " + repr(x.rhs); },

      [](const Expr::Alias &x) { return "(~>" + repr(x.ref) + ")"; },
      [](const Expr::Invoke &x) {
        return repr(x.lhs) + "`" + x.name + "`" +
               mk_string<Term::Any>(
                   x.args, [&](auto x) { return repr(x); }, ",") +
               ":" + repr(x.tpe);
      },
      [](const Expr::Index &x) { return repr(x.lhs) + "[" + repr(x.idx) + "]"; });
}

[[nodiscard]] string polyast::repr(const Stmt::Any &stmt) {
  return variants::total(
      *stmt, //
      [](const Stmt::Comment &x) { return "// " + x.value; },
      [](const Stmt::Var &x) { return "var " + repr(x.name) + " = " + (x.expr ? repr(*x.expr) : "_"); },
      [](const Stmt::Mut &x) { return repr(x.name) + " := " + repr(x.expr); },
      [](const Stmt::Update &x) { return repr(x.lhs) + "[" + repr(x.idx) + "] = " + repr(x.value); },
      [](const Stmt::Effect &x) {
        return repr(x.lhs) + "`" + x.name + "`" +
               mk_string<Term::Any>(
                   x.args, [&](auto x) { return repr(x); }, ",") +
               " : Unit";
      },
      [](const Stmt::While &x) {
        return "while(" + repr(x.cond) + "){\n" +
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
  return "def " + fn.name + "(" +
         mk_string<Named>(
             fn.args, [&](auto x) { return repr(x); }, ",") +
         ") : " + repr(fn.rtn) + " = {\n" +
         mk_string<Stmt::Any>(
             fn.body, [&](auto x) { return repr(x); }, "\n") +
         "\n}";
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

Named polyast::head(const Term::Select &select) { return select.init.empty() ? select.last : select.init[0]; }

std::vector<Named> polyast::tail(const Term::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}