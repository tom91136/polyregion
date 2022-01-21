#include <iostream>

#include "opencl.h"
#include "utils.hpp"
#include "variants.hpp"

using namespace polyregion;
using namespace std::string_literals;

std::string backend::OpenCL::mkTpe(const Type::Any &tpe) {
  return variants::total(
      *tpe,                                                          //
      [&](const Type::Float &x) { return "float"s; },                //
      [&](const Type::Double &x) { return "double"s; },              //
      [&](const Type::Bool &x) { return "bool"s; },                  //
      [&](const Type::Byte &x) { return "int8_t"s; },                //
      [&](const Type::Char &x) { return "uint16_t"s; },              //
      [&](const Type::Short &x) { return "int16_t"s; },              //
      [&](const Type::Int &x) { return "int32_t"s; },                //
      [&](const Type::Long &x) { return "int64_t"s; },               //
      [&](const Type::String &x) { return "char *"s; },              //
      [&](const Type::Unit &x) { return "void"s; },                  //
      [&](const Type::Struct &x) { return "???"s; },                 //
      [&](const Type::Array &x) { return mkTpe(x.component) + "*"; } //
  );
}

std::string backend::OpenCL::mkRef(const Term::Any &ref) {
  return variants::total(
      *ref,                                                                  //
      [](const Term::Select &x) { return x.last.symbol; },                   //
      [](const Term::UnitConst &x) { return "/*void*/"s; },                  //
      [](const Term::BoolConst &x) { return x.value ? "true"s : "false"s; }, //
      [](const Term::ByteConst &x) { return std::to_string(x.value); },      //
      [](const Term::CharConst &x) { return std::to_string(x.value); },      //
      [](const Term::ShortConst &x) { return std::to_string(x.value); },     //
      [](const Term::IntConst &x) { return std::to_string(x.value); },       //
      [](const Term::LongConst &x) { return std::to_string(x.value); },      //
      [](const Term::DoubleConst &x) { return std::to_string(x.value); },    //
      [](const Term::FloatConst &x) { return std::to_string(x.value); },     //
      [](const Term::StringConst &x) { return "" + x.value + ""; }           //
  );                                                                         // FIXME escape string
}

std::string backend::OpenCL::mkExpr(const Expr::Any &expr, const std::string &key) {
  return variants::total(
      *expr,                                                           //
      [&](const Expr::Sin &x) { return "sin(" + mkRef(x.lhs) + ")"; }, //
      [&](const Expr::Cos &x) { return "cos(" + mkRef(x.lhs) + ")"; }, //
      [&](const Expr::Tan &x) { return "tan(" + mkRef(x.lhs) + ")"; }, //
      [&](const Expr::Abs &x) { return "abs(" + mkRef(x.lhs) + ")"; }, //

      [&](const Expr::Add &x) { return mkRef(x.lhs) + " + " + mkRef(x.rhs); }, //
      [&](const Expr::Sub &x) { return mkRef(x.lhs) + " - " + mkRef(x.rhs); }, //
      [&](const Expr::Div &x) { return mkRef(x.lhs) + " / " + mkRef(x.rhs); }, //
      [&](const Expr::Mul &x) { return mkRef(x.lhs) + " * " + mkRef(x.rhs); }, //
      [&](const Expr::Rem &x) { return mkRef(x.lhs) + " % " + mkRef(x.rhs); }, //
      [&](const Expr::Pow &x) { return mkRef(x.lhs) + " ^ " + mkRef(x.rhs); }, //

      [](const Expr::BNot &x) { return "^" + repr(x.lhs); },                 //
      [](const Expr::BAnd &x) { return repr(x.lhs) + " & " + repr(x.rhs); }, //
      [](const Expr::BOr &x) { return repr(x.lhs) + " | " + repr(x.rhs); },  //
      [](const Expr::BXor &x) { return repr(x.lhs) + " ^ " + repr(x.rhs); }, //
      [](const Expr::BSL &x) { return repr(x.lhs) + " >> " + repr(x.rhs); }, //
      [](const Expr::BSR &x) { return repr(x.lhs) + " << " + repr(x.rhs); }, //

      [&](const Expr::Not &x) { return "!(" + mkRef(x.lhs) + ")"; },            //
      [&](const Expr::Eq &x) { return mkRef(x.lhs) + " == " + mkRef(x.rhs); },  //
      [](const Expr::Neq &x) { return repr(x.lhs) + " != " + repr(x.rhs); },    //
      [](const Expr::And &x) { return repr(x.lhs) + " && " + repr(x.rhs); },    //
      [](const Expr::Or &x) { return repr(x.lhs) + " || " + repr(x.rhs); },     //
      [&](const Expr::Lte &x) { return mkRef(x.lhs) + " <= " + mkRef(x.rhs); }, //
      [&](const Expr::Gte &x) { return mkRef(x.lhs) + " >= " + mkRef(x.rhs); }, //
      [&](const Expr::Lt &x) { return mkRef(x.lhs) + " < " + mkRef(x.rhs); },   //
      [&](const Expr::Gt &x) { return mkRef(x.lhs) + " > " + mkRef(x.rhs); },   //

      [&](const Expr::Alias &x) { return mkRef(x.ref); },                               //
      [&](const Expr::Invoke &x) { return "???"s; },                                    //
      [&](const Expr::Index &x) { return qualified(x.lhs) + "[" + mkRef(x.idx) + "]"; } //
  );
}

std::string backend::OpenCL::mkStmt(const Stmt::Any &stmt) {
  return variants::total(
      *stmt, //
      [&](const Stmt::Comment &x) {
        auto lines = split(x.value, '\n');
        auto commented = map_vec<std::string, std::string>(lines, [](auto &&line) { return "// " + line; });
        return mk_string<std::string>(
            commented, [](auto &&x) { return x; }, "\n");
      },
      [&](const Stmt::Var &x) {
        auto line =
            mkTpe(x.name.tpe) + " " + x.name.symbol + (x.expr ? (" = " + mkExpr(*x.expr, x.name.symbol)) : "") + ";";
        return line;
      },
      [&](const Stmt::Mut &x) { return x.name.last.symbol + " = " + mkExpr(x.expr, "?") + ";"; },
      [&](const Stmt::Update &x) {
        auto idx = mkRef(x.idx);
        auto val = mkRef(x.value);
        return qualified(x.lhs) + "[" + idx + "] = " + val + ";";
      },
      [&](const Stmt::Effect &x) -> std::string {
        auto lhs = (x.lhs.last.symbol);
        throw std::logic_error("no impl");
      },
      [&](const Stmt::While &x) {
        auto body = mk_string<Stmt::Any>(
            x.body, [&](auto &stmt) { return mkStmt(stmt); }, "\n");
        return "while (" + mkExpr(x.cond, "?") + ") {\n" + body + "\n}";
      },
      [&](const Stmt::Break &x) { return "break;"s; },   //
      [&](const Stmt::Cont &x) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        return "if(" + mkExpr(x.cond, "if") + ") { \n" +
               mk_string<Stmt::Any>(
                   x.trueBr, [&](auto x) { return mkStmt(x); }, "\n") +
               "} else {\n" +
               mk_string<Stmt::Any>(
                   x.falseBr, [&](auto x) { return mkStmt(x); }, "\n") +
               "}";
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value, "rtn") + ";"; } //
  );
}

compiler::Compilation backend::OpenCL::run(const Function &fnTree) {

  auto start = compiler::nowMono();

  auto args = mk_string<Named>(
      fnTree.args, [&](auto x) { return mkTpe(x.tpe) + " " + x.symbol; }, ", ");

  auto prototype = mkTpe(fnTree.rtn) + " " + fnTree.name + "(" + args + ")";

  auto body = mk_string<Stmt::Any>(
      fnTree.body, [&](auto &stmt) { return mkStmt(stmt); }, "\n");

  auto def = prototype + "{\n" + body + "\n}";
  std::cout << def << std::endl;

  std::vector<uint8_t> data(def.c_str(), def.c_str() + def.length() + 1);

  return compiler::Compilation(                                               //
      data,                                                                   //
      {},                                                                     //
      {{compiler::nowMs(), "polyast_to_opencl", compiler::elapsedNs(start)}}, //
      ""                                                                      //
  );
}
