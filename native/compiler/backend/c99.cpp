#include <iostream>

#include "c99.h"
#include "utils.hpp"
#include "variants.hpp"

using namespace polyregion;
using namespace std::string_literals;

std::string backend::C99::mkTpe(const Type::Any &tpe) {
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
      [&](const Type::Struct &x) { return qualified(x.name); },      //
      [&](const Type::Array &x) { return mkTpe(x.component) + "*"; } //
  );
}

std::string backend::C99::mkRef(const Term::Any &ref) {
  return variants::total(
      *ref,                                                                  //
      [](const Term::Select &x) { return polyast::qualified(x); },           //
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

std::string backend::C99::mkExpr(const Expr::Any &expr, const std::string &key) {
  return variants::total(
      *expr, //
      [](const Expr::UnaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind, //
            [](const UnaryIntrinsicKind::Sin &x) { return "sin"; },
            [](const UnaryIntrinsicKind::Cos &x) { return "cos"; },
            [](const UnaryIntrinsicKind::Tan &x) { return "tan"; },
            [](const UnaryIntrinsicKind::Abs &x) { return "abs"; },
            [](const UnaryIntrinsicKind::BNot &x) { return "~"; });
        return std::string(op) + "(" + repr(x.lhs) + ")";
      },
      [](const Expr::BinaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind, //
            [](const BinaryIntrinsicKind::Add &x) { return "+"; },
            [](const BinaryIntrinsicKind::Sub &x) { return "-"; },
            [](const BinaryIntrinsicKind::Div &x) { return "/"; },
            [](const BinaryIntrinsicKind::Mul &x) { return "*"; },
            [](const BinaryIntrinsicKind::Rem &x) { return "%"; },
            [](const BinaryIntrinsicKind::Pow &x) { return "**"; },

            [](const BinaryIntrinsicKind::BAnd &x) { return "&"; },
            [](const BinaryIntrinsicKind::BOr &x) { return "|"; },
            [](const BinaryIntrinsicKind::BXor &x) { return "^"; },
            [](const BinaryIntrinsicKind::BSL &x) { return ">>"; },
            [](const BinaryIntrinsicKind::BSR &x) { return "<<"; });
        return repr(x.lhs) + " " + std::string(op) + " " + repr(x.rhs);
      },
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

std::string backend::C99::mkStmt(const Stmt::Any &stmt) {
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
      [&](const Stmt::Mut &x) {
        if (x.copy) {
          return "memcpy(" + //
                 polyast::qualified(x.name) + ", " + mkExpr(x.expr, "?") + ", sizeof(" + mkTpe(tpe(x.expr)) + "));";
        } else {
          return polyast::qualified(x.name) + " = " + mkExpr(x.expr, "?") + ";";
        }
      },
      [&](const Stmt::Update &x) {
        auto idx = mkRef(x.idx);
        auto val = mkRef(x.value);
        return qualified(x.lhs) + "[" + idx + "] = " + val + ";";
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

compiler::Compilation backend::C99::run(const Program &program) {
  auto fnTree = program.entry;

  auto start = compiler::nowMono();

  auto args = mk_string<Named>(
      fnTree.args, [&](auto x) { return mkTpe(x.tpe) + " " + x.symbol; }, ", ");

  auto prototype = mkTpe(fnTree.rtn) + " " + qualified(fnTree.name) + "(" + args + ")";

  auto body = mk_string<Stmt::Any>(
      fnTree.body, [&](auto &stmt) { return "  " + mkStmt(stmt); }, "\n");

  auto structDefs = mk_string<StructDef>(
      program.defs,
      [&](auto &x) {
        return std::accumulate(                                                                           //
                   x.members.begin(), x.members.end(),                                                    //
                   "typedef struct {"s,                                                                   //
                   [&](auto &&acc, auto m) { return acc + "\n  " + mkTpe(m.tpe) + " " + m.symbol + ";"; } //
                   ) +
               "\n} " + qualified(x.name) + ";";
      },
      "\n");

  auto def = structDefs + "\n" + prototype + "{\n" + body + "\n}";
  std::cout << def << std::endl;
  std::vector<uint8_t> data(def.c_str(), def.c_str() + def.length() + 1);

  return compiler::Compilation(                                                   //
      data,                                                                       //
      {{compiler::nowMs(), compiler::elapsedNs(start), "polyast_to_opencl", ""}}, //
      ""                                                                          //
  );
}
