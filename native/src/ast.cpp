#include <string>

#include "ast.h"
#include "utils.hpp"

using namespace polyregion::ast;
using std::string;

[[nodiscard]] string DebugPrinter::repr(const Sym &sym) { return mk_string<string>(sym.fqn(), std::identity(), "."); }

[[nodiscard]] string DebugPrinter::repr(const Types_Type &type) {
  if (auto reftpe = POLY_OPT(type, reftpe); reftpe) {
    return reftpe->args().empty() //
               ? repr(reftpe->name())
               : repr(reftpe->name()) + //
                     "[" +
                     mk_string<Types_Type>(
                         reftpe->args(), [&](auto x) { return repr(x); }, ",") + //
                     "]";
  }
  if (auto arrtpe = POLY_OPT(type, arraytpe); arrtpe) {
    return "Array[" + repr(arrtpe->tpe()) + "]";
  }
  if (type.has_booltpe()) return "Bool";
  if (type.has_bytetpe()) return "Byte";
  if (type.has_chartpe()) return "Char";
  if (type.has_shorttpe()) return "Short";
  if (type.has_inttpe()) return "Int";
  if (type.has_longtpe()) return "Long";
  if (type.has_doubletpe()) return "Double";
  if (type.has_floattpe()) return "Float";
  if (type.has_stringtpe()) return "String";
  return "Unit";
}

[[nodiscard]] string DebugPrinter::repr(const Named &path) { return "(" + path.name() + ":" + repr(path.tpe()) + ")"; }

[[nodiscard]] string DebugPrinter::repr(const Refs_Ref &ref) {
  if (auto select = POLY_OPT(ref, select); select) {
    return select->tail().empty() ? repr(select->head())
                                  : repr(select->head()) + "." +
                                        mk_string<Named>(
                                            select->tail(), [&](auto x) { return repr(x); }, ".");
  }
  if (auto c = POLY_OPT(ref, boolconst); c) return "Bool(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, byteconst); c) return "Byte(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, charconst); c) return "Char(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, shortconst); c) return "Short(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, intconst); c) return "Int(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, longconst); c) return "Long(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, doubleconst); c) return "Double(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, floatconst); c) return "Float(" + std::to_string(c->value()) + ")";
  if (auto c = POLY_OPT(ref, stringconst); c) return "String(" + c->value() + ")";
  return "Unit()";
}

[[nodiscard]] string DebugPrinter::repr(const Tree_Expr &expr) {
  if (auto alias = POLY_OPT(expr, alias); alias) {
    return "(~>" + repr(alias->ref()) + ")";
  }
  if (auto invoke = POLY_OPT(expr, invoke); invoke) {
    return repr(invoke->lhs()) + "<" + invoke->name() + ">" +
           mk_string<Refs_Ref>(
               invoke->args(), [&](auto x) { return repr(x); }, ",") +
           ":" + repr(invoke->tpe());
  }
  return "(unknown repr)";
}

[[nodiscard]] string DebugPrinter::repr(const Tree_Stmt &stmt) {
  if (auto comment = POLY_OPT(stmt, comment); comment) {
    return "// " + comment->value();
  }
  if (auto var = POLY_OPT(stmt, var); var) {
    return "var " + repr(var->name()) + " = " + repr(var->rhs());
  }
  if (auto effect = POLY_OPT(stmt, effect); effect) {
    return repr(effect->lhs()) + "<" + effect->name() + ">" +
           mk_string<Refs_Ref>(
               effect->args(), [&](auto x) { return repr(x); }, ",") +
           " : Unit";
  }
  if (auto mut = POLY_OPT(stmt, mut); mut) {
    return repr(mut->name()) + " := " + repr(mut->expr());
  }
  if (auto while_ = POLY_OPT(stmt, while_); while_) {
    return "while(" + repr(while_->cond()) + "){\n" +
           mk_string<Tree_Stmt>(
               while_->body(), [&](auto x) { return repr(x); }, "\n") +
           "}";
  }
  return "(unknown stmt)";
}

[[nodiscard]] string DebugPrinter::repr(const Tree_Function &fn) {
  return "def " + fn.name() + "(" +
         mk_string<Named>(
             fn.args(), [&](auto x) { return repr(x); }, ",") +
         ") : " + repr(fn.returntpe()) + " = {\n" +
         mk_string<Tree_Stmt>(
             fn.statements(), [&](auto x) { return repr(x); }, "\n") +
         "\n}";
}

[[nodiscard]] string DebugPrinter::repr(const Program &program) {
  return mk_string<Tree_Function>(
      program.functions(), [&](auto x) { return repr(x); }, "\n");
}
