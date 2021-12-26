#include <string>

#include "ast.h"
#include "utils.hpp"

using namespace polyregion::ast;
using namespace polyregion;
using std::string;

[[nodiscard]] string ast::repr(const Sym &sym) { return mk_string<string>(sym.fqn(), std::identity(), "."); }

[[nodiscard]] string ast::repr(const Types_Type &type) {
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

[[nodiscard]] string ast::repr(const Named &path) { return "(" + path.symbol() + ":" + repr(path.tpe()) + ")"; }

[[nodiscard]] string ast::repr(const Refs_Select &select) {
  return select.tail().empty() //
             ? repr(select.head())
             : repr(select.head()) + "." +
                   mk_string<Named>(
                       select.tail(), [&](auto x) { return repr(x); }, ".");
}

[[nodiscard]] string ast::repr(const Refs_Ref &ref) {
  if (auto c = POLY_OPT(ref, select); c) return repr(*c);
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

[[nodiscard]] string ast::repr(const Tree_Expr &expr) {
  if (auto alias = POLY_OPT(expr, alias); alias) {
    return "(~>" + repr(alias->ref()) + ")";
  }
  if (auto invoke = POLY_OPT(expr, invoke); invoke) {
    return repr(invoke->lhs()) + "`" + invoke->name() + "`" +
           mk_string<Refs_Ref>(
               invoke->args(), [&](auto x) { return repr(x); }, ",") +
           ":" + repr(invoke->tpe());
  }
  if (auto index = POLY_OPT(expr, index); index) {
    return repr(index->lhs()) + "[" + repr(index->idx()) + "]";
  }
  return "(unknown repr)";
}

[[nodiscard]] string ast::repr(const Tree_Stmt &stmt) {
  if (auto comment = POLY_OPT(stmt, comment); comment) {
    return "// " + comment->value();
  }
  if (auto var = POLY_OPT(stmt, var); var) {
    return "var " + repr(var->name()) + " = " + repr(var->rhs());
  }
  if (auto effect = POLY_OPT(stmt, effect); effect) {
    return repr(effect->lhs().head()) + "`" + effect->name() + "`" +
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
  if (auto update = POLY_OPT(stmt, update); update) {
    return repr(update->lhs()) + "[" + repr(update->idx()) + "] = " + repr(update->value());
  }

  return "(unknown stmt)";
}

[[nodiscard]] string ast::repr(const Tree_Function &fn) {
  return "def " + fn.name() + "(" +
         mk_string<Named>(
             fn.args(), [&](auto x) { return repr(x); }, ",") +
         ") : " + repr(fn.returntpe()) + " = {\n" +
         mk_string<Tree_Stmt>(
             fn.statements(), [&](auto x) { return repr(x); }, "\n") +
         "\n}";
}

[[nodiscard]] string repr(const Program &program) {
  return mk_string<Tree_Function>(
      program.functions(), [&](auto x) { return repr(x); }, "\n");
}

Named ast::selectLast(const Refs_Select &select) {
  return select.tail_size() != 0 ? *select.tail().rbegin() : select.head();
}

std::optional<ast::NumKind> ast::numKind(const Types_Type &tpe) {
  if (tpe.has_booltpe()     //
      || tpe.has_bytetpe()  //
      || tpe.has_chartpe()  //
      || tpe.has_shorttpe() //
      || tpe.has_inttpe()   //
      || tpe.has_longtpe()) //
    return {ast::NumKind::Integral};
  else if (tpe.has_doubletpe() || tpe.has_floattpe())
    return {ast::NumKind::Fractional};
  else
    return {};
}

std::optional<ast::NumKind> ast::numKind(const Refs_Ref &ref) {
  if (ref.has_boolconst()     //
      || ref.has_byteconst()  //
      || ref.has_charconst()  //
      || ref.has_shortconst() //
      || ref.has_intconst()   //
      || ref.has_longconst()) //
    return {ast::NumKind::Integral};
  else if (ref.has_doubleconst() || ref.has_floatconst())
    return {ast::NumKind::Fractional};
  else if (ref.has_select())
    return numKind(selectLast(ref.select()).tpe());
  else
    return {};
}

std::string ast::name(NumKind k) {
  switch (k) {
  case NumKind::Integral:
    return "Integral";
  case NumKind::Fractional:
    return "Fractional";
  default:
    static_assert("unimplemented KindCase");
  }
}

std::string ast::qualified(const Refs_Select &select) {
  return select.tail_size() == 0 //
             ? select.head().symbol()
             : select.head().symbol() + "." +
                   polyregion::mk_string<Named>(
                       select.tail(), [](auto &n) { return n.symbol(); }, ".");
}
