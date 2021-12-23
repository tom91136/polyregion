#include "opencl.h"
#include "utils.hpp"

using namespace polyregion;

std::string codegen::OpenCLCodeGen::mkTpe(const Types_Type &tpe) {

  if (tpe.has_booltpe()) return "false";
  if (tpe.has_bytetpe()) return "int8_t";
  if (tpe.has_chartpe()) return "uint16_t";
  if (tpe.has_shorttpe()) return "int16_t";
  if (tpe.has_inttpe()) return "int32_t";
  if (tpe.has_longtpe()) return "int64_t";
  if (tpe.has_doubletpe()) return "double";
  if (tpe.has_floattpe()) return "float";
  if (tpe.has_stringtpe()) return "char *";
  if (auto arrtpe = POLY_OPT(tpe, arraytpe); arrtpe) {
    return mkTpe(arrtpe->tpe()) + "*";
  }
  if (auto reftpe = POLY_OPT(tpe, reftpe); reftpe) {
    return undefined();
  }
  return "void";
}
std::string codegen::OpenCLCodeGen::mkRef(const Refs_Ref &ref) {

  if (auto select = POLY_OPT(ref, select); select) {
    return select->head().name();
  }

  if (auto c = POLY_OPT(ref, boolconst); c) return c->value() ? "true" : "false";
  if (auto c = POLY_OPT(ref, byteconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, charconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, shortconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, intconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, longconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, doubleconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, floatconst); c) return std::to_string(c->value());
  if (auto c = POLY_OPT(ref, stringconst); c) return "\"" + c->value() + "\""; // FIXME escape string
  return undefined("Unimplemented ref:" + ref.DebugString());
}
std::string codegen::OpenCLCodeGen::mkExpr(const Tree_Expr &expr, const std::string &key) {

  if (auto alias = POLY_OPT(expr, alias); alias) {
    return mkRef(alias->ref());
  }
  if (auto invoke = POLY_OPT(expr, invoke); invoke) {

    auto name = invoke->name();
    auto lhs = mkRef(invoke->lhs());

    if (invoke->args_size() == 1) {
      auto rhs = mkRef(invoke->args(0));
      switch (hash(name)) {
      case "+"_:
        return lhs + " + " + rhs;
      case "-"_:
        return lhs + " - " + rhs;
      case "*"_:
        return lhs + " * " + rhs;
      case "/"_:
        return lhs + " / " + rhs;
      case "%"_:
        return lhs + " % " + rhs;
      }
    }
    return "???";
  }
  return "???";
}
std::string codegen::OpenCLCodeGen::mkStmt(const Tree_Stmt &stmt) {

  if (auto comment = POLY_OPT(stmt, comment); comment) {
    auto lines = split(comment->value(), '\n');
    auto commented = map_vec<std::string, std::string>(lines, [](auto &&line) { return "// " + line; });
    return mk_string<std::string>(commented, std::identity(), "\n");
  }
  if (auto var = POLY_OPT(stmt, var); var) {
    auto line =
        mkTpe(var->name().tpe()) + " " + var->name().name() + " = " + mkExpr(var->rhs(), var->name().name()) + ";";
    return line;
  }

  //
  if (auto effect = POLY_OPT(stmt, effect); effect) {
  }
  if (auto mut = POLY_OPT(stmt, mut); mut) {
  }
  if (auto while_ = POLY_OPT(stmt, while_); while_) {
    auto body = mk_string<Tree_Stmt>(
        while_->body(), [&](auto &stmt) { return mkStmt(stmt); }, "\n");
    return "while (" + mkExpr(while_->cond(), "?") + ") {\n" + body + "}";
  }
  return "???";
}
void codegen::OpenCLCodeGen::run(const Tree_Function &fnTree) {

  auto args = mk_string<Named>(
      fnTree.args(), [&](auto x) { return mkTpe(x.tpe()) + " " + x.name(); }, ", ");

  auto prototype = mkTpe(fnTree.returntpe()) + " " + fnTree.name() + "(" + args + ")";

  auto body = mk_string<Tree_Stmt>(
      fnTree.statements(), [&](auto &stmt) { return mkStmt(stmt); }, "\n");

  auto def = prototype + "{\n" + body + "\n}";
  std::cout << def << std::endl;
}
