#pragma once

#include "PolyAst.pb.h"

#define POLY_OPT(PARENT, MEMBER) ((PARENT).has_##MEMBER() ? std::make_optional((PARENT).MEMBER()) : std::nullopt)

#define POLY_OPTM(PREV, PARENT, MEMBER) ( (PREV.has_value()) ? (POLY_OPT((PREV)->PARENT(), MEMBER)) : std::nullopt )

namespace polyregion::ast {
template <typename T> class AstVisitor {
public:
  virtual T repr(const Sym &sym) = 0;
  virtual T repr(const Types_Type &type) = 0;
  virtual T repr(const Named &path) = 0;
  virtual T repr(const Refs_Ref &ref) = 0;
  virtual T repr(const Tree_Expr &expr) = 0;
  virtual T repr(const Tree_Stmt &stmt) = 0;
  virtual T repr(const Tree_Function &fn) = 0;
  virtual T repr(const Program &program) = 0;
};

class DebugPrinter : public ast::AstVisitor<std::string> {
public:
  std::string repr(const Sym &sym) override;
  std::string repr(const Types_Type &type) override;
  std::string repr(const Named &path) override;
  std::string repr(const Refs_Ref &ref) override;
  std::string repr(const Tree_Expr &expr) override;
  std::string repr(const Tree_Stmt &stmt) override;
  std::string repr(const Tree_Function &fn) override;
  std::string repr(const Program &program) override;
};


} // namespace polyregion::ast