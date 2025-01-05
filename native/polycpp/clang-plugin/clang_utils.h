#pragma once

#include <clang/AST/ASTContext.h>
#include <string>

namespace polyregion::polystl {
template <typename Node> std::string pretty_string(Node *node, const clang::ASTContext &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  node->printPretty(os, nullptr, c.getPrintingPolicy());
  return s;
}

template <typename Node> std::string to_string(Node *node) {
  std::string s;
  llvm::raw_string_ostream os(s);
  node->print(os);
  return s;
}

std::string dump_to_string(const clang::Type &tpe, const clang::ASTContext &c);

std::string dump_to_string(const clang::Decl *decl);

std::string dump_to_string(const clang::Expr *decl, const clang::ASTContext &c);

template <typename Node> std::string to_string(Node node) { return to_string(&node); }

std::string print_type(clang::QualType type, clang::ASTContext &c);

std::string replaceAllInplace(std::string subject, const std::string &search, const std::string &replace);

std::string underlyingToken(clang::Expr *stmt, clang::ASTContext &c);

struct Location {
  std::string filename;
  size_t line, col;
};

Location getLocation(const clang::SourceLocation &e, clang::ASTContext &c);
Location getLocation(const clang::Expr &e, clang::ASTContext &c);

clang::DeclRefExpr *mkDeclRef(const clang::ASTContext &C, clang::VarDecl *lhs);
clang::QualType mkConstArrTy(const clang::ASTContext &C, clang::QualType componentTpe, size_t size);
clang::StringLiteral *mkStrLit(const clang::ASTContext &C, const std::string &str);
clang::IntegerLiteral *mkIntLit(const clang::ASTContext &C, clang::QualType tpe, uint64_t value);
clang::CXXNullPtrLiteralExpr *mkNullPtrLit(const clang::ASTContext &C, clang::QualType componentTpe);
clang::CXXBoolLiteralExpr *mkBoolLit(const clang::ASTContext &C, bool value);
clang::ImplicitCastExpr *mkArrayToPtrDecay(const clang::ASTContext &C, clang::QualType to, clang::Expr *expr);
clang::InitListExpr *mkInitList(const clang::ASTContext &C, clang::QualType ty, const std::vector<clang::Expr *> &initExprs);
clang::MemberExpr *mkMemberExpr(const clang::ASTContext &C, clang::Expr *lhs, clang::ValueDecl *field, const bool arrow = false);
clang::QualType constCharStarTy(const clang::ASTContext &C);
clang::VarDecl *mkStaticVarDecl(clang::ASTContext &C, clang::DeclContext *calleeDecl, const std::string &name, clang::QualType ty,
                                const std::vector<clang::Expr *> &initExprs);

} // namespace polyregion::polystl