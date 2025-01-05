#include "clang_utils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace polyregion::polystl {

std::string print_type(clang::QualType type, clang::ASTContext &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os, c.getPrintingPolicy());
  return s;
}

std::string replaceAllInplace(std::string subject, const std::string &search, const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}

Location getLocation(const clang::SourceLocation &l, clang::ASTContext &c) {
  return {.filename = llvm::sys::path::filename(c.getSourceManager().getFilename(l)).str(),
          .line = c.getSourceManager().getSpellingLineNumber(l),
          .col = c.getSourceManager().getSpellingColumnNumber(l)};
}

Location getLocation(const clang::Expr &e, clang::ASTContext &c) { return getLocation(e.getExprLoc(), c); }

std::string underlyingToken(clang::Expr *stmt, clang::ASTContext &c) {
  auto range = clang::CharSourceRange::getTokenRange(stmt->getBeginLoc(), stmt->getEndLoc());
  return clang::Lexer::getSourceText(range, c.getSourceManager(), c.getLangOpts()).str();
}
std::string dump_to_string(const clang::Type &tpe, const clang::ASTContext &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  tpe.dump(os, c);
  return s;
}
std::string dump_to_string(const clang::Decl *decl) {
  std::string s;
  llvm::raw_string_ostream os(s);
  decl->dump(os);
  return s;
}

std::string dump_to_string(const clang::Expr *decl, const clang::ASTContext &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  decl->dump(os, c);
  return s;
}

clang::DeclRefExpr *mkDeclRef(const clang::ASTContext &C, clang::VarDecl *lhs) {
  return clang::DeclRefExpr::Create(C, {}, {}, lhs, false, clang::SourceLocation{}, lhs->getType(), clang::ExprValueKind::VK_LValue);
}

clang::QualType mkConstArrTy(const clang::ASTContext &C, const clang::QualType componentTpe, size_t size) {
  return C.getConstantArrayType(componentTpe, llvm::APInt(C.getTypeSize(C.IntTy), size), nullptr, clang::ArraySizeModifier::Normal, 0);
}

clang::StringLiteral *mkStrLit(const clang::ASTContext &C, const std::string &str) {
  return clang::StringLiteral::Create(C, str, clang::StringLiteralKind::Ordinary, false,
                                      C.getConstantArrayType(C.getConstType(C.CharTy),
                                                             llvm::APInt(C.getTypeSize(C.IntTy), str.length() + 1), nullptr,
                                                             clang::ArraySizeModifier::Normal, 0),
                                      {});
}

clang::IntegerLiteral *mkIntLit(const clang::ASTContext &C, clang::QualType tpe, uint64_t value) {
  return clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(tpe), value), tpe, {});
}

clang::CXXNullPtrLiteralExpr *mkNullPtrLit(const clang::ASTContext &C, clang::QualType componentTpe) {
  return new (C) clang::CXXNullPtrLiteralExpr(C.getPointerType(componentTpe), {});
}

clang::CXXBoolLiteralExpr *mkBoolLit(const clang::ASTContext &C, bool value) {
  return clang::CXXBoolLiteralExpr::Create(C, value, C.BoolTy, {});
}

clang::ImplicitCastExpr *mkArrayToPtrDecay(const clang::ASTContext &C, clang::QualType to, clang::Expr *expr) {
  return clang::ImplicitCastExpr::Create(C, to, clang::CK_ArrayToPointerDecay, expr, nullptr, clang::VK_PRValue, {});
}

clang::InitListExpr *mkInitList(const clang::ASTContext &C, clang::QualType ty, const std::vector<clang::Expr *> &initExprs) {
  auto init = new (C) clang::InitListExpr(C, {}, initExprs, {});
  init->setType(ty);
  return init;
}

clang::MemberExpr *mkMemberExpr(const clang::ASTContext &C, clang::Expr *lhs, clang::ValueDecl *field, const bool arrow) {
  return clang::MemberExpr::CreateImplicit( //
      C,
      /*Base*/ lhs,
      /*IsArrow*/ arrow,
      /*MemberDecl*/ field,
      /*T*/ field->getType(),
      /*VK*/ clang::ExprValueKind::VK_LValue,
      /*OK*/ clang::ExprObjectKind::OK_Ordinary);
}

clang::QualType constCharStarTy(const clang::ASTContext &C) { return C.getPointerType(C.CharTy.withConst()); }

clang::VarDecl *mkStaticVarDecl(clang::ASTContext &C, clang::DeclContext *calleeDecl, const std::string &name, clang::QualType ty,
                                const std::vector<clang::Expr *> &initExprs) {
  const auto decl = clang::VarDecl::Create(C, calleeDecl, {}, {}, &C.Idents.get(name), ty, nullptr, clang::SC_Static);
  decl->setInit(mkInitList(C, ty, initExprs));
  decl->setInitStyle(clang::VarDecl::InitializationStyle::ListInit);
  return decl;
}

} // namespace polyregion::polystl