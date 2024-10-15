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


Location getLocation(const clang::Expr &e, clang::ASTContext &c) {
  return getLocation(e.getExprLoc(), c);
}

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
std::string dump_to_string(const clang::Decl *decl ) {
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

} // namespace polyregion::polystl