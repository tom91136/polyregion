#include "clang_utils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
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

std::string underlyingToken(clang::Expr *stmt, clang::ASTContext &c) {
  auto range = clang::CharSourceRange::getTokenRange(stmt->getBeginLoc(), stmt->getEndLoc());
  return clang::Lexer::getSourceText(range, c.getSourceManager(), c.getLangOpts()).str();
}

} // namespace polyregion::polystl