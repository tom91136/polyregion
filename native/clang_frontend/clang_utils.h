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

template <typename Node> std::string to_string(Node node) { return to_string(&node); }

std::string print_type(clang::QualType type, clang::ASTContext &c);

std::string replaceAllInplace(std::string subject, const std::string &search, const std::string &replace);

std::string underlyingToken(clang::Expr *stmt, clang::ASTContext &c);

struct Location {
  std::string filename;
  size_t line, col;
};

Location getLocation(const clang::SourceLocation &e, clang::ASTContext &c);
Location getLocation(const clang::Expr &e, clang::ASTContext &c) ;

} // namespace polyregion::polystl