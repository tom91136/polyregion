#pragma once

#include <optional>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/Casting.h"

#include "ast_visitors.h"

namespace polyregion::polystl {

// Recursively (following CallExpr too) finds the first call to a () operator and records the concrete method called
class OverloadTargetVisitor : public clang::RecursiveASTVisitor<OverloadTargetVisitor> {
  std::optional<clang::CXXMethodDecl *> target;
  const clang::CXXRecordDecl *owner;

public:
  explicit OverloadTargetVisitor(const clang::CXXRecordDecl *owner);
  std::optional<clang::CXXMethodDecl *> run(clang::Stmt *stmt);
  bool VisitCallExpr(clang::CallExpr *S);
  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S);
};

class SpecialisationPathVisitor : public clang::RecursiveASTVisitor<SpecialisationPathVisitor> {
  std::unordered_map<clang::FunctionDecl *, std::pair<clang::FunctionDecl *, clang::CallExpr *>> map;
  clang::ASTContext &context;

public:
  explicit SpecialisationPathVisitor(clang::ASTContext &context);
  bool shouldVisitTemplateInstantiations() const;
  bool VisitDecl(clang::Decl *decl);
  bool VisitCallExpr(clang::CallExpr *expr);
  std::vector<std::pair<clang::FunctionDecl *, clang::CallExpr *>> resolve(clang::FunctionDecl *decl) const;
};

} // namespace polyregion::polystl
