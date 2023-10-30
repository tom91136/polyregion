#include <optional>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/Casting.h"

#include "ast_visitors.h"

using namespace polyregion::polystl;

OverloadTargetVisitor::OverloadTargetVisitor(const clang::CXXRecordDecl *owner) : owner(owner) {}
std::optional<clang::CXXMethodDecl *> OverloadTargetVisitor ::run(clang::Stmt *stmt) {
  TraverseStmt(stmt);
  return target;
}
bool OverloadTargetVisitor::VisitCallExpr(clang::CallExpr *S) {
  if (auto decl = S->getCalleeDecl(); decl) {
    target = OverloadTargetVisitor(owner).run(decl->getBody());
    return !target.has_value();
  } else {
    return true; // keep going
  }
}

bool OverloadTargetVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S) {
  if (S->getOperator() != clang::OverloadedOperatorKind::OO_Call) return true;
  if (auto cxxMethodDecl = llvm::dyn_cast_if_present<clang::CXXMethodDecl>(S->getCalleeDecl())) {
    if (cxxMethodDecl->getParent() == owner) target.emplace(cxxMethodDecl);
  }
  return !target.has_value();
}

SpecialisationPathVisitor::SpecialisationPathVisitor(clang::ASTContext &context) : context(context) {
  SpecialisationPathVisitor::TraverseDecl(context.getTranslationUnitDecl());
}
clang::FunctionDecl *currentFnDecl{};

bool SpecialisationPathVisitor::shouldVisitTemplateInstantiations() const { return true; }

bool SpecialisationPathVisitor::VisitDecl(clang::Decl *decl) {
  if (decl && decl->isFunctionOrFunctionTemplate()) currentFnDecl = decl->getAsFunction();
  return true;
}
bool SpecialisationPathVisitor::VisitCallExpr(clang::CallExpr *expr) {
  if (expr) {
    if (auto decl = dyn_cast_or_null<clang::FunctionDecl>(expr->getCalleeDecl()); decl && decl->isFunctionTemplateSpecialization()) {
      map.emplace(decl, std::pair{currentFnDecl, expr});
    }
  }
  return true;
}
std::vector<std::pair<clang::FunctionDecl *, clang::CallExpr *>> SpecialisationPathVisitor::resolve(clang::FunctionDecl *decl) const {
  std::vector<std::pair<clang::FunctionDecl *, clang::CallExpr *>> xs;
  auto start = decl;
  while (true) {
    if (auto it = map.find(start); it != map.end()) {
      xs.emplace_back(it->second);
      start = it->second.first;
    } else
      break;
  }
  return xs;
}
