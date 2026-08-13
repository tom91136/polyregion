#include "clang_utils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

#include "aspartame/all.hpp"

#include "ast.h"

using namespace aspartame;

namespace polyregion::polystl {

Location getLocation(const clang::SourceLocation &l, clang::ASTContext &c) {
  return {.filename = llvm::sys::path::filename(c.getSourceManager().getFilename(l)).str(),
          .line = c.getSourceManager().getSpellingLineNumber(l),
          .col = c.getSourceManager().getSpellingColumnNumber(l)};
}

Location getLocation(const clang::Expr &e, clang::ASTContext &c) { return getLocation(e.getExprLoc(), c); }

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

bool hasAnnotation(const clang::Decl *decl, const llvm::StringRef annotation) {
  return decl->attrs() | exists([&](const auto &a) {
           const auto *ann = llvm::dyn_cast<clang::AnnotateAttr>(a);
           return ann && ann->getAnnotation() == annotation;
         });
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
                                      {{}});
}

clang::IntegerLiteral *mkIntLit(const clang::ASTContext &C, clang::QualType tpe, uint64_t value) {
  return clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(tpe), value), tpe, {});
}

clang::CXXNullPtrLiteralExpr *mkNullPtrLit(const clang::ASTContext &C, clang::QualType componentTpe) {
  return new (C) clang::CXXNullPtrLiteralExpr(C.getPointerType(componentTpe), {});
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

clang::FunctionDecl *mkExternCFn(clang::ASTContext &C, const std::string &name, clang::QualType retTy,
                                 const std::vector<clang::QualType> &paramTys) {
  auto *tu = C.getTranslationUnitDecl();
  auto *linkage = clang::LinkageSpecDecl::Create(C, tu, {}, {}, clang::LinkageSpecLanguageIDs::C, false);
  const auto fnTy = C.getFunctionType(retTy, paramTys, clang::FunctionProtoType::ExtProtoInfo());
  auto *fn = clang::FunctionDecl::Create(C, linkage, {}, {}, clang::DeclarationName(&C.Idents.get(name)), fnTy,
                                         C.getTrivialTypeSourceInfo(fnTy), clang::SC_Extern);
  std::vector<clang::ParmVarDecl *> params;
  for (const auto &paramTy : paramTys)
    params.emplace_back(
        clang::ParmVarDecl::Create(C, fn, {}, {}, nullptr, paramTy, C.getTrivialTypeSourceInfo(paramTy), clang::SC_None, nullptr));
  fn->setParams(params);
  linkage->addDecl(fn);
  tu->addDecl(linkage);
  return fn;
}

clang::CallExpr *mkCall(clang::ASTContext &C, clang::FunctionDecl *fn, const std::vector<clang::Expr *> &args) {
  auto *ref = clang::DeclRefExpr::Create(C, {}, {}, fn, false, clang::SourceLocation{}, fn->getType(), clang::VK_LValue);
  auto *decay = clang::ImplicitCastExpr::Create(C, C.getPointerType(fn->getType()), clang::CK_FunctionToPointerDecay, ref, nullptr,
                                                clang::VK_PRValue, {});
  return clang::CallExpr::Create(C, decay, args, fn->getReturnType(), clang::VK_PRValue, {}, {});
}

clang::Expr *mkLoad(clang::ASTContext &C, clang::VarDecl *var) {
  auto *ref = mkDeclRef(C, var);
  if (var->getType()->getAsCXXRecordDecl()) return ref;
  return clang::ImplicitCastExpr::Create(C, var->getType(), clang::CK_LValueToRValue, ref, nullptr, clang::VK_PRValue, {});
}

} // namespace polyregion::polystl
