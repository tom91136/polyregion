#include <fmt/format.h>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Support/Casting.h"

#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"

#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"
#include "rewriter.h"

using namespace polyregion::polystl;
using namespace aspartame;

namespace {

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename F> class InlineMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  F f;
  void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override { f(result); }

public:
  explicit InlineMatchCallback(F f) : f(f) {}
};

template <typename M, typename F> void runMatch(M matcher, F callback, clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  InlineMatchCallback cb(callback);
  MatchFinder finder;
  finder.addMatcher(matcher, &cb);
  finder.matchAST(context);
}
} // namespace

struct Callsite {
  clang::CallExpr *callExpr;         // decl of the std::transform call
  clang::Expr *callLambdaArgExpr;    // decl of the lambda arg
  clang::FunctionDecl *calleeDecl;   // decl of the specialised std::transform
  clang::CXXMethodDecl *functorDecl; // decl of the specialised lambda functor, this is the root of the lambda body
  polyregion::runtime::PlatformKind kind;
};
struct Failure {
  const clang::Stmt *callExpr;
  std::string reason;
};

constexpr static auto offloadFunctionName = "__polyregion_offload__";

static std::vector<std::variant<Failure, Callsite>> outlinePolyregionOffload(clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  std::vector<std::variant<Failure, Callsite>> results;
  runMatch(
      callExpr(callee(functionDecl(hasName(offloadFunctionName)))).bind(offloadFunctionName),
      [&](const MatchFinder::MatchResult &result) {
        if (auto offloadCallExpr = result.Nodes.getNodeAs<clang::CallExpr>(offloadFunctionName)) {
          auto lastArgExpr = offloadCallExpr->getArg(offloadCallExpr->getNumArgs() - 1)->IgnoreUnlessSpelledInSource();
          auto fnDecl = offloadCallExpr->getDirectCallee();
          if (auto lambdaArgCxxRecordDecl = lastArgExpr->getType()->getAsCXXRecordDecl()) {
            // TODO we should support explicit structs with () operator and not just lambdas
            if (auto op = lambdaArgCxxRecordDecl->getLambdaCallOperator(); lambdaArgCxxRecordDecl->isLambda() && op) {

              // prototype is <polyregion::runtime::PlatformKind, typename F>; we check the first template arg's type and value
              auto templateArgs = fnDecl->getTemplateSpecializationArgs();
              if (templateArgs->size() != 2) {
                results.emplace_back(
                    Failure{offloadCallExpr, "Template arity mismatch for " + std::string(offloadFunctionName) + ", expecting 2"});
              } else {
                auto templateArg0 = templateArgs->get(0);
                if (templateArg0.getKind() == clang::TemplateArgument::Integral &&
                    templateArg0.getIntegralType()->getAsTagDecl()->getName().str() == "PlatformKind") {
                  auto kind = static_cast<polyregion::runtime::PlatformKind>(templateArg0.getAsIntegral().getExtValue());
                  results.emplace_back(Callsite{const_cast<clang::CallExpr *>(offloadCallExpr), const_cast<clang::Expr *>(lastArgExpr),
                                                const_cast<clang::FunctionDecl *>(fnDecl), op, kind});
                } else {
                  results.emplace_back(Failure{offloadCallExpr, "First template kind is not a PlatformKind"});
                }
              }
            } else {
              results.emplace_back(Failure{offloadCallExpr, "Last arg is not a lambda or does not provide a operator ()"});
            }

          } else {
            results.emplace_back(Failure{offloadCallExpr, "Last arg is not a valid synthesised lambda record type"});
          }
        } else {
          auto root = result.Nodes.getNodeAs<clang::Stmt>(offloadFunctionName);
          results.emplace_back(Failure{root, "Unexpected offload definition:" + pretty_string(root, context)});
        }
      },
      context);
  return results;
}

static std::string createHumanReadableFunctionIdentifier(clang::ASTContext &c, clang::FunctionDecl *decl) {
  SpecialisationPathVisitor spv(c);
  auto xs = spv.resolve(decl);
  std::string identifier;
  for (std::make_signed_t<size_t> i = xs.size() - 1; i >= 0; --i) {
    auto loc = getLocation(*xs[i].second, c);
    identifier += xs[i].first->getName();
    identifier += "<";
    identifier += loc.filename;
    identifier += ":";
    identifier += std::to_string(loc.line);
    identifier += ">";
    if (i != 0) identifier += "->";
  }
  return identifier;
}

void insertKernelImage(clang::ASTContext &C, Callsite &c, const polyregion::runtime::KernelBundle &bundle) {
  auto varDeclWithName = [](clang::Stmt *stmt, const std::string &name) -> clang::VarDecl * {
    if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt); declStmt && declStmt->isSingleDecl()) {
      if (auto varDecl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl()); varDecl && varDecl->getName() == name) {
        return varDecl;
      }
    }
    return {};
  };

  auto cxxOperatorCallFromParamWithName = [](clang::Stmt *stmt, const std::string &name) -> clang::ParmVarDecl * {
    if (auto varDecl = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stmt); varDecl && varDecl->getNumArgs() == 1) {
      if (auto paramVarDecl = llvm::dyn_cast_or_null<clang::ParmVarDecl>(varDecl->getArg(0)->getReferencedDeclOfCallee());
          paramVarDecl && paramVarDecl->getName() == name) {
        return paramVarDecl;
      }
    }
    return {};
  };

  auto createDeclRef = [&](clang::VarDecl *lhs) {
    return clang::DeclRefExpr::Create(C, {}, {}, lhs, false, clang::SourceLocation{}, lhs->getType(), clang::ExprValueKind::VK_LValue);
  };

  auto createAssignStmt = [&](clang::VarDecl *lhs, clang::Expr *rhs) {
    return clang::BinaryOperator::Create(C, createDeclRef(lhs), rhs, clang::BinaryOperator::Opcode::BO_Assign, lhs->getType(),
                                         clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary, {}, {});
  };

  auto createConstArrayTpe = [&](clang::QualType componentTpe, size_t size) {
    return C.getConstantArrayType(componentTpe, llvm::APInt(C.getTypeSize(C.IntTy), size), nullptr, clang::ArrayType::Normal, 0);
  };

  auto createStringLiteral = [&](const std::string &str) {
    return clang::StringLiteral::Create(C, str, clang::StringLiteral::StringKind::Ordinary, false,
                                        C.getConstantArrayType(C.getConstType(C.CharTy),
                                                               llvm::APInt(C.getTypeSize(C.IntTy), str.length() + 1), nullptr,
                                                               clang::ArrayType::Normal, 0),
                                        {});
  };

  auto createInitExpr = [&](clang::QualType tpe, const clang::ArrayRef<clang::Expr *> &&xs) {
    auto expr = new (C) clang::InitListExpr(C, {}, xs, {});
    expr->setType(tpe);
    return expr;
  };

  auto createArrayToPtrDecay = [&](clang::QualType to, clang::Expr *expr) {
    return clang::ImplicitCastExpr::Create(C, to, clang::CK_ArrayToPointerDecay, expr, nullptr, clang::VK_PRValue, {});
  };

  auto existingStmts = c.calleeDecl->getBody()->children();

  auto image = bundle.toMsgPack();

  std::vector<clang::Stmt *> newStmts;
  for (auto stmt : existingStmts) {
    if (auto kernelImageDecl = varDeclWithName(stmt, "__stub_kernelImageBytes__"); kernelImageDecl) {

      auto component = kernelImageDecl->getType()->getAs<clang::PointerType>()->getPointeeType();

      // TODO add section attributes to support kernel image extraction
      auto varDecl = clang::VarDecl::Create(C, c.calleeDecl, {}, {}, &C.Idents.get("__kernel_image__"),
                                            createConstArrayTpe(component, image.size()), nullptr, clang::SC_Static);
      std::vector<clang::Expr *> initExprs(image.size());
      std::transform(image.begin(), image.end(), initExprs.begin(), [&](auto &c) {
        return clang::ImplicitCastExpr::Create(C, component, clang::CK_IntegralCast,
                                               clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(C.IntTy), c), C.IntTy, {}),
                                               nullptr, clang::VK_PRValue, {});
      });
      auto init = new (C) clang::InitListExpr(C, {}, initExprs, {});
      init->setType(varDecl->getType());
      varDecl->setInit(init);

      newStmts.push_back(stmt);

      newStmts.push_back(new (C) clang::DeclStmt(clang::DeclGroupRef(varDecl), {}, {}));
      newStmts.push_back(
          createAssignStmt(kernelImageDecl, clang::ImplicitCastExpr::Create(C, kernelImageDecl->getType(), clang::CK_ArrayToPointerDecay,
                                                                            createDeclRef(varDecl), nullptr, clang::VK_PRValue, {})));

    } else if (auto kernelImageSizeDecl = varDeclWithName(stmt, "__stub_kernelImageSize__"); kernelImageSizeDecl) {
      auto tpe = kernelImageSizeDecl->getType();
      newStmts.push_back(stmt);
      newStmts.push_back(
          createAssignStmt(kernelImageSizeDecl, clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(tpe), image.size()), tpe, {})));
    } else {
      newStmts.push_back(stmt);
    }
  }

  c.calleeDecl->setBody(clang::CompoundStmt::Create(C, newStmts, {}, {}, {}));
  //  c.calleeDecl->dumpColor(); // void __polyregion_offload__(F __polyregion__f)
  c.calleeDecl->print(llvm::outs());
}

OffloadRewriteConsumer::OffloadRewriteConsumer(clang::DiagnosticsEngine &diag) : clang::ASTConsumer(), diag(diag) {}

template <typename Parent, typename Node> const Parent *findParentOfType(clang::ASTContext &context, Node *from) {
  for (auto node = context.getParents(*from).begin()->template get<clang::Decl>(); node;
       node = context.getParents(*node).begin()->template get<clang::Decl>()) {
    //
    //    clang::Stmt *ax = {};
    //    context.getParents(*ax).begin()->get<clang::Decl>();

    if (auto parent = llvm::dyn_cast<Parent>(node); parent) {
      return parent;
    }
  }
  return {};
}

void OffloadRewriteConsumer::HandleTranslationUnit(clang::ASTContext &C) {
  std::cout << "[TU] >>>" << std::endl;

  std::vector<std::variant<Failure, Callsite>> results = outlinePolyregionOffload(C);

  for (auto r : results) {
    std::visit(overloaded{
                   [&](Failure &f) { //
                     diag.Report(f.callExpr->getBeginLoc(),
                                 diag.getCustomDiagID(clang::DiagnosticsEngine::Warning, "[PolySTL] Outline failed: %0"))
                         .AddString(f.reason);

                   },
                   [&](Callsite &c) { //
                     auto moduleId = createHumanReadableFunctionIdentifier(C, c.calleeDecl);


                     // if target ==
//                     Object_LLVM_HOST ,
//                     Object_LLVM_x86_64,
//                     Object_LLVM_AArch64,
//                     Object_LLVM_ARM,
//                     Source_C_C11 ,

                     std::vector<std::pair<compiletime::Target, std::string>> targets = {
                         {compiletime::Target::Object_LLVM_HOST, "native"},
                         {compiletime::Target::Object_LLVM_NVPTX64, "sm_60"},
                     };

                     auto bundle =
                         generate(C, diag, moduleId, *c.functorDecl,
                                  targets ^ filter([&](auto &target, auto &) { return c.kind == runtime::targetPlatformKind(target); }));


                     diag.Report(c.callLambdaArgExpr->getExprLoc(),
                                 diag.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[PolySTL] Outlined function: %0 for %1 (%2)\n"))
                         << moduleId << to_string(c.kind)
                         << (bundle.objects //
                             | map([](auto &o) {
                                 return std::string(to_string(o.format)) + "=" +
                                        std::to_string(static_cast<float>(o.moduleImage.size()) / 1000) + "KB";
                               }) //
                             | mk_string(", "));

                     insertKernelImage(C, c, bundle);

                   },
               },
               r);
  }
}
