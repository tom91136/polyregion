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

#include "aspartame/optional.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"

#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"
#include "options.h"
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

// static std::string createHumanReadableFunctionIdentifier(clang::ASTContext &c, clang::FunctionDecl *decl) {
//   SpecialisationPathVisitor spv(c);
//   auto xs = spv.resolve(decl);
//   std::string identifier;
//   for (std::make_signed_t<size_t> i = xs.size() - 1; i >= 0; --i) {
//     auto loc = getLocation(*xs[i].second, c);
//     identifier += xs[i].first->getName();
//     identifier += "<";
//     identifier += loc.filename;
//     identifier += ":";
//     identifier += std::to_string(loc.line);
//     identifier += ">";
//     if (i != 0) identifier += "->";
//   }
//   return identifier;
// }

void insertKernelImage(clang::DiagnosticsEngine &diag, clang::ASTContext &C, Callsite &c, const KernelBundle &bundle) {

  auto createDeclRef = [&](clang::VarDecl *lhs) {
    return clang::DeclRefExpr::Create(C, {}, {}, lhs, false, clang::SourceLocation{}, lhs->getType(), clang::ExprValueKind::VK_LValue);
  };

  auto createAssignStmt = [&](clang::VarDecl *lhs, clang::Expr *rhs) {
    return clang::BinaryOperator::Create(C, createDeclRef(lhs), rhs, clang::BinaryOperator::Opcode::BO_Assign, lhs->getType(),
                                         clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary, {}, {});
  };

  auto mkConstArrayTpe = [&](clang::QualType componentTpe, size_t size) {
    return C.getConstantArrayType(componentTpe, llvm::APInt(C.getTypeSize(C.IntTy), size), nullptr, clang::ArrayType::Normal, 0);
  };

  auto mkStringLiteral = [&](const std::string &str) {
    return clang::StringLiteral::Create(C, str, clang::StringLiteral::StringKind::Ordinary, false,
                                        C.getConstantArrayType(C.getConstType(C.CharTy),
                                                               llvm::APInt(C.getTypeSize(C.IntTy), str.length() + 1), nullptr,
                                                               clang::ArrayType::Normal, 0),
                                        {});
  };

  auto mkIntegerLiteral = [&](clang::QualType tpe, uint64_t value) {
    return clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(tpe), value), tpe, {});
  };

  auto mkArrayToPtrDecay = [&](clang::QualType to, clang::Expr *expr) {
    return clang::ImplicitCastExpr::Create(C, to, clang::CK_ArrayToPointerDecay, expr, nullptr, clang::VK_PRValue, {});
  };

  auto mkStaticVarDecl = [&](const std::string &name, clang::QualType ty, const std::vector<clang::Expr *> &initExprs) {
    auto decl = clang::VarDecl::Create(C, c.calleeDecl, {}, {}, &C.Idents.get(name), ty, nullptr, clang::SC_Static);

    auto init = new (C) clang::InitListExpr(C, {}, initExprs, {});
    init->setType(decl->getType());
    decl->setInit(init);
    return decl;
  };

  auto existingStmts = std::vector<clang::Stmt *>(c.calleeDecl->getBody()->child_begin(), c.calleeDecl->getBody()->child_end());

  auto varDeclWithName = [&](clang::Stmt *stmt, const std::string &name) -> std::optional<clang::VarDecl *> {
    if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt); declStmt && declStmt->isSingleDecl()) {
      if (auto varDecl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl()); varDecl && varDecl->getName() == name) {
        return varDecl;
      }
    }
    return {};
  };

  auto resolveDeclIssueDiag = [&](const std::string &name) {
    auto decl = existingStmts | collect([&](clang::Stmt *s) { return varDeclWithName(s, name); }) | head_maybe();
    if (!decl) {
      diag.Report({}, diag.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                           "[PolySTL] Callsite method is malformed: missing VarDecl `%0`. This is a bug."))
          .AddString(name);
    }
    return decl;
  };

  auto moduleNameDecl = resolveDeclIssueDiag("__moduleName"); // const char*
  auto metadataDecl = resolveDeclIssueDiag("__metadata");     // const char*
  auto objectSizeDecl = resolveDeclIssueDiag("__objectSize"); // size_t
  auto formatsDecl = resolveDeclIssueDiag("__formats");       // uint8_t*
  auto kindsDecl = resolveDeclIssueDiag("__kinds");           // uint8_t*
  auto featuresDecl = resolveDeclIssueDiag("__features");     // const char***
  auto imageSizesDecl = resolveDeclIssueDiag("__imageSizes"); // size_t*
  auto imagesDecl = resolveDeclIssueDiag("__images");         // const unsigned char**

  if (!moduleNameDecl || !metadataDecl || !objectSizeDecl || !formatsDecl || !kindsDecl || !featuresDecl || !imageSizesDecl || !imagesDecl)
    return;

  auto insertPoint = existingStmts ^ index_where([](clang::Stmt *s) {
                       auto l = llvm::dyn_cast<clang::LabelStmt>(s);
                       return l && l->getName() == std::string("__insert_point");
                     });
  if (insertPoint == -1) {
    diag.Report({}, diag.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                         "[PolySTL] Callsite method is malformed: missing label __insert_point. This is a bug."));
  }

  auto constCharStarTpe = C.getPointerType(C.CharTy.withConst());
  //  auto constUnsignedCharStarTpe = C.getConstType(C.getPointerType(C.UnsignedCharTy));
  auto formatsComponentTpe = (*formatsDecl)->getType()->getPointeeType();
  auto platformsComponentTpe = (*kindsDecl)->getType()->getPointeeType();
  auto imageSizesComponentTpe = (*imageSizesDecl)->getType()->getPointeeType();

  auto kernelImageDecls =                                               //
      bundle.objects                                                    //
      | zip_with_index()                                                //
      | map([&](auto ko, auto idx) {                                    //
          return mkStaticVarDecl(                                       //
              "__image_data_" + std::to_string(idx),                    //
              mkConstArrayTpe(C.UnsignedCharTy, ko.moduleImage.size()), //
              ko.moduleImage | map([&](const unsigned char c) -> clang::Expr * {
                return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast, mkIntegerLiteral(C.IntTy, c), nullptr,
                                                       clang::VK_PRValue, {});
              }) | to_vector());
        }) //
      | to_vector();

  auto kernelFeatureDecls =                                          //
      bundle.objects                                                 //
      | zip_with_index()                                             //
      | map([&](auto ko, auto idx) {                                 //
          return mkStaticVarDecl(                                    //
              "__feature_data_" + std::to_string(idx),               //
              mkConstArrayTpe(constCharStarTpe, ko.features.size()), //
              ko.features | map([&](auto &feature) -> clang::Expr * {
                return mkArrayToPtrDecay(C.getConstType(C.getPointerType(C.CharTy)), mkStringLiteral(feature));
              }) | to_vector());
        }) //
      | to_vector();

  std::vector<clang::Stmt *> newStmts;

  kernelImageDecls | concat(kernelFeatureDecls) |
      for_each([&](auto dcl) { newStmts.push_back(new (C) clang::DeclStmt(clang::DeclGroupRef(dcl), {}, {})); });

  std::vector<std::tuple<clang::VarDecl *, clang::CastKind, clang::VarDecl *>> dataDecls{
      {*moduleNameDecl, clang::CK_LValueToRValue,
       mkStaticVarDecl("__moduleName_data", constCharStarTpe, {mkArrayToPtrDecay(constCharStarTpe, mkStringLiteral(bundle.moduleName))})},
      {*metadataDecl, clang::CK_LValueToRValue,
       mkStaticVarDecl("__metadata_data", constCharStarTpe, {mkArrayToPtrDecay(constCharStarTpe, mkStringLiteral(bundle.metadata))})},
      {*objectSizeDecl, clang::CK_LValueToRValue,
       mkStaticVarDecl("__objectSize_data", (*objectSizeDecl)->getType(),
                       {mkIntegerLiteral((*objectSizeDecl)->getType(), bundle.objects.size())})},
      {*formatsDecl, clang::CK_ArrayToPointerDecay,
       mkStaticVarDecl("__formats_data", mkConstArrayTpe(formatsComponentTpe, bundle.objects.size()),
                       bundle.objects ^ map([&](auto &ko) -> clang::Expr * {
                         return mkIntegerLiteral(formatsComponentTpe, static_cast<std::underlying_type_t<decltype(ko.format)>>(ko.format));
                       }))},
      {*kindsDecl, clang::CK_ArrayToPointerDecay,
       mkStaticVarDecl("__kinds_data", mkConstArrayTpe(platformsComponentTpe, bundle.objects.size()),
                       bundle.objects ^ map([&](auto &ko) -> clang::Expr * {
                         return mkIntegerLiteral(platformsComponentTpe, static_cast<std::underlying_type_t<decltype(ko.kind)>>(ko.kind));
                       }))},
      {*featuresDecl, clang::CK_ArrayToPointerDecay,
       mkStaticVarDecl("__features_data", mkConstArrayTpe(C.getPointerType(constCharStarTpe), kernelFeatureDecls.size()),
                       kernelFeatureDecls ^ map([&](auto x) -> clang::Expr * {
                         return mkArrayToPtrDecay(C.getPointerType(C.CharTy.withConst()), createDeclRef(x));
                       }))},
      {*imageSizesDecl, clang::CK_ArrayToPointerDecay,
       mkStaticVarDecl("__imageSizes_data", mkConstArrayTpe(imageSizesComponentTpe, bundle.objects.size()),
                       bundle.objects ^
                           map([&](auto ko) -> clang::Expr * { return mkIntegerLiteral(imageSizesComponentTpe, ko.moduleImage.size()); }))},
      {*imagesDecl, clang::CK_ArrayToPointerDecay,
       mkStaticVarDecl("__images_data", mkConstArrayTpe(C.getPointerType(C.UnsignedCharTy.withConst()), kernelImageDecls.size()),
                       kernelImageDecls ^ map([&](auto x) -> clang::Expr * {
                         return mkArrayToPtrDecay(C.getPointerType(C.UnsignedCharTy.withConst()), createDeclRef(x));
                       }))}};

  dataDecls | for_each([&](auto, auto, auto decl) { newStmts.push_back(new (C) clang::DeclStmt(clang::DeclGroupRef(decl), {}, {})); });
  dataDecls | for_each([&](clang::VarDecl *lhs, clang::CastKind ck, clang::VarDecl *decl) {
    newStmts.push_back(
        createAssignStmt(lhs, clang::ImplicitCastExpr::Create(C, lhs->getType(), ck, createDeclRef(decl), nullptr, clang::VK_PRValue, {})));
  });

  c.calleeDecl->setBody(clang::CompoundStmt::Create(      //
      C,                                                  //
      existingStmts                                       //
          | take(insertPoint)                             //
          | concat(newStmts)                              //
          | concat(existingStmts | drop(insertPoint + 1)) //
          | to_vector(),
      {}, {}, {}));
  //  c.calleeDecl->dumpColor(); // void __polyregion_offload__(F __polyregion__f)
  c.calleeDecl->print(llvm::outs());
}

OffloadRewriteConsumer::OffloadRewriteConsumer(clang::DiagnosticsEngine &diag, const DriverContext &ctx)
    : clang::ASTConsumer(), diag(diag), ctx(ctx) {}

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
  //  std::cout << "[TU] >>>" << std::endl;

  std::vector<std::variant<Failure, Callsite>> results = outlinePolyregionOffload(C);

  for (auto r : results) {
    std::visit(overloaded{
                   [&](Failure &f) { //
                     diag.Report(f.callExpr->getBeginLoc(),
                                 diag.getCustomDiagID(clang::DiagnosticsEngine::Warning, "[PolySTL] Outline failed: %0"))
                         .AddString(f.reason);

                   },
                   [&](Callsite &c) { //
                     SpecialisationPathVisitor spv(C);
                     auto specialisationPath = spv.resolve(c.calleeDecl) ^ reverse();
                     auto moduleId = specialisationPath ^ mk_string("->", [&](auto fnDecl, auto callExpr) {
                                       auto l = getLocation(*callExpr, C);
                                       std::string moduleId;
                                       moduleId += "<";
                                       moduleId += l.filename;
                                       moduleId += ":";
                                       moduleId += std::to_string(l.line);
                                       moduleId += ">";
                                       return moduleId;
                                     });

                     auto bundle = generate(ctx, C, diag, moduleId, *c.functorDecl,
                                            specialisationPath ^ head_maybe() ^
                                                fold([](auto, auto callExpr) { return callExpr->getExprLoc(); },
                                                     [&]() { return c.callLambdaArgExpr->getExprLoc(); }),
                                            c.kind);

                     if (!ctx.opts.quiet) {
                       diag.Report(c.callLambdaArgExpr->getExprLoc(),
                                   diag.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[PolySTL] Outlined function: %0 for %1 (%2)\n"))
                           << moduleId << to_string(c.kind)
                           << (bundle.objects //
                               | map([](auto &o) {
                                   return std::string(to_string(o.format)) + "=" +
                                          std::to_string(static_cast<float>(o.moduleImage.size()) / 1000) + "KB";
                                 }) //
                               | mk_string(", "));
                     }

                     insertKernelImage(diag, C, c, bundle);

                   },
               },
               r);
  }
}
