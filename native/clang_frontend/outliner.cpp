#include <fmt/format.h>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/Support/Casting.h"

#include "clang_utils.h"
#include "outliner.h"
#include "remapper.h"

using namespace polyregion::polystl;

namespace {

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename F> class InlineMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  F f;

public:
  explicit InlineMatchCallback(F f) : f(f) {}

private:
  void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override { f(result); }
};

// Recursively (following CallExpr too) finds the first call to a () operator and records the concrete method called
class OverloadTargetVisitor : public clang::RecursiveASTVisitor<OverloadTargetVisitor> {
  std::optional<clang::CXXMethodDecl *> target;
  const clang::CXXRecordDecl *owner;

public:
  explicit OverloadTargetVisitor(const clang::CXXRecordDecl *owner) : owner(owner) {}
  std::optional<clang::CXXMethodDecl *> run(clang::Stmt *stmt) {
    TraverseStmt(stmt);
    return target;
  }
  bool VisitCallExpr(clang::CallExpr *S) {
    target = OverloadTargetVisitor(owner).run(S->getCalleeDecl()->getBody());
    return !target.has_value();
  }

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S) {
    if (S->getOperator() != clang::OverloadedOperatorKind::OO_Call) return true;
    if (auto cxxMethodDecl = llvm::dyn_cast_if_present<clang::CXXMethodDecl>(S->getCalleeDecl())) {
      if (cxxMethodDecl->getParent() == owner) target.emplace(cxxMethodDecl);
    }
    return !target.has_value();
  }
};

struct RecordSource {
  struct Decl {
    std::string type;
    std::string name;
  };
  struct Fn {
    std::string returnType;
    std::string name;
    std::vector<Decl> parameters;
    std::vector<std::string> body;
  };
  std::string name;
  std::vector<Decl> members;
  std::vector<Fn> functions;

};

} // namespace

OutlineConsumer::OutlineConsumer(clang::Rewriter &rewriter, std::atomic_bool &error) : rewriter(rewriter), error(error) {}
// bool OutlineConsumer::HandleTopLevelDecl(clang::DeclGroupRef DG) {
//   for (const auto &D : DG) {
//     // For each lambda expression, there's a top-level specialisation
//     if (const auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(D)) {
//       if (const auto &templateDecl = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(methodDecl->getPrimaryTemplate())) {
//         for (const auto &fnDecl : templateDecl->specializations()) {
//           if (const auto &record = llvm::dyn_cast<clang::CXXRecordDecl>(templateDecl->getTemplatedDecl()->getParent())) {
//             if (record->isLambda()) {
//               drain.insert({fnDecl->getID(), std::pair{fnDecl, record}});
//             }
//           }
//         }
//       }
//     }
//   }
//
//   return true;
// }

struct Callsite {
  const clang::CallExpr *callExpr;         // decl of the std::transform call
  const clang::Expr *callLambdaArgExpr;    // decl of the lambda arg
  const clang::FunctionDecl *calleeDecl;   // decl of the specialised std::transform
  const clang::CXXMethodDecl *functorDecl; // decl of the specialised lambda functor, this is the root of the lambda body
};
struct Failure {
  const clang::Stmt *callExpr;
  std::string reason;
};

template <typename M, typename F> void runMatch(M matcher, F callback, clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  InlineMatchCallback cb(callback);
  MatchFinder finder;
  finder.addMatcher(matcher, &cb);
  finder.matchAST(context);
}

static std::vector<std::variant<Failure, Callsite>> outlinePSTLCallsites(clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  static constexpr const char *policyTag = "policyTag";
  static constexpr const char *transformCall = "transformCall";
  static const std::unordered_set<std::string> pstlParallelPolicies = {
      "__pstl::execution::parallel_unsequenced_policy",
      "__pstl::execution::parallel_policy",
  };
  std::vector<std::variant<Failure, Callsite>> results;
  runMatch(
      callExpr(                          //
          callee(functionDecl(anyOf(     //
              hasName("std::transform"), //
              hasName("std::for_each"),  //
              hasName("std::for_each_n") //
              ))),
          hasArgument(0, declRefExpr(to(varDecl(hasType(cxxRecordDecl().bind(policyTag)))))))
          .bind(transformCall),
      [&](const MatchFinder::MatchResult &result) {
        if (auto record = result.Nodes.getNodeAs<clang::CXXRecordDecl>(policyTag)) {
          // Make sure the execution policy tag is a parallel one first
          if (pstlParallelPolicies.count(record->getQualifiedNameAsString()) == 0) return;
          if (auto algCallExpr = result.Nodes.getNodeAs<clang::CallExpr>(transformCall)) {
            auto lastArgExpr = algCallExpr->getArg(algCallExpr->getNumArgs() - 1)->IgnoreUnlessSpelledInSource();
            auto fnDecl = algCallExpr->getDirectCallee();
            if (auto lambdaArgCxxRecordDecl = lastArgExpr->getType()->getAsCXXRecordDecl()) {
              if (auto target = OverloadTargetVisitor(lambdaArgCxxRecordDecl).run(fnDecl->getBody()); target) {
                // We found a valid overload target, validate this by checking whether the decl belongs to the last arg's implicit class.
                if ((*target)->getParent() == lambdaArgCxxRecordDecl)
                  results.emplace_back(Callsite{algCallExpr, lastArgExpr, fnDecl, *target});
                else
                  results.emplace_back(Failure{algCallExpr, "Target record mismatch:\nLast arg=" + to_string((*target)->getParent()) +
                                                                "\nTarget=" + to_string(lambdaArgCxxRecordDecl)});
              } else {
                results.emplace_back(Failure{algCallExpr, "Cannot find any call `()` to last arg:" + pretty_string(lastArgExpr, context)});
              }
            } else {
              results.emplace_back(Failure{algCallExpr, "Last arg does is not a valid synthesised record type"});
            }
          } else {
            auto root = result.Nodes.getNodeAs<clang::Stmt>(transformCall);
            results.emplace_back(Failure{root, "Unexpected algorithm call expression:" + pretty_string(root, context)});
          }
        } else {
          auto root = result.Nodes.getNodeAs<clang::Stmt>(transformCall);
          results.emplace_back(Failure{root, "Unexpected algorithm execution policy:" +
                                                 pretty_string(result.Nodes.getNodeAs<clang::Stmt>(policyTag), context)});
        }
      },
      context);
  return results;
}

static std::vector<std::variant<Failure, Callsite>> outlinePolyregionOffload(clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  static constexpr const char *offloadCall = "offloadCall";
  std::vector<std::variant<Failure, Callsite>> results;
  runMatch(
      callExpr(callee(functionDecl(hasName("__polyregion_offload__")))).bind(offloadCall),
      [&](const MatchFinder::MatchResult &result) {
        if (auto offloadCallExpr = result.Nodes.getNodeAs<clang::CallExpr>(offloadCall)) {
          auto lastArgExpr = offloadCallExpr->getArg(offloadCallExpr->getNumArgs() - 1)->IgnoreUnlessSpelledInSource();
          auto fnDecl = offloadCallExpr->getDirectCallee();
          if (auto lambdaArgCxxRecordDecl = lastArgExpr->getType()->getAsCXXRecordDecl()) {
            if (auto target = OverloadTargetVisitor(lambdaArgCxxRecordDecl).run(fnDecl->getBody()); target) {
              // We found a valid overload target, validate this by checking whether the decl belongs to the last arg's implicit class.
              if ((*target)->getParent() == lambdaArgCxxRecordDecl)
                results.emplace_back(Callsite{offloadCallExpr, lastArgExpr, fnDecl, *target});
              else
                results.emplace_back(Failure{offloadCallExpr, "Target record mismatch:\nLast arg=" + to_string((*target)->getParent()) +
                                                                  "\nTarget=" + to_string(lambdaArgCxxRecordDecl)});
            } else {
              results.emplace_back(
                  Failure{offloadCallExpr, "Cannot find any call `()` to last arg:" + pretty_string(lastArgExpr, context)});
            }
          } else {
            results.emplace_back(Failure{offloadCallExpr, "Last arg does is not a valid synthesised record type"});
          }
        } else {
          auto root = result.Nodes.getNodeAs<clang::Stmt>(offloadCall);
          results.emplace_back(Failure{root, "Unexpected offload definition:" + pretty_string(root, context)});
        }
      },
      context);
  return results;
}

void OutlineConsumer::HandleTranslationUnit(clang::ASTContext &context) {

  std::cout << "===" << std::endl;

  std::vector<std::variant<Failure, Callsite>> results = outlinePolyregionOffload(context);

  for (auto r : results) {
    std::visit(
        overloaded{
            [&](Failure &f) { //
              std::cout << "Failed:" << pretty_string(f.callExpr, context) << "\nReason:" << f.reason << "\n";
            },
            [&](Callsite &c) { //
              std::cout << pretty_string(c.callExpr, context) << "\n";
              c.callExpr->dumpColor();
              std::cout << "decl=" << std::endl;
              c.functorDecl->dumpColor();
              std::cout << "fnDecl=" << std::endl;
              c.calleeDecl->dumpColor();

              polyregion::polystl::Remapper remapper(context);

              auto owningClass = c.functorDecl->getParent();
              std::vector<polyregion::polyast::Arg> captures;
              for (auto cap : owningClass->captures()) {

                if (cap.capturesVariable()) {
                  auto var = cap.getCapturedVar();
                  captures.push_back(
                      {polyregion::polyast::Named(var->getDeclName().getAsString(), remapper.handleType(var->getType())), {}});
                } else if (cap.capturesThis()) {
                  captures.push_back(
                      {polyregion::polyast::Named("this", remapper.handleType(owningClass->getTypeForDecl()->getCanonicalTypeInternal())),
                       {}});
                } else {
                  throw std::logic_error("Illegal capture");
                }
              }
              std::vector<polyregion::polyast::Arg> args;

              for (auto arg : c.functorDecl->parameters()) {
                args.push_back({polyregion::polyast::Named(arg->getDeclName().getAsString(), remapper.handleType(arg->getType())), {}});
              }

              std::cout << "=========" << std::endl;
              auto r = polyregion::polystl::Remapper::RemapContext{};
              remapper.handleStmt(c.functorDecl->getBody(), r);
              auto f0 = polyregion::polyast::Function(polyregion::polyast::Sym({"kernel"}), {}, {}, args, {}, captures,
                                                      remapper.handleType(c.functorDecl->getReturnType()), r.stmts);

              std::vector<Function> fns;
              std::vector<StructDef> structDefs;
              for (auto &[_, s] : r.structs) {
                structDefs.push_back(s);
                std::cout << repr(s) << std::endl;
              }
              for (auto &[_, f] : r.functions) {
                fns.push_back(f);
                std::cout << repr(f) << std::endl;
              }
              std::cout << repr(f0) << std::endl;

              auto p = Program(f0, fns, structDefs);

              //              auto result = compileIt(p);
              //              if (result) {
              //                std::cout << repr(*result) << std::endl;
              //              } else {
              //                std::cout << "No compile!" << std::endl;
              //              }

              std::vector<std::string> fieldDecl;
              std::vector<std::string> ctorArgs;
              std::vector<std::string> ctorInits;
              std::vector<std::string> ctorAps;

              for (auto c : c.functorDecl->getParent()->captures()) {

                switch (c.getCaptureKind()) {
                  case clang::LambdaCaptureKind::LCK_This: break;
                  case clang::LambdaCaptureKind::LCK_StarThis: break;
                  case clang::LambdaCaptureKind::LCK_ByCopy: break;
                  case clang::LambdaCaptureKind::LCK_ByRef: break;
                  case clang::LambdaCaptureKind::LCK_VLAType: break;
                }

                if (c.capturesVariable()) {
                  auto var = c.getCapturedVar();
                  auto tpe = print_type(var->getType().getDesugaredType(context), context);
                  auto name = var->getQualifiedNameAsString();
                  fieldDecl.push_back(fmt::format("{} {};", tpe, name));
                  ctorArgs.push_back(fmt::format("{} {}", tpe, name));
                  ctorInits.push_back(fmt::format("{}({})", name, name));
                  ctorAps.push_back(name);

                } else if (c.capturesThis()) {

                } else {
                  throw std::logic_error("Illegal capture");
                }
              }

              constexpr static const char *s1 = R"cpp(
                       struct {name} {{
                         {fields}
                         {name}({ctorArgs}) :{ctorInits} {{}}
                         inline {applyReturnTpe} operator()({applyArgs}) {{ return 0; }}
                       }};
                       )cpp";

              auto fileName = context.getSourceManager().getFilename(c.callLambdaArgExpr->getExprLoc());
              auto line = context.getSourceManager().getSpellingLineNumber(c.callLambdaArgExpr->getExprLoc());
              auto col = context.getSourceManager().getSpellingColumnNumber(c.callLambdaArgExpr->getExprLoc());

              auto identifier =
                  fmt::format("lambda_{}_{}_{}__", replaceAllInplace(replaceAllInplace(fileName.str(), "/", "_"), ".", "_"), line, col);

              //                rewriter.InsertText(c.callExpr->getBeginLoc(),
              //                                         fmt::format(s1,                                                          //
              //                                                     fmt::arg("name", identifier),                                //
              //                                                     fmt::arg("fields", fmt::join(fieldDecl, "\n")),              //
              //                                                     fmt::arg("ctorArgs", fmt::join(ctorArgs, ", ")),             //
              //                                                     fmt::arg("ctorInits", fmt::join(ctorInits, ", ")),           //
              //                                                     fmt::arg("applyReturnTpe", "int"),                           //
              //                                                     fmt::arg("applyArgs", fmt::join(std::vector{"int a"}, ", ")) //
              //                                                     ));
              //
              //                rewriter.ReplaceText(c.callLambdaArgExpr->getSourceRange(),
              //                                          fmt::format("{}({})", identifier, fmt::join(ctorAps, ", ")));

              // TODO invoke

              //                       std::cout << result << std::endl;

            },
        },
        r);
  }
}