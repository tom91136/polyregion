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

#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"
#include "rewriter.h"

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

static void checkFunctionPrototype(const clang::FunctionDecl *funcDecl, clang::ASTContext &context) {
  // Check if the FunctionDecl is valid
  if (!funcDecl || funcDecl->isInvalidDecl()) {
    llvm::errs() << "(Invalid)";
    return;
  }

  // Access the function's return type
  clang::QualType returnType = funcDecl->getReturnType();

  // Access the function's name
  clang::DeclarationNameInfo nameInfo = funcDecl->getNameInfo();
  std::string functionName = nameInfo.getAsString();

  // Access the function's parameters
  llvm::SmallVector<clang::ParmVarDecl *, 4> parameters;
  for (clang::ParmVarDecl *param : funcDecl->parameters()) {
    parameters.push_back(param);
  }

  // Print or check the prototype information

  returnType.print(llvm::outs(), funcDecl->getASTContext().getPrintingPolicy());
  llvm::outs() << " " << functionName << "(";

  for (size_t i = 0; i < parameters.size(); ++i) {
    parameters[i]->print(llvm::outs(), funcDecl->getASTContext().getPrintingPolicy());
    if (i < parameters.size() - 1) {
      llvm::outs() << ", ";
    }
  }

  clang::SourceLocation loc = funcDecl->getBeginLoc();
  llvm::outs() << ") @ " << loc.printToString(context.getSourceManager());
}

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

struct Callsite {
  clang::CallExpr *callExpr;         // decl of the std::transform call
  clang::Expr *callLambdaArgExpr;    // decl of the lambda arg
  clang::FunctionDecl *calleeDecl;   // decl of the specialised std::transform
  clang::CXXMethodDecl *functorDecl; // decl of the specialised lambda functor, this is the root of the lambda body
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
                if ((*target)->getParent() == lambdaArgCxxRecordDecl) {

                  results.emplace_back(Callsite{const_cast<clang::CallExpr *>(algCallExpr), const_cast<clang::Expr *>(lastArgExpr),
                                                const_cast<clang::FunctionDecl *>(fnDecl), *target});
                } else
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
                results.emplace_back(Callsite{const_cast<clang::CallExpr *>(offloadCallExpr), const_cast<clang::Expr *>(lastArgExpr),
                                              const_cast<clang::FunctionDecl *>(fnDecl), *target});
              else
                results.emplace_back(Failure{offloadCallExpr, "Target record mismatch:\nLast arg=" + to_string((*target)->getParent()) +
                                                                  "\nTarget=" + to_string(lambdaArgCxxRecordDecl)});
            } else {
              results.emplace_back(
                  Failure{offloadCallExpr, "Cannot find any call `operator ()` to last arg:" + pretty_string(lastArgExpr, context)});
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

void insertKernelImage(clang::ASTContext &C, Callsite &c, const SpecialisationPathVisitor &spVisitor, const std::string &image) {
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

  auto xs = spVisitor.resolve(c.calleeDecl);
  std::string identifier;
  for (std::make_signed_t<size_t> i = xs.size() - 1; i >= 0; --i) {
    auto loc = getLocation(*xs[i].second, C);
    identifier += xs[i].first->getName();
    identifier += "<";
    identifier += loc.filename;
    identifier += ":";
    identifier += std::to_string(loc.line);
    identifier += ">";
    if (i != 0) identifier += "->";
  }

  auto existingStmts = c.calleeDecl->getBody()->children();
  std::vector<clang::Stmt *> newStmts;
  for (auto stmt : existingStmts) {
    if (auto kernelImageDecl = varDeclWithName(stmt, "__stub_kernelImageBytes__"); kernelImageDecl) {

      auto component = kernelImageDecl->getType()->getAs<clang::PointerType>()->getPointeeType();

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
    } else if (auto kernelNameDecl = varDeclWithName(stmt, "__stub_kernelName__"); kernelNameDecl) {
      auto rhs = clang::ImplicitCastExpr::Create(
          C, kernelNameDecl->getType(), clang::CK_ArrayToPointerDecay,
          clang::StringLiteral::Create(C, identifier, clang::StringLiteral::StringKind::Ordinary, false,
                                       C.getConstantArrayType(C.CharTy, llvm::APInt(C.getTypeSize(C.IntTy), identifier.length() + 1),
                                                              nullptr, clang::ArrayType::Normal, 0),
                                       {}),
          nullptr, clang::VK_PRValue, {});
      newStmts.push_back(stmt);
      newStmts.push_back(createAssignStmt(kernelNameDecl, rhs));

    } else if (cxxOperatorCallFromParamWithName(stmt, "__stub_polyregion__f__")) {
      // delete the direct call on the lambda
    } else {
      newStmts.push_back(stmt);
    }
  }

  c.calleeDecl->setBody(clang::CompoundStmt::Create(C, newStmts, {}, {}, {}));
  std::cout << "<<<" << std::endl;
  //  c.calleeDecl->dumpColor(); // void __polyregion_offload__(F __polyregion__f)
  //  c.calleeDecl->print(llvm::outs());
}

void handleDecl(clang::ASTContext &C) {
  std::vector<std::variant<Failure, Callsite>> results = outlinePolyregionOffload(C);

  SpecialisationPathVisitor spVisitor(C);

  for (auto r : results) {
    std::visit(overloaded{
                   [&](Failure &f) { //
                     llvm::errs() << "Failed:" << pretty_string(f.callExpr, C) << "\nReason:" << f.reason << "\n";
                   },
                   [&](Callsite &c) { //
                                      //              llvm::outs() << pretty_string(c.callExpr, context) << "\n";
                     llvm::errs() << "callLambdaArgExpr=" << "\n";
                     //                     c.callLambdaArgExpr->dumpColor();
                     llvm::errs() << "callExpr=" << "\n";
                     //                     c.callExpr->dumpColor();
                     llvm::errs() << "functorDecl=" << "\n";
                                          c.functorDecl->dumpColor(); // F __polyregion__f; F = [&]() { __polyregion__v[0] =
                     //                     __polyregion__f(); }
                     llvm::errs() << "calleeDecl=" << "\n";

                     c.calleeDecl->dump();
                     llvm::errs() << ">>>" << "\n";
                     std::string image = {0x12, 0x34, 0x54};



                     auto m = generate(C, c.functorDecl->getParent(), c.functorDecl->getReturnType(), c.functorDecl->getBody());

                     insertKernelImage(C, c, spVisitor, image);
                   },
               },
               r);
  }
}

OffloadRewriteConsumer::OffloadRewriteConsumer() : clang::ASTConsumer() {}

template <typename Parent, typename Node> const Parent *findParentOfType(clang::ASTContext &context, Node *from) {
  for (auto node = context.getParents(*from).begin()->template get<clang::Decl>(); node;
       node = context.getParents(*node).begin()->template get<clang::Decl>()) {
    //
    //    clang::Stmt *ax = {};
    //    context.getParents(*ax).begin()->get<clang::Decl>();

    if (auto parent = dyn_cast<Parent>(node); parent) {
      return parent;
    }
  }
  return {};
}

void OffloadRewriteConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
  std::cout << "[TU] >>>" << std::endl;

  //  //  Context.getTranslationUnitDecl()->dumpColor();
  //
  //    SpecialisationPathVisitor a(Context);
  //    a.TraverseDecl(Context.getTranslationUnitDecl());
  //    std::cout << a.map.size() << "\n";

  //    for (auto &[called, src] : a.map) {
  //      //    k->dump(llvm::outs(), Context);
  //      //    clang::SourceLocation loc = called->getBeginLoc();
  //      //    llvm::outs() << "CallExpr " << loc.printToString(Context.getSourceManager()) << " '" <<
  //      //    called->getCallReturnType(Context).getAsString()
  //      //                 << "' ";
  //
  //      checkFunctionPrototype(called, Context);
  //      llvm::outs() << " <- \n\t";
  //      checkFunctionPrototype(src.first, Context);
  //      llvm::outs() << "\n";
  //    }

  handleDecl(Context);

  //  context.PrintStats();
  //  Initialize(context);

  using namespace clang::ast_matchers;

  //  runMatch(
  //       ( (functionDecl(hasName("main")))).bind("that"),
  //      [&](const MatchFinder::MatchResult &result) {
  //        if (auto main = result.Nodes.getNodeAs<clang::FunctionDecl>("that")) {
  //          main->dumpColor();
  //          auto nc = const_cast<clang::FunctionDecl*>( main);
  //          auto stmts = clang::CompoundStmt::Create(context, {}, {}, {}, {});
  //          nc->setBody(stmts);
  //          nc->dumpColor();
  //        }
  //      },
  //      context);

  //    context.getTranslationUnitDecl()->dumpColor();

  return;
}
