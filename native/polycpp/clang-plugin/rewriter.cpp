#include <iostream>
#include <string>
#include <vector>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"

#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"
#include "rewriter.h"

#include "aspartame/all.hpp"
#include "magic_enum.hpp"

using namespace polyregion::polystl;
using namespace aspartame;

namespace {

template <typename T> constexpr const char *typenameOfImpl(const char *v) {
  static_assert(std::is_class_v<T> || std::is_enum_v<T>, "Symbol is not a class, struct, or enum.");
  return v;
}
#define typenameOf(Type) typenameOfImpl<Type>(#Type);

template <typename F> class InlineMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  F f;
  void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override { f(result); }

public:
  explicit InlineMatchCallback(F f) : f(f) {}
};

template <typename F, typename... M> void runMatch(clang::ASTContext &context, F callback, M... matcher) {
  using namespace clang::ast_matchers;
  InlineMatchCallback cb(callback);
  MatchFinder finder;
  (finder.addMatcher(matcher, &cb), ...);
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
      context,
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
      callExpr(callee(functionDecl(hasName(offloadFunctionName)))).bind(offloadFunctionName));
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

template <typename T> T *findDecl(clang::DiagnosticsEngine &D, clang::Sema &S, clang::ASTContext &C, const char *name) {
  clang::LookupResult result(S, clang::DeclarationName(&C.Idents.get(name)), clang::SourceLocation(), clang::Sema::LookupAnyName);
  S.LookupName(result, S.getScopeForContext(C.getTranslationUnitDecl()));
  if (result.isSingleResult()) {
    auto decl = result.getFoundDecl();
    if (const auto record = llvm::dyn_cast<T>(decl)) return record;
    else
      D.Report({}, D.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                     "[PolySTL] Name lookup for %0 resulted in unexpected type %1; this is a bug.\n"))
          << name << decl->getDeclKindName();
  } else
    D.Report({}, D.getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolySTL] Name lookup for %0 unsuccessful (%1); this is a bug.\n"))
        << name << magic_enum::enum_name(result.getResultKind());
  return {};
}

void insertKernelImage(clang::DiagnosticsEngine &D, clang::Sema &S, clang::ASTContext &C, const Callsite &c, const KernelBundle &bundle) {

  auto typeOfFieldWithName = [&](clang::QualType ty, const auto &fieldName) -> std::optional<clang::QualType> {
    if (auto decl = ty->getAsCXXRecordDecl()) {
      return (ty->getAsCXXRecordDecl()->fields() | find([&](auto f) { return f->getName() == fieldName; })) //
             ^ map([&](auto x) { return x->getType().getDesugaredType(C); });
    }
    D.Report({},
             D.getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolySTL] Type %0 cannot be resolved to a CXXRecordDecl. This is a bug."))
        << ty;
    return {};
  };

  auto RuntimeKernelBundleTy = c.calleeDecl->getReturnType()->getPointeeType();
  auto RuntimeKernelObjectTy = typeOfFieldWithName(RuntimeKernelBundleTy, "objects") ^ map([](auto &t) { return t->getPointeeType(); });
  auto PlatformKindTy = typeOfFieldWithName(*RuntimeKernelObjectTy, "kind");
  auto ModuleFormatTy = typeOfFieldWithName(*RuntimeKernelObjectTy, "format");
  auto RuntimeStructTy = typeOfFieldWithName(RuntimeKernelBundleTy, "structs") ^ map([](auto &t) { return t->getPointeeType(); });
  auto RuntimeStructMemberTy = typeOfFieldWithName(*RuntimeStructTy, "members") ^ map([](auto &t) { return t->getPointeeType(); });

  RuntimeStructMemberTy->dump();
  RuntimeKernelObjectTy->dump();
  PlatformKindTy->dump();
  ModuleFormatTy->dump();

  // findDecl<clang::CXXRecordDecl>(D, S, C, "::polyregion::runtime::RuntimeKernelObject");
  // findDecl<clang::EnumDecl>(D, S, C, "PlatformKind");
  // findDecl<clang::EnumDecl>(D, S, C, "ModuleFormat");

  auto createDeclRef = [&](clang::VarDecl *lhs) {
    return clang::DeclRefExpr::Create(C, {}, {}, lhs, false, clang::SourceLocation{}, lhs->getType(), clang::ExprValueKind::VK_LValue);
  };

  auto mkConstArrTy = [&](clang::QualType componentTpe, size_t size) {
    return C.getConstantArrayType(componentTpe, llvm::APInt(C.getTypeSize(C.IntTy), size), nullptr, clang::ArraySizeModifier::Normal, 0);
  };

  auto mkStrLit = [&](const std::string &str) {
    return clang::StringLiteral::Create(C, str, clang::StringLiteralKind::Ordinary, false,
                                        C.getConstantArrayType(C.getConstType(C.CharTy),
                                                               llvm::APInt(C.getTypeSize(C.IntTy), str.length() + 1), nullptr,
                                                               clang::ArraySizeModifier::Normal, 0),
                                        {});
  };

  auto mkIntLit = [&](clang::QualType tpe, uint64_t value) {
    return clang::IntegerLiteral::Create(C, llvm::APInt(C.getTypeSize(tpe), value), tpe, {});
  };

  auto mkBoolLit = [&](bool value) { return clang::CXXBoolLiteralExpr::Create(C, value, C.BoolTy, {}); };

  auto mkArrayToPtrDecay = [&](clang::QualType to, clang::Expr *expr) {
    return clang::ImplicitCastExpr::Create(C, to, clang::CK_ArrayToPointerDecay, expr, nullptr, clang::VK_PRValue, {});
  };

  auto mkInitList = [&](clang::QualType ty, const std::vector<clang::Expr *> &initExprs) {
    auto init = new (C) clang::InitListExpr(C, {}, initExprs, {});
    init->setType(ty);
    return init;
  };

  auto mkStaticVarDecl = [&](const std::string &name, clang::QualType ty, const std::vector<clang::Expr *> &initExprs) {
    auto decl = clang::VarDecl::Create(C, c.calleeDecl, {}, {}, &C.Idents.get(name), ty, nullptr, clang::SC_Static);
    decl->setInit(mkInitList(ty, initExprs));
    decl->setInitStyle(clang::VarDecl::InitializationStyle::ListInit);
    return decl;
  };

  auto varDeclWithName = [&](clang::Stmt *stmt, const std::string &name) -> std::optional<clang::VarDecl *> {
    if (auto declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt); declStmt && declStmt->isSingleDecl()) {
      if (auto varDecl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl()); varDecl && varDecl->getName() == name) {
        return varDecl;
      }
    }
    return {};
  };

  auto constCharStarTy = C.getPointerType(C.CharTy.withConst());

  auto kernelImageDecls =                                            //
      bundle.objects                                                 //
      | zip_with_index()                                             //
      | map([&](auto ko, auto idx) {                                 //
          return mkStaticVarDecl(                                    //
              "__kernelobject_image_data_" + std::to_string(idx),    //
              mkConstArrTy(C.UnsignedCharTy, ko.moduleImage.size()), //
              ko.moduleImage | map([&](const unsigned char c) -> clang::Expr * {
                return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast, mkIntLit(C.IntTy, c), nullptr,
                                                       clang::VK_PRValue, {});
              }) | to_vector());
        }) //
      | to_vector();

  auto kernelFeatureDecls =                                         //
      bundle.objects                                                //
      | zip_with_index()                                            //
      | map([&](auto &ko, auto idx) {                               //
          return mkStaticVarDecl(                                   //
              "__kernelobject_feature_data_" + std::to_string(idx), //
              mkConstArrTy(constCharStarTy, ko.features.size()),    //
              ko.features | map([&](auto &feature) -> clang::Expr * {
                return mkArrayToPtrDecay(C.getConstType(C.getPointerType(C.CharTy)), mkStrLit(feature));
              }) | to_vector());
        }) //
      | to_vector();

  auto kernelObjectArrayDecl = mkStaticVarDecl(                    //
      "__kernelobject_data",                                       //
      mkConstArrTy(*RuntimeKernelObjectTy, bundle.objects.size()), //
      bundle.objects                                               //
          | zip_with_index()                                       //
          | map([&](auto &ko, auto idx) -> clang::Expr * {         //
              return mkInitList(                                   //
                  *RuntimeKernelObjectTy,                          //
                  {
                      S.ImpCastExprToType(mkIntLit(C.IntTy, static_cast<std::underlying_type_t<decltype(ko.kind)>>(ko.kind)),
                                          *PlatformKindTy, clang::CastKind::CK_IntegralCast)
                          .get(),
                      S.ImpCastExprToType(mkIntLit(C.IntTy, static_cast<std::underlying_type_t<decltype(ko.format)>>(ko.format)),
                                          *ModuleFormatTy, clang::CastKind::CK_IntegralCast)
                          .get(),

                      mkArrayToPtrDecay(C.getPointerType(C.CharTy.withConst()), createDeclRef(kernelFeatureDecls[idx])),
                      mkIntLit(C.getSizeType(), ko.moduleImage.size()),
                      mkArrayToPtrDecay(C.getPointerType(C.UnsignedCharTy.withConst()), createDeclRef(kernelImageDecls[idx])),
                  });
            }) //
          | to_vector());

  auto nameToIndex = bundle.layouts | map([](auto, auto &l) { return l.name; }) | zip_with_index() | to<std::unordered_map>();

  // name -> idx

  auto kernelStructMemberArrayDecl = //
      bundle.layouts | zip_with_index() | map([&](auto &k, auto idx) {
        return mkStaticVarDecl(                                                                     //
            "__kernelstruct_member_data_" + std::to_string(idx),                                    //
            mkConstArrTy(*RuntimeStructMemberTy, k.second.members.size()),                          //
            k.second.members                                                                        //
                | zip_with_index()                                                                  //
                | map([&](auto &m, auto idx) -> clang::Expr * {                                     //
                    return mkInitList(*RuntimeStructMemberTy,                                       //
                                      {mkArrayToPtrDecay(constCharStarTy, mkStrLit(m.name.symbol)), //
                                       mkIntLit(C.getSizeType(), m.offsetInBytes),                  //
                                       mkIntLit(C.getSizeType(), m.sizeInBytes),                    //
                                       mkIntLit(C.getIntTypeForBitwidth(64, true),
                                                m.name.tpe.template get<Type::Struct>() ^
                                                    bind([&](auto &s) { return nameToIndex ^ get(s.name); }) ^ get_or_else(-1))

                                      });
                  }) //
                | to_vector());
      }) |
      to_vector();

  auto kernelStructArrayDecl = mkStaticVarDecl(                                         //
      "__kernelstruct_data",                                                            //
      mkConstArrTy(*RuntimeStructTy, bundle.layouts.size()),                            //
      bundle.layouts | zip_with_index() | map([&](auto &k, auto idx) -> clang::Expr * { //
        auto &[exported, ks] = k;
        return mkInitList(    //
            *RuntimeStructTy, //
            {
                mkArrayToPtrDecay(constCharStarTy, mkStrLit(ks.name)), //
                mkBoolLit(exported),                                   //
                mkIntLit(C.getSizeType(), ks.members.size()),          //
                mkArrayToPtrDecay(C.getPointerType(*RuntimeStructMemberTy), createDeclRef(kernelStructMemberArrayDecl[idx])),

                //                      S.ImpCastExprToType(mkIntegerLiteral(C.IntTy,
                //                      static_cast<std::underlying_type_t<decltype(ko.kind)>>(ko.kind)),
                //                                          *PlatformKindTy, clang::CastKind::CK_IntegralCast)
                //                          .get(),
                //                      S.ImpCastExprToType(mkIntegerLiteral(C.IntTy,
                //                      static_cast<std::underlying_type_t<decltype(ko.format)>>(ko.format)),
                //                                          *ModuleFormatTy, clang::CastKind::CK_IntegralCast)
                //                          .get(),
                //
                //                      mkArrayToPtrDecay(C.getPointerType(C.CharTy.withConst()),
                //                      createDeclRef(kernelFeatureDecls[idx])),
                //                      mkIntegerLiteral(C.getSizeType(), ko.moduleImage.size()),
                //                      mkArrayToPtrDecay(C.getPointerType(C.UnsignedCharTy.withConst()),
                //                      createDeclRef(kernelImageDecls[idx])),
            });
      }) //
          | to_vector());

  auto kernelBundleDecl = mkStaticVarDecl( //
      "__kb",                              //
      RuntimeKernelBundleTy.withConst(),   //
      {
          mkArrayToPtrDecay(constCharStarTy, mkStrLit(bundle.moduleName)),

          mkIntLit(C.getSizeType(), bundle.objects.size()),
          mkArrayToPtrDecay(C.getPointerType(*RuntimeKernelObjectTy), createDeclRef(kernelObjectArrayDecl)),

          mkIntLit(C.getSizeType(), bundle.layouts.size()),
          mkArrayToPtrDecay(C.getPointerType(*RuntimeStructTy), createDeclRef(kernelStructArrayDecl)),

          mkArrayToPtrDecay(constCharStarTy, mkStrLit(bundle.metadata)),
      });

  std::vector<clang::Stmt *> newStmts =
      kernelImageDecls                                                                                            //
      | concat(kernelFeatureDecls)                                                                                //
      | concat(kernelStructMemberArrayDecl)                                                                       //
      | concat(std::vector{kernelObjectArrayDecl, kernelStructArrayDecl, kernelBundleDecl})                       //
      | map([&](auto dcl) -> clang::Stmt * { return new (C) clang::DeclStmt(clang::DeclGroupRef(dcl), {}, {}); }) //
      | append(clang::ReturnStmt::Create(C, {}, createDeclRef(kernelBundleDecl), {}))                             //
      | to_vector();

  c.calleeDecl->setBody(clang::CompoundStmt::Create( //
      C,                                             //
      newStmts, {}, {}, {}));
  //  c.calleeDecl->dumpColor(); // void __polyregion_offload__(F __polyregion__f)
  c.calleeDecl->print(llvm::outs());
}

OffloadRewriteConsumer::OffloadRewriteConsumer(clang::CompilerInstance &CI, const Options &opts)
    : clang::ASTConsumer(), CI(CI), opts(opts) {}

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
  auto &D = CI.getDiagnostics();
  for (auto r : outlinePolyregionOffload(C))
    r ^ foreach_total(
            [&](const Failure &f) { //
              D.Report(f.callExpr->getBeginLoc(), D.getCustomDiagID(clang::DiagnosticsEngine::Warning, "[PolySTL] Outline failed: %0"))
                  .AddString(f.reason);
            },
            [&](const Callsite &c) { //
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

              std::cout << moduleId << std::endl;

              auto bundle = generateBundle(
                  opts, C, D, moduleId, *c.functorDecl,
                  specialisationPath ^ head_maybe() ^
                      fold([](auto, auto callExpr) { return callExpr->getExprLoc(); }, [&]() { return c.callLambdaArgExpr->getExprLoc(); }),
                  c.kind);

              if (opts.verbose) {
                D.Report(c.callLambdaArgExpr->getExprLoc(),
                         D.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[PolySTL] Outlined function: %0 for %1 (%2)\n"))
                    << moduleId << to_string(c.kind)
                    << (bundle.objects //
                        | map([](auto &o) {
                            return std::string(to_string(o.format)) + "=" +
                                   std::to_string(static_cast<float>(o.moduleImage.size()) / 1000) + "KB";
                          }) //
                        | mk_string(", "));
              }

              insertKernelImage(D, CI.getSema(), C, c, bundle);

            });
}
