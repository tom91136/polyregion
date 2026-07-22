#include "rewriter.h"

#include <string>
#include <vector>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyregion/conventions.h"

#include "ast.h"
#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"

using namespace polyregion::polystl;
using namespace polyregion;
using polyregion::polyfront::emit;
using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;
using namespace aspartame;

namespace {

template <typename F> class InlineMatchCallback final : public clang::ast_matchers::MatchFinder::MatchCallback {
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

static Vector<std::variant<Failure, Callsite>> outlinePolyregionOffload(clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  Vector<std::variant<Failure, Callsite>> results;
  runMatch(
      context,
      [&](const MatchFinder::MatchResult &result) {
        if (const auto offloadCallExpr = result.Nodes.getNodeAs<clang::CallExpr>(offloadFunctionName)) {
          const auto lastArgExpr = offloadCallExpr->getArg(offloadCallExpr->getNumArgs() - 1)->IgnoreUnlessSpelledInSource();
          const auto fnDecl = offloadCallExpr->getDirectCallee();
          if (const auto lambdaArgCxxRecordDecl = lastArgExpr->getType()->getAsCXXRecordDecl()) {
            // TODO we should support explicit structs with () operator and not just lambdas
            if (const auto op = lambdaArgCxxRecordDecl->getLambdaCallOperator(); lambdaArgCxxRecordDecl->isLambda() && op) {

              // prototype is <polyregion::runtime::PlatformKind, typename F>; we check the first template arg's type and value
              const auto templateArgs = fnDecl->getTemplateSpecializationArgs();
              if (templateArgs->size() != 2) {
                results.emplace_back(
                    Failure{offloadCallExpr, "Template arity mismatch for " + std::string(offloadFunctionName) + ", expecting 2"});
              } else {
                if (const auto templateArg0 = templateArgs->get(0);
                    templateArg0.getKind() == clang::TemplateArgument::Integral &&
                    templateArg0.getIntegralType()->getAsTagDecl()->getName().str() == "PlatformKind") {
                  const auto kind = static_cast<polyregion::runtime::PlatformKind>(templateArg0.getAsIntegral().getExtValue());
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
          const auto root = result.Nodes.getNodeAs<clang::Stmt>(offloadFunctionName);
          results.emplace_back(Failure{root, "Unexpected offload definition:" + pretty_string(root, context)});
        }
      },
      callExpr(callee(functionDecl(hasName(offloadFunctionName)))).bind(offloadFunctionName));
  return results;
}

void insertKernelImage(clang::DiagnosticsEngine &D, clang::Sema &S, clang::ASTContext &C, const Callsite &c,
                       const polyregion::polyfront::KernelBundle &bundle) {
  const auto fieldWithName = [&](const clang::QualType ty, const auto &fieldName) -> Opt<clang::FieldDecl *> {
    if (const auto decl = ty->getAsCXXRecordDecl()) {
      return decl->fields() | find([&](auto f) { return f->getName() == fieldName; });
    }
    emit(D, clang::DiagnosticsEngine::Error, POLYREGION_DIAG_POLYSTL "Type %0 cannot be resolved to a CXXRecordDecl. This is a bug.", ty);
    return {};
  };

  const auto typeOfFieldWithName = [&](clang::QualType ty, const auto &fieldName) -> Opt<clang::QualType> {
    return fieldWithName(ty, fieldName) ^ map([&](auto f) { return f->getType().getDesugaredType(C); });
  };

  const auto KernelBundleTy = c.calleeDecl->getReturnType()->getPointeeType();
  const auto KernelObjectTy = typeOfFieldWithName(KernelBundleTy, "objects") ^ map([](auto &t) { return t->getPointeeType(); });
  const auto PlatformKindTy = typeOfFieldWithName(*KernelObjectTy, "kind");
  const auto ModuleFormatTy = typeOfFieldWithName(*KernelObjectTy, "format");
  const auto TargetTy = typeOfFieldWithName(*KernelObjectTy, "target");
  const auto OptLevelTy = typeOfFieldWithName(*KernelObjectTy, "opt");
  const auto TypeLayoutTy = typeOfFieldWithName(KernelBundleTy, "structs") ^ map([](auto &t) { return t->getPointeeType(); });
  const auto AggregateMemberTy = typeOfFieldWithName(*TypeLayoutTy, "members") ^ map([](auto &t) { return t->getPointeeType(); });
  const auto TypeLayoutMembersField = fieldWithName(*TypeLayoutTy, "members");

  auto kernelImageDecls =
      bundle.objects | zip_with_index() | map([&](auto &ko, auto idx) {
        return mkStaticVarDecl(C, c.calleeDecl, fmt::format("__ko_image_data_{}", idx),
                               mkConstArrTy(C, C.UnsignedCharTy, ko.moduleImage.size()),
                               ko.moduleImage | map([&](const unsigned char x) -> clang::Expr * {
                                 return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast,
                                                                        mkIntLit(C, C.IntTy, x), nullptr, clang::VK_PRValue, {});
                               }) | to_vector());
      }) //
      | to_vector();

  auto kernelFeatureDecls =
      bundle.objects | zip_with_index() | map([&](auto &ko, auto idx) {
        return mkStaticVarDecl(C, c.calleeDecl, fmt::format("__ko_feature_data_{}", idx),
                               mkConstArrTy(C, constCharStarTy(C), ko.features.size()),
                               ko.features | map([&](auto &feature) -> clang::Expr * {
                                 return mkArrayToPtrDecay(C, C.getConstType(C.getPointerType(C.CharTy)), mkStrLit(C, feature));
                               }) | to_vector());
      }) //
      | to_vector();

  Opt<clang::VarDecl *> kernelProgramDecl;
  if (!bundle.program.empty()) {
    kernelProgramDecl = mkStaticVarDecl(C, c.calleeDecl, "__ko_program_data", //
                                        mkConstArrTy(C, C.UnsignedCharTy, bundle.program.size()),
                                        bundle.program | map([&](const unsigned char x) -> clang::Expr * {
                                          return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast,
                                                                                 mkIntLit(C, C.IntTy, x), nullptr, clang::VK_PRValue, {});
                                        }) | to_vector());
  }

  auto kernelObjectArrayDecl = mkStaticVarDecl(
      C, c.calleeDecl,                                         //
      "__ko_data",                                             //
      mkConstArrTy(C, *KernelObjectTy, bundle.objects.size()), //
      bundle.objects                                           //
          | zip_with_index()                                   //
          | map([&](auto &ko, auto idx) -> clang::Expr * {     //
              return mkInitList(
                  C,               //
                  *KernelObjectTy, //
                  {
                      /*kind         */ S
                          .ImpCastExprToType(mkIntLit(C, C.IntTy, static_cast<std::underlying_type_t<decltype(ko.kind)>>(ko.kind)),
                                             *PlatformKindTy, clang::CastKind::CK_IntegralCast)
                          .get(),
                      /*format       */
                      S.ImpCastExprToType(mkIntLit(C, C.IntTy, static_cast<std::underlying_type_t<decltype(ko.format)>>(ko.format)),
                                          *ModuleFormatTy, clang::CastKind::CK_IntegralCast)
                          .get(),
                      /*featureCount */ mkIntLit(C, C.getSizeType(), ko.features.size()),
                      /*features     */ mkArrayToPtrDecay(C, C.getPointerType(C.CharTy.withConst()), mkDeclRef(C, kernelFeatureDecls[idx])),
                      /*imageLength  */ mkIntLit(C, C.getSizeType(), ko.moduleImage.size()),
                      /*image        */
                      mkArrayToPtrDecay(C, C.getPointerType(C.UnsignedCharTy.withConst()), mkDeclRef(C, kernelImageDecls[idx])),
                      /*target       */
                      S.ImpCastExprToType(mkIntLit(C, C.IntTy, static_cast<std::underlying_type_t<decltype(ko.target)>>(ko.target)),
                                          *TargetTy, clang::CastKind::CK_IntegralCast)
                          .get(),
                      /*arch         */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, ko.arch)),
                      /*pipelineSpec */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, ko.pipelineSpec)),
                      /*opt          */
                      S.ImpCastExprToType(mkIntLit(C, C.IntTy, static_cast<int>(polyregion::compiletime::OptLevel::O3)), *OptLevelTy,
                                          clang::CastKind::CK_IntegralCast)
                          .get(),
                      /*programLength*/ mkIntLit(C, C.getSizeType(), bundle.program.size()),
                      /*program      */
                      bundle.program.empty() ? static_cast<clang::Expr *>(mkNullPtrLit(C, C.UnsignedCharTy.withConst()))
                                             : static_cast<clang::Expr *>(mkArrayToPtrDecay(
                                                   C, C.getPointerType(C.UnsignedCharTy.withConst()), mkDeclRef(C, *kernelProgramDecl))),
                  });
            }) //
          | to_vector());

  auto table = bundle.layouts | values() | map([&](auto &sl) { return std::pair{Type::Struct(Sym({sl.name}), {}), sl}; }) | to<Map>();

  auto primitiveTypeLayoutsDecls =
      Vector<Type::Any>{
          Type::Float16(), Type::Float32(), Type::Float64(),                 //
          Type::IntU8(),   Type::IntU16(),  Type::IntU32(),  Type::IntU64(), //
          Type::IntS8(),   Type::IntS16(),  Type::IntS32(),  Type::IntS64(), //
          Type::Unit0(),   Type::Bool1(),                                    //
      } //
      | collect([&](auto &t) {
          return primitiveSize(t) ^ map([&](auto sizeInBytes) {
                   return std::pair{t, mkStaticVarDecl(C, c.calleeDecl, fmt::format("__primitive_type_layout_{}", repr(t)), *TypeLayoutTy,
                                                       {
                                                           /*name        */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, repr(t))),
                                                           /*sizeInBytes */ mkIntLit(C, C.getSizeType(), sizeInBytes),
                                                           /*alignment   */ mkIntLit(C, C.getSizeType(), sizeInBytes),
                                                           /*attrs       */
                                                           mkIntLit(C, C.getSizeType(),
                                                                    to_underlying(polyregion::runtime::LayoutAttrs::Opaque |     //
                                                                                  polyregion::runtime::LayoutAttrs::SelfOpaque | //
                                                                                  polyregion::runtime::LayoutAttrs::Primitive)),
                                                           /*memberCount */ mkIntLit(C, C.getSizeType(), 0),
                                                           /*member      */ mkNullPtrLit(C, *TypeLayoutTy),
                                                       })};
                 });
        }) //
      | to<Map>();

  auto TypeLayoutTyNoConst = TypeLayoutTy->withoutLocalFastQualifiers();
  auto structTypeLayoutArrayDecl =
      mkStaticVarDecl(C, c.calleeDecl, "__struct_type_layouts", mkConstArrTy(C, TypeLayoutTyNoConst, bundle.layouts.size()),
                      bundle.layouts | map([&](auto, auto &sl) -> clang::Expr * {
                        auto attrs = polyregion::runtime::LayoutAttrs::None;
                        if (isSelfOpaque(sl)) attrs |= polyregion::runtime::LayoutAttrs::SelfOpaque;
                        if (isOpaque(sl, table)) attrs |= polyregion::runtime::LayoutAttrs::Opaque;
                        return mkInitList(C, TypeLayoutTyNoConst,
                                          {
                                              /*name        */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, sl.name)), //
                                              /*sizeInBytes */ mkIntLit(C, C.getSizeType(), sl.sizeInBytes),                   //
                                              /*alignment   */ mkIntLit(C, C.getSizeType(), sl.alignment),                     //
                                              /*attrs       */ mkIntLit(C, C.getSizeType(), to_underlying(attrs)),             //
                                              /*memberCount */ mkIntLit(C, C.getSizeType(), sl.members.size()),                //
                                              /*member      */ mkNullPtrLit(C, *AggregateMemberTy), // XXX assigned later
                                          });
                      }) | to_vector());

  auto structNameToTypeLayoutIdx = bundle.layouts | values() | map([](auto &sl) { return sl.name; }) | zip_with_index() | to<Map>();

  auto aggregateMemberArrayDecls = //
      bundle.layouts | values() | zip_with_index() | map([&](auto &sl, auto idx) {
        return std::pair{
            sl.name,
            mkStaticVarDecl(
                C, c.calleeDecl,                                        //
                fmt::format("__aggregate_member_{}", idx),              //
                mkConstArrTy(C, *AggregateMemberTy, sl.members.size()), //
                sl.members | map([&](auto &m) -> clang::Expr * {        //
                  const auto [indirections, componentSize] = countIndirectionsAndComponentSize(m.name.tpe, table);
                  const auto typeDecl =
                      extractComponent(m.name.tpe) ^ flat_map([&](auto &t) {
                        return primitiveTypeLayoutsDecls                                                                             //
                               ^ get_maybe(t)                                                                                        //
                               ^ map([&](auto &decl) -> clang::Expr * {                                                              //
                                   return S.CreateBuiltinUnaryOp({}, clang::UnaryOperatorKind::UO_AddrOf, mkDeclRef(C, decl)).get(); //
                                 })                                                                                                  //
                               ^ or_else([&]() {
                                   return t.template get<Type::Struct>() ^ flat_map([&](auto &s) {
                                            return structNameToTypeLayoutIdx ^ get_maybe(repr(s.name)) ^ map([&](auto layoutIdx) {
                                                     return S
                                                         .CreateBuiltinBinOp({}, clang::BinaryOperatorKind::BO_Add,
                                                                             mkDeclRef(C, structTypeLayoutArrayDecl),
                                                                             mkIntLit(C, C.getSizeType(), layoutIdx))
                                                         .get();
                                                   });
                                          });
                                 });
                      });

                  const bool readOnly =
                      bundle.readOnlyMembers ^ get_maybe(sl.name) ^ exists([&](auto &ms) { return ms ^ contains(m.name.symbol); });
                  return mkInitList(C,
                                    *AggregateMemberTy,                                                                         //
                                    {/*name            */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, m.name.symbol)), //
                                     /*offsetInBytes   */ mkIntLit(C, C.getSizeType(), m.offsetInBytes),                        //
                                     /*sizeInBytes     */ mkIntLit(C, C.getSizeType(), m.sizeInBytes),                          //
                                     /*ptrIndirections */ mkIntLit(C, C.getSizeType(), indirections),                           //
                                     /*componentSize   */ mkIntLit(C, C.getSizeType(), componentSize.value_or(m.sizeInBytes)),  //
                                     /*type            */ typeDecl ^ get_or_else(mkNullPtrLit(C, *TypeLayoutTy)),               //
                                     /*readOnly        */ mkIntLit(C, C.getSizeType(), readOnly ? 1 : 0)});
                }) | to_vector())};
      }) //
      | to<Map>();

  auto assignTypeLayoutMembers =
      structNameToTypeLayoutIdx ^ to_vector() ^ map([&](auto &name, auto &idx) -> clang::Stmt * {
        const auto typeLayoutExpr = new (C) clang::ArraySubscriptExpr(
            mkArrayToPtrDecay(C, TypeLayoutTyNoConst, mkDeclRef(C, structTypeLayoutArrayDecl)), mkIntLit(C, C.getSizeType(), idx),
            TypeLayoutTyNoConst, clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary, {});
        const auto lhs = mkMemberExpr(C, typeLayoutExpr, *TypeLayoutMembersField);
        const auto rhs = mkArrayToPtrDecay(C, C.getPointerType(*AggregateMemberTy),
                                           aggregateMemberArrayDecls ^ get_maybe(name) ^
                                               fold([&](auto &d) -> clang::Expr * { return mkDeclRef(C, d); },
                                                    [&]() -> clang::Expr * { return mkNullPtrLit(C, *AggregateMemberTy); }));
        return S.CreateBuiltinBinOp({}, clang::BinaryOperatorKind::BO_Assign, lhs, rhs).get();
      });

  auto interfaceLayoutIdx = bundle.layouts | index_where([&](auto exported, auto &) { return exported; });

  auto kernelBundleDecl = mkStaticVarDecl(
      C, c.calleeDecl, "__kb", KernelBundleTy.withConst(),
      {
          /*moduleName         */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, bundle.moduleName)),
          /*objectCount        */ mkIntLit(C, C.getSizeType(), bundle.objects.size()),
          /*objects            */ mkArrayToPtrDecay(C, C.getPointerType(*KernelObjectTy), mkDeclRef(C, kernelObjectArrayDecl)),
          /*structCount        */ mkIntLit(C, C.getSizeType(), bundle.layouts.size()),
          /*structs            */ mkArrayToPtrDecay(C, C.getPointerType(*TypeLayoutTy), mkDeclRef(C, structTypeLayoutArrayDecl)),
          /*interfaceLayoutIdx */ mkIntLit(C, C.getSizeType(), interfaceLayoutIdx),
          /*metadata           */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, bundle.metadata)),
          /*mirrorId           */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, bundle.mirrorId)),
          /*prelude           */ mkNullPtrLit(C, (*typeOfFieldWithName(KernelBundleTy, "prelude"))->getPointeeType()),
          /*postlude          */ mkNullPtrLit(C, (*typeOfFieldWithName(KernelBundleTy, "postlude"))->getPointeeType()),
          /*asserts           */ new (C) clang::CXXBoolLiteralExpr(bundle.asserts, C.BoolTy, {}),
      });

  // embed the host mirroring BC
  Opt<clang::VarDecl *> hostBcDecl;
  if (!bundle.hostMirrorBitcode.empty()) {
    const auto arrTy =
        C.getConstantArrayType(C.getConstType(C.CharTy), llvm::APInt(C.getTypeSize(C.IntTy), bundle.hostMirrorBitcode.size()), nullptr,
                               clang::ArraySizeModifier::Normal, 0);
    auto *lit = clang::StringLiteral::Create(C, bundle.hostMirrorBitcode, clang::StringLiteralKind::Ordinary, false, arrTy, {{}});
    auto d = clang::VarDecl::Create(C, c.calleeDecl, {}, {}, &C.Idents.get(std::string("__") + conventions::reflect::MirrorBitcodeGlobal),
                                    arrTy, nullptr, clang::SC_Static);
    d->setInit(lit);
    d->addAttr(clang::UsedAttr::CreateImplicit(C));
    hostBcDecl = d;
  }

  // program data must precede __ko_data, whose initialiser takes its address
  Vector<clang::Stmt *> newStmts =                                                                                //
      (kernelProgramDecl | to_vector())                                                                           //
      | concat(hostBcDecl | to_vector())                                                                          //
      | concat(kernelImageDecls)                                                                                  //
      | concat(primitiveTypeLayoutsDecls | values())                                                              //
      | append(structTypeLayoutArrayDecl)                                                                         //
      | concat(aggregateMemberArrayDecls | values())                                                              //
      | concat(kernelFeatureDecls)                                                                                //
      | append(kernelObjectArrayDecl)                                                                             //
      | append(kernelBundleDecl)                                                                                  //
      | map([&](auto dcl) -> clang::Stmt * { return new (C) clang::DeclStmt(clang::DeclGroupRef(dcl), {}, {}); }) //
      | concat(assignTypeLayoutMembers)                                                                           //
      | append(clang::ReturnStmt::Create(C, {}, mkDeclRef(C, kernelBundleDecl), {}))                              //
      | to_vector();

  c.calleeDecl->setBody(clang::CompoundStmt::Create(C, newStmts, {}, {}, {}));
}

OffloadRewriteConsumer::OffloadRewriteConsumer(clang::CompilerInstance &CI, const polyfront::Options &opts)
    : clang::ASTConsumer(), CI(CI), opts(opts) {}

void OffloadRewriteConsumer::HandleTranslationUnit(clang::ASTContext &C) {
  auto &D = CI.getDiagnostics();
  for (auto r : outlinePolyregionOffload(C))
    r ^ foreach_total(
            [&](const Failure &f) { //
              emit(D, f.callExpr->getBeginLoc(), clang::DiagnosticsEngine::Warning, POLYREGION_DIAG_POLYSTL "Outline failed: %0", f.reason);
            },
            [&](const Callsite &c) { //
              const SpecialisationPathVisitor spv(C);
              const auto specialisationPath = spv.resolve(c.calleeDecl) ^ reverse();
              auto moduleId = specialisationPath | values() | mk_string("->", [&](auto callExpr) {
                                const auto l = getLocation(*callExpr, C);
                                std::string name;
                                name += "<";
                                name += l.filename;
                                name += ":";
                                name += std::to_string(l.line);
                                name += ">";
                                return name;
                              });
              // Source locations alone collide across template instantiations (miniBUDE's
              // `fasten_main<PPWI>` has the same line numbers for every PPWI). The runtime's
              // flat moduleName->object map keeps the first kernel and silently mis-dispatches
              // the rest. Disambiguate with the lambda's CXXRecordDecl ID.
              moduleId += fmt::format("@{:x}", c.functorDecl->getParent()->getID());

              const auto bundle = compileRegion(
                  opts, C, D, moduleId, *c.functorDecl,
                  specialisationPath ^ head_maybe() ^
                      fold([](auto, auto callExpr) { return callExpr->getExprLoc(); }, [&] { return c.callLambdaArgExpr->getExprLoc(); }),
                  c.kind);

              if (opts.verbose) {
                emit(D, c.callLambdaArgExpr->getExprLoc(), clang::DiagnosticsEngine::Remark,
                     POLYREGION_DIAG_POLYSTL "Outlined function: %0 for %1 (%2)\n", moduleId, std::string(magic_enum::enum_name(c.kind)),
                     (bundle.objects //
                      | map([](auto &o) {
                          return std::string(magic_enum::enum_name(o.format)) + "=" +
                                 std::to_string(static_cast<float>(o.moduleImage.size()) / 1000) + "KB";
                        }) //
                      | mk_string(", ")));
              }

              insertKernelImage(D, CI.getSema(), C, c, bundle);

            });
}
