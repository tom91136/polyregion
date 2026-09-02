#include "rewriter.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MD5.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyfront/package.hpp"
#include "polyfront/package_program.hpp"
#include "polyfront/polyc_client.hpp"
#include "polyregion/conventions.h"
#include "polyregion/program_import.hpp"

#include "ast.h"
#include "ast_visitors.h"
#include "clang_utils.h"
#include "codegen.h"
#include "remapper.h"

using namespace polyregion::polystl;
using namespace polyregion;
using polyregion::polyfront::emit;
using namespace polyregion::polyast;
using namespace polyregion::polyast::dsl;
using namespace aspartame;

namespace {

constexpr static llvm::StringLiteral packageExportPrefix = "polyregion_export:";
constexpr static llvm::StringLiteral packageExportTemplatePrefix = "polyregion_export_template:";
constexpr static llvm::StringLiteral packageImplementsPrefix = "polyregion_implements:";
constexpr static llvm::StringLiteral packageRequiresPrefix = "polyregion_requires:";
constexpr static llvm::StringLiteral typeVariablePrefix = "polyregion_type_variable:";

static std::optional<unsigned> packageTypeWidth(const clang::TemplateArgument &argument) {
  if (argument.getKind() != clang::TemplateArgument::Type) return {};
  const auto *record = argument.getAsType()->getAs<clang::RecordType>();
  if (!record) return {};
  for (const auto *attr : record->getDecl()->attrs()) {
    const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attr);
    if (!annotation) continue;
    const auto value = annotation->getAnnotation();
    if (!value.starts_with(typeVariablePrefix)) continue;
    constexpr llvm::StringLiteral marker = ":size=";
    const auto offset = value.find(marker);
    unsigned width = 0;
    if (offset == llvm::StringRef::npos || value.drop_front(offset + marker.size()).getAsInteger(10, width) || width == 0) return {};
    return width;
  }
  return {};
}

struct PackageExportMetadata {
  std::optional<Sym> name;
  std::optional<Sym> implements;
  Vector<std::string> requiredCapabilities;
  bool invalid;
};

static PackageExportMetadata packageExportMetadata(const clang::FunctionDecl *decl, clang::DiagnosticsEngine &diag) {
  bool invalid = false;
  Vector<Sym> explicitNames;
  Vector<Sym> templateNames;
  Vector<Sym> implements;
  Vector<std::string> requiredCapabilities;
  for (const auto *attr : decl->attrs()) {
    const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attr);
    if (!annotation) continue;
    const auto value = annotation->getAnnotation();
    const auto parseSym = [&](const llvm::StringLiteral prefix, Vector<Sym> &out, const std::string_view kind) {
      if (!value.starts_with(prefix)) return false;
      const auto parts = value.drop_front(prefix.size()).str() ^ split('.');
      if (parts.empty() || parts ^ exists([](const auto &part) { return part.empty(); })) {
        emit(diag, annotation->getLocation(), clang::DiagnosticsEngine::Level::Error,
             POLYREGION_DIAG_POLYSTL "Malformed package %0 identity: %1", std::string(kind), value.str());
        invalid = true;
      } else out.emplace_back(parts);
      return true;
    };
    if (parseSym(packageExportPrefix, explicitNames, "export")) continue;
    if (parseSym(packageExportTemplatePrefix, templateNames, "export template")) continue;
    if (parseSym(packageImplementsPrefix, implements, "implementation")) continue;
    if (!value.starts_with(packageRequiresPrefix)) continue;
    const auto capability = value.drop_front(packageRequiresPrefix.size());
    if (capability.empty()) {
      emit(diag, annotation->getLocation(), clang::DiagnosticsEngine::Level::Error,
           POLYREGION_DIAG_POLYSTL "Malformed package capability: %0", value.str());
      invalid = true;
    } else requiredCapabilities.emplace_back(capability.str());
  }

  const auto names = explicitNames | distinct() | to_vector();
  const auto templates = templateNames | distinct() | to_vector();
  const auto implementationNames = implements | distinct() | to_vector();
  if (names.size() > 1) {
    emit(diag, decl->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
         POLYREGION_DIAG_POLYSTL "Conflicting package export identities: %0",
         names | map([](const auto &name) { return fqcn(name); }) | mk_string(", "));
    invalid = true;
  }
  if (templates.size() > 1 || (!names.empty() && !templates.empty())) {
    emit(diag, decl->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
         POLYREGION_DIAG_POLYSTL "Conflicting package export identities on template: %0",
         templates | map([](const auto &name) { return fqcn(name); }) | mk_string(", "));
    invalid = true;
  }
  if (implementationNames.size() > 1) {
    emit(diag, decl->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
         POLYREGION_DIAG_POLYSTL "Conflicting package implementation identities: %0",
         implementationNames | map([](const auto &name) { return fqcn(name); }) | mk_string(", "));
    invalid = true;
  }
  const auto implementation = implementationNames.empty() ? std::optional<Sym>{} : std::optional<Sym>{implementationNames.front()};
  const auto capabilities = requiredCapabilities | distinct() | to_vector();
  if (invalid) return {{}, {}, {}, true};
  if (!names.empty()) return {names.front(), implementation, capabilities, false};
  if (!templates.empty()) {
    const auto *arguments = decl->getTemplateSpecializationArgs();
    if (!arguments) return {{}, implementation, capabilities, false};
    auto name = templates.front();
    bool foundWidth = false;
    for (const auto &argument : arguments->asArray())
      if (const auto width = packageTypeWidth(argument)) {
        name.fqn.back() += "_w" + std::to_string(*width);
        foundWidth = true;
      }
    if (!foundWidth) {
      emit(diag, decl->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
           POLYREGION_DIAG_POLYSTL "Package export template specialization has no sized type-variable argument: %0",
           decl->getQualifiedNameAsString());
      return {{}, {}, {}, true};
    }
    return {name, implementation, capabilities, false};
  }
  return {{}, implementation, capabilities, false};
}

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
constexpr static llvm::StringLiteral interfaceAnnotation = "polyregion_interface:";

struct InterfaceSite {
  clang::CallExpr *call;
  clang::FunctionDecl *callee;
  std::string packageName;
  std::string declaration;
};

static std::optional<std::pair<std::string, std::string>> interfaceIdentity(const clang::FunctionDecl *decl) {
  return decl->attrs() | collect_first([](const auto *attr) -> std::optional<std::pair<std::string, std::string>> {
           const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attr);
           if (!annotation || !annotation->getAnnotation().starts_with(interfaceAnnotation)) return {};
           const auto payload = annotation->getAnnotation().drop_front(interfaceAnnotation.size());
           const auto [packageName, declaration] = payload.split(':');
           if (!packageName.empty() && !declaration.empty() && !declaration.contains(':'))
             return std::pair(packageName.str(), declaration.str());
           return {};
         });
}

static Vector<InterfaceSite> interfaceSites(clang::ASTContext &context) {
  using namespace clang::ast_matchers;
  Vector<InterfaceSite> results;
  runMatch(
      context,
      [&](const MatchFinder::MatchResult &result) {
        const auto *call = result.Nodes.getNodeAs<clang::CallExpr>("interfaceCall");
        const auto *callee = result.Nodes.getNodeAs<clang::FunctionDecl>("interfaceDecl");
        if (call && callee)
          if (const auto identity = interfaceIdentity(callee))
            results.emplace_back(InterfaceSite{const_cast<clang::CallExpr *>(call), const_cast<clang::FunctionDecl *>(callee),
                                               identity->first, identity->second});
      },
      callExpr(callee(functionDecl(hasAttr(clang::attr::Annotate)).bind("interfaceDecl"))).bind("interfaceCall"));
  return results;
}

static Vector<const clang::FunctionDecl *> interfaceCallableIdentities(const InterfaceSite &site) {
  Vector<const clang::FunctionDecl *> identities;
  identities.reserve(site.call->getNumArgs());
  for (size_t i = 0; i < site.call->getNumArgs(); ++i) {
    const auto *argument = site.call->getArg(i)->IgnoreUnlessSpelledInSource();
    const auto *record = argument->getType()->getAsCXXRecordDecl();
    const clang::FunctionDecl *callable = nullptr;
    if (record) {
      if (record->isLambda()) callable = record->getLambdaCallOperator();
      else {
        const auto methods = record->methods() | filter([](const auto *method) {
                               return method->getOverloadedOperator() == clang::OO_Call && method->doesThisDeclarationHaveABody();
                             })
                             | to_vector();
        if (methods.size() == 1) callable = methods.front();
      }
    } else if (argument->getType()->isFunctionPointerType()) {
      const clang::Expr *referenced = argument->IgnoreParenImpCasts();
      if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(referenced); address && address->getOpcode() == clang::UO_AddrOf)
        referenced = address->getSubExpr()->IgnoreParenImpCasts();
      if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(referenced)) callable = llvm::dyn_cast<clang::FunctionDecl>(ref->getDecl());
    }
    identities.emplace_back(callable ? callable->getCanonicalDecl() : nullptr);
  }
  return identities;
}

static Opt<std::string> interfaceCallableShapeError(const InterfaceSite &site) {
  for (size_t i = 0; i < site.call->getNumArgs(); ++i) {
    const auto *argument = site.call->getArg(i)->IgnoreUnlessSpelledInSource();
    const auto *record = argument->getType()->getAsCXXRecordDecl();
    if (!record) continue;
    if (record->isLambda()) {
      if (record->capture_size() != 0) return std::string("capturing library callables are not supported");
      continue;
    }
    bool callable = false;
    for (const auto *method : record->methods())
      if (method->getOverloadedOperator() == clang::OO_Call && method->doesThisDeclarationHaveABody()) {
        callable = true;
        break;
      }
    if (callable && !record->isEmpty()) return std::string("stateful library callables are not supported");
  }
  return {};
}

static std::string interfaceKey(const clang::CompilerInstance &CI, const clang::ASTContext &C, const InterfaceSite &site) {
  llvm::MD5 md5;
  md5.update(CI.getFrontendOpts().Inputs.empty() ? llvm::StringRef{} : CI.getFrontendOpts().Inputs.front().getFile());
  md5.update(site.packageName);
  md5.update(site.declaration);
  md5.update(site.callee->getQualifiedNameAsString());
  md5.update(site.callee->getType().getAsString(C.getPrintingPolicy()));
  const auto location = C.getSourceManager().getExpansionLoc(site.callee->getLocation());
  md5.update(std::to_string(C.getSourceManager().getFileOffset(location)));
  return md5.final().digest().str().str();
}

struct PreparedInterfaceCall {
  InterfaceSite site;
  polyast::Package pkg;
  std::vector<Type::Any> argumentTypes;
  std::vector<ProgramTypeSize> typeSizes;
  std::vector<Function> consumerFunctions;
  std::vector<StructDef> consumerDefinitions;
  std::string suffix;
  std::string entryName;
};

static std::optional<PreparedInterfaceCall> prepareInterfaceCall(clang::CompilerInstance &CI, clang::ASTContext &C,
                                                                 const InterfaceSite &site, const polyast::Package &pkg) {
  auto &D = CI.getDiagnostics();
  Remapper remapper(C);
  Remapper::RemapContext context;
  std::vector<Type::Any> argumentTypes;
  std::vector<ProgramTypeSize> typeSizes;
  for (const auto *parameter : site.callee->parameters()) {
    const auto type = remapper.handleType(parameter->getType(), context);
    argumentTypes.emplace_back(type);
    const auto sized = parameter->getType()->isPointerType() ? parameter->getType()->getPointeeType() : parameter->getType();
    const auto sizeKey = type.get<Type::Ptr>() ? type.get<Type::Ptr>()->comp : type;
    const auto sizeInBytes = static_cast<int32_t>(C.getTypeSize(sized) / 8);
    if (!sizeKey.is<Type::Nothing>() && sizeInBytes > 0) typeSizes.emplace_back(sizeKey, sizeInBytes);
  }
  for (size_t i = 0; i < site.call->getNumArgs(); ++i) {
    const auto *argument = site.call->getArg(i)->IgnoreUnlessSpelledInSource();
    const auto *record = argument->getType()->getAsCXXRecordDecl();
    const clang::FunctionDecl *callable = nullptr;
    if (record) {
      if (record->isLambda()) callable = record->getLambdaCallOperator();
      else {
        const auto methods = record->methods() | filter([](const auto *method) {
                               return method->getOverloadedOperator() == clang::OO_Call && method->doesThisDeclarationHaveABody();
                             })
                             | to_vector();
        if (methods.size() > 1) {
          emit(D, argument->getExprLoc(), clang::DiagnosticsEngine::Error,
               POLYREGION_DIAG_POLYSTL "library callable has an ambiguous operator()");
          return {};
        }
        callable = methods | head_maybe() | get_or_else(static_cast<clang::CXXMethodDecl *>(nullptr));
      }
    } else if (argument->getType()->isFunctionPointerType()) {
      const clang::Expr *referenced = argument->IgnoreParenImpCasts();
      if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(referenced); address && address->getOpcode() == clang::UO_AddrOf)
        referenced = address->getSubExpr()->IgnoreParenImpCasts();
      if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(referenced)) callable = llvm::dyn_cast<clang::FunctionDecl>(ref->getDecl());
    }
    if (!callable) continue;
    if (record && record->isLambda() && record->capture_size() != 0) {
      emit(D, argument->getExprLoc(), clang::DiagnosticsEngine::Error,
           POLYREGION_DIAG_POLYSTL "capturing library callables are not supported");
      return {};
    }
    if (record && !record->isLambda() && !record->isEmpty()) {
      emit(D, argument->getExprLoc(), clang::DiagnosticsEngine::Error,
           POLYREGION_DIAG_POLYSTL "stateful library callables are not supported");
      return {};
    }
    if (callable->isTemplated()) {
      emit(D, argument->getExprLoc(), clang::DiagnosticsEngine::Error,
           POLYREGION_DIAG_POLYSTL "generic library callables require a concrete operator() specialization");
      return {};
    }
    auto [name, function] = remapper.handleCall(callable, context);
    if (record && (function->collect_all<Term::Select>() ^ exists([](const auto &select) {
                     return select.root.symbol == conventions::ThisReceiver;
                   }))) {
      emit(D, argument->getExprLoc(), clang::DiagnosticsEngine::Error,
           POLYREGION_DIAG_POLYSTL "library callable operator() depends on its receiver");
      return {};
    }
    auto args = function->decl.args;
    if (!args.empty() && args.front().named.symbol == conventions::ThisReceiver) args.erase(args.begin());
    function->decl = function->decl.withArgs(args).withAffinity(FunctionAffinity::Host());
    argumentTypes[i] = Type::FnRef(function->decl.name);
  }
  const auto returnType = remapper.handleType(site.callee->getReturnType(), context);
  if (!site.callee->getReturnType()->isVoidType())
    typeSizes.emplace_back(returnType, static_cast<int32_t>(C.getTypeSize(site.callee->getReturnType()) / 8));
  const auto publicName = Sym(site.declaration ^ split('.'));
  const auto suffix = interfaceKey(CI, C, site);
  const auto entryName = "__polyregion_package_program_" + suffix;
  auto callerFns = context.functions | values() | map([](const auto &function) { return *function; }) | to_vector();
  const auto callerDefs = context.structs | values() | map([](const auto &definition) { return *definition; }) | to_vector();
  callerFns.emplace_back(program::importRoot(entryName, publicName, argumentTypes, returnType));
  return PreparedInterfaceCall{site,   pkg,      std::move(argumentTypes), std::move(typeSizes), std::move(callerFns), callerDefs,
                               suffix, entryName};
}

struct InterfaceModuleResource {
  clang::VarDecl *image;
  clang::VarDecl *features;
};

static void materializeInterfaceCall(clang::CompilerInstance &CI, clang::ASTContext &C, const PreparedInterfaceCall &prepared,
                                     const polyast::CompileBundle &compiled, const std::vector<InterfaceModuleResource> &resources) {
  const auto &site = prepared.site;
  const auto &argumentTypes = prepared.argumentTypes;
  const auto &suffix = prepared.suffix;
  const auto &entryName = prepared.entryName;
  auto &S = CI.getSema();
  std::vector<clang::QualType> entryTypes;
  std::vector<clang::Expr *> entryArgs;
  auto *contextFn = mkExternCFn(C, "polyrt_context_current", C.VoidPtrTy, {});
  auto *contextArg = mkCall(C, contextFn, {});
  entryTypes.emplace_back(C.VoidPtrTy);
  entryArgs.emplace_back(contextArg);
  clang::VarDecl *resultDecl = nullptr;
  for (size_t index = 0; index < argumentTypes.size(); ++index) {
    if (program::importArgumentMode(argumentTypes[index]) == program::ImportArgumentMode::OmittedCallable) continue;
    auto *parameter = site.callee->getParamDecl(index);
    if (program::importArgumentMode(argumentTypes[index]) == program::ImportArgumentMode::DirectPointer) {
      entryTypes.emplace_back(parameter->getType());
      entryArgs.emplace_back(mkLoad(C, parameter));
    } else {
      entryTypes.emplace_back(C.getPointerType(parameter->getType()));
      entryArgs.emplace_back(S.CreateBuiltinUnaryOp({}, clang::UO_AddrOf, mkDeclRef(C, parameter)).get());
    }
  }
  if (!site.callee->getReturnType()->isVoidType()) {
    const auto type = site.callee->getReturnType();
    resultDecl = clang::VarDecl::Create(C, site.callee, {}, {}, &C.Idents.get("__polyregion_interface_result"), type,
                                        C.getTrivialTypeSourceInfo(type), clang::SC_None);
    entryTypes.emplace_back(C.getPointerType(type));
    entryArgs.emplace_back(S.CreateBuiltinUnaryOp({}, clang::UO_AddrOf, mkDeclRef(C, resultDecl)).get());
  }
  auto *entry = mkExternCFn(C, entryName, C.VoidTy, entryTypes);
  auto *contextAcquire = mkExternCFn(C, "polyrt_context_acquire", C.VoidTy, {C.VoidPtrTy});
  auto *contextRelease = mkExternCFn(C, "polyrt_context_release", C.VoidTy, {C.VoidPtrTy});
  std::vector<clang::Stmt *> statements;
  if (resultDecl) statements.emplace_back(new (C) clang::DeclStmt(clang::DeclGroupRef(resultDecl), {}, {}));
  statements.emplace_back(mkCall(C, contextAcquire, {contextArg}));
  const auto protectedBegin = statements.size();
  auto *remoteLoad = mkExternCFn(C, "polyrt_remote_load", C.BoolTy,
                                 {C.VoidPtrTy, constCharStarTy(C), C.IntTy, C.IntTy, C.getSizeType(), C.getPointerType(constCharStarTy(C)),
                                  C.getSizeType(), C.getPointerType(C.getConstType(C.UnsignedCharTy))});
  std::map<std::string, clang::VarDecl *> loadedModules;
  for (size_t index = 0; index < compiled.remoteModules.size(); ++index) {
    const auto &object = compiled.remoteModules[index];
    const auto &moduleName = object.moduleName;
    auto [loadedEntry, inserted] = loadedModules.try_emplace(moduleName, nullptr);
    if (inserted) {
      auto *loaded = clang::VarDecl::Create(
          C, site.callee, {}, {}, &C.Idents.get("__polyregion_package_loaded_" + suffix + "_" + std::to_string(loadedModules.size() - 1)),
          C.BoolTy, C.getTrivialTypeSourceInfo(C.BoolTy), clang::SC_None);
      loaded->setInit(new (C) clang::CXXBoolLiteralExpr(false, C.BoolTy, {}));
      loadedEntry->second = loaded;
      statements.emplace_back(new (C) clang::DeclStmt(clang::DeclGroupRef(loaded), {}, {}));
    }
    auto *image = resources[index].image;
    clang::Expr *featureData = mkNullPtrLit(C, constCharStarTy(C));
    if (!object.features.empty()) {
      auto *features = resources[index].features;
      featureData = mkArrayToPtrDecay(C, C.getPointerType(constCharStarTy(C)), mkDeclRef(C, features));
    }
    auto *load =
        mkCall(C, remoteLoad,
               {contextArg, mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, moduleName)),
                mkIntLit(C, C.IntTy, static_cast<int>(object.kind)), mkIntLit(C, C.IntTy, static_cast<int>(object.format)),
                mkIntLit(C, C.getSizeType(), object.features.size()), featureData, mkIntLit(C, C.getSizeType(), object.image.size()),
                mkArrayToPtrDecay(C, C.getPointerType(C.getConstType(C.UnsignedCharTy)), mkDeclRef(C, image))});
    auto *loaded = loadedEntry->second;
    auto *anyLoaded = S.CreateBuiltinBinOp({}, clang::BO_LOr, load, mkLoad(C, loaded)).get();
    statements.emplace_back(S.CreateBuiltinBinOp({}, clang::BO_Assign, mkDeclRef(C, loaded), anyLoaded).get());
  }
  auto *requireLoaded = mkExternCFn(C, "polyrt_remote_require_loaded", C.VoidTy, {C.VoidPtrTy, constCharStarTy(C), C.BoolTy});
  for (const auto &[moduleName, loaded] : loadedModules) {
    statements.emplace_back(
        mkCall(C, requireLoaded, {contextArg, mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, moduleName)), mkLoad(C, loaded)}));
  }
  statements.emplace_back(mkCall(C, entry, entryArgs));
  std::vector<clang::Stmt *> protectedStatements(statements.begin() + protectedBegin, statements.end());
  statements.erase(statements.begin() + protectedBegin, statements.end());
  auto *release = mkCall(C, contextRelease, {contextArg});
  auto *rethrow = new (C) clang::CXXThrowExpr(nullptr, C.VoidTy, {}, false);
  const std::vector<clang::Stmt *> handlerStatements{release, rethrow};
  auto *handlerBody = clang::CompoundStmt::Create(C, handlerStatements, {}, {}, {});
  auto *handler = new (C) clang::CXXCatchStmt({}, nullptr, handlerBody);
  auto *tryBody = clang::CompoundStmt::Create(C, protectedStatements, {}, {}, {});
  const std::vector<clang::Stmt *> handlers{handler};
  statements.emplace_back(clang::CXXTryStmt::Create(C, {}, tryBody, handlers));
  statements.emplace_back(mkCall(C, contextRelease, {contextArg}));
  if (resultDecl) statements.emplace_back(clang::ReturnStmt::Create(C, {}, mkLoad(C, resultDecl), nullptr));
  site.callee->setBody(clang::CompoundStmt::Create(C, statements, {}, {}, {}));
}

static void compileInterfaceCalls(const polyfront::Options &opts, clang::CompilerInstance &CI, clang::ASTContext &C,
                                  const std::vector<PreparedInterfaceCall> &prepared, std::vector<int8_t> &packageProgramBitcode) {
  if (prepared.empty()) return;
  auto &D = CI.getDiagnostics();
  const auto entryTarget = polyfront::objectTargetFor(CI.getTarget().getTriple());
  if (!entryTarget) {
    emit(D, prepared.front().site.call->getExprLoc(), clang::DiagnosticsEngine::Error,
         POLYREGION_DIAG_POLYSTL "linked package program does not support target architecture: %0",
         CI.getTarget().getTriple().getArchName());
    return;
  }
  std::vector<Package> packages;
  std::vector<Function> functions;
  std::vector<StructDef> definitions;
  std::vector<ProgramTypeSize> typeSizes;
  const auto appendDistinct = []<typename T>(std::vector<T> &target, const std::vector<T> &values) {
    for (const auto &value : values)
      if (std::find(target.begin(), target.end(), value) == target.end()) target.emplace_back(value);
  };
  for (const auto &item : prepared) {
    if (std::find(packages.begin(), packages.end(), item.pkg) == packages.end()) packages.emplace_back(item.pkg);
    appendDistinct(functions, item.consumerFunctions);
    appendDistinct(definitions, item.consumerDefinitions);
    appendDistinct(typeSizes, item.typeSizes);
  }
  const auto capabilities = std::vector<std::string>(opts.libraryCapabilities.begin(), opts.libraryCapabilities.end());
  const auto request = ProgramLinkRequest(std::move(packages), polyfront::packageProgram(std::move(functions), std::move(definitions)),
                                          capabilities, std::move(typeSizes));
  const auto &targetCPU = CI.getTarget().getTargetOpts().CPU;
  auto compiled =
      polyfront::package::compileProgram(request, opts.executable, *entryTarget,
                                         polyfront::objectCPUFor(CI.getTarget().getTriple(), targetCPU), opts.targets, opts.stackDepth);
  if (!compiled) {
    emit(D, prepared.front().site.call->getExprLoc(), clang::DiagnosticsEngine::Error, POLYREGION_DIAG_POLYSTL "%0",
         compiled.errors ^ mk_string("; "));
    return;
  }
  packageProgramBitcode = compiled.value->hostObject;
  std::vector<InterfaceModuleResource> resources;
  resources.reserve(compiled.value->remoteModules.size());
  auto *translationUnit = C.getTranslationUnitDecl();
  const auto &batchSuffix = prepared.front().suffix;
  for (size_t index = 0; index < compiled.value->remoteModules.size(); ++index) {
    const auto &object = compiled.value->remoteModules[index];
    auto *image = mkStaticVarDecl(C, translationUnit, "__polyregion_package_image_" + batchSuffix + "_" + std::to_string(index),
                                  mkConstArrTy(C, C.UnsignedCharTy, object.image.size()),
                                  object.image | map([&](const auto byte) -> clang::Expr * {
                                    return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast,
                                                                           mkIntLit(C, C.IntTy, static_cast<unsigned char>(byte)), nullptr,
                                                                           clang::VK_PRValue, {});
                                  }) | to_vector());
    translationUnit->addDecl(image);
    CI.getASTConsumer().HandleTopLevelDecl(clang::DeclGroupRef(image));
    clang::VarDecl *features = nullptr;
    if (!object.features.empty()) {
      features = mkStaticVarDecl(C, translationUnit, "__polyregion_package_features_" + batchSuffix + "_" + std::to_string(index),
                                 mkConstArrTy(C, constCharStarTy(C), object.features.size()),
                                 object.features | map([&](const auto &feature) -> clang::Expr * {
                                   return mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, feature));
                                 }) | to_vector());
      translationUnit->addDecl(features);
      CI.getASTConsumer().HandleTopLevelDecl(clang::DeclGroupRef(features));
    }
    resources.emplace_back(InterfaceModuleResource{image, features});
  }
  for (const auto &item : prepared)
    materializeInterfaceCall(CI, C, item, *compiled.value, resources);
}

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
                    templateArg0.getKind() == clang::TemplateArgument::Integral
                    && templateArg0.getIntegralType()->getAsTagDecl()->getName().str() == "PlatformKind") {
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
      return decl->fields() | find([&](const auto &f) { return f->getName() == fieldName; });
    }
    emit(D, clang::DiagnosticsEngine::Error, POLYREGION_DIAG_POLYSTL "Type %0 cannot be resolved to a CXXRecordDecl. This is a bug.", ty);
    return {};
  };

  const auto typeOfFieldWithName = [&](clang::QualType ty, const auto &fieldName) -> Opt<clang::QualType> {
    return fieldWithName(ty, fieldName) ^ map([&](const auto &f) { return f->getType().getDesugaredType(C); });
  };

  const auto KernelBundleTy = c.calleeDecl->getReturnType()->getPointeeType();
  const auto KernelObjectTy = typeOfFieldWithName(KernelBundleTy, "objects") ^ map([](const auto &t) { return t->getPointeeType(); });
  const auto PlatformKindTy = typeOfFieldWithName(*KernelObjectTy, "kind");
  const auto ModuleFormatTy = typeOfFieldWithName(*KernelObjectTy, "format");
  const auto TargetTy = typeOfFieldWithName(*KernelObjectTy, "target");
  const auto OptLevelTy = typeOfFieldWithName(*KernelObjectTy, "opt");
  const auto TypeLayoutTy = typeOfFieldWithName(KernelBundleTy, "structs") ^ map([](const auto &t) { return t->getPointeeType(); });
  const auto AggregateMemberTy = typeOfFieldWithName(*TypeLayoutTy, "members") ^ map([](const auto &t) { return t->getPointeeType(); });
  const auto TypeLayoutMembersField = fieldWithName(*TypeLayoutTy, "members");

  auto kernelImageDecls =
      bundle.objects     //
      | zip_with_index() //
      | map([&](const auto &ko, const auto &idx) {
          return mkStaticVarDecl(
              C, c.calleeDecl, fmt::format("__ko_image_data_{}", idx), mkConstArrTy(C, C.UnsignedCharTy, ko.moduleImage.size()),
              ko.moduleImage ^ map([&](const auto &x) -> clang::Expr * {
                return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast,
                                                       mkIntLit(C, C.IntTy, static_cast<unsigned char>(x)), nullptr, clang::VK_PRValue, {});
              }));
        }) //
      | to_vector();

  auto kernelFeatureDecls =
      bundle.objects     //
      | zip_with_index() //
      | map([&](const auto &ko, const auto &idx) {
          return mkStaticVarDecl(C, c.calleeDecl, fmt::format("__ko_feature_data_{}", idx),
                                 mkConstArrTy(C, constCharStarTy(C), ko.features.size()),
                                 ko.features ^ map([&](const auto &feature) -> clang::Expr * {
                                   return mkArrayToPtrDecay(C, C.getConstType(C.getPointerType(C.CharTy)), mkStrLit(C, feature));
                                 }));
        }) //
      | to_vector();

  Opt<clang::VarDecl *> kernelProgramDecl;
  if (!bundle.program.empty()) {
    kernelProgramDecl = mkStaticVarDecl(C, c.calleeDecl, "__ko_program_data", //
                                        mkConstArrTy(C, C.UnsignedCharTy, bundle.program.size()),
                                        bundle.program ^ map([&](const auto &x) -> clang::Expr * {
                                          return clang::ImplicitCastExpr::Create(C, C.UnsignedCharTy, clang::CK_IntegralCast,
                                                                                 mkIntLit(C, C.IntTy, static_cast<unsigned char>(x)),
                                                                                 nullptr, clang::VK_PRValue, {});
                                        }));
  }

  auto kernelObjectArrayDecl = mkStaticVarDecl(
      C, c.calleeDecl,                                                  //
      "__ko_data",                                                      //
      mkConstArrTy(C, *KernelObjectTy, bundle.objects.size()),          //
      bundle.objects                                                    //
          | zip_with_index()                                            //
          | map([&](const auto &ko, const auto &idx) -> clang::Expr * { //
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

  auto table = bundle.layouts | values() | map([&](const auto &sl) { return std::pair{Type::Struct(Sym({sl.name}), {}), sl}; }) | to<Map>();

  auto primitiveTypeLayoutsDecls =
      Vector<Type::Any>{
          Type::Float16(), Type::Float32(), Type::Float64(),                 //
          Type::IntU8(),   Type::IntU16(),  Type::IntU32(),  Type::IntU64(), //
          Type::IntS8(),   Type::IntS16(),  Type::IntS32(),  Type::IntS64(), //
          Type::Unit0(),   Type::Bool1(),                                    //
      } //
      | collect([&](const auto &t) {
          return primitiveSize(t) ^ map([&](const auto &sizeInBytes) {
                   return std::pair{
                       t, mkStaticVarDecl(C, c.calleeDecl, fmt::format("__primitive_type_layout_{}", canonicalName(t)), *TypeLayoutTy,
                                          {
                                              /*name        */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, canonicalName(t))),
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
                      bundle.layouts | map([&](const auto &, const auto &sl) -> clang::Expr * {
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

  auto structNameToTypeLayoutIdx = bundle.layouts | values() | map([](const auto &sl) { return sl.name; }) | zip_with_index() | to<Map>();

  auto aggregateMemberArrayDecls = //
      bundle.layouts | values() | zip_with_index() | map([&](const auto &sl, const auto &idx) {
        return std::pair{
            sl.name,
            mkStaticVarDecl(
                C, c.calleeDecl,                                        //
                fmt::format("__aggregate_member_{}", idx),              //
                mkConstArrTy(C, *AggregateMemberTy, sl.members.size()), //
                sl.members                                              //
                    | map([&](const auto &m) -> clang::Expr * {         //
                        const auto [indirections, componentSize] = countIndirectionsAndComponentSize(m.name.tpe, table);
                        const auto typeDecl =
                            extractComponent(m.name.tpe) ^ flat_map([&](const auto &t) {
                              return primitiveTypeLayoutsDecls                      //
                                     ^ get_maybe(t)                                 //
                                     ^ map([&](const auto &decl) -> clang::Expr * { //
                                         return S.CreateBuiltinUnaryOp({}, clang::UnaryOperatorKind::UO_AddrOf, mkDeclRef(C, decl))
                                             .get(); //
                                       })            //
                                     ^ or_else([&]() {
                                         return t.template get<Type::Struct>() ^ flat_map([&](const auto &s) {
                                                  return structNameToTypeLayoutIdx //
                                                         ^ get_maybe(fqcn(s.name)) //
                                                         ^ map([&](const auto &layoutIdx) {
                                                             return S
                                                                 .CreateBuiltinBinOp({}, clang::BinaryOperatorKind::BO_Add,
                                                                                     mkDeclRef(C, structTypeLayoutArrayDecl),
                                                                                     mkIntLit(C, C.getSizeType(), layoutIdx))
                                                                 .get();
                                                           });
                                                });
                                       });
                            });

                        const bool readOnly = bundle.readOnlyMembers //
                                              ^ get_maybe(sl.name)   //
                                              ^ exists([&](const auto &ms) { return ms ^ contains(m.name.symbol); });
                        return mkInitList(C,
                                          *AggregateMemberTy,                                                                         //
                                          {/*name            */ mkArrayToPtrDecay(C, constCharStarTy(C), mkStrLit(C, m.name.symbol)), //
                                           /*offsetInBytes   */ mkIntLit(C, C.getSizeType(), m.offsetInBytes),                        //
                                           /*sizeInBytes     */ mkIntLit(C, C.getSizeType(), m.sizeInBytes),                          //
                                           /*ptrIndirections */ mkIntLit(C, C.getSizeType(), indirections),                           //
                                           /*componentSize   */ mkIntLit(C, C.getSizeType(), componentSize.value_or(m.sizeInBytes)),  //
                                           /*type            */ typeDecl ^ get_or_else(mkNullPtrLit(C, *TypeLayoutTy)),               //
                                           /*readOnly        */ mkIntLit(C, C.getSizeType(), readOnly ? 1 : 0)});
                      })
                    | to_vector())};
      }) //
      | to<Map>();

  auto assignTypeLayoutMembers =
      structNameToTypeLayoutIdx //
      ^ to_vector()             //
      ^ map([&](const auto &name, const auto &idx) -> clang::Stmt * {
          const auto typeLayoutExpr = new (C) clang::ArraySubscriptExpr(
              mkArrayToPtrDecay(C, TypeLayoutTyNoConst, mkDeclRef(C, structTypeLayoutArrayDecl)), mkIntLit(C, C.getSizeType(), idx),
              TypeLayoutTyNoConst, clang::ExprValueKind::VK_LValue, clang::ExprObjectKind::OK_Ordinary, {});
          const auto lhs = mkMemberExpr(C, typeLayoutExpr, *TypeLayoutMembersField);
          const auto rhs = mkArrayToPtrDecay(C, C.getPointerType(*AggregateMemberTy),
                                             aggregateMemberArrayDecls //
                                                 ^ get_maybe(name)     //
                                                 ^ fold([&](const auto &d) -> clang::Expr * { return mkDeclRef(C, d); },
                                                        [&]() -> clang::Expr * { return mkNullPtrLit(C, *AggregateMemberTy); }));
          return S.CreateBuiltinBinOp({}, clang::BinaryOperatorKind::BO_Assign, lhs, rhs).get();
        });

  auto interfaceLayoutIdx = bundle.layouts | index_where([&](const auto &exported, const auto &) { return exported; });

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
  Vector<clang::Stmt *> newStmts =                                                                                       //
      kernelProgramDecl                                                                                                  //
      | to_vector()                                                                                                      //
      | concat(hostBcDecl)                                                                                               //
      | concat(kernelImageDecls)                                                                                         //
      | concat(primitiveTypeLayoutsDecls | values())                                                                     //
      | append(structTypeLayoutArrayDecl)                                                                                //
      | concat(aggregateMemberArrayDecls | values())                                                                     //
      | concat(kernelFeatureDecls)                                                                                       //
      | append(kernelObjectArrayDecl)                                                                                    //
      | append(kernelBundleDecl)                                                                                         //
      | map([&](const auto &dcl) -> clang::Stmt * { return new (C) clang::DeclStmt(clang::DeclGroupRef(dcl), {}, {}); }) //
      | concat(assignTypeLayoutMembers)                                                                                  //
      | append(clang::ReturnStmt::Create(C, {}, mkDeclRef(C, kernelBundleDecl), {}))                                     //
      | to_vector();

  c.calleeDecl->setBody(clang::CompoundStmt::Create(C, newStmts, {}, {}, {}));
}

namespace polyregion::polystl {
class OffloadRewriteConsumer final : public clang::ASTConsumer {
  clang::CompilerInstance &CI;
  polyfront::Options opts;
  std::shared_ptr<std::vector<int8_t>> packageProgramBitcode;

public:
  OffloadRewriteConsumer(clang::CompilerInstance &CI, const polyfront::Options &opts,
                         std::shared_ptr<std::vector<int8_t>> packageProgramBitcode);
  void HandleTranslationUnit(clang::ASTContext &C) override;
};
} // namespace polyregion::polystl

OffloadRewriteConsumer::OffloadRewriteConsumer(clang::CompilerInstance &CI, const polyfront::Options &opts,
                                               std::shared_ptr<std::vector<int8_t>> packageProgramBitcode)
    : clang::ASTConsumer(), CI(CI), opts(opts), packageProgramBitcode(std::move(packageProgramBitcode)) {}

namespace {
struct ExportCollector final : clang::RecursiveASTVisitor<ExportCollector> {
  clang::ASTContext &context;
  clang::DiagnosticsEngine &diag;
  std::vector<PackageExport> exports;
  std::vector<const clang::FunctionDecl *> deviceKernels;
  std::unordered_map<const clang::FunctionDecl *, bool> unsupportedDeviceClosures;
  bool invalid = false;

  ExportCollector(clang::ASTContext &context, clang::DiagnosticsEngine &diag) : context(context), diag(diag) {}

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool hasUnsupportedDeviceClosure(const clang::FunctionDecl *fd) {
    if (const auto found = unsupportedDeviceClosures.find(fd); found != unsupportedDeviceClosures.end()) return found->second;
    std::unordered_set<const clang::FunctionDecl *> visited;
    std::function<bool(const clang::FunctionDecl *)> inspect;
    struct ClosureVisitor final : clang::RecursiveASTVisitor<ClosureVisitor> {
      std::function<bool(const clang::FunctionDecl *)> &inspect;
      bool &unsupported;

      ClosureVisitor(std::function<bool(const clang::FunctionDecl *)> &inspect, bool &unsupported)
          : inspect(inspect), unsupported(unsupported) {}

      bool VisitGCCAsmStmt(clang::GCCAsmStmt *stmt) {
        const auto text = stmt->getAsmString();
        unsupported |= text.find("b128") != std::string::npos || text.find("dwordx4") != std::string::npos;
        return !unsupported;
      }

      bool VisitCallExpr(clang::CallExpr *expr) {
        const auto *callee = expr->getDirectCallee();
        const auto *definition = callee ? callee->getDefinition() : nullptr;
        unsupported |= definition && inspect(definition);
        return !unsupported;
      }
    };
    inspect = [&](const clang::FunctionDecl *current) {
      if (!visited.emplace(current).second) return false;
      bool unsupported = false;
      ClosureVisitor(inspect, unsupported).TraverseStmt(current->getBody());
      return unsupported;
    };
    const auto unsupported = inspect(fd);
    unsupportedDeviceClosures.emplace(fd, unsupported);
    return unsupported;
  }

  bool isLiveDeviceKernel(const clang::FunctionDecl *fd) const {
    if (fd->getQualifiedNameAsString() != "rocprim::detail::trampoline_kernel") return true;
    const auto *body = llvm::dyn_cast<clang::CompoundStmt>(fd->getBody());
    if (!body) return false;
    for (const auto *stmt : body->body()) {
      const auto *branch = llvm::dyn_cast<clang::IfStmt>(stmt);
      bool enabled = false;
      if (branch && branch->isConstexpr() && branch->getCond()->EvaluateAsBooleanCondition(enabled, context)) return enabled;
    }
    return false;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *fd) {
    if (fd->doesThisDeclarationHaveABody()) {
      const bool isSpecialisation = fd->getTemplateSpecializationKind() != clang::TSK_Undeclared;
      if ((context.getLangOpts().CUDA || context.getLangOpts().HIP) && context.getLangOpts().CUDAIsDevice
          && fd->hasAttr<clang::CUDAGlobalAttr>() && !fd->getType()->isDependentType()
          && (fd->getDescribedFunctionTemplate() == nullptr || isSpecialisation) && isLiveDeviceKernel(fd)) {
        if (!hasUnsupportedDeviceClosure(fd)) deviceKernels.push_back(fd);
      }
      const auto metadata = packageExportMetadata(fd, diag);
      invalid |= metadata.invalid;
      if (metadata.name) exports.push_back({fd, *metadata.name, metadata.implements, metadata.requiredCapabilities});
    }
    return true;
  }
};
} // namespace

void OffloadRewriteConsumer::HandleTranslationUnit(clang::ASTContext &C) {
  auto &D = CI.getDiagnostics();
  if (!opts.emitLibraryPath.empty()) {
    ExportCollector collector(C, D);
    collector.TraverseDecl(C.getTranslationUnitDecl());
    if (collector.invalid) return;
    if (collector.exports.empty())
      emit(D, clang::DiagnosticsEngine::Warning,
           POLYREGION_DIAG_POLYSTL
           "-fstdpar-emit-library set but no [[clang::annotate(\"polyregion_export:<identity>\")]] functions found");
    compilePackageProgram(opts, C, D, collector.exports, collector.deviceKernels, opts.emitLibraryPath);
    return;
  }
  for (auto r : outlinePolyregionOffload(C))
    r //
        ^ foreach_total(
            [&](const Failure &f) { //
              emit(D, f.callExpr->getBeginLoc(), clang::DiagnosticsEngine::Warning, POLYREGION_DIAG_POLYSTL "Outline failed: %0", f.reason);
            },
            [&](const Callsite &c) { //
              const SpecialisationPathVisitor spv(C);
              const auto specialisationPath = spv.resolve(c.calleeDecl) ^ reverse();
              auto moduleId = specialisationPath | values() | mk_string("->", [&](const auto &callExpr) {
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

              const auto bundle = compileRegion(opts, C, D, moduleId, *c.functorDecl,
                                                specialisationPath //
                                                    ^ head_maybe() //
                                                    ^ fold([](const auto &, const auto &callExpr) { return callExpr->getExprLoc(); },
                                                           [&] { return c.callLambdaArgExpr->getExprLoc(); }),
                                                c.kind);

              if (opts.verbose) {
                emit(D, c.callLambdaArgExpr->getExprLoc(), clang::DiagnosticsEngine::Remark,
                     POLYREGION_DIAG_POLYSTL "Outlined function: %0 for %1 (%2)\n", moduleId, std::string(magic_enum::enum_name(c.kind)),
                     (bundle.objects | map([](const auto &o) {
                        return std::string(magic_enum::enum_name(o.format)) + "="
                               + std::to_string(static_cast<float>(o.moduleImage.size()) / 1000) + "KB";
                      }) //
                      | mk_string(", ")));
              }

              insertKernelImage(D, CI.getSema(), C, c, bundle);

            });

  const auto action = CI.getFrontendOpts().ProgramAction;
  const bool emitsCode = action == clang::frontend::EmitAssembly || action == clang::frontend::EmitBC || action == clang::frontend::EmitLLVM
                         || action == clang::frontend::EmitLLVMOnly || action == clang::frontend::EmitCodeGenOnly
                         || action == clang::frontend::EmitObj;
  if (!emitsCode) return;
  std::map<const clang::FunctionDecl *, Vector<const clang::FunctionDecl *>> interfaceCallables;
  std::map<std::string, polyast::Package> interfacePackages;
  std::vector<PreparedInterfaceCall> preparedInterfaces;
  for (const auto &site : interfaceSites(C)) {
    if (const auto error = interfaceCallableShapeError(site)) {
      emit(D, site.call->getExprLoc(), clang::DiagnosticsEngine::Error, POLYREGION_DIAG_POLYSTL "%0", *error);
      continue;
    }
    const auto callableIdentities = interfaceCallableIdentities(site);
    const auto [existing, inserted] = interfaceCallables.emplace(site.callee->getCanonicalDecl(), callableIdentities);
    if (!inserted) {
      if (existing->second != callableIdentities)
        emit(D, site.call->getExprLoc(), clang::DiagnosticsEngine::Error,
             POLYREGION_DIAG_POLYSTL "one interface specialization cannot use conflicting callable identities");
      continue;
    }
    if (!C.getLangOpts().CXXExceptions) {
      emit(D, site.call->getExprLoc(), clang::DiagnosticsEngine::Error,
           POLYREGION_DIAG_POLYSTL "package interface compilation requires C++ exceptions for failure-safe context cleanup");
      continue;
    }
    auto found = interfacePackages.find(site.packageName);
    if (found == interfacePackages.end()) {
      const auto package = polyfront::package::loadPackage(site.packageName, opts.libraryPath.empty()
                                                                                 ? polyfront::package::packageRoots()
                                                                                 : polyfront::package::splitPackageRoots(opts.libraryPath));
      if (!package) {
        for (const auto &error : package.errors)
          emit(D, site.call->getExprLoc(), clang::DiagnosticsEngine::Error, POLYREGION_DIAG_POLYSTL "%0", error);
        continue;
      }
      found = interfacePackages.emplace(site.packageName, *package.value).first;
    }
    if (auto prepared = prepareInterfaceCall(CI, C, site, found->second)) preparedInterfaces.emplace_back(std::move(*prepared));
  }
  compileInterfaceCalls(opts, CI, C, preparedInterfaces, *packageProgramBitcode);
}

std::unique_ptr<clang::ASTConsumer>
polyregion::polystl::makeOffloadRewriteConsumer(clang::CompilerInstance &CI, const polyfront::Options &opts,
                                                std::shared_ptr<std::vector<int8_t>> packageProgramBitcode) {
  return std::make_unique<OffloadRewriteConsumer>(CI, opts, std::move(packageProgramBitcode));
}
