#include "codegen.h"

#include "clang/AST/Attr.h"
#include "clang/AST/RecordLayout.h"

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyregion/conventions.h"
#include "polyregion/types.h"

#include "ast.h"
#include "clang_utils.h"
#include "remapper.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;
using polyregion::polyfront::emit;

polyfront::KernelBundle polystl::compileRegion(const polyfront::Options &opts,
                                               clang::ASTContext &C,                //
                                               clang::DiagnosticsEngine &diag,      //
                                               const std::string &moduleId,         //
                                               const clang::CXXMethodDecl &functor, //
                                               const clang::SourceLocation &loc,    //
                                               runtime::PlatformKind kind) {
  Remapper remapper(C);

  auto parent = functor.getParent();
  auto returnTpe = functor.getReturnType();
  auto body = functor.getBody();

  auto r = Remapper::RemapContext{};
  auto parentDef = remapper.handleRecord(parent, r);

  auto rtnTpe = remapper.handleType(returnTpe, r);

  auto stmts = r.scoped([&](auto &r) { remapper.handleStmt(body, r); }, false, rtnTpe, parentDef);
  stmts.push_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));

  auto recv = Arg(Named(conventions::ThisReceiver, Type::Ptr(Type::Struct(parentDef->name, {}), TypeSpace::Global())), {});

  // The kernel ABI prepends a thread-id Int64 to Entry functions and the offload lambdas take
  // an int64 tid as their first parameter -- the runtime fills the same slot for both. Drop
  // the lambda's leading int64 from the arg list and alias it to `__tid` at the top of the body.
  Vector<Stmt::Any> tidAliases;
  Vector<const clang::ParmVarDecl *> userParams =
      functor.parameters() | map([](auto *p) -> const clang::ParmVarDecl * { return p; }) | to_vector();
  if (!userParams.empty() && remapper.handleType(userParams.front()->getType(), r).is<Type::IntS64>()) {
    // declName() carries the per-decl ID suffix so the alias matches what DeclRefExpr emits.
    auto name = declName(userParams.front());
    if (!name.empty()) {
      tidAliases.push_back(Stmt::Var(Named(name, Type::IntS64()), Expr::Alias(dsl::Select(Vector<Named>{}, Named("__tid", Type::IntS64()))),
                                     /*isMutable*/ false));
    }
    userParams.erase(userParams.begin());
  }
  stmts.insert(stmts.begin(), tidAliases.begin(), tidAliases.end());

  auto args = userParams | map([&](const clang::ParmVarDecl *x) {
                auto local = x->attrs() | exists([](const clang::Attr *a) {
                               if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated) {
                                 return annotated->getAnnotation() == "__polyregion_local";
                               }
                               return false;
                             });

                auto tpe = remapper.handleType(x->getType(), r);

                auto annotatedTpe =
                    tpe.get<Type::Ptr>() ^
                    fold([&](auto p) { return Type::Ptr(p.comp, local ? TypeSpace::Local() : p.space).widen(); }, [&]() { return tpe; });

                return Arg(Named(declName(x), annotatedTpe), {});
              })             //
              | append(recv) //
              | to_vector();

  auto f0 =
      std::make_shared<Function>(Sym({conventions::EntryName}), std::vector<std::string>{}, std::optional<Arg>{}, args, std::vector<Arg>{},
                                 std::vector<Arg>{}, rtnTpe, stmts, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);

  auto program = Program(*f0, r.functions | values() | map([&](auto &x) { return *x; }) | to_vector(),
                         r.structs | values() | map([&](auto &x) { return *x; }) | to_vector(), PassPhase::Initial());

  auto exportedFns = (std::vector<Function>{program.entry} ^ concat(program.functions))                         //
                     | filter([](auto &f) { return f.visibility.template is<FunctionVisibility::Exported>(); }) //
                     | to_vector();
  auto exportedStructNames =
      exportedFns                                                                                                                         //
      | flat_map([](auto &f) { return f.args; })                                                                                          //
      | collect([](auto &a) { return extractComponent(a.named.tpe) ^ flat_map([](auto &t) { return t.template get<Type::Struct>(); }); }) //
      | map([](auto &s) { return repr(s.name); })                                                                                         //
      | to<std::unordered_set>();

  auto layouts = r.layouts | values() | map([&](auto &x) { return std::pair{exportedStructNames ^ contains(x->name), *x}; }) | to_vector();

  if (opts.verbose) {
    emit(diag, loc, clang::DiagnosticsEngine::Level::Remark, POLYREGION_DIAG_POLYSTL "Remapped program [%0, sizeof capture=%1]\n%2",
         moduleId, C.getTypeSize(C.getCanonicalTagType(parent)), repr(program));
  }

  auto objects = opts.targets                                                                                //
                 | filter([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); }) //
                 | collect([&](auto &target, auto &features) {
                     return compileProgram(opts, program, target, features) ^
                            fold_total([&](const CompileResult &r) -> std::optional<CompileResult> { return r; },
                                       [&](const std::string &err) -> std::optional<CompileResult> {
                                         emit(diag, clang::DiagnosticsEngine::Level::Warning,
                                              POLYREGION_DIAG_POLYSTL "Frontend failed to compile program [%0, target=%1, features=%2]\n%3",
                                              moduleId, std::string(magic_enum::enum_name(target)), features, err);
                                         return std::nullopt;
                                       }) ^
                            map([&](auto &x) { return std::tuple{target, features, x}; });
                   }) //
                 | collect([&](auto &target, auto &features, auto &result) -> std::optional<polyfront::KernelObject> {
                     emit(diag, loc, clang::DiagnosticsEngine::Level::Remark,
                          POLYREGION_DIAG_POLYSTL "Compilation events for [%0, target=%1, features=%2]\n%3", moduleId,
                          std::string(magic_enum::enum_name(target)), features, repr(result));

                     if (auto bin = result.binary; !bin) {
                       emit(diag, loc, clang::DiagnosticsEngine::Level::Warning,
                            POLYREGION_DIAG_POLYSTL "Backend failed to compile program [%0, target=%1, features=%2]\nReason: %3", moduleId,
                            std::string(magic_enum::enum_name(target)), features, result.messages);
                       return std::nullopt;
                     } else {

                       if (!result.messages.empty()) {
                         emit(diag, loc, clang::DiagnosticsEngine::Level::Warning,
                              POLYREGION_DIAG_POLYSTL "Backend emitted binary (%0KB) with warnings [%1, target=%2, features=%3]\n%4",
                              std::to_string(static_cast<float>(bin->size()) / 1000.f), moduleId,
                              std::string(magic_enum::enum_name(target)), features, result.messages);

                       } else {
                         emit(diag, loc, clang::DiagnosticsEngine::Level::Remark,
                              POLYREGION_DIAG_POLYSTL "Backend emitted binary (%0KB) [%1, target=%2, features=%3]",
                              std::to_string(static_cast<float>(bin->size()) / 1000.f), moduleId,
                              std::string(magic_enum::enum_name(target)), features, result.messages);
                       }

                       if (auto format = runtime::moduleFormatOf(target)) {
                         return polyfront::KernelObject{
                             *format,                              //
                             runtime::targetPlatformKind(target),  //
                             result.features,                      //
                             std::string(bin->begin(), bin->end()) //
                         };
                       } else {
                         emit(diag, loc, clang::DiagnosticsEngine::Level::Remark,
                              POLYREGION_DIAG_POLYSTL "Backend emitted binary for unknown target [%1, target=%2, features=%3]", moduleId,
                              std::string(magic_enum::enum_name(target)), features, result.messages);
                         return std::nullopt;
                       }
                     }
                   }) //
                 | to_vector();
  // If targets were requested for this kind but every one of them failed to compile, escalate
  // to a hard error: a kernel bundle with zero objects compiles cleanly but then fails at run
  // time with "no compatible image", which is much harder to diagnose than a compile-time
  // failure that surfaces the original backend error.
  const auto requestedForKind = opts.targets ^ count([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); });
  if (requestedForKind > 0 && objects.empty()) {
    emit(diag, loc, clang::DiagnosticsEngine::Level::Error,
         POLYREGION_DIAG_POLYSTL "No kernels compiled successfully for [%0, kind=%1] (requested %2 target(s)); "
                                 "see prior diagnostics for the per-target failure",
         moduleId, std::string(magic_enum::enum_name(kind)), static_cast<int>(requestedForKind));
  }
  return polyfront::KernelBundle{moduleId, objects, layouts, program_to_json(program).dump()};
}
