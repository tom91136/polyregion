#include "codegen.h"

#include "clang/AST/Attr.h"
#include "clang/AST/RecordLayout.h"

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyfront/pass_specs.hpp"
#include "polyregion/conventions.h"
#include "polyregion/env_keys.h"
#include "polyregion/mirror_names.h"
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

  auto args = userParams |
              map([&](const clang::ParmVarDecl *x) { return Arg(Named(declName(x), remapper.annotateLocalSpace(x, r)), {}); }) //
              | append(recv)                                                                                                   //
              | to_vector();

  auto f0 = std::make_shared<Function>(Sym({conventions::EntryName}), std::vector<std::string>{}, std::optional<Arg>{}, args,
                                       std::vector<Arg>{}, std::vector<Arg>{}, rtnTpe, stmts, FunctionVisibility::Exported(),
                                       FunctionFpMode::Relaxed(), true, FunctionAffinity::Offload());

  auto program = Program(*f0, r.functions | values() | map([&](auto &x) { return *x; }) | to_vector(),
                         r.structs | values() | map([&](auto &x) { return *x; }) | to_vector(), PassPhase::Initial(), {});

  auto exportedStructNames = std::unordered_set<std::string>{repr(parentDef->name)};

  auto layouts = r.layouts | values() | map([&](auto &x) { return std::pair{exportedStructNames ^ contains(x->name), *x}; }) | to_vector();

  if (opts.verbose) {
    emit(diag, loc, clang::DiagnosticsEngine::Level::Remark, POLYREGION_DIAG_POLYSTL "Remapped program [%0, sizeof capture=%1]\n%2",
         moduleId, C.getTypeSize(C.getCanonicalTagType(parent)), repr(program));
  }

  if (opts.jit) {
    auto jitObjects = opts.targets                                                                                //
                      | filter([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); }) //
                      | collect([&](auto &target, auto &arch) -> std::optional<polyfront::KernelObject> {
                          auto format = runtime::moduleFormatOf(target);
                          if (!format) return std::nullopt;
                          const auto pp = polyfront::passes::arenaPassesFor(target, opts.stackDepth);
                          polyfront::KernelObject ko;
                          ko.format = *format;
                          ko.kind = runtime::targetPlatformKind(target);
                          ko.target = target;
                          ko.arch = arch;
                          ko.pipelineSpec = pp.size() >= 2 ? pp[1] : std::string{};
                          return ko;
                        }) //
                      | to_vector();
    const auto packed = polyast::hashed_program_to_msgpack(program);
    if (opts.verbose)
      emit(diag, loc, clang::DiagnosticsEngine::Level::Remark, POLYREGION_DIAG_POLYSTL "JIT deferred [%0]: %1 target(s), program %2 bytes",
           moduleId, std::to_string(jitObjects.size()), std::to_string(packed.size()));
    const bool jitAsserts = !program.template collect_all<polyast::Spec::Assert>().empty();
    return polyfront::KernelBundle{moduleId,
                                   jitObjects,
                                   layouts,
                                   remapper.readOnlyMembers,
                                   program_to_json(program).dump(),
                                   {},
                                   {},
                                   jitAsserts,
                                   std::string(packed.begin(), packed.end())};
  }

  const auto compiled =
      opts.targets                                                                                //
      | filter([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); }) //
      | collect([&](auto &target, auto &features) {
          return compileProgram(opts, program, target, features, polyfront::passes::arenaPassesFor(target, opts.stackDepth)) ^
                 fold_total([&](const CompileResult &r) -> std::optional<CompileResult> { return r; },
                            [&](const std::string &err) -> std::optional<CompileResult> {
                              emit(diag, clang::DiagnosticsEngine::Level::Warning,
                                   POLYREGION_DIAG_POLYSTL "Frontend failed to compile program [%0, target=%1, features=%2]\n%3", moduleId,
                                   std::string(magic_enum::enum_name(target)), features, err);
                              return std::nullopt;
                            }) ^
                 map([&](auto &x) { return std::tuple{target, features, x}; });
        }) //
      | to_vector();

  const bool asserts = compiled ^ exists([](auto &, auto &, auto &result) { return polyfront::entryNeedsErrorBuffer(result); });

  auto objects = compiled //
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
  auto mir = polyfront::compileManagedHostMirror(opts, program, kind, moduleId);
  if (mir.error)
    emit(diag, loc, clang::DiagnosticsEngine::Level::Warning, POLYREGION_DIAG_POLYSTL "Host mirroring compile failed [%0]: %1", moduleId,
         *mir.error);
  return polyfront::KernelBundle{moduleId,    objects,      layouts, remapper.readOnlyMembers, program_to_json(program).dump(),
                                 mir.bitcode, mir.mirrorId, asserts};
}
