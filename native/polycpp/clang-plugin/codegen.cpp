#include "codegen.h"

#include <string_view>

#include "clang/AST/Attr.h"
#include "clang/AST/RecordLayout.h"

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/diag.hpp"
#include "polyfront/package_program.hpp"
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

namespace {
constexpr std::string_view ContextArgument = "#context";

Type::Any eraseCallableBinders(const Type::Any &type, const Set<Type::Var> &bound = {}) {
  return type.match_total([](const Type::Float16 &x) -> Type::Any { return x; }, [](const Type::Float32 &x) -> Type::Any { return x; },
                          [](const Type::Float64 &x) -> Type::Any { return x; }, [](const Type::IntU8 &x) -> Type::Any { return x; },
                          [](const Type::IntU16 &x) -> Type::Any { return x; }, [](const Type::IntU32 &x) -> Type::Any { return x; },
                          [](const Type::IntU64 &x) -> Type::Any { return x; }, [](const Type::IntS8 &x) -> Type::Any { return x; },
                          [](const Type::IntS16 &x) -> Type::Any { return x; }, [](const Type::IntS32 &x) -> Type::Any { return x; },
                          [](const Type::IntS64 &x) -> Type::Any { return x; }, [](const Type::Nothing &x) -> Type::Any { return x; },
                          [](const Type::Unit0 &x) -> Type::Any { return x; }, [](const Type::Bool1 &x) -> Type::Any { return x; },
                          [&](const Type::Struct &x) -> Type::Any {
                            return x.withArgs(x.args ^ map([&](const auto &arg) { return eraseCallableBinders(arg, bound); }));
                          },
                          [&](const Type::Ptr &x) -> Type::Any { return x.withComp(eraseCallableBinders(x.comp, bound)); },
                          [&](const Type::Arr &x) -> Type::Any { return x.withComp(eraseCallableBinders(x.comp, bound)); },
                          [&](const Type::Var &x) -> Type::Any { return bound ^ contains(x) ? Type::Nothing().widen() : x.widen(); },
                          [&](const Type::Exec &x) -> Type::Any {
                            const auto nested = bound | concat(x.tpeVars) | to<Set>();
                            return Type::Exec({}, x.args ^ map([&](const auto &arg) { return eraseCallableBinders(arg, nested); }),
                                              eraseCallableBinders(x.rtn, nested));
                          },
                          [](const Type::FnRef &x) -> Type::Any { return x; });
}

template <typename T> Vector<Type::Var> freeTypeVariables(const T &value) {
  return value.template modify_all<Type::Exec>([](const auto &exec) { return *eraseCallableBinders(exec).template get<Type::Exec>(); })
      .template collect_all<Type::Var>();
}
} // namespace

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
  r.entryCapture = parent->getCanonicalDecl();
  auto parentDef = remapper.handleRecord(parent, r);

  auto rtnTpe = remapper.handleType(returnTpe, r);

  auto stmts = r.scoped([&](auto &r) { remapper.handleStmt(body, r); }, false, rtnTpe, parentDef);
  stmts.push_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));

  auto recv = Arg(Named(conventions::ThisReceiver, Type::Ptr(Type::Struct(parentDef->name, {}), TypeSpace::Global())), {});

  // The kernel ABI prepends a thread-id Int64 to Entry functions and the offload lambdas take
  // an int64 tid as their first parameter -- the runtime fills the same slot for both. Drop
  // the lambda's leading int64 from the arg list and alias it to `__tid` at the top of the body.
  Vector<Stmt::Any> tidAliases;
  Vector<const clang::ParmVarDecl *> userParams = functor.parameters()                                                 //
                                                  | map([](const auto &p) -> const clang::ParmVarDecl * { return p; }) //
                                                  | to_vector();
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

  auto args = userParams                                                                                           //
              | map([&](const auto &x) { return Arg(Named(declName(x), remapper.annotateLocalSpace(x, r)), {}); }) //
              | append(recv)                                                                                       //
              | to_vector();

  auto f0 = std::make_shared<Function>(FunctionDecl(Sym({conventions::EntryName}), std::vector<Type::Var>{}, std::optional<Arg>{}, args,
                                                    std::vector<Arg>{}, std::vector<Arg>{}, rtnTpe, FunctionAffinity::Offload()),
                                       stmts, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());

  auto program = Program(*f0, r.functions | values() | map([&](const auto &x) { return *x; }) | to_vector(),
                         r.structs | values() | map([&](const auto &x) { return *x; }) | to_vector(), PassPhase::Initial(), {});

  auto exportedStructNames = std::unordered_set<std::string>{fqcn(parentDef->name)};

  auto layouts = r.layouts                                                                                    //
                 | values()                                                                                   //
                 | map([&](const auto &x) { return std::pair{exportedStructNames ^ contains(x->name), *x}; }) //
                 | to_vector();

  if (opts.verbose) {
    emit(diag, loc, clang::DiagnosticsEngine::Level::Remark, POLYREGION_DIAG_POLYSTL "Remapped program [%0, sizeof capture=%1]\n%2",
         moduleId, C.getTypeSize(C.getCanonicalTagType(parent)), repr(program));
  }

  if (opts.jit) {
    auto jitObjects = opts.targets                                                                                            //
                      | filter([&](const auto &target, const auto &) { return kind == runtime::targetPlatformKind(target); }) //
                      | collect([&](const auto &target, const auto &arch) -> std::optional<polyfront::KernelObject> {
                          auto format = runtime::moduleFormatOf(target);
                          if (!format) return std::nullopt;
                          const auto pp = polyfront::passes::arenaPassesFor(target, opts.stackDepth);
                          polyfront::KernelObject ko;
                          ko.format = *format;
                          ko.kind = runtime::targetPlatformKind(target);
                          ko.features = polyfront::passes::jitFeaturesFor(target);
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
    // the deferred path decides before lowering, so a raise counts too: both end up writing the #error buffer
    const bool jitAsserts =
        !program.template collect_all<polyast::Spec::Assert>().empty() || !program.template collect_all<polyast::Stmt::Raise>().empty();
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
      opts.targets                                                                                            //
      | filter([&](const auto &target, const auto &) { return kind == runtime::targetPlatformKind(target); }) //
      | collect([&](const auto &target, const auto &features) {
          return compileProgram(opts, program, target, features, polyfront::passes::arenaPassesFor(target, opts.stackDepth)) //
                 ^ fold_total([&](const CompileResult &r) -> std::optional<CompileResult> { return r; },
                              [&](const std::string &err) -> std::optional<CompileResult> {
                                emit(diag, clang::DiagnosticsEngine::Level::Warning,
                                     POLYREGION_DIAG_POLYSTL "Frontend failed to compile program [%0, target=%1, features=%2]\n%3",
                                     moduleId, std::string(magic_enum::enum_name(target)), features, err);
                                return std::nullopt;
                              }) //
                 ^ map([&](const auto &x) { return std::tuple{target, features, x}; });
        }) //
      | to_vector();

  const bool asserts =
      compiled ^ exists([](const auto &, const auto &, const auto &result) { return polyfront::entryNeedsErrorBuffer(result); });

  auto objects = compiled //
                 | collect([&](const auto &target, const auto &features, const auto &result) -> std::optional<polyfront::KernelObject> {
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
  const auto requestedForKind =
      opts.targets ^ count([&](const auto &target, const auto &) { return kind == runtime::targetPlatformKind(target); });
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

void polystl::compilePackageProgram(const polyfront::Options &opts,                                //
                                    clang::ASTContext &C,                                          //
                                    clang::DiagnosticsEngine &diag,                                //
                                    const std::vector<PackageExport> &exports,                     //
                                    const std::vector<const clang::FunctionDecl *> &deviceKernels, //
                                    const std::string &outPath) {
  Remapper remapper(C);
  remapper.emitPackageProgramMode = true;
  Remapper::RemapContext r;
  Map<Sym, Sym> exportNames;
  Set<std::string> exported;

  if ((C.getLangOpts().CUDA || C.getLangOpts().HIP) && C.getLangOpts().CUDAIsDevice) {
    for (const auto *kernel : deviceKernels) {
      auto fn = remapper.handleCall(kernel, r).second;
      fn->decl.affinity = FunctionAffinity::Offload();
      fn->convention = CallConvention::OffloadEntry();
    }
  }

  for (const auto &exportedFunction : exports) {
    const auto *decl = exportedFunction.decl;
    const auto &exportName = exportedFunction.name;
    auto [name, fn] = remapper.handleCall(decl, r);
    exportNames.emplace(fn->decl.name, exportName);
    exported.emplace(name);
    fn->visibility = FunctionVisibility::Exported();
    if (opts.verbose)
      emit(diag, decl->getBeginLoc(), clang::DiagnosticsEngine::Level::Remark, POLYREGION_DIAG_POLYSTL "Exporting package symbol: %0",
           fqcn(exportName));
  }

  const auto completeStructArgs = [&](const Type::Struct &structType) {
    const auto definition = r.structs ^ get_maybe(fqcn(structType.name));
    if (!definition || structType.args.size() >= (*definition)->tpeVars.size()) return structType;
    auto args = structType.args;
    for (size_t i = args.size(); i < (*definition)->tpeVars.size(); ++i)
      args.emplace_back((*definition)->tpeVars[i]);
    return structType.withArgs(args);
  };
  const auto convergenceLimit = std::max<size_t>(64, (r.structs.size() + r.functions.size() + 1) * 8);
  size_t structIterations = 0;
  for (bool changed = true; changed;) {
    if (++structIterations > convergenceLimit) {
      emit(diag, C.getTranslationUnitDecl()->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
           POLYREGION_DIAG_POLYSTL "Package struct completion did not converge after %0 iterations",
           static_cast<unsigned long long>(convergenceLimit));
      return;
    }
    changed = false;
    for (const auto &definition : r.structs ^ values()) {
      auto next = definition->template modify_all<Type::Struct>(completeStructArgs);
      next.tpeVars = next.tpeVars | concat(next.template collect_all<Type::Var>()) | distinct() | to_vector();
      if (next == *definition) continue;
      *definition = std::move(next);
      changed = true;
    }
  }
  size_t functionIterations = 0;
  for (bool changed = true; changed;) {
    if (++functionIterations > convergenceLimit) {
      emit(diag, C.getTranslationUnitDecl()->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
           POLYREGION_DIAG_POLYSTL "Package function completion did not converge after %0 iterations",
           static_cast<unsigned long long>(convergenceLimit));
      return;
    }
    changed = false;
    const auto functionsByName = r.functions | values() | map([](const auto &fn) { return std::pair{fn->decl.name, fn}; }) | to<Map>();
    for (const auto &fn : r.functions ^ values()) {
      auto next = fn->template modify_all<Type::Struct>(completeStructArgs);
      next = next.template modify_all<Expr::Invoke>([&](const Expr::Invoke &invoke) {
        const auto callee =
            invoke.callee.template get<Type::FnRef>() ^ flat_map([&](const auto &ref) { return functionsByName ^ get_maybe(ref.name); });
        if (!callee || invoke.tpeArgs.size() >= (*callee)->decl.tpeVars.size()) return invoke;
        Map<std::string, Type::Any> inferred;
        std::function<void(const Type::Any &, const Type::Any &)> infer = [&](const auto &expected, const auto &actual) {
          if (const auto variable = expected.template get<Type::Var>()) inferred.emplace(variable->name, actual);
          else if (const auto pointer = expected.template get<Type::Ptr>()) {
            if (const auto value = actual.template get<Type::Ptr>()) infer(pointer->comp, value->comp);
          } else if (const auto array = expected.template get<Type::Arr>()) {
            if (const auto value = actual.template get<Type::Arr>()) infer(array->comp, value->comp);
          } else if (const auto structure = expected.template get<Type::Struct>()) {
            if (const auto value = actual.template get<Type::Struct>(); value && value->name == structure->name)
              for (size_t i = 0; i < std::min(structure->args.size(), value->args.size()); ++i)
                infer(structure->args[i], value->args[i]);
          }
        };
        if ((*callee)->decl.receiver && invoke.receiver) infer((*callee)->decl.receiver->named.tpe, invoke.receiver->tpe());
        for (size_t i = 0; i < std::min((*callee)->decl.args.size(), invoke.args.size()); ++i)
          infer((*callee)->decl.args[i].named.tpe, invoke.args[i].tpe());
        infer((*callee)->decl.rtn, invoke.rtn);
        auto args = invoke.tpeArgs;
        for (size_t i = args.size(); i < (*callee)->decl.tpeVars.size(); ++i) {
          const auto &variable = (*callee)->decl.tpeVars[i];
          if (const auto value = inferred ^ get_maybe(variable.name)) args.emplace_back(*value);
          else if (r.packageVariables ^ contains(variable.name)) args.emplace_back(variable);
          else break;
        }
        return invoke.withTpeArgs(args);
      });
      next.decl.tpeVars =
          next.decl.tpeVars | concat(freeTypeVariables(next.decl))
          | concat(freeTypeVariables(next) | filter([&](const auto &variable) { return r.packageVariables ^ contains(variable.name); }))
          | distinct() | to_vector();
      if (next == *fn) continue;
      *fn = std::move(next);
      changed = true;
    }
  }

  const auto callees = [](const Function &fn) {
    return fn.collect_all<Expr::Invoke>() | collect([](const auto &invoke) { return invoke.callee.template get<Type::FnRef>(); })
           | map([](const auto &ref) { return ref.name; }) | to_vector();
  };
  const auto keyByName = r.functions | map([](const auto &entry) { return std::pair{entry.second->decl.name, entry.first}; }) | to<Map>();
  Set<std::string> host = exported;
  Set<std::string> contextual;
  for (const auto &[name, fn] : r.functions)
    if (fn->template collect_all<Term::Select>() ^ exists([](const auto &selection) { return selection.root.symbol == ContextArgument; }))
      contextual.emplace(name);
  size_t closureIterations = 0;
  for (bool changed = true; changed;) {
    if (++closureIterations > convergenceLimit) {
      emit(diag, C.getTranslationUnitDecl()->getBeginLoc(), clang::DiagnosticsEngine::Level::Error,
           POLYREGION_DIAG_POLYSTL "Package call closure did not converge after %0 iterations",
           static_cast<unsigned long long>(convergenceLimit));
      return;
    }
    changed = false;
    for (const auto &[name, fn] : r.functions) {
      const auto invoked = callees(*fn) | collect([&](const auto &callee) { return keyByName ^ get_maybe(callee); }) | to_vector();
      if (host ^ contains(name))
        for (const auto &callee : invoked)
          if (host.emplace(callee).second) changed = true;
      if (!(contextual ^ contains(name)) && invoked ^ exists([&](const auto &callee) { return contextual ^ contains(callee); })) {
        contextual.emplace(name);
        changed = true;
      }
    }
  }
  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const auto context = Term::Select(Named(std::string(ContextArgument), contextType), {}, contextType).widen();
  for (const auto &[name, fn] : r.functions) {
    if (host ^ contains(name)) fn->decl.affinity = FunctionAffinity::Host();
    if (!(contextual ^ contains(name))) continue;
    if (fn->decl.args.empty() || fn->decl.args.front().named.symbol != ContextArgument)
      fn->decl.args.insert(fn->decl.args.begin(), Arg(Named(std::string(ContextArgument), contextType), {}));
    *fn = fn->template modify_all<Expr::Invoke>([&](const Expr::Invoke &invoke) -> Expr::Invoke {
      const auto callee =
          invoke.callee.template get<Type::FnRef>() ^ flat_map([&](const auto &ref) { return keyByName ^ get_maybe(ref.name); });
      if (!callee || !(contextual ^ contains(*callee))) return invoke;
      if (!invoke.args.empty())
        if (const auto selected = invoke.args.front().template get<Term::Select>(); selected && selected->root.symbol == ContextArgument)
          return invoke;
      return invoke.withArgs(invoke.args ^ prepend(context));
    });
  }
  for (const auto &exportedFunction : exports) {
    auto fn = remapper.handleCall(exportedFunction.decl, r).second;
    fn->decl.name = exportedFunction.name;
    fn->implements = exportedFunction.implements;
    fn->requiredCapabilities = exportedFunction.requiredCapabilities;
  }

  const auto functions =
      r.functions //
      | values()  //
      | map([&](const auto &x) {
          return x->template modify_all<Type::FnRef>([&](const auto &ref) {
            return exportNames ^ get_maybe(ref.name) ^ map([&](const auto &name) { return ref.withName(name); }) ^ get_or_else(ref);
          });
        }) //
      | to_vector();
  Set<Sym> functionNames;
  const auto duplicateName = functions | collect_first([&](const auto &fn) -> Opt<Sym> {
                               if (functionNames.emplace(fn.decl.name).second) return {};
                               return fn.decl.name;
                             });
  if (duplicateName) {
    const auto site = exports | collect_first([&](const auto &x) -> Opt<const clang::FunctionDecl *> {
                        if (x.name == *duplicateName) return x.decl;
                        return {};
                      });
    emit(diag, site ? (*site)->getBeginLoc() : clang::SourceLocation{}, clang::DiagnosticsEngine::Level::Error,
         POLYREGION_DIAG_POLYSTL "Duplicate package function identity: %0", fqcn(*duplicateName));
    return;
  }
  const auto unboundFunctionVariable =
      functions | collect_first([](const auto &fn) -> Opt<std::pair<Sym, Type::Var>> {
        const auto bound = fn.decl.tpeVars | to<Set>();
        return freeTypeVariables(fn) | collect_first([&](const auto &variable) -> Opt<std::pair<Sym, Type::Var>> {
                 return bound ^ contains(variable) ? Opt<std::pair<Sym, Type::Var>>{} : std::pair{fn.decl.name, variable};
               });
      });
  if (unboundFunctionVariable) {
    emit(diag, clang::DiagnosticsEngine::Level::Error, POLYREGION_DIAG_POLYSTL "Package function %0 contains unbound type variable %1",
         fqcn(unboundFunctionVariable->first), canonicalName(unboundFunctionVariable->second.widen()));
    return;
  }
  const auto unboundStructVariable =
      r.structs | values() | collect_first([](const auto &definition) -> Opt<std::pair<Sym, Type::Var>> {
        const auto bound = definition->tpeVars | to<Set>();
        return freeTypeVariables(*definition) | collect_first([&](const auto &variable) -> Opt<std::pair<Sym, Type::Var>> {
                 return bound ^ contains(variable) ? Opt<std::pair<Sym, Type::Var>>{} : std::pair{definition->name, variable};
               });
      });
  if (unboundStructVariable) {
    emit(diag, clang::DiagnosticsEngine::Level::Error, POLYREGION_DIAG_POLYSTL "Package struct %0 contains unbound type variable %1",
         fqcn(unboundStructVariable->first), canonicalName(unboundStructVariable->second.widen()));
    return;
  }
  const auto program =
      polyfront::packageProgram(functions, r.structs | values() | map([](const auto &x) { return std::move(*x); }) | to_vector());

  auto phasePath = outPath;
  if (C.getLangOpts().CUDA || C.getLangOpts().HIP) phasePath += C.getLangOpts().CUDAIsDevice ? ".device" : ".host";
  polyfront::writeProgramMsgpack(program, phasePath) //
      ^ foreach_total(
          [&](const std::error_code &ec) {
            emit(diag, clang::DiagnosticsEngine::Level::Error, POLYREGION_DIAG_POLYSTL "Cannot open package program output %0: %1",
                 phasePath, ec.message());
          },
          [&](const size_t bytes) {
            emit(diag, clang::DiagnosticsEngine::Level::Remark,
                 POLYREGION_DIAG_POLYSTL "Wrote PolyAST package program %0 (%1 symbols, %2 functions, %3 bytes)", phasePath,
                 std::to_string(exports.size()), std::to_string(program.functions.size()), std::to_string(bytes));
          });
}
