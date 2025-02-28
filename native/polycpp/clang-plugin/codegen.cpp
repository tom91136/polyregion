#include <iostream>

#include "aspartame/all.hpp"
#include "clang_utils.h"
#include "codegen.h"
#include "polyregion/types.h"
#include "remapper.h"

#include "clang/AST/Attr.h"
#include "clang/AST/RecordLayout.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

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
  stmts.push_back(Stmt::Return(Expr::Unit0Const()));

  auto recv = Arg(Named("#this", Type::Ptr(Type::Struct(parentDef->name), {}, TypeSpace::Global())), {});

  auto args = functor.parameters() | map([&](const clang::ParmVarDecl *x) {
                auto local = x->attrs() | exists([](const clang::Attr *a) {
                               if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated) {
                                 return annotated->getAnnotation() == "__polyregion_local";
                               }
                               return false;
                             });

                auto tpe = remapper.handleType(x->getType(), r);

                auto annotatedTpe = tpe.get<Type::Ptr>() ^
                                    fold([&](auto p) { return Type::Ptr(p.comp, p.length, local ? TypeSpace::Local() : p.space).widen(); },
                                         [&]() { return tpe; });

                return Arg(Named(x->getName().str(), annotatedTpe), {});
              })             //
              | append(recv) //
              | to_vector();

  auto f0 = std::make_shared<Function>("_main", args, rtnTpe, stmts,
                                       std::set<FunctionAttr::Any>{FunctionAttr::Exported(), FunctionAttr::Entry()});

  auto program = Program(r.structs | values() | map([&](auto &x) { return *x; }) | to_vector(),
                         r.functions | values() | append(f0) | map([&](auto &x) { return *x; }) | to_vector());

  auto exportedStructNames =
      program.functions                                                                                                                   //
      | filter([](auto &f) { return f.attrs ^ contains(FunctionAttr::Exported()); })                                                      //
      | flat_map([](auto &f) { return f.args; })                                                                                          //
      | collect([](auto &a) { return extractComponent(a.named.tpe) ^ flat_map([](auto &t) { return t.template get<Type::Struct>(); }); }) //
      | map([](auto &s) { return s.name; })                                                                                               //
      | to<std::unordered_set>();

  auto layouts = r.layouts | values() | map([&](auto &x) { return std::pair{exportedStructNames ^ contains(x->name), *x}; }) | to_vector();

  if (opts.verbose) {
    diag.Report(loc,
                diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Remark, "[PolySTL] Remapped program [%0, sizeof capture=%1]\n%2"))
        << moduleId << C.getTypeSize(parent->getTypeForDecl()) << repr(program);
  }

  auto objects =
      opts.targets                                                                                //
      | filter([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); }) //
      | collect([&](auto &target, auto &features) {
          return compileProgram(opts, program, target, features) ^
                 fold_total([&](const CompileResult &r) -> std::optional<CompileResult> { return r; },
                            [&](const std::string &err) -> std::optional<CompileResult> {
                              diag.Report(
                                  diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Warning,
                                                       "[PolySTL] Frontend failed to compile program [%0, target=%1, features=%2]\n%3"))
                                  << moduleId << std::string(to_string(target)) << features << err;
                              return std::nullopt;
                            }) ^
                 map([&](auto &x) { return std::tuple{target, features, x}; });
        }) //
      |
      collect([&](auto &target, auto &features, auto &result) -> std::optional<polyfront::KernelObject> {
        diag.Report(loc, diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Remark,
                                              "[PolySTL] Compilation events for [%0, target=%1, features=%2]\n%3"))
            << moduleId << std::string(to_string(target)) << features << repr(result);

        if (auto bin = result.binary; !bin) {
          diag.Report(loc, diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Warning,
                                                "[PolySTL] Backend failed to compile program [%0, target=%1, features=%2]\nReason: %3"))
              << moduleId << std::string(to_string(target)) << features << result.messages;
          return std::nullopt;
        } else {

          if (!result.messages.empty()) {
            diag.Report(loc, diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Warning,
                                                  "[PolySTL] Backend emitted binary (%0KB) with warnings [%1, target=%2, features=%3]\n%4"))
                << std::to_string(static_cast<float>(bin->size()) / 1000.f) << moduleId << std::string(to_string(target)) << features
                << result.messages;

          } else {
            diag.Report(loc, diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Remark,
                                                  "[PolySTL] Backend emitted binary (%0KB) [%1, target=%2, features=%3]"))
                << std::to_string(static_cast<float>(bin->size()) / 1000.f) << moduleId << std::string(to_string(target)) << features
                << result.messages;
          }

          if (auto format = runtime::moduleFormatOf(target)) {
            return polyfront::KernelObject{
                *format,                                                                                                         //
                *format == runtime::ModuleFormat::Object ? runtime::PlatformKind::HostThreaded : runtime::PlatformKind::Managed, //
                result.features,                                                                                                 //
                std::string(bin->begin(), bin->end())                                                                            //
            };
          } else {
            diag.Report(loc, diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Remark,
                                                  "[PolySTL] Backend emitted binary for unknown target [%1, target=%2, features=%3]"))
                << moduleId << std::string(to_string(target)) << features << result.messages;
            return std::nullopt;
          }
        }
      }) //
      | to_vector();
  return polyfront::KernelBundle{moduleId, objects, layouts, program_to_json(program).dump()};
}
