#include <iostream>

#include "aspartame/all.hpp"
#include "clang_utils.h"
#include "codegen.h"
#include "polyregion/types.h"
#include "remapper.h"

#include "clang/AST/Attr.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

static std::variant<std::string, CompileResult> compileProgram(const polystl::Options &opts,      //
                                                               const Program &p,                  //
                                                               const compiletime::Target &target, //
                                                               const std::string &arch) {
  auto data = nlohmann::json::to_msgpack(hashed_to_json(program_to_json(p)));

  llvm::SmallString<64> inputPath;
  auto inputCreateEC = llvm::sys::fs::createTemporaryFile("", "", inputPath);
  if (inputCreateEC) return "Failed to create temp input file: " + inputCreateEC.message();

  llvm::SmallString<64> outputPath;
  auto outputCreateEC = llvm::sys::fs::createTemporaryFile("", "", outputPath);
  if (outputCreateEC) return "Failed to create temp output file: " + outputCreateEC.message();

  std::error_code streamEC;
  llvm::raw_fd_ostream file(inputPath, streamEC, llvm::sys::fs::OF_None);
  if (streamEC) return "Failed to open file: " + streamEC.message();

  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  file.flush();

  std::vector<llvm::StringRef> args{//
                                    "",         "--polyc",         inputPath.str(), "--out", outputPath.str(),
                                    "--target", to_string(target), "--arch",        arch};

  if (opts.verbose) {
    (llvm::errs() << (args | prepend(opts.executable) | mk_string(" ", [](auto &s) { return s.data(); })) << "\n").flush();
  }

  if (int code = llvm::sys::ExecuteAndWait(opts.executable, args); code != 0)
    return "Non-zero exit code for task: " + (args ^ mk_string(" ", [](auto &s) { return s.str(); }));

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputPath);

  if (auto Err = BufferOrErr.getError()) return "Failed to read output buffer: " + toString(llvm::errorCodeToError(Err));
  else return compileresult_from_json(nlohmann::json::from_msgpack((*BufferOrErr)->getBufferStart(), (*BufferOrErr)->getBufferEnd()));
}

polystl::KernelBundle polystl::generate(const Options &opts,
                                        clang::ASTContext &C,                //
                                        clang::DiagnosticsEngine &diag,      //
                                        const std::string &moduleId,         //
                                        const clang::CXXMethodDecl &functor, //
                                        const clang::SourceLocation &loc,    //
                                        runtime::PlatformKind kind) {

  polystl::Remapper remapper(C);

  auto parent = functor.getParent();
  auto returnTpe = functor.getReturnType();
  auto body = functor.getBody();

  auto r = polystl::Remapper::RemapContext{};
  auto parentName = remapper.handleRecord(parent, r);
  StructDef &parentDef = r.structs.find(parentName)->second;

  auto rtnTpe = remapper.handleType(returnTpe, r);

  auto stmts = r.scoped([&](auto &r) { remapper.handleStmt(body, r); }, false, rtnTpe, parentName);
  stmts.push_back(Stmt::Return(Expr::Alias(Term::Unit0Const())));

  auto recv =
      Arg(Named("this", Type::Ptr(Type::Struct(Sym({parentName}), parentDef.tpeVars, {}, parentDef.parents), {}, TypeSpace::Global())), {});

  auto args = functor.parameters() //
              | map([&](const clang::ParmVarDecl *x) {
                  auto local = x->attrs() | exists([](const clang::Attr *a) {
                                 if (auto annotated = llvm::dyn_cast<clang::AnnotateAttr>(a); annotated) {
                                   return annotated->getAnnotation() == "__polyregion_local";
                                 }
                                 return false;
                               });

                  auto tpe = remapper.handleType(x->getType(), r);

                  auto annotatedTpe =
                      tpe.get<Type::Ptr>() ^
                      fold([&](auto p) { return Type::Ptr(p.component, p.length, local ? TypeSpace::Local() : p.space).widen(); },
                           [&]() { return tpe; });

                  return Arg(Named(x->getName().str(), annotatedTpe), {});
                })           //
              | append(recv) //
              | to_vector();

  auto f0 = polyast::Function(polyast::Sym({"kernel"}), {}, {}, args, {}, {}, rtnTpe, stmts, FunctionKind::Exported());

  auto p = Program(f0, r.functions ^ values(), r.structs ^ values());

  if (opts.verbose) {
    diag.Report(loc,
                diag.getCustomDiagID(clang::DiagnosticsEngine::Level::Remark, "[PolySTL] Remapped program [%0, sizeof capture=%1]\n%2"))
        << moduleId << C.getTypeSize(parent->getTypeForDecl()) << repr(p);
  }

  auto objects =
      opts.targets                                                                                //
      | filter([&](auto &target, auto &) { return kind == runtime::targetPlatformKind(target); }) //
      | collect([&](auto &target, auto &features) {
          return compileProgram(opts, p, target, features) ^
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
      collect([&](auto &target, auto &features, auto &result) -> std::optional<KernelObject> {
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

          if (auto format = std::optional{runtime::targetFormat(target)}; format) {
            return KernelObject{
                //
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
  return KernelBundle{moduleId, objects, program_to_json(p).dump()};
}
