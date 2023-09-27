#include <iostream>

#include "clang_utils.h"
#include "codegen.h"
#include "remapper.h"

#include "clang/AST/RecordLayout.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include "fmt/format.h"

using namespace polyregion::variants;
using namespace polyregion::polyast;

static std::optional<CompileResult> compileIt(Program &p) {
  auto data = nlohmann::json::to_msgpack(hashed_to_json(program_to_json(p)));

  //    llvm::sys::fs::createTemporaryFile("","", )

  llvm::SmallString<64> inputPath;
  auto inputCreateEC = llvm::sys::fs::createTemporaryFile("", "", inputPath);
  if (inputCreateEC) {
    llvm::errs() << "Failed to create temp input file: " << inputCreateEC.message() << "\n";
    return {};
  }

  llvm::SmallString<64> outputPath;
  auto outputCreateEC = llvm::sys::fs::createTemporaryFile("", "", outputPath);
  if (outputCreateEC) {
    llvm::errs() << "Failed to create temp output file: " << outputCreateEC.message() << "\n";
    return {};
  }

  std::error_code streamEC;
  llvm::raw_fd_ostream File(inputPath, streamEC, llvm::sys::fs::OF_None);
  if (streamEC) {
    llvm::errs() << "Failed to open file: " << streamEC.message() << "\n";
    return {};
  }
  File.write(reinterpret_cast<const char *>(data.data()), data.size());
  File.flush();
  llvm::outs() << "Wrote " << inputPath.str() << " \n";

  int code = llvm::sys::ExecuteAndWait(                                                              //
      "/home/tom/polyregion/native/cmake-build-debug-clang/compiler/polyc",                          //
      {"polyc", inputPath.str(), "--out", outputPath.str(), "--target", "host", "--arch", "native"}, //
      {{}}                                                                                           //
  );

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputPath);

  if (auto Err = BufferOrErr.getError()) {
    llvm::errs() << llvm::errorCodeToError(Err) << "\n";
    return {};
  } else {

    return compileresult_from_json(nlohmann::json::from_msgpack((*BufferOrErr)->getBufferStart(), (*BufferOrErr)->getBufferEnd()));
  }
}

std::string polyregion::polystl::generate(clang::ASTContext &C, const clang::CXXRecordDecl *parent, clang::QualType returnTpe,
                                          const clang::Stmt *body) {
  polyregion::polystl::Remapper remapper(C);

  auto r = polyregion::polystl::Remapper::RemapContext{};
  r.parent = r.structs.find(remapper.handleRecord(parent, r))->second;
  remapper.handleStmt(body, r);
  //
  //
  auto f0 =
      polyregion::polyast::Function(polyregion::polyast::Sym({"kernel"}), {}, {}, {}, {}, {}, remapper.handleType(returnTpe), r.stmts);

  std::vector<Function> fns;
  std::vector<StructDef> structDefs;
  std::cout << "=========" << std::endl;

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

  auto &layout = C.getASTRecordLayout(parent);

  std::cout << C.getTypeSize(parent->getTypeForDecl())  << "\n";

//  auto result = compileIt(p);
//  if (result) {
//    std::cout << repr(*result) << std::endl;
//  } else {
//    std::cout << "No compile!" << std::endl;
//  }
  //
  std::vector<std::string> fieldDecl;
  std::vector<std::string> ctorArgs;
  std::vector<std::string> ctorInits;
  std::vector<std::string> ctorAps;

  for (auto c : parent->captures()) {

    switch (c.getCaptureKind()) {
      case clang::LambdaCaptureKind::LCK_This: break;
      case clang::LambdaCaptureKind::LCK_StarThis: break;
      case clang::LambdaCaptureKind::LCK_ByCopy: break;
      case clang::LambdaCaptureKind::LCK_ByRef: break;
      case clang::LambdaCaptureKind::LCK_VLAType: break;
    }

    if (c.capturesVariable()) {
      auto var = c.getCapturedVar();
      auto tpe = print_type(var->getType().getDesugaredType(C), C);
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
  return "";
}
