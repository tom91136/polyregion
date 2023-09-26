#include <iostream>

#include "clang_utils.h"
#include "codegen.h"
#include "remapper.h"

#include "fmt/format.h"

using namespace polyregion::variants;
using namespace polyregion::polyast;

std::string polyregion::polyast::generate(clang::ASTContext &C, const clang::CXXRecordDecl *parent, clang::QualType returnTpe, const clang::Stmt *body) {
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

  //              auto p = Program(f0, fns, structDefs);
  //
  //              //              auto result = compileIt(p);
  //              //              if (result) {
  //              //                std::cout << repr(*result) << std::endl;
  //              //              } else {
  //              //                std::cout << "No compile!" << std::endl;
  //              //              }
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
      auto tpe = polyregion::polystl::print_type(var->getType().getDesugaredType(C), C);
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
