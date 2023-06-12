#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "fmt/core.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "ast.h"
#include "clang_utils.h"
#include "compiler.h"
#include "remapper.h"
namespace {

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

using namespace polyregion::polystl;

template <typename F> class InlineMatchCallback : public clang::ast_matchers::MatchFinder::MatchCallback {
  F f;

public:
  explicit InlineMatchCallback(F f) : f(f) {}

private:
  void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override { f(result); }
};

static std::string printType(clang::QualType type, clang::ASTContext &c) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os, c.getPrintingPolicy());
  return s;
}

static std::string replaceAll(std::string subject, const std::string &search, const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}

constexpr const char *policyTag = "policyTag";
constexpr const char *operatorCallTag = "operatorCallTag";
constexpr const char *transformCall = "transformCall";

static std::string underlyingToken(clang::Expr *stmt, clang::ASTContext &c) {
  auto range = clang::CharSourceRange::getTokenRange(stmt->getBeginLoc(), stmt->getEndLoc());
  return clang::Lexer::getSourceText(range, c.getSourceManager(), c.getLangOpts()).str();
}

// Recursively (following CallExpr too) finds the first call to a () operator and records the concrete method called
class OverloadTargetVisitor : public clang::RecursiveASTVisitor<OverloadTargetVisitor> {
  std::optional<clang::CXXMethodDecl *> target;
  const clang::CXXRecordDecl *owner;

public:
  explicit OverloadTargetVisitor(const clang::CXXRecordDecl *owner) : owner(owner) {}
  std::optional<clang::CXXMethodDecl *> run(clang::Stmt *stmt) {
    TraverseStmt(stmt);
    return target;
  }
  bool VisitCallExpr(clang::CallExpr *S) {
    target = OverloadTargetVisitor(owner).run(S->getCalleeDecl()->getBody());
    return !target.has_value();
  }

  bool VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S) {
    if (S->getOperator() != clang::OverloadedOperatorKind::OO_Call) return true;
    if (auto cxxMethodDecl = llvm::dyn_cast_if_present<clang::CXXMethodDecl>(S->getCalleeDecl())) {
      if (cxxMethodDecl->getParent() == owner) target.emplace(cxxMethodDecl);
    }
    return !target.has_value();
  }
};

// Recursively (following CallExpr too) finds the first call to a () operator and records the concrete method called
class PolyASTRewriteVisitor : public clang::RecursiveASTVisitor<PolyASTRewriteVisitor> {

public:
  bool VisitReturnStmt(clang::ReturnStmt *S) {
    //    S->getRetValue();
    return true;
  }
};

class OutlineConsumer : public clang::ASTConsumer {

private:
  clang::FileID &FileID_;
  clang::Rewriter &FileRewriter_;
  bool &FileRewriteError_;

public:
  OutlineConsumer(clang::FileID &FileID, clang::Rewriter &FileRewriter, bool &FileRewriteError)
      : FileID_(FileID), FileRewriter_(FileRewriter), FileRewriteError_(FileRewriteError) {}

  std::unordered_map<int64_t, std::pair<clang::FunctionDecl *, clang::CXXRecordDecl *>> drain{};

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override {

    for (const auto &D : DG) {
      // For each lambda expression, there's a top-level specialisation
      if (const auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(D)) {
        if (const auto &templateDecl = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(methodDecl->getPrimaryTemplate())) {
          for (const auto &fnDecl : templateDecl->specializations()) {
            if (const auto &record = llvm::dyn_cast<clang::CXXRecordDecl>(templateDecl->getTemplatedDecl()->getParent())) {
              if (record->isLambda()) {
                drain.insert({fnDecl->getID(), std::pair{fnDecl, record}});
              }
            }
          }
        }
      }
    }

    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &context) override {

    //    for (auto [id, value] : drain) {
    //      std::cout << "=====" << id << std::endl;
    //      auto &[fn, record] = value;
    //      fn->getSourceRange().dump(context.getSourceManager());
    //      record->dump();
    //      fn->dumpColor();
    //      for (auto c : record->captures()) {
    //        c.getCapturedVar()->dumpColor();
    //      }
    //    }

    std::cout << "===" << std::endl;

    using namespace clang::ast_matchers;

    const static std::unordered_set<std::string> pstlParallelPolicies = {
        "__pstl::execution::parallel_unsequenced_policy",
        "__pstl::execution::parallel_policy",
    };

    auto transformCallMatcher =            //
        callExpr(                          //
            callee(functionDecl(anyOf(     //
                hasName("std::transform"), //
                hasName("std::for_each"),  //
                hasName("std::for_each_n") //
                ))),
            hasArgument(0, declRefExpr(to(varDecl(hasType(cxxRecordDecl().bind(policyTag)))))))
            .bind(transformCall);

    struct Callsite {
      const clang::CallExpr *callExpr;         // decl of the std::transform call
      const clang::Expr *callLambdaArgExpr;    // decl of the lambda arg
      const clang::FunctionDecl *calleeDecl;   // decl of the specialised std::transform
      const clang::CXXMethodDecl *functorDecl; // decl of the specialised lambda functor
    };
    struct Failure {
      const clang::Stmt *callExpr;
      std::string reason;
    };
    std::vector<std::variant<Failure, Callsite>> results;

    InlineMatchCallback cb([&](const MatchFinder::MatchResult &result) {
      if (auto record = result.Nodes.getNodeAs<clang::CXXRecordDecl>(policyTag)) {
        // Make sure the execution policy tag is a parallel one first
        if (pstlParallelPolicies.count(record->getQualifiedNameAsString()) == 0) return;
        if (auto algCallExpr = result.Nodes.getNodeAs<clang::CallExpr>(transformCall)) {
          auto lastArgExpr = algCallExpr->getArg(algCallExpr->getNumArgs() - 1)->IgnoreUnlessSpelledInSource();
          auto fnDecl = algCallExpr->getDirectCallee();
          if (auto lambdaArgCxxRecordDecl = lastArgExpr->getType()->getAsCXXRecordDecl()) {
            if (auto target = OverloadTargetVisitor(lambdaArgCxxRecordDecl).run(fnDecl->getBody()); target) {
              // We found a valid overload target, validate this by checking whether the decl belongs to the last arg's implicit class.
              if ((*target)->getParent() == lambdaArgCxxRecordDecl)
                results.emplace_back(Callsite{algCallExpr, lastArgExpr, fnDecl, *target});
              else
                results.emplace_back(Failure{algCallExpr, "Target record mismatch:\nLast arg=" + to_string((*target)->getParent()) +
                                                              "\nTarget=" + to_string(lambdaArgCxxRecordDecl)});
            } else {
              results.emplace_back(Failure{algCallExpr, "Cannot find any call `()` to last arg:" + pretty_string(lastArgExpr, context)});
            }
          } else {
            results.emplace_back(Failure{algCallExpr, "Last arg does is not a valid synthesised record type"});
          }
        } else {
          auto root = result.Nodes.getNodeAs<clang::Stmt>(transformCall);
          results.emplace_back(Failure{root, "Unexpected algorithm call expression:" + pretty_string(root, context)});
        }
      } else {
        auto root = result.Nodes.getNodeAs<clang::Stmt>(transformCall);
        results.emplace_back(Failure{root, "Unexpected algorithm execution policy:" +
                                               pretty_string(result.Nodes.getNodeAs<clang::Stmt>(policyTag), context)});
      }
    });
    MatchFinder Finder;
    Finder.addMatcher(transformCallMatcher, &cb);
    Finder.matchAST(context);

    for (auto r : results) {
      std::visit(
          overloaded{
              [&](Failure &f) { //
                std::cout << "Failed:" << pretty_string(f.callExpr, context) << "\nReason:" << f.reason << "\n";
              },
              [&](Callsite &c) { //
                std::cout << pretty_string(c.callExpr, context) << "\n";
                c.callExpr->dumpColor();
                std::cout << "decl=" << std::endl;
                c.functorDecl->dumpColor();
                std::cout << "fnDecl=" << std::endl;
                c.calleeDecl->dumpColor();

                polyregion::polystl::Remapper remapper(context);

                auto owningClass = c.functorDecl->getParent();
                std::vector<polyregion::polyast::Arg> captures;
                for (auto cap : owningClass->captures()) {

                  if (cap.capturesVariable()) {
                    auto var = cap.getCapturedVar();
                    captures.push_back(
                        {polyregion::polyast::Named(var->getDeclName().getAsString(), remapper.handleType(var->getType())), {}});
                  } else if (cap.capturesThis()) {
                    captures.push_back(
                        {polyregion::polyast::Named("this", remapper.handleType(owningClass->getTypeForDecl()->getCanonicalTypeInternal())),
                         {}});
                  } else {
                    throw std::logic_error("Illegal capture");
                  }
                }

                std::cout << "=========" << std::endl;
                auto r = polyregion::polystl::Remapper::RemapContext{};
                remapper.handleStmt(c.functorDecl->getBody(), r);
                auto f0 = polyregion::polyast::Function(polyregion::polyast::Sym({"kernel"}), {}, {}, captures, {}, {},
                                                        remapper.handleType(c.functorDecl->getReturnType()), r.stmts);

                std::vector<Function> fns;
                std::vector<StructDef> structDefs;
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

                std::cout << "AAA" << std::endl;

                std::vector<std::string> fieldDecl;
                std::vector<std::string> ctorArgs;
                std::vector<std::string> ctorInits;
                std::vector<std::string> ctorAps;

                for (auto c : c.functorDecl->getParent()->captures()) {

                  switch (c.getCaptureKind()) {
                    case clang::LambdaCaptureKind::LCK_This: break;
                    case clang::LambdaCaptureKind::LCK_StarThis: break;
                    case clang::LambdaCaptureKind::LCK_ByCopy: break;
                    case clang::LambdaCaptureKind::LCK_ByRef: break;
                    case clang::LambdaCaptureKind::LCK_VLAType: break;
                  }

                  if (c.capturesVariable()) {
                    auto var = c.getCapturedVar();
                    auto tpe = printType(var->getType().getDesugaredType(context), context);
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

                constexpr static const char *s1 = R"cpp(
                       struct {name} {{
                         {fields}
                         {name}({ctorArgs}) :{ctorInits} {{}}
                         inline {applyReturnTpe} operator()({applyArgs}) {{ return 0; }}
                       }};
                       )cpp";

                auto fileName = context.getSourceManager().getFilename(c.callLambdaArgExpr->getExprLoc());
                auto line = context.getSourceManager().getSpellingLineNumber(c.callLambdaArgExpr->getExprLoc());
                auto col = context.getSourceManager().getSpellingColumnNumber(c.callLambdaArgExpr->getExprLoc());

                auto identifier = fmt::format("lambda_{}_{}_{}__", replaceAll(replaceAll(fileName.str(), "/", "_"), ".", "_"), line, col);

                FileRewriter_.InsertText(c.callExpr->getBeginLoc(),
                                         fmt::format(s1,                                                          //
                                                     fmt::arg("name", identifier),                                //
                                                     fmt::arg("fields", fmt::join(fieldDecl, "\n")),              //
                                                     fmt::arg("ctorArgs", fmt::join(ctorArgs, ", ")),             //
                                                     fmt::arg("ctorInits", fmt::join(ctorInits, ", ")),           //
                                                     fmt::arg("applyReturnTpe", "int"),                           //
                                                     fmt::arg("applyArgs", fmt::join(std::vector{"int a"}, ", ")) //
                                                     ));

                FileRewriter_.ReplaceText(c.callLambdaArgExpr->getSourceRange(),
                                          fmt::format("{}({})", identifier, fmt::join(ctorAps, ", ")));

                // TODO invoke

                //                       std::cout << result << std::endl;

              },
          },
          r);
    }

    FileID_ = context.getSourceManager().getMainFileID();
    auto B = FileRewriter_.getRewriteBufferFor(FileID_);
    std::cout << "Mod=" << (B == nullptr) << std::endl;
  }
};

class OutlineAction : public clang::PluginASTAction {
private:
  clang::CompilerInstance *CI_{};
  std::string FileName_;
  clang::FileID FileID_;
  clang::Rewriter FileRewriter_;
  bool FileRewriteError_ = false;

protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef FileName) override {
    CI_ = &CI;
    FileName_ = FileName.str();
    std::cerr << "In: " << FileName_ << std::endl;

    auto &SourceManager = CI.getSourceManager();
    auto &LangOpts = CI.getLangOpts();
    FileRewriter_.setSourceMgr(SourceManager, LangOpts);
    return std::make_unique<OutlineConsumer>(FileID_, FileRewriter_, FileRewriteError_);
  }

  bool ParseArgs(clang::CompilerInstance const &, std::vector<std::string> const &) override { return true; }

  clang::PluginASTAction::ActionType getActionType() override { return clang::PluginASTAction::ReplaceAction; }

  void EndSourceFileAction() override {
    if (FileRewriteError_) return;

    std::cerr << "End Action" << std::endl;
    auto FileRewriteBuffer = FileRewriter_.getRewriteBufferFor(FileID_);

    auto &CommandLineArgs = CI_->getCodeGenOpts().CommandLineArgs;
    std::vector<const char *> ConstCommandLineArgs(CommandLineArgs.size());
    std::transform(CommandLineArgs.begin(), CommandLineArgs.end(), ConstCommandLineArgs.begin(), [](auto &s) { return s.c_str(); });
    //    std::cerr << "Args:"
    //              << "\n";
    //    for (auto &arg : CommandLineArgs) {
    //      std::cerr << "\t" << arg << "\n";
    //    }

    auto &Target = CI_->getTarget();

    // create new compiler instance
    auto CInvNew = std::make_shared<clang::CompilerInvocation>();
    bool CInvNewCreated = clang::CompilerInvocation::CreateFromArgs(*CInvNew, ConstCommandLineArgs, CI_->getDiagnostics());

    assert(CInvNewCreated);

    clang::CompilerInstance CINew;
    CINew.setInvocation(CInvNew);
    CINew.setTarget(&Target);
    CINew.createDiagnostics();

    // create "virtual" input file
    auto &PreprocessorOpts = CINew.getPreprocessorOpts();

    std::vector<std::unique_ptr<llvm::MemoryBuffer>> buffers;
    if (FileRewriteBuffer) {
      // create rewrite buffer
      std::string FileContent = {FileRewriteBuffer->begin(), FileRewriteBuffer->end()};
      //      std::string FileContent = " int main (){return 0;} ";
      std::cerr << "====" << FileName_ << "====" << std::endl;
      std::cerr << FileContent << std::endl;
      std::cerr << "========" << std::endl;
      //      buffers.emplace_back((llvm::MemoryBuffer::getMemBufferCopy(FileContent)));

      PreprocessorOpts.addRemappedFile(FileName_, llvm::MemoryBuffer::getMemBufferCopy(FileContent).release());
    }
    //    for (auto &b : buffers) {
    //      PreprocessorOpts.addRemappedFile(FileName_, b.get());
    //    }

    // generate code
    clang::EmitObjAction EmitObj;
    auto success = CINew.ExecuteAction(EmitObj);

    std::cout << "action=" << success << std::endl;
    //    buffers.clear();

    Program p(Function(Sym({"fn"}), {}, {}, {Arg(Named("a", Type::IntS64()), {})}, {}, {}, Type::IntS64(), //
                       {
                           Stmt::Var(Named("out", Type::IntS64()), {Expr::IntrOp(Intr::Add(Term::Select({}, Named("a", Type::IntS64())),
                                                                                           Term::IntS64Const(42), Type::IntS64()))}),

                           Stmt::Return(Expr::Alias(Term::Select({}, Named("out", Type::IntS64())))),
                       }),
              {}, {});

    auto data = nlohmann::json::to_msgpack(hashed_to_json(program_to_json(p)));

    //    llvm::sys::fs::createTemporaryFile("","", )

    llvm::SmallString<64> inputPath;
    auto inputCreateEC = llvm::sys::fs::createTemporaryFile("", "", inputPath);
    if (inputCreateEC) {
      llvm::errs() << "Failed to create temp input file: " << inputCreateEC.message() << "\n";
      return;
    }

    llvm::SmallString<64> outputPath;
    auto outputCreateEC = llvm::sys::fs::createTemporaryFile("", "", outputPath);
    if (outputCreateEC) {
      llvm::errs() << "Failed to create temp output file: " << outputCreateEC.message() << "\n";
      return;
    }

    std::error_code streamEC;
    llvm::raw_fd_ostream File(inputPath, streamEC, llvm::sys::fs::OF_None);
    if (streamEC) {
      llvm::errs() << "Failed to open file: " << streamEC.message() << "\n";
      return;
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
    } else {

      auto result = compileresult_from_json(nlohmann::json::from_msgpack((*BufferOrErr)->getBufferStart(), (*BufferOrErr)->getBufferEnd()));
      std::cout << repr(result) << std::endl;
    }

    //    llvm::outs() << code << "\n";

    // Git

    //    PreprocessorOpts.clearRemappedFiles();
    //    compile(CI_, FileName_, FileRewriteBuffer->begin(), FileRewriteBuffer->end());
  }
};

} // end namespace

static clang::FrontendPluginRegistry::Add<OutlineAction> X("fire", "create CLI from functions or classes");

static int init() {
  fprintf(stderr, ">>>>>>>>>>>hi\n");
  return 1;
}

static int A = init();
