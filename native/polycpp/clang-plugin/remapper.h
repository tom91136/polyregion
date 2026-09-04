#pragma once

#include <optional>

#include "clang/AST/ASTContext.h"

#include "fmt/format.h"

#include "polyregion/error.h"

#include "ast.h"

namespace polyregion::polystl {

using namespace polyregion::polyast;
using polyregion::raise;

[[nodiscard]] std::string declName(const clang::NamedDecl *decl);

struct Remapper {
  clang::ASTContext &context;
  bool emitPackageProgramMode = false;
  mutable Map<std::string, Set<std::string>> readOnlyMembers{};
  struct BitFieldInfo {
    Named storage;
    Type::Any valueTpe;
    uint64_t bitOffset;
    uint64_t bitWidth;
  };
  struct Cleanup {
    clang::QualType type;
    Named instance;
  };
  struct RemapContext {
    std::shared_ptr<StructDef> parent = {};
    const clang::FunctionDecl *function = {};
    const clang::CXXRecordDecl *entryCapture = {};
    Set<const clang::CXXRecordDecl *> globalCaptures{};
    Set<const clang::ValueDecl *> capturesInScope{};
    Map<const clang::ValueDecl *, Named> captureValueBindings{};
    TypeSpace::Any thisSpace = TypeSpace::Global();
    bool ctorChain = false;
    Opt<Named> constructInto{};
    Opt<Term::Select> constructArrayInto{};
    Opt<Named> aggregateThis{};
    Type::Any rtnType = Type::Unit0();
    size_t counter{};
    Vector<Stmt::Any> stmts{};
    Map<std::string, std::shared_ptr<Function>> functions{};
    Map<std::string, std::shared_ptr<StructDef>> structs{};
    Map<std::string, std::shared_ptr<StructLayout>> layouts{};
    Map<std::string, Vector<std::shared_ptr<StructDef>>> parents{};
    Map<std::string, BitFieldInfo> bitFields{};
    Set<std::string> packageVariables{};
    Map<std::string, Type::Var> packageVariableTypes{};
    Set<std::string> callableVariables{};
    Map<std::string, std::string> indirectOffloadEntries{};
    Map<std::string, uint64_t> indirectLaunchBlockSizes{};
    Vector<Named> syclLogicalGlobalSizes{};
    Map<const clang::ValueDecl *, Type::Any> valueTypes{};
    Map<std::string, Named> exceptionWhats{};
    Map<std::string, Named> exceptionCodes{};
    Set<std::string> incompleteExceptionWhats{};
    Vector<std::function<void(RemapContext &)>> onContinue{};
    Vector<std::function<void(RemapContext &)>> onBreak{};
    Vector<Vector<Cleanup>> cleanups{};
    size_t loopFrame{};
    // A raise unwinds to the innermost enclosing try, or the whole function when none exists.
    size_t tryFrame{};
    bool inCatch = false;
    bool cleanupsSuspended = false;
    Opt<clang::QualType> boundTemporaryType{};

    template <typename T>
    [[nodiscard]] Pair<T, Vector<Stmt::Any>> scoped(const std::function<T(RemapContext &)> &f,              //
                                                    const Opt<bool> &scopeCtorChain = {},                   //
                                                    const Opt<Type::Any> &scopeRtnType = {},                //
                                                    const std::shared_ptr<StructDef> &scopeStructName = {}, //
                                                    const bool persistFunctionState = true) {
      RemapContext r = *this;
      r.parent = scopeStructName ? scopeStructName : parent;
      r.ctorChain = scopeCtorChain.value_or(ctorChain);
      r.constructInto.reset();
      r.constructArrayInto.reset();
      r.aggregateThis.reset();
      r.rtnType = scopeRtnType.value_or(rtnType);
      r.stmts.clear();
      if (!persistFunctionState) {
        r.counter = 0;
        r.exceptionWhats.clear();
        r.exceptionCodes.clear();
        r.incompleteExceptionWhats.clear();
        r.cleanups.clear();
        r.loopFrame = 0;
        r.tryFrame = 0;
        r.inCatch = false;
        r.cleanupsSuspended = false;
        r.boundTemporaryType.reset();
      }
      auto result = f(r);
      if (persistFunctionState) {
        counter = r.counter;
        incompleteExceptionWhats.insert(r.incompleteExceptionWhats.begin(), r.incompleteExceptionWhats.end());
      }
      functions = r.functions;
      structs = r.structs;
      layouts = r.layouts;
      parents = r.parents;
      bitFields = r.bitFields;
      packageVariables = r.packageVariables;
      packageVariableTypes = r.packageVariableTypes;
      callableVariables = r.callableVariables;
      indirectOffloadEntries = r.indirectOffloadEntries;
      indirectLaunchBlockSizes = r.indirectLaunchBlockSizes;
      globalCaptures = r.globalCaptures;
      return {result, r.stmts};
    }
    // Function-level scopes pass false to reset temporaries, cleanups and exception state.
    [[nodiscard]] Vector<Stmt::Any> scoped(const std::function<void(RemapContext &)> &f,           //
                                           const Opt<bool> &scopeCtorChain = {},                   //
                                           const Opt<Type::Any> &scopeRtnType = {},                //
                                           const std::shared_ptr<StructDef> &scopeStructName = {}, //
                                           bool persistFunctionState = true);

    [[nodiscard]] std::shared_ptr<StructDef> findStruct(const std::string &name, const std::string &reason) const;
    [[nodiscard]] bool emptyStruct(const StructDef &def);
    [[nodiscard]] bool isEmpty(const Type::Struct &s);

    void push(const Stmt::Any &stmt);
    void push(const Vector<Stmt::Any> &xs);

    [[nodiscard]] Named newName(const Type::Any &tpe);
    [[nodiscard]] Term::Any newVar(const Expr::Any &expr);
    [[nodiscard]] Named newVar(const Type::Any &tpe);
  };

  explicit Remapper(clang::ASTContext &context);
  [[nodiscard]] static Expr::Any integralConstOfType(const Type::Any &tpe, uint64_t value);
  [[nodiscard]] static Expr::Any floatConstOfType(const Type::Any &tpe, double value);

  [[nodiscard]] std::string typeName(const Type::Any &tpe) const;
  [[nodiscard]] std::string nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const;
  [[nodiscard]] Named namedOfDecl(const clang::NamedDecl *decl, const Type::Any &tpe) const;
  [[nodiscard]] Term::Select selectPath(RemapContext &r, const Vector<Named> &prefix, const Named &leaf) const;
  [[nodiscard]] Pair<std::string, std::shared_ptr<Function>> handleCall(const clang::FunctionDecl *decl, RemapContext &r);
  [[nodiscard]] Type::Any handleType(clang::QualType qual, RemapContext &r) const;
  [[nodiscard]] Expr::Any conform(RemapContext &r, const Expr::Any &expr, const Type::Any &targetTpe);
  [[nodiscard]] Expr::Any zeroInitialise(RemapContext &r, const Type::Any &tpe);
  void recordExceptionCode(const clang::Stmt &stmt, const Named &code, RemapContext &r) const;
  [[nodiscard]] Opt<Expr::Any> lowerSpecialCall(const clang::CallExpr &call, const clang::FunctionDecl &decl, RemapContext &r);
  [[nodiscard]] bool specialCallPreservesExceptionMetadata(const clang::CallExpr &call, const clang::FunctionDecl &decl) const;
  [[nodiscard]] Type::Any annotateLocalSpace(const clang::ValueDecl *decl, RemapContext &r) const;
  [[nodiscard]] std::shared_ptr<StructDef> handleRecord(const clang::RecordDecl *decl, RemapContext &r) const;
  [[nodiscard]] Expr::Any handleExpr(const clang::Expr *expr, RemapContext &r);
  void handleStmt(const clang::Stmt *root, RemapContext &expr);
  // `downTo` is inclusive: an exit edge destroys every frame it leaves, innermost first
  void unwindCleanups(RemapContext &r, size_t downTo);
  void destroyValue(RemapContext &r, clang::QualType type, const Term::Select &instance);
  void destroyRecord(RemapContext &r, const clang::CXXRecordDecl *record, const Term::Select &instance);
};

} // namespace polyregion::polystl
