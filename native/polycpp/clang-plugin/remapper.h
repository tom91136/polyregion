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
  mutable Map<std::string, Set<std::string>> readOnlyMembers{};
  struct RemapContext {
    std::shared_ptr<StructDef> parent = {};
    bool ctorChain = false;
    Type::Any rtnType = Type::Unit0();
    size_t counter{};
    Vector<Stmt::Any> stmts{};
    Map<std::string, std::shared_ptr<Function>> functions{};
    Map<std::string, std::shared_ptr<StructDef>> structs{};
    Map<std::string, std::shared_ptr<StructLayout>> layouts{};
    Map<std::string, Vector<std::shared_ptr<StructDef>>> parents{};

    template <typename T>
    [[nodiscard]] Pair<T, Vector<Stmt::Any>> scoped(const std::function<T(RemapContext &)> &f,              //
                                                    const Opt<bool> &scopeCtorChain = {},                   //
                                                    const Opt<Type::Any> &scopeRtnType = {},                //
                                                    const std::shared_ptr<StructDef> &scopeStructName = {}, //
                                                    const bool persistCounter = true) {
      const std::shared_ptr<StructDef> nextParent = scopeStructName ? scopeStructName : parent;
      RemapContext r{nextParent,
                     scopeCtorChain.value_or(ctorChain),
                     scopeRtnType.value_or(rtnType),
                     persistCounter ? counter : 0,
                     {}, //
                     functions,
                     structs,
                     layouts,
                     parents};
      auto result = f(r);
      if (persistCounter) {
        counter = r.counter;
      }
      functions = r.functions;
      structs = r.structs;
      layouts = r.layouts;
      parents = r.parents;
      return {result, r.stmts};
    }
    // persistCounter=true keeps `_v<N>` temporaries unique across sibling scopes within one
    // function -- polyc's flat stackVarPtrs LUT rejects same-name re-declarations of differing
    // types. Function-level scopes pass false to reset.
    [[nodiscard]] Vector<Stmt::Any> scoped(const std::function<void(RemapContext &)> &f,           //
                                           const Opt<bool> &scopeCtorChain = {},                   //
                                           const Opt<Type::Any> &scopeRtnType = {},                //
                                           const std::shared_ptr<StructDef> &scopeStructName = {}, //
                                           bool persistCounter = true);

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
  [[nodiscard]] Pair<std::string, std::shared_ptr<Function>> handleCall(const clang::FunctionDecl *decl, RemapContext &r);
  [[nodiscard]] Type::Any handleType(clang::QualType qual, RemapContext &r) const;
  [[nodiscard]] Type::Any annotateLocalSpace(const clang::ValueDecl *decl, RemapContext &r) const;
  [[nodiscard]] std::shared_ptr<StructDef> handleRecord(const clang::RecordDecl *decl, RemapContext &r) const;
  [[nodiscard]] Expr::Any handleExpr(const clang::Expr *expr, RemapContext &r);
  void handleStmt(const clang::Stmt *root, RemapContext &expr);
};

} // namespace polyregion::polystl
