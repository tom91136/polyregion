#pragma once

#include <iostream>
#include <optional>

#include "ast.h"
#include "fmt/format.h"

#include "clang/AST/ASTContext.h"

namespace polyregion::polystl {

using namespace polyregion::polyast;

[[noreturn]] static void raise(const std::string &message, const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
  std::cerr << fmt::format("[{}:{}] {}", file, line, message) << std::endl;
  std::abort();
}

struct Remapper {
  clang::ASTContext &context;
  struct RemapContext {
    std::shared_ptr<StructDef> parent = {};
    bool ctorChain = false;
    Type::Any rtnType = Type::Unit0();
    size_t counter{};
    Vec<Stmt::Any> stmts{};
    Map<std::string, std::shared_ptr<Function>> functions{};
    Map<std::string, std::shared_ptr<StructDef>> structs{};
    Map<std::string, std::shared_ptr<StructLayout>> layouts{};
    Map<std::string, Vec<std::shared_ptr<StructDef>>> parents{};

    template <typename T>
    [[nodiscard]] Pair<T, Vec<Stmt::Any>> scoped(const std::function<T(RemapContext &)> &f,              //
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
    [[nodiscard]] Vec<Stmt::Any> scoped(const std::function<void(RemapContext &)> &f,           //
                                        const Opt<bool> &scopeCtorChain = {},                   //
                                        const Opt<Type::Any> &scopeRtnType = {},                //
                                        const std::shared_ptr<StructDef> &scopeStructName = {}, //
                                        bool persistCounter = false);

    [[nodiscard]] std::shared_ptr<StructDef> findStruct(const std::string &name, const std::string &reason) const;
    [[nodiscard]] bool emptyStruct(const StructDef &def);

    void push(const Stmt::Any &stmt);
    void push(const Vec<Stmt::Any> &xs);

    [[nodiscard]] Named newName(const Type::Any &tpe);
    [[nodiscard]] Expr::Any newVar(const Expr::Any &expr);
    [[nodiscard]] Named newVar(const Type::Any &tpe);
    //    void operator+=(const Remapper &that);
  };

  explicit Remapper(clang::ASTContext &context);
  [[nodiscard]] static Expr::Any integralConstOfType(const Type::Any &tpe, uint64_t value);
  [[nodiscard]] static Expr::Any floatConstOfType(const Type::Any &tpe, double value);

  [[nodiscard]] std::string typeName(const Type::Any &tpe) const;
  [[nodiscard]] std::string nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const;
  [[nodiscard]] Pair<std::string, std::shared_ptr<Function>> handleCall(const clang::FunctionDecl *decl, RemapContext &r);
  [[nodiscard]] Type::Any handleType(clang::QualType qual, RemapContext &r) const;
  [[nodiscard]] std::shared_ptr<StructDef> handleRecord(const clang::RecordDecl *decl, RemapContext &r) const;
  [[nodiscard]] Expr::Any handleExpr(const clang::Expr *expr, RemapContext &r);
  void handleStmt(const clang::Stmt *root, RemapContext &expr);
};

} // namespace polyregion::polystl
