#pragma once

#include "generated/polyast.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Casting.h"

#include <optional>
#include <type_traits>

namespace polyregion::polystl {

template <typename Ret, typename Arg, typename... Rest> Arg arg0_helper(Ret (*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest> Arg arg0_helper(Ret (F::*)(Arg, Rest...));
template <typename Ret, typename F, typename Arg, typename... Rest> Arg arg0_helper(Ret (F::*)(Arg, Rest...) const);
template <typename F> decltype(arg0_helper(&F::operator())) arg0_helper(F);
template <typename T> using arg0_t = decltype(arg0_helper(std::declval<T>()));
template <typename T, typename Node, typename... Fs> std::optional<T> visitDyn(Node n, Fs... fs) {
  std::optional<T> result{};
  auto _ = {[&]() {
    if (!result) {
      if (auto x = llvm::dyn_cast<std::remove_pointer_t<arg0_t<Fs>>>(n)) {
        result = T(fs(x));
      }
    }
    return 0;
  }()...};
  return result;
}

using namespace polyregion::polyast;

struct Remapper {
  clang::ASTContext &context;
  struct RemapContext {
    std::optional<std::reference_wrapper<StructDef>> parent;
    bool ctorChain = false;
    Type::Any rtnType = Type::Unit0();
    size_t counter{};
    std::vector<Stmt::Any> stmts{};
    std::unordered_map<std::string, Function> functions{};
    std::unordered_map<std::string, StructDef> structs{};

    template <typename T>
    [[nodiscard]] std::pair<T, std::vector<Stmt::Any>> scoped(const std::function<T(RemapContext &)> &f,         //
                                                              const std::optional<bool> &scopeCtorChain = {},    //
                                                              const std::optional<Type::Any> &scopeRtnType = {}, //
                                                              std::optional<std::string> scopeStructName = {},   //
                                                              bool persistCounter = true) {
      std::optional<std::reference_wrapper<StructDef>> nextParent = parent;
      if (scopeStructName) {
        if (auto it = structs.find(*scopeStructName); it != structs.end()) {
          nextParent = std::optional<std::reference_wrapper<StructDef>>{it->second};
        } else {
          throw std::logic_error("Unexpected parent scope: " + *scopeStructName);
        }
      }
      RemapContext r{
          nextParent, scopeCtorChain.value_or(ctorChain), scopeRtnType.value_or(rtnType), persistCounter ? counter : 0, {}, functions,
          structs};
      auto result = f(r);
      if (persistCounter) {
        counter = r.counter;
      }
      functions = r.functions;
      structs = r.structs;
      return {result, r.stmts};
    }
    [[nodiscard]] std::vector<Stmt::Any> scoped(const std::function<void(RemapContext &)> &f,      //
                                                const std::optional<bool> &scopeCtorChain = {},    //
                                                const std::optional<Type::Any> &scopeRtnType = {}, //
                                                std::optional<std::string> scopeStructName = {},   //
                                                bool persistCounter = false);

    void push(const Stmt::Any &stmt);
    void push(const std::vector<Stmt::Any> &xs);

    Named newName(const Type::Any &tpe);
    Term::Any newVar(const Expr::Any &expr);
    void newVar0(const Expr::Any &expr);
    Named newVar(const Type::Any &tpe);
    //    void operator+=(const Remapper &that);
  };

public:
  explicit Remapper(clang::ASTContext &context);
  [[nodiscard]] static Term::Any integralConstOfType(const Type::Any &tpe, uint64_t value);
  [[nodiscard]] static Term::Any floatConstOfType(const Type::Any &tpe, double value);

  [[nodiscard]] std::string typeName(const Type::Any &tpe) const;
  [[nodiscard]] std::string nameOfRecord(const clang::RecordType *tpe, RemapContext &r) const;
  [[nodiscard]] std::pair<std::string, Function> handleCall(const clang::FunctionDecl *decl, RemapContext &r);
  [[nodiscard]] Type::Any handleType(clang::QualType tpe, RemapContext &r) const;
  [[nodiscard]] std::string handleRecord(const clang::RecordDecl *decl, RemapContext &r) const;
  [[nodiscard]] Expr::Any handleExpr(const clang::Expr *expr, RemapContext &r);
  void handleStmt(const clang::Stmt *root, RemapContext &expr);
};

} // namespace polyregion::polystl
