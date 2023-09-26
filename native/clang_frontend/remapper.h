#pragma once

#include "clang/AST/ASTContext.h"
#include "llvm/Support/Casting.h"
#include "generated/polyast.h"
#include "variants.hpp"

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
    size_t counter{};
    std::vector<Stmt::Any> stmts{};
    std::unordered_map<std::string, Function> functions{};
    std::unordered_map<std::string, StructDef> structs{};

    template <typename T>
    [[nodiscard]] std::pair<T, std::vector<Stmt::Any>>
    scoped(const std::function<T(RemapContext &)> &f, std::optional<std::string> scopeStructName = {}, bool persistCounter = false) {
      std::optional<std::reference_wrapper<StructDef>> nextParent = parent;
      if (scopeStructName) {
        if (auto it = structs.find(*scopeStructName); it != structs.end()) {
          nextParent = std::optional<std::reference_wrapper<StructDef>>{it->second};
        } else {
          throw std::logic_error("Unexpected parent scope: " + *scopeStructName);
        }
      }
      RemapContext r{nextParent, persistCounter ? 0 : counter, {}, functions, structs};
      auto result = f(r);
      if(!persistCounter){
        counter = r.counter;
      }
      functions = r.functions;
      structs = r.structs;
      return {result, r.stmts};
    }
    [[nodiscard]] std::vector<Stmt::Any> scoped(const std::function<void(RemapContext &)> &f,
                                                std::optional<std::string> scopeStructName = {}, bool persistCounter = false);

    void push(const Stmt::Any &stmt);
    Named newName(const Type::Any &tpe);
    Term::Any newVar(const Expr::Any &expr);
//    void operator+=(const Remapper &that);
  };

public:
  explicit Remapper(clang::ASTContext &context);
  static Term::Any integralConstOfType(const Type::Any &tpe, uint64_t value);
  static Term::Any floatConstOfType(const Type::Any &tpe, double value);
  [[nodiscard]] Type::Any handleType(clang::QualType tpe) const;
  [[nodiscard]] std::string typeName(const Type::Any &tpe) const;
  [[nodiscard]] std::string nameOfRecord(const clang::RecordType *tpe) const;

  std::string handleCall(const clang::FunctionDecl *decl, RemapContext &r);
  std::string handleRecord(const  clang::RecordDecl *decl, RemapContext &r);

  [[nodiscard]] Expr::Any handleExpr(const clang::Expr *root, RemapContext &r);
  void handleStmt(const clang::Stmt *root, RemapContext &r);
};

} // namespace polyregion::polystl
