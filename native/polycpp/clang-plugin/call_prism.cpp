#include <string_view>

#include "clang/AST/Expr.h"

#include "aspartame/all.hpp"

#include "call_prism_internal.hpp"
#include "clang_utils.h"

namespace polyregion::polystl {

using namespace aspartame;

namespace call_prism {

struct PrismGroup {
  std::string_view name;
  Vector<CallPrism> prisms;
};

Vector<Term::Any> lowerArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r) {
  Vector<Term::Any> result;
  result.reserve(call.getNumArgs());
  for (const auto *arg : call.arguments())
    result.emplace_back(r.newVar(self.handleExpr(arg, r)));
  return result;
}

Expr::Any unitAfterArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r) {
  (void)lowerArguments(call, self, r);
  return Expr::Alias(Term::Unit0Const());
}

static Opt<MatchedCall> match(const clang::CallExpr &call, const clang::FunctionDecl &decl, const clang::ASTContext &context) {
  // These are source API dialects, not output targets: the remapper creates one
  // target-neutral PolyAST program before backend fanout. Add oneDPL and future
  // CUDA/HIP source rewrites here rather than branching on the selected backend.
  const static Vector<PrismGroup> groups{
      {"core", corePrisms()}, {"polyregion", polyregionPrisms()}, {"cuda", cudaPrisms()}, {"hip", hipPrisms()}};
  const static Vector<PrismGroup> fallbacks{{"host-fallback", coreFallbackPrisms()}};
  Vector<Pair<std::string_view, MatchedCall>> matches;
  const auto collectMatches = [&](const auto &candidates) {
    for (const auto &group : candidates)
      for (const auto prism : group.prisms)
        if (auto matched = prism(call, decl)) matches.emplace_back(group.name, std::move(*matched));
  };
  collectMatches(groups);
  // Attribute-based host sinks are intentionally lower priority than semantic
  // dialect matches (real vendor helpers may themselves be [[noreturn]]).
  if (matches.empty()) collectMatches(fallbacks);
  if (matches.size() > 1) {
    const auto names = matches | map([](const auto &entry) { return std::string(entry.first); }) | mk_string(", ");
    raise("Multiple special-call prisms (" + names + ") matched " + pretty_string(&call, context));
  }
  return matches ^ head_maybe() ^ map([](auto &entry) { return std::move(entry.second); });
}

} // namespace call_prism

Opt<Expr::Any> Remapper::lowerSpecialCall(const clang::CallExpr &call, const clang::FunctionDecl &decl, RemapContext &r) {
  return call_prism::match(call, decl, context) ^ map([&](const auto &matched) { return matched.lower(*this, r); });
}

bool Remapper::specialCallPreservesExceptionMetadata(const clang::CallExpr &call, const clang::FunctionDecl &decl) const {
  return call_prism::match(call, decl, context) ^ exists([](const auto &matched) { return matched.preservesExceptionMetadata; });
}

} // namespace polyregion::polystl
