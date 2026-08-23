#pragma once

#include <functional>

#include "remapper.h"

namespace polyregion::polystl::call_prism {

using Lowering = std::function<Expr::Any(Remapper &, Remapper::RemapContext &)>;

struct MatchedCall {
  Lowering lower;
  bool preservesExceptionMetadata;
};

using CallPrism = Opt<MatchedCall> (*)(const clang::CallExpr &, const clang::FunctionDecl &);

[[nodiscard]] Vector<Term::Any> lowerArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r);
[[nodiscard]] Expr::Any unitAfterArguments(const clang::CallExpr &call, Remapper &self, Remapper::RemapContext &r);

[[nodiscard]] Vector<CallPrism> corePrisms();
[[nodiscard]] Vector<CallPrism> coreFallbackPrisms();
[[nodiscard]] Vector<CallPrism> polyregionPrisms();
[[nodiscard]] Vector<CallPrism> cudaPrisms();
[[nodiscard]] Vector<CallPrism> hipPrisms();

} // namespace polyregion::polystl::call_prism
