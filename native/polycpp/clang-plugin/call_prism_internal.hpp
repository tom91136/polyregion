#pragma once

#include <functional>
#include <string_view>

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
[[nodiscard]] Term::Any packageContext();
[[nodiscard]] Term::Select termToSelect(const Term::Any &term, Remapper::RemapContext &r);
[[nodiscard]] Expr::Any referenceTerm(const Term::Any &term, Remapper::RemapContext &r);
[[nodiscard]] Vector<Type::Any> functionTypeArguments(const FunctionDecl &decl);
[[nodiscard]] Opt<MatchedCall> remoteRuntimePrism(const clang::CallExpr &call, const clang::FunctionDecl &decl, std::string_view prefix);

[[nodiscard]] Vector<CallPrism> corePrisms();
[[nodiscard]] Vector<CallPrism> coreFallbackPrisms();
[[nodiscard]] Vector<CallPrism> polyregionPrisms();
[[nodiscard]] Vector<CallPrism> cudaPrisms();
[[nodiscard]] Vector<CallPrism> hipPrisms();
[[nodiscard]] Vector<CallPrism> syclPrisms();

[[nodiscard]] const clang::FunctionDecl *resolveHipKernel(const clang::FunctionDecl &decl);
[[nodiscard]] bool isHipIndirectKernel(const std::string &name);
[[nodiscard]] Opt<uint64_t> hipIndirectKernelBlockSize(const clang::FunctionDecl &decl, const Type::Any &argumentType);

} // namespace polyregion::polystl::call_prism
