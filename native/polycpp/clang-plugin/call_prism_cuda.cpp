#include <string_view>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

namespace {

constexpr std::string_view cubShuffleDownLegacy = "cub::SHFL_DOWN_SYNC";
constexpr std::string_view cubShuffleUpLegacy = "cub::SHFL_UP_SYNC";
constexpr std::string_view cubVaPrintf = "cub::va_printf";
constexpr std::string_view thrustThrowOnError = "thrust::cuda_cub::throw_on_error";
constexpr std::string_view thrustTerminateWithMessage = "thrust::system::cuda::detail::terminate_with_message";

} // namespace

static Opt<MatchedCall> cubLegacyShuffle(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if (call.getNumArgs() != 4 || (name != cubShuffleDownLegacy && name != cubShuffleUpLegacy)) return {};
  const auto down = name == cubShuffleDownLegacy;
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, down](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto args = lowerArguments(*expression, self, r);
                       const auto tpe = self.handleType(expression->getType(), r);
                       return Expr::SpecOp(down ? Spec::Any(Spec::GpuShuffleDown(args[0], args[1], args[2], args[3], tpe))
                                                : Spec::Any(Spec::GpuShuffleUp(args[0], args[1], args[2], args[3], tpe)));
                     }},
                     false};
}

static Opt<MatchedCall> cudaIgnoredHelper(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool error = name == thrustThrowOnError && call.getNumArgs() >= 1;
  const bool sink = name == thrustTerminateWithMessage || name == cubVaPrintf;
  if (!error && !sink) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) { return unitAfterArguments(*expression, self, r); }},
                     false};
}

Vector<CallPrism> cudaPrisms() { return {cubLegacyShuffle, cudaIgnoredHelper}; }

} // namespace polyregion::polystl::call_prism
