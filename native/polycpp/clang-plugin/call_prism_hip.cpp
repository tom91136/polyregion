#include <string_view>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

namespace {

constexpr std::string_view rocprimIsSleepScanStateUsed = "rocprim::detail::is_sleep_scan_state_used";
constexpr std::string_view thrustThrowOnError = "thrust::hip_rocprim::throw_on_error";
constexpr std::string_view thrustTerminateWithMessage = "thrust::system::hip::detail::terminate_with_message";

} // namespace

static Opt<MatchedCall> hipSleepScanState(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (call.getNumArgs() != 2 || decl.getQualifiedNameAsString() != rocprimIsSleepScanStateUsed) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto args = lowerArguments(*expression, self, r);
                       const auto output = args[1].get<Term::Select>();
                       if (!output) raise("rocPRIM sleep scan state output did not lower to a selectable value");
                       r.push(Stmt::Mut(*output, Expr::Alias(Term::Bool1Const(false))));
                       return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
                     }},
                     false};
}

static Opt<MatchedCall> hipIgnoredHelper(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool error = name == thrustThrowOnError && call.getNumArgs() >= 1;
  const bool sink = name == thrustTerminateWithMessage;
  if (!error && !sink) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) { return unitAfterArguments(*expression, self, r); }},
                     false};
}

Vector<CallPrism> hipPrisms() { return {hipSleepScanState, hipIgnoredHelper}; }

} // namespace polyregion::polystl::call_prism
