#include <string_view>

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

namespace {

constexpr std::string_view rocprimIsSleepScanStateUsed = "rocprim::detail::is_sleep_scan_state_used";
constexpr std::string_view thrustThrowOnError = "thrust::hip_rocprim::throw_on_error";
constexpr std::string_view thrustTerminateWithMessage = "thrust::system::hip::detail::terminate_with_message";
constexpr std::string_view trampolineKernel = "trampoline_kernel";
constexpr std::string_view devicePartition = "device_partition";
constexpr std::string_view warpShuffleDown = "rocprim::warp_shuffle_down";
constexpr std::string_view warpShuffleUp = "rocprim::warp_shuffle_up";
constexpr std::string_view hostTargetArchitecture = "rocprim::detail::host_target_arch";
constexpr std::string_view deviceArchitecture = "rocprim::detail::get_device_arch";

[[nodiscard]] bool warpSizeAttribute(const clang::Expr &expression) {
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression.IgnoreParenImpCasts());
  const auto *constant = reference ? llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl()) : nullptr;
  return constant && constant->getName().contains("WarpSize");
}

} // namespace

const clang::FunctionDecl *resolveHipKernel(const clang::FunctionDecl &decl) {
  if (decl.getNameAsString().find(trampolineKernel) == std::string::npos) return &decl;
  const auto *primary = decl.getPrimaryTemplate();
  const auto *arguments = decl.getTemplateSpecializationArgs();
  if (!primary || !arguments || arguments->size() < 2 || arguments->get(1).getKind() != clang::TemplateArgument::Integral) return &decl;
  constexpr uint64_t UnknownArchitecture = 0xFFFFFFFFULL;
  if (arguments->get(1).getAsIntegral().getZExtValue() == UnknownArchitecture) return &decl;
  for (const auto *specialisation : primary->specializations()) {
    const auto *candidate = specialisation->getTemplateSpecializationArgs();
    if (!candidate || candidate->size() != arguments->size() || candidate->get(1).getKind() != clang::TemplateArgument::Integral
        || candidate->get(1).getAsIntegral().getZExtValue() != UnknownArchitecture)
      continue;
    bool same = true;
    for (size_t i = 0; i < candidate->size() && same; ++i) {
      if (i == 1) continue;
      llvm::FoldingSetNodeID lhs;
      llvm::FoldingSetNodeID rhs;
      candidate->get(i).Profile(lhs, decl.getASTContext());
      arguments->get(i).Profile(rhs, decl.getASTContext());
      same = lhs == rhs;
    }
    if (same && specialisation->doesThisDeclarationHaveABody()) return specialisation;
  }
  return &decl;
}

bool isHipIndirectKernel(const std::string &name) { return name.find(trampolineKernel) != std::string::npos; }

Opt<uint64_t> hipIndirectKernelBlockSize(const clang::FunctionDecl &decl, const Type::Any &argumentType) {
  if (canonicalName(argumentType).find(devicePartition) == std::string::npos) return {};
  const auto *bounds = decl.getAttr<clang::AMDGPUFlatWorkGroupSizeAttr>();
  if (!bounds || !bounds->getMax()) return {};
  clang::Expr::EvalResult value;
  if (!bounds->getMax()->EvaluateAsInt(value, decl.getASTContext()) || !value.Val.isInt()) return {};
  return value.Val.getInt().getLimitedValue();
}

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

static Opt<MatchedCall> hipShuffle(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if (call.getNumArgs() != 3 || (name != warpShuffleDown && name != warpShuffleUp)) return {};
  const auto down = name == warpShuffleDown;
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, down](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto result = self.handleType(expression->getType(), r);
                       const auto value = r.newVar(self.conform(r, self.handleExpr(expression->getArg(0), r), result));
                       const auto delta = r.newVar(self.conform(r, self.handleExpr(expression->getArg(1), r), Type::IntU32()));
                       const auto width = r.newVar(self.conform(r, self.handleExpr(expression->getArg(2), r), Type::IntU32()));
                       const auto clamp = r.newVar(Expr::IntrOp(Intr::Sub(width, Term::IntU32Const(1), Type::IntU32())));
                       const auto mask = Term::IntU32Const(0xFFFFFFFFu);
                       return Expr::SpecOp(down ? Spec::Any(Spec::GpuShuffleDown(value, delta, clamp, mask, result))
                                                : Spec::Any(Spec::GpuShuffleUp(value, delta, clamp, mask, result)));
                     }},
                     false};
}

static Opt<MatchedCall> hipBuiltin(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool barrier = name == "__builtin_amdgcn_s_barrier";
  const bool waveSize = name == "__builtin_amdgcn_wavefrontsize" && call.getNumArgs() == 0;
  const bool countLo = name == "__builtin_amdgcn_mbcnt_lo" && call.getNumArgs() == 2;
  const bool countHi = name == "__builtin_amdgcn_mbcnt_hi" && call.getNumArgs() == 2;
  const bool stream = name == "hipStreamSynchronize" && call.getNumArgs() == 1;
  const bool device = name == "hipDeviceSynchronize" && call.getNumArgs() == 0;
  if (!barrier && !waveSize && !countLo && !countHi && !stream && !device) return {};
  if (countLo || countHi) {
    clang::Expr::EvalResult mask;
    if (!call.getArg(0)->EvaluateAsInt(mask, decl.getASTContext()) || !mask.Val.isInt()
        || mask.Val.getInt().getLimitedValue() != 0xFFFFFFFFu)
      raise("HIP mbcnt currently requires a constant all-lanes mask");
  }
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, barrier, waveSize, countLo, countHi, stream, device](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        if (barrier) {
          (void)lowerArguments(*expression, self, r);
          return Expr::SpecOp(Spec::GpuBarrierLocal());
        }
        if (waveSize) return self.conform(r, Expr::SpecOp(Spec::GpuSubgroupSize()), self.handleType(expression->getType(), r));
        if (countLo || countHi) {
          (void)r.newVar(self.handleExpr(expression->getArg(0), r));
          const auto base = r.newVar(self.conform(r, self.handleExpr(expression->getArg(1), r), Type::IntU32()));
          const auto lane = r.newVar(Expr::SpecOp(Spec::GpuLaneIdx()));
          const auto count =
              countLo ? r.newVar(Expr::IntrOp(Intr::Min(lane, Term::IntU32Const(32), Type::IntU32())))
                      : r.newVar(Expr::IntrOp(Intr::Sub(r.newVar(Expr::IntrOp(Intr::Max(lane, Term::IntU32Const(32), Type::IntU32()))),
                                                        Term::IntU32Const(32), Type::IntU32())));
          return self.conform(r, Expr::IntrOp(Intr::Add(base, count, Type::IntU32())), self.handleType(expression->getType(), r));
        }
        if (stream || device) {
          (void)lowerArguments(*expression, self, r);
          if (stream && !expression->getArg(0)->isNullPointerConstant(self.context, clang::Expr::NPC_ValueDependentIsNotNull))
            raise(fmt::format("Non-default HIP streams are not supported in package code at {}",
                              expression->getArg(0)->getExprLoc().printToString(self.context.getSourceManager())));
          (void)r.newVar(Expr::SpecOp(Spec::RemoteSync(packageContext())));
          return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
        }
        return Expr::Alias(Term::Poison(self.handleType(expression->getType(), r)));
      }},
      false};
}

static Opt<MatchedCall> hipOcklIndex(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  char operation = 0;
  if (name == "__ockl_get_local_id") operation = 'l';
  else if (name == "__ockl_get_group_id") operation = 'g';
  else if (name == "__ockl_get_local_size") operation = 'L';
  else if (name == "__ockl_get_num_groups") operation = 'G';
  else if (name == "__ockl_get_global_id") operation = 'i';
  else if (name == "__ockl_get_global_size") operation = 'S';
  if (!operation || call.getNumArgs() != 1) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, operation](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto dimension = r.newVar(self.conform(r, self.handleExpr(expression->getArg(0), r), Type::IntU32()));
                       const Spec::Any spec = operation == 'l'   ? Spec::GpuLocalIdx(dimension).widen()
                                              : operation == 'g' ? Spec::GpuGroupIdx(dimension).widen()
                                              : operation == 'L' ? Spec::GpuLocalSize(dimension).widen()
                                              : operation == 'G' ? Spec::GpuGroupSize(dimension).widen()
                                              : operation == 'i' ? Spec::GpuGlobalIdx(dimension).widen()
                                                                 : Spec::GpuGlobalSize(dimension).widen();
                       return self.conform(r, Expr::SpecOp(spec), self.handleType(expression->getType(), r));
                     }},
                     false};
}

static Opt<MatchedCall> hipBallot(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if ((name != "__builtin_amdgcn_ballot_w32" && name != "__builtin_amdgcn_ballot_w64") || call.getNumArgs() != 1) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto predicate = r.newVar(self.handleExpr(expression->getArg(0), r));
                       const auto ballot = Expr::SpecOp(Spec::GpuBallot(Term::IntU32Const(0xffffffffu), predicate));
                       // PolyAST defines a portable 32-lane logical subgroup. A w64 source alternative may still be
                       // instantiated in host code for a wave32 target; widen the logical mask to its declared type.
                       return self.conform(r, ballot, self.handleType(expression->getType(), r));
                     }},
                     false};
}

static Opt<MatchedCall> hipRuntime(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  return remoteRuntimePrism(call, decl, "hip");
}

static Opt<MatchedCall> hipErrorState(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if (call.getNumArgs() != 0 || (name != "hipPeekAtLastError" && name != "hipGetLastError")) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       (void)lowerArguments(*expression, self, r);
                       return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
                     }},
                     false};
}

static Opt<MatchedCall> hipHostQuery(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool architecture = (name == hostTargetArchitecture || name == deviceArchitecture) && call.getNumArgs() >= 2;
  const bool deviceOrdinal = name == "hipGetDevice" && call.getNumArgs() == 1;
  const bool attribute = name == "hipDeviceGetAttribute" && call.getNumArgs() == 3;
  const bool warpSize = attribute && warpSizeAttribute(*call.getArg(1));
  if (!architecture && !deviceOrdinal && !attribute) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, architecture, deviceOrdinal, warpSize](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto arguments = lowerArguments(*expression, self, r);
                       if (!architecture && !deviceOrdinal && !warpSize) raise("Unsupported HIP device attribute query");
                       const auto output = arguments[architecture ? 1 : 0];
                       const auto selection = output.get<Term::Select>();
                       if (!selection) raise("HIP host query output did not lower to a selectable value");
                       if (const auto pointer = output.tpe().get<Type::Ptr>()) {
                         const auto stored =
                             architecture    ? r.newVar(self.integralConstOfType(pointer->comp, 0xFFFFFFFFULL))
                             : deviceOrdinal ? r.newVar(self.integralConstOfType(pointer->comp, 0))
                                             : r.newVar(self.conform(
                                                   r, Expr::ForeignCall("polyrt_device_subgroup_size", {packageContext()}, Type::IntU64()),
                                                   pointer->comp));
                         r.push(Stmt::Update(*selection, Term::IntU64Const(0), stored));
                       } else {
                         if (!architecture) raise("HIP host query output is not a pointer");
                         r.push(Stmt::Mut(*selection, self.integralConstOfType(output.tpe(), 0xFFFFFFFFULL)));
                       }
                       return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
                     }},
                     false};
}

Vector<CallPrism> hipPrisms() {
  return {hipSleepScanState, hipShuffle, hipBuiltin, hipOcklIndex, hipBallot, hipRuntime, hipErrorState, hipHostQuery, hipIgnoredHelper};
}

} // namespace polyregion::polystl::call_prism
