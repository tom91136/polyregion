#include <string_view>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

static Opt<MatchedCall> polyregionIntrinsic(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  constexpr std::string_view prefix = "__polyregion_gpu_";
  const auto qualifiedName = decl.getQualifiedNameAsString();
  const bool assertion = qualifiedName == "__polyregion_builtin_assert";
  if (!assertion && !std::string_view(qualifiedName).starts_with(prefix)) return {};

  const auto builtinName = assertion ? std::string{"assert"} : "gpu_" + qualifiedName.substr(prefix.size());
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, builtinName](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto args = lowerArguments(*expression, self, r);
                       const auto resultType = [&] { return self.handleType(expression->getType(), r); };
                       const auto poison = [&] { return Expr::Alias(Term::Poison(resultType())).widen(); };
                       const auto spec = [&](const size_t arity, const auto &mk) -> Expr::Any {
                         if (args.size() != arity) return poison();
                         return Expr::SpecOp(mk());
                       };
                       const auto atomic = [&](const AtomicOp::Any &op) {
                         return Spec::GpuAtomicRMW(op, args[0], args[1], MemScope::Device(), MemOrder::Relaxed(), resultType());
                       };

                       if (builtinName == "gpu_global_idx") return spec(1, [&] { return Spec::GpuGlobalIdx(args[0]); });
                       if (builtinName == "gpu_global_size") return spec(1, [&] { return Spec::GpuGlobalSize(args[0]); });
                       if (builtinName == "gpu_group_idx") return spec(1, [&] { return Spec::GpuGroupIdx(args[0]); });
                       if (builtinName == "gpu_group_size") return spec(1, [&] { return Spec::GpuGroupSize(args[0]); });
                       if (builtinName == "gpu_local_idx") return spec(1, [&] { return Spec::GpuLocalIdx(args[0]); });
                       if (builtinName == "gpu_local_size") return spec(1, [&] { return Spec::GpuLocalSize(args[0]); });
                       if (builtinName == "gpu_barrier_global") return spec(0, [&] { return Spec::GpuBarrierGlobal(); });
                       if (builtinName == "gpu_barrier_local") return spec(0, [&] { return Spec::GpuBarrierLocal(); });
                       if (builtinName == "gpu_barrier_all") return spec(0, [&] { return Spec::GpuBarrierAll(); });
                       if (builtinName == "gpu_fence_global") return spec(0, [&] { return Spec::GpuFenceGlobal(); });
                       if (builtinName == "gpu_fence_local") return spec(0, [&] { return Spec::GpuFenceLocal(); });
                       if (builtinName == "gpu_fence_all") return spec(0, [&] { return Spec::GpuFenceAll(); });
                       if (builtinName == "gpu_lane_idx") return spec(0, [&] { return Spec::GpuLaneIdx(); });
                       if (builtinName == "gpu_subgroup_size") return spec(0, [&] { return Spec::GpuSubgroupSize(); });
                       if (builtinName == "gpu_shuffle_down_u32")
                         return spec(4, [&] { return Spec::GpuShuffleDown(args[0], args[1], args[2], args[3], resultType()); });
                       if (builtinName == "gpu_shuffle_up_u32")
                         return spec(4, [&] { return Spec::GpuShuffleUp(args[0], args[1], args[2], args[3], resultType()); });
                       if (builtinName == "gpu_shuffle_idx_u32")
                         return spec(4, [&] { return Spec::GpuShuffleIdx(args[0], args[1], args[2], args[3], resultType()); });
                       if (builtinName == "gpu_shuffle_xor_u32")
                         return spec(4, [&] { return Spec::GpuShuffleXor(args[0], args[1], args[2], args[3], resultType()); });
                       if (builtinName == "gpu_subgroup_barrier") return spec(1, [&] { return Spec::GpuSubgroupBarrier(args[0]); });
                       if (builtinName == "gpu_ballot") return spec(2, [&] { return Spec::GpuBallot(args[0], args[1]); });
                       if (builtinName == "gpu_vote_any") return spec(2, [&] { return Spec::GpuVoteAny(args[0], args[1]); });
                       if (builtinName == "gpu_vote_all") return spec(2, [&] { return Spec::GpuVoteAll(args[0], args[1]); });
                       if (builtinName == "gpu_atomic_xchg_u32") return spec(2, [&] { return atomic(AtomicOp::Xchg()); });
                       if (builtinName == "gpu_atomic_add_u32") return spec(2, [&] { return atomic(AtomicOp::Add()); });
                       if (builtinName == "gpu_atomic_sub_u32") return spec(2, [&] { return atomic(AtomicOp::Sub()); });
                       if (builtinName == "gpu_atomic_min_u32") return spec(2, [&] { return atomic(AtomicOp::Min()); });
                       if (builtinName == "gpu_atomic_max_u32") return spec(2, [&] { return atomic(AtomicOp::Max()); });
                       if (builtinName == "gpu_atomic_and_u32") return spec(2, [&] { return atomic(AtomicOp::And()); });
                       if (builtinName == "gpu_atomic_or_u32") return spec(2, [&] { return atomic(AtomicOp::Or()); });
                       if (builtinName == "gpu_atomic_xor_u32") return spec(2, [&] { return atomic(AtomicOp::Xor()); });
                       if (builtinName == "gpu_volatile_load_u32")
                         return spec(1, [&] { return Spec::GpuVolatileLoad(args[0], resultType()); });
                       if (builtinName == "gpu_volatile_store_u32")
                         return spec(2, [&] { return Spec::GpuVolatileStore(args[0], args[1]); });
                       if (builtinName == "assert") return spec(2, [&] { return Spec::Assert(args[0], args[1]); });
                       return poison();
                     }},
                     false};
}

static Opt<MatchedCall> usmHostAccess(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool acquire = name == "polyrt_device_usm_host_acquire" && call.getNumArgs() == 3;
  const bool release = name == "polyrt_device_usm_host_release" && call.getNumArgs() == 4;
  if (!acquire && !release) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, acquire, name](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto raw = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
                       const auto remote = r.newVar(self.conform(r, self.handleExpr(expression->getArg(0), r), raw));
                       if (acquire) {
                         const auto bytes = r.newVar(self.conform(r, self.handleExpr(expression->getArg(1), r), Type::IntU64()));
                         const auto mode = r.newVar(self.conform(r, self.handleExpr(expression->getArg(2), r), Type::IntS32()));
                         return Expr::ForeignCall(name, {packageContext(), remote, bytes, mode}, self.handleType(expression->getType(), r));
                       }
                       const auto local = r.newVar(self.conform(r, self.handleExpr(expression->getArg(1), r), raw));
                       const auto bytes = r.newVar(self.conform(r, self.handleExpr(expression->getArg(2), r), Type::IntU64()));
                       const auto mode = r.newVar(self.conform(r, self.handleExpr(expression->getArg(3), r), Type::IntS32()));
                       return Expr::ForeignCall(name, {packageContext(), remote, local, bytes, mode}, Type::Unit0());
                     }},
                     false};
}

Vector<CallPrism> polyregionPrisms() { return {polyregionIntrinsic, usmHostAccess}; }

} // namespace polyregion::polystl::call_prism
