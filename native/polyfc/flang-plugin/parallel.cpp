#include "parallel.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace polyregion::polyast;
using namespace polyregion::polyfc;
using namespace dsl;

using Stmts = std::vector<Stmt::Any>;

namespace {

size_t gFreshId = 0;
std::string fresh(const std::string &hint) { return fmt::format("#{}_{}", hint, ++gFreshId); }

Term::Any letBind(Stmts &out, const std::string &hint, const Expr::Any &e) {
  Named n(fresh(hint), e.tpe());
  out.emplace_back(Stmt::Var(n, e, /*isMutable*/ false));
  return Term::Select(n, {}, n.tpe);
}

Stmts mappedInductionStmts(const Named &induction, const Term::Any &lowerBound, const Term::Any &step) {
  // induction := lowerBound + (#i * step), all at induction.tpe
  Stmts out;
  const auto i = Term::Select(Named("#i", Long), {}, Long);
  const auto product = letBind(out, "iStep", Expr::IntrOp(Intr::Mul(i, step, induction.tpe)));
  out.emplace_back(Stmt::Var(induction, Expr::IntrOp(Intr::Add(lowerBound, product, induction.tpe)), /*isMutable*/ false));
  return out;
}

template <typename... Vs> Stmts splice(Vs &&...vs) {
  Stmts out;
  (out.insert(out.end(), std::begin(vs), std::end(vs)), ...);
  return out;
}

} // namespace

Stmt::Any parallel_ops::SingleVarReduction::partialVar() const {
  // var target = init
  return Stmt::Var(target, Expr::Alias(init), /*isMutable*/ true);
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Term::Select &lhs, const Term::Any &idx) const {
  // lhs[idx] = target
  return Stmt::Update(lhs, idx, Term::Select(target, {}, target.tpe));
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Term::Any &idx) const { return drainPartial(partialArray, idx); }

Stmts parallel_ops::SingleVarReduction::applyPartial(const Term::Any &lhs, const Term::Any &idx) const {
  // tmp = lhs[idx]; target = binOp(tmp, target)
  Stmts out;
  const auto loaded = letBind(out, "applyTmp", Expr::Index(lhs, idx, target.tpe));
  out.emplace_back(Stmt::Mut(Term::Select(target, {}, target.tpe), binaryOp(loaded, Term::Select(target, {}, target.tpe))));
  return out;
}
Stmts parallel_ops::SingleVarReduction::applyPartial(const Term::Any &idx) const { return applyPartial(partialArray, idx); }

Function parallel_ops::forEach(const std::string &fnName, const Named &capture, const OpParams &params) {
  return params ^ //
         fold_total(
             [&](const CPUParams &p) {
               Stmts body;
               const auto begin = letBind(body, "begin", Expr::Index(p.begins, "__tid"_(Long), Long));
               const auto end = letBind(body, "end", Expr::Index(p.ends, "__tid"_(Long), Long));
               body.emplace_back(Stmt::ForRange(Named("#i", Long), begin, end, Term::IntS64Const(1),
                                                splice(mappedInductionStmts(p.induction, p.lowerBound, p.step), p.body)));
               body.emplace_back(ret());
               return Function(Sym({fnName}), {}, std::optional<Arg>{},
                               std::vector<Arg>{Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, {}, {}, Unit, body,
                               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
             },
             [&](const GPUParams &p) {
               Stmts body;
               const auto gsU = letBind(body, "gsU", call(Spec::GpuGlobalSize(0_(UInt))));
               const auto gs = letBind(body, "gs", Expr::Cast(gsU, Long));
               const auto gidU = letBind(body, "gidU", call(Spec::GpuGlobalIdx(0_(UInt))));
               const auto gid = letBind(body, "gid", Expr::Cast(gidU, Long));
               body.emplace_back(Stmt::ForRange(Named("#i", Long), gid, p.tripCount, gs,
                                                splice(mappedInductionStmts(p.induction, p.lowerBound, p.step), p.body)));
               body.emplace_back(ret());
               return Function(Sym({fnName}), {}, std::optional<Arg>{},
                               std::vector<Arg>{Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, {}, {}, Unit, body,
                               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
             });
}

Function parallel_ops::reduce(const std::string &fnName, const Named &capture, const Named &unmanaged, const OpParams &params,
                              const std::vector<SingleVarReduction> &reductions) {
  return params ^
         fold_total(
             [&](const CPUParams &p) {
               Stmts body;
               for (auto &r : reductions)
                 body.emplace_back(r.partialVar());
               const auto begin = letBind(body, "begin", Expr::Index(p.begins, "__tid"_(Long), Long));
               const auto end = letBind(body, "end", Expr::Index(p.ends, "__tid"_(Long), Long));
               body.emplace_back(Stmt::ForRange(Named("#i", Long), begin, end, Term::IntS64Const(1),
                                                splice(mappedInductionStmts(p.induction, p.lowerBound, p.step), p.body)));
               for (auto &r : reductions)
                 body.emplace_back(r.drainPartial("__tid"_(Long)));
               body.emplace_back(ret());
               return Function(Sym({fnName}), {}, std::optional<Arg>{}, std::vector<Arg>{Arg(capture, {}), Arg(unmanaged, {})}, {}, {},
                               Unit, body, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
             },
             [&](const GPUParams &p) {
               // GPU tree reduction: per-thread accumulate into target, drain to local memory, tree-reduce, drain group result.
               // The local memory layout is contiguous; each reduction component lives at a thread-strided offset.
               const auto localMemArg = Named("#localMem", Ptr(Byte, TypeSpace::Local()));
               const auto localMemSel = Term::Select(localMemArg, {}, localMemArg.tpe);

               Stmts body;
               const auto gsU = letBind(body, "gsU", call(Spec::GpuGlobalSize(0_(UInt))));
               const auto gs = letBind(body, "gs", Expr::Cast(gsU, Long));
               const auto gidU = letBind(body, "gidU", call(Spec::GpuGlobalIdx(0_(UInt))));
               const auto gid = letBind(body, "gid", Expr::Cast(gidU, Long));
               for (auto &r : reductions)
                 body.emplace_back(r.partialVar());
               body.emplace_back(Stmt::ForRange(Named("#i", Long), gid, p.tripCount, gs,
                                                splice(mappedInductionStmts(p.induction, p.lowerBound, p.step), p.body)));

               if (reductions.empty()) {
                 // XXX Skip the tree-reduce barrier loop: miscompiles on some SPIR-V targets at small workgroup sizes.
                 body.emplace_back(ret());
                 return Function(Sym({fnName}), {}, std::optional<Arg>{},
                                 std::vector<Arg>{Arg(capture, {}), Arg(unmanaged, {}), Arg(localMemArg, {})}, {}, {}, Unit, body,
                                 FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
               }

               const auto liU = letBind(body, "liU", call(Spec::GpuLocalIdx(0_(UInt))));
               const auto li = letBind(body, "li", Expr::Cast(liU, Long));
               const auto lsU = letBind(body, "lsU", call(Spec::GpuLocalSize(0_(UInt))));
               const auto ls = letBind(body, "ls", Expr::Cast(lsU, Long));

               Term::Any offset = letBind(body, "off", Expr::Alias(Term::IntS64Const(0)));
               std::vector<Named> localTargets;
               for (auto &r : reductions) {
                 const auto primSize = polyast::primitiveSize(r.target.tpe);
                 if (!primSize) {
                   llvm::errs() << "polyfc: GPU reduction target type has no primitive size: " << polyast::repr(r.target.tpe) << "\n";
                   std::abort();
                 }
                 const Term::Any compSize = Term::IntS64Const(static_cast<int64_t>(*primSize));
                 const auto localPtrTpe = Ptr(r.target.tpe, TypeSpace::Local());
                 const Named localTgt(fresh("lref"), localPtrTpe);
                 localTargets.push_back(localTgt);
                 const auto bytePtr = letBind(body, "bytePtr", Expr::RefTo(localMemSel, offset, Byte, TypeSpace::Local()));
                 body.emplace_back(Stmt::Var(localTgt, Expr::Cast(bytePtr, localPtrTpe), /*isMutable*/ false));
                 const auto stride = letBind(body, "stride", Expr::IntrOp(Intr::Mul(ls, compSize, Long)));
                 offset = letBind(body, "off", Expr::IntrOp(Intr::Add(offset, stride, Long)));
               }

               // localTgt[li] = target
               for (size_t i = 0; i < reductions.size(); ++i) {
                 const auto &r = reductions[i];
                 body.emplace_back(
                     Stmt::Update(Term::Select(localTargets[i], {}, localTargets[i].tpe), li, Term::Select(r.target, {}, r.target.tpe)));
               }

               // Tree reduction:
               //  var #off = ls / 2;
               //  while (#off > 0) {
               //    barrier;
               //    if (li < #off) localTgt[li] = op(localTgt[li], localTgt[li+#off]);
               //    #off /= 2;
               //  }
               const Named offVar("#off", Long);
               body.emplace_back(Stmt::Var(offVar, Expr::IntrOp(Intr::Div(ls, Term::IntS64Const(2), Long)), /*isMutable*/ true));

               Stmts whileBody;
               whileBody.emplace_back(Stmt::Var(Named(fresh("barrier"), Unit), call(Spec::GpuBarrierLocal()), /*isMutable*/ false));

               Stmts ifBody;
               const auto liPlusOff = letBind(ifBody, "liOff", Expr::IntrOp(Intr::Add(li, Term::Select(offVar, {}, Long), Long)));
               for (size_t i = 0; i < reductions.size(); ++i) {
                 const auto &r = reductions[i];
                 const auto localTgtSel = Term::Select(localTargets[i], {}, localTargets[i].tpe);
                 const auto a = letBind(ifBody, "lhs", Expr::Index(localTgtSel, li, r.target.tpe));
                 const auto b = letBind(ifBody, "rhs", Expr::Index(localTgtSel, liPlusOff, r.target.tpe));
                 const auto combined = letBind(ifBody, "comb", r.binaryOp(a, b));
                 ifBody.emplace_back(Stmt::Update(localTgtSel, li, combined));
               }
               const auto cond = letBind(whileBody, "cond", Expr::IntrOp(Intr::LogicLt(li, Term::Select(offVar, {}, Long))));
               whileBody.emplace_back(Stmt::Cond(cond, ifBody, {}));
               whileBody.emplace_back(Stmt::Mut(Term::Select(offVar, {}, Long),
                                                Expr::IntrOp(Intr::Div(Term::Select(offVar, {}, Long), Term::IntS64Const(2), Long))));

               // XXX Stmt::While does not re-evaluate its cond Term; carry it in a mutable var.
               const Named whileCondVar(fresh("whileCond"), Bool);
               body.emplace_back(Stmt::Var(whileCondVar, Expr::IntrOp(Intr::LogicGt(Term::Select(offVar, {}, Long), Term::IntS64Const(0))),
                                           /*isMutable*/ true));
               whileBody.emplace_back(Stmt::Mut(Term::Select(whileCondVar, {}, Bool),
                                                Expr::IntrOp(Intr::LogicGt(Term::Select(offVar, {}, Long), Term::IntS64Const(0)))));
               body.emplace_back(Stmt::While(Term::Select(whileCondVar, {}, Bool), whileBody));

               // if (li == 0) { partialArray[groupIdx] = localTgt[li] }
               const auto giU = letBind(body, "giU", call(Spec::GpuGroupIdx(0_(UInt))));
               const auto gi = letBind(body, "gi", Expr::Cast(giU, Long));
               const auto liEqZero = letBind(body, "liEqZero", Expr::IntrOp(Intr::LogicEq(li, Term::IntS64Const(0))));
               Stmts drainBody;
               for (size_t i = 0; i < reductions.size(); ++i) {
                 const auto &r = reductions[i];
                 const auto localTgtSel = Term::Select(localTargets[i], {}, localTargets[i].tpe);
                 const auto loaded = letBind(drainBody, "drain", Expr::Index(localTgtSel, li, r.target.tpe));
                 drainBody.emplace_back(Stmt::Update(r.partialArray, gi, loaded));
               }
               body.emplace_back(Stmt::Cond(liEqZero, drainBody, {}));
               body.emplace_back(ret());

               return Function(Sym({fnName}), {}, std::optional<Arg>{},
                               std::vector<Arg>{Arg(capture, {}), Arg(unmanaged, {}), Arg(localMemArg, {})}, {}, {}, Unit, body,
                               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), /*isEntry*/ true);
             });
}
