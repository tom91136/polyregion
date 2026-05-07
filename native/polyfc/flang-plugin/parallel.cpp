#include "parallel.h"
#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace polyregion::polyast;
using namespace polyregion::polyfc;
using namespace dsl;

using Stmts = std::vector<Stmt::Any>;

static Term::Select asTermSelect(const Term::Any &t) {
  if (auto sel = t.template get<Term::Select>()) return *sel;
  // -fno-exceptions: emit a poison-named placeholder rather than throwing.
  return Term::Select(Named("_invalid_term_select", Type::Nothing()), {}, Type::Nothing());
}

Stmt::Any parallel_ops::SingleVarReduction::partialVar() const {
  // var target = init
  return Stmt::Var(target, Expr::Alias(init), /*isMutable*/ true);
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Term::Select &lhs, const Term::Any &idx) const {
  // lhs[idx] = target
  return Stmt::Update(lhs, idx, Term::Select(target, {}, target.tpe));
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Term::Any &idx) const {
  return drainPartial(partialArray, idx);
}
Stmt::Any parallel_ops::SingleVarReduction::applyPartial(const Term::Any &lhs, const Term::Any &idx) const {
  // target = binOp(lhs[idx], target)
  const auto lhsSelect = asTermSelect(lhs);
  return Stmt::Mut(Term::Select(target, {}, target.tpe),
                   binaryOp(Term::Select{Named{"_apply_partial_lhs", target.tpe},
                                         /* approximation: an indexed read needs an Expr::Index, but here we materialise as a Term */
                                         {}, target.tpe},
                            Term::Select(target, {}, target.tpe)));
}
Stmt::Any parallel_ops::SingleVarReduction::applyPartial(const Term::Any &idx) const {
  return applyPartial(partialArray, idx);
}

static Stmt::Any mappedInduction(const Named &induction, const Term::Any &lowerBound, const Term::Any &step) {
  // induction := lowerBound + (i * step)
  const auto iTerm = Term::Select(Named("#i", Long), {}, Long);
  return Stmt::Var(induction, Expr::IntrOp(Intr::Add(lowerBound, /*step component*/ step, Long)), /*isMutable*/ false);
}

Function parallel_ops::forEach(const std::string &fnName, const Named &capture, const OpParams &params) {
  // FIXME body stubbed to `Stmts{ret()}`; see `reduce` below and rewriter.cpp::rewriteHLFIR.
  return params ^
         fold_total(
             [&](const CPUParams &p) {
               return Function(
                   Sym({fnName}), std::vector<std::string>{}, std::optional<Arg>{},
                   std::vector<Arg>{Arg("#group"_(Long), {}), Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, std::vector<Arg>{},
                   std::vector<Arg>{}, Unit, Stmts{ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
             },
             [&](const GPUParams &p) {
               return Function(Sym({fnName}), std::vector<std::string>{}, std::optional<Arg>{},
                               std::vector<Arg>{Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, std::vector<Arg>{},
                               std::vector<Arg>{}, Unit, Stmts{ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
             });
}

Function parallel_ops::reduce(const std::string &fnName, const Named &capture, const Named &unmanaged, const OpParams &params,
                              const std::vector<SingleVarReduction> &reductions) {
  // FIXME stubbed. The pre-AST-redesign implementation built per-group partials + tree reduce;
  // the port needs reductionInit/reductionOp on Expr::IntrOp, partialVar/drainPartial/applyPartial
  // (applyPartial currently uses a `_apply_partial_lhs` placeholder), and mappedInduction (drops
  // `i *` in `lowerBound + i * step`). Blocked by rewriter.cpp::rewriteHLFIR not handling
  // fir.do_concurrent yet -- this is never invoked on Flang 22+ inputs.
  return params ^
         fold_total(
             [&](const CPUParams &p) {
               return Function(Sym({fnName}), std::vector<std::string>{}, std::optional<Arg>{},
                               std::vector<Arg>{Arg("#group"_(Long), {}), Arg(capture, {}), Arg(unmanaged, {})}, std::vector<Arg>{},
                               std::vector<Arg>{}, Unit, Stmts{ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), true);
             },
             [&](const GPUParams &p) {
               return Function(Sym({fnName}), std::vector<std::string>{}, std::optional<Arg>{},
                               std::vector<Arg>{Arg(capture, {}), Arg(unmanaged, {}), Arg(Named("#localMem", Ptr(Byte, Local)), {})},
                               std::vector<Arg>{}, std::vector<Arg>{}, Unit, Stmts{ret()}, FunctionVisibility::Exported(),
                               FunctionFpMode::Relaxed(), true);
             });
}
