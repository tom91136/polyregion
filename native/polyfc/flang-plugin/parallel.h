#pragma once

#include "polyast.h"

namespace polyregion::polyfc::parallel_ops {

struct SingleVarReduction {
  polyast::Named target;
  polyast::Term::Any init;
  polyast::Term::Select partialArray;
  std::function<polyast::Expr::Any(const polyast::Term::Any &, const polyast::Term::Any &)> binaryOp;
  polyast::Stmt::Any partialVar() const;
  polyast::Stmt::Any drainPartial(const polyast::Term::Select &lhs, const polyast::Term::Any &idx) const;
  polyast::Stmt::Any drainPartial(const polyast::Term::Any &idx) const;
  std::vector<polyast::Stmt::Any> applyPartial(const polyast::Term::Any &lhs, const polyast::Term::Any &idx) const;
  std::vector<polyast::Stmt::Any> applyPartial(const polyast::Term::Any &idx) const;
};

struct CPUParams {
  polyast::Named induction;                          //
  polyast::Term::Any lowerBound, step, begins, ends; //
  std::vector<polyast::Stmt::Any> body;
};

struct GPUParams {
  polyast::Named induction;                       //
  polyast::Term::Any lowerBound, step, tripCount; //
  std::vector<polyast::Stmt::Any> body;
};

using OpParams = std::variant<CPUParams, GPUParams>;

polyast::Function forEach(const std::string &fnName,     //
                          const polyast::Named &capture, //
                          const OpParams &params);

polyast::Function reduce(const std::string &fnName,       //
                         const polyast::Named &capture,   //
                         const polyast::Named &unmanaged, //
                         const OpParams &params,          //
                         const std::vector<SingleVarReduction> &reductions);

} // namespace polyregion::polyfc::parallel_ops
