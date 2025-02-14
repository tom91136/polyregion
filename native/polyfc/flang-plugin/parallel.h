#pragma once

#include "polyast.h"

namespace polyregion::polyfc::parallel_ops {

struct SingleVarReduction {
  polyast::Named target;
  polyast::Expr::Any init;
  polyast::Expr::Select partialArray;
  std::function<polyast::Expr::Any(const polyast::Expr::Any &, const polyast::Expr::Any &)> binaryOp;
  polyast::Stmt::Any partialVar() const;
  polyast::Stmt::Any drainPartial(const polyast::Expr::Any &lhs, const polyast::Expr::Any &idx) const;
  polyast::Stmt::Any drainPartial(const polyast::Expr::Any &idx) const;
  polyast::Stmt::Any applyPartial(const polyast::Expr::Any &lhs, const polyast::Expr::Any &idx) const;
  polyast::Stmt::Any applyPartial(const polyast::Expr::Any &idx) const;
};

struct CPUParams {
  polyast::Named induction;                          //
  polyast::Expr::Any lowerBound, step, begins, ends; //
  std::vector<polyast::Stmt::Any> body;
};

struct GPUParams {
  polyast::Named induction;                       //
  polyast::Expr::Any lowerBound, step, tripCount; //
  std::vector<polyast::Stmt::Any> body;
};

using OpParams = std::variant<CPUParams, GPUParams>;

polyast::Function forEach(const std::string &fnName,     //
                          const polyast::Named &capture, //
                          const OpParams &params);

polyast::Function reduce(const std::string &fnName,       //
                         const polyast::Named &capture,   //
                         const polyast::Named &unmanaged, //
                         const OpParams &params,            //
                         const std::vector<SingleVarReduction> &reductions);

} // namespace polyregion::polyfc::parallel_ops