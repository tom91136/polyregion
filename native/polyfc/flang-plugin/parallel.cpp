#include "parallel.h"
#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace polyregion::polyast;
using namespace polyregion::polyfc;
using namespace dsl;

using Stmts = std::vector<Stmt::Any>;

Stmt::Any parallel_ops::SingleVarReduction::partialVar() const {
  return Var(target, init); // var target = init
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Expr::Any &lhs, const Expr::Any &idx) const {
  return Update(lhs, idx, Select({}, target)); // lhs[idx] = target
}
Stmt::Any parallel_ops::SingleVarReduction::drainPartial(const Expr::Any &idx) const {
  return drainPartial(partialArray, idx); // partialArray[idx] = target
}
Stmt::Any parallel_ops::SingleVarReduction::applyPartial(const Expr::Any &lhs, const Expr::Any &idx) const {
  return Mut(Select({}, target),              // target = binOp(lhs[g], target)
             binaryOp(                        //
                 Index(lhs, idx, target.tpe), //
                 Select({}, target)));
}
Stmt::Any parallel_ops::SingleVarReduction::applyPartial(const Expr::Any &idx) const {
  return applyPartial(partialArray, idx); // target = binOp(partialArray[g], target)
}

static Stmt::Any mappedInduction(const Named &induction, const Expr::Any &lowerBound, const Expr::Any &step) {
  return let(induction.symbol) = call(Intr::Add(lowerBound, call(Intr::Mul("#i"_(Long), step, Long)), Long));
}

Function parallel_ops::forEach(const std::string &fnName, const Named &capture, const OpParams &params) {
  return params ^
         fold_total(
             [&](const CPUParams &p) {
               return Function(
                   fnName, {Arg("#group"_(Long), {}), Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, Unit,                   //
                   Stmts{let("#i") = 0_(Long),                                                                                         //
                         ForRange("#i"_(Long), Index(p.begins, "#group"_(Long), Long), Index(p.ends, "#group"_(Long), Long), 1_(Long), //
                                  {
                                      mappedInduction(p.induction, p.lowerBound, p.step), //
                                      Block(p.body),
                                  }),
                         ret()}, //
                   {FunctionAttr::Entry(), FunctionAttr::Exported()});
             },
             [&](const GPUParams &p) {
               return Function(fnName, {Arg(capture, {}), Arg(Named("#unused", Ptr(Byte)), {})}, Unit, //
                               Stmts{let("#gs") = Cast(call(Spec::GpuGlobalSize(0_(UInt))), Long),     //
                                     let("#i") = 0_(Long),                                             //
                                     ForRange("#i"_(Long), Cast(call(Spec::GpuGlobalIdx(0_(UInt))), Long), p.tripCount, "#gs"_(Long),
                                              {
                                                  mappedInduction(p.induction, p.lowerBound, p.step), //
                                                  Block(p.body),
                                              }),
                                     ret()}, //
                               {FunctionAttr::Entry(), FunctionAttr::Exported()});
             });
}

struct LocalMemSVR {
  parallel_ops::SingleVarReduction reduction;
  Named localMemTarget;
  Expr::Any compSize;

  Expr::Any targetOffset(const Expr::Any &offset, const Expr::Any &count) const { // offset + (local_size() * sizeof(T))
    return call(Intr::Add(offset, call(Intr::Mul(count, compSize, Long)), Long));
  }

  Var var(const Select &byteStorage, const Expr::Any &offset) const { // var localTgt = (T*) localMem[offset]
    return Var(localMemTarget, Cast(RefTo(byteStorage, {offset}, Byte, Local), localMemTarget.tpe));
  }

  Stmt::Any drainLocal(const Expr::Any &localIdx) const { // localTgt[localIdx] = target
    return Update(Select({}, localMemTarget), localIdx, Select({}, reduction.target));
  }

  Stmt::Any drainReduceLocal(const Expr::Any &localIdx, const Expr::Any &offset) const {
    return Update(Select({}, localMemTarget), localIdx, // localTgt[localIdx] = reduce(localTgt[localIdx], localTgt[localIdx + offset]);
                  reduction.binaryOp(Index(Select({}, localMemTarget), localIdx, reduction.target.tpe),
                                     Index(Select({}, localMemTarget), call(Intr::Add(localIdx, offset, Long)), reduction.target.tpe)));
  }

  Stmt::Any drainPartial(const Expr::Any &groupIdx, const Expr::Any &localIdx) const { // partialArray[groupIdx] = localTgt[localIdx];
    return Update(reduction.partialArray, groupIdx, Index(Select({}, localMemTarget), localIdx, reduction.target.tpe));
  }
};

Function parallel_ops::reduce(const std::string &fnName, const Named &capture, const Named &unmanaged, const OpParams &params,
                              const std::vector<SingleVarReduction> &reductions) {
  return params ^
         fold_total(
             [&](const CPUParams &p) {
               return Function(
                   fnName, {Arg("#group"_(Long), {}), Arg(capture, {}), Arg(unmanaged, {})}, Unit, //
                   Stmts{Block(reductions ^ map([](auto &r) { return r.partialVar(); })),
                         let("#i") = 0_(Long),                                                                                         //
                         ForRange("#i"_(Long), Index(p.begins, "#group"_(Long), Long), Index(p.ends, "#group"_(Long), Long), 1_(Long), //
                                  {
                                      mappedInduction(p.induction, p.lowerBound, p.step), //
                                      Block(p.body),                                      //
                                  }),
                         Block(reductions ^ map([](auto &r) { return r.drainPartial("#group"_(Long)); })), //
                         ret()},                                                                           //
                   {FunctionAttr::Entry(), FunctionAttr::Exported()});
             },
             [&](const GPUParams &p) {
               const auto localMemSVRs =
                   reductions ^ map([](const SingleVarReduction &r) {
                     return LocalMemSVR{
                         .reduction = r,
                         .localMemTarget = Named(fmt::format("#local_ref<{}>", r.target.symbol), Ptr(r.target.tpe, {}, Local)),
                         .compSize =
                             primitiveSize(r.target.tpe) ^
                             fold(
                                 [](const size_t size) { return IntS64Const(size).widen(); }, //
                                 [&]() {
                                   return Expr::Annotated(
                                              Poison(Long), {},
                                              fmt::format(
                                                  "Reduction of non-primitive type ({}) are not supported as the size cannot be determined",
                                                  repr(r.target.tpe)))
                                       .widen();
                                 })};
                   });

               return Function(
                   fnName, {Arg(capture, {}), Arg(unmanaged, {}), Arg(Named("#localMem", Ptr(Byte, {}, Local)), {})}, Unit, //
                   Stmts{let("#gs") = Cast(call(Spec::GpuGlobalSize(0_(UInt))), Long),                                      //
                         let("#i") = 0_(Long),                                                                              //
                         Block(reductions ^ map([](auto &r) { return r.partialVar(); })),                                   //
                         ForRange("#i"_(Long), Cast(call(Spec::GpuGlobalIdx(0_(UInt))), Long), p.tripCount, "#gs"_(Long),
                                  {
                                      mappedInduction(p.induction, p.lowerBound, p.step), //
                                      Block(p.body),                                      //
                                  }),
                         // XXX Actual local storage reduction starts after the body:
                         let("#localIdx") = Cast(call(Spec::GpuLocalIdx(0_(UInt))), Long),   //
                         let("#localSize") = Cast(call(Spec::GpuLocalSize(0_(UInt))), Long), //
                         // XXX Local (shared) memory is exposed as a contiguous block, so we work out the offset of each partial target for
                         // the supported primitive types as follows:
                         //  (T*) localMem(0)
                         //  (U*) localMem(0 + local_size() * sizeof(T))
                         //  (V*) localMem(0 + local_size() * sizeof(T) + local_size() * sizeof(U))
                         // ...
                         Block((localMemSVRs | fold_left(std::pair{0_(Long), Stmts{}},
                                                         [](auto &&acc, auto &r) {
                                                           const auto &[offset, stmts] = acc;
                                                           return std::pair{r.targetOffset(offset, "#localSize"_(Long)), //
                                                                            stmts ^
                                                                                append(r.var("#localMem"_(Ptr(Byte, {}, Local)), offset))};
                                                         }))
                                   .second),                                                                      //
                         Block(localMemSVRs ^ map([](auto &svr) { return svr.drainLocal("#localIdx"_(Long)); })), //

                         // XXX We have local storage initialised to reduction values now, proceed to do a partial tree reduction
                         let("#offset") = call(Intr::Div("#localSize"_(Long), 2_(Long), Long)), // var offset = get_local_size() / 2
                         While({let("#cont") = call(Intr::LogicGt("#offset"_(Long), 0_(Long)))}, "#cont"_(Bool), // while(offset > 0) {
                               {
                                   let("#_") = call(Spec::GpuBarrierLocal()),                      //
                                   Cond(call(Intr::LogicLt("#localIdx"_(Long), "#offset"_(Long))), // if (localIdx < offset) {
                                        {
                                            Block(localMemSVRs ^ map([](auto &svr) {
                                                    return svr.drainReduceLocal("#localIdx"_(Long), "#offset"_(Long));
                                                  })),
                                        },
                                        {}),
                                   "#offset"_(Long) = call(Intr::Div("#offset"_(Long), 2_(Long), Long)) // offset /= 2
                               }),

                         Cond(call(Intr::LogicEq("#localIdx"_(Long), 0_(Long))), // if (localIdx == 0) {
                              {
                                  let("#groupIdx") = Cast(call(Spec::GpuGroupIdx(0_(UInt))), Long), //
                                  Block(localMemSVRs ^
                                        map([](auto &svr) { return svr.drainPartial("#groupIdx"_(Long), "#localIdx"_(Long)); })), //
                              },
                              {}),
                         ret()}, //
                   {FunctionAttr::Entry(), FunctionAttr::Exported()});
             });
}
