#include "llvm_cpu.h"

using namespace polyregion::backend::details;

void CPUTargetSpecificHandler::witnessEntry(CodeGen &ctx, llvm::Function &fn) {}
ValPtr CPUTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  return expr.op.match_total(                                            //
      [&](const Spec::Assert &) -> ValPtr { return cg.invokeAbort(); }, //
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuBarrierAll &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceLocal &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceAll &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalSize &) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuGroupIdx &) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuGroupSize &) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Spec::GpuLocalIdx &) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuLocalSize &) -> ValPtr { throw BackendException("unimplemented"); }   //
  );
}
ValPtr CPUTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total(
      [&](const Math::Abs &v) -> ValPtr {
        return cg.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto) { return cg.intr1(llvm::Intrinsic::abs, v.tpe, v.x); },
            [&](auto) { return cg.intr1(llvm::Intrinsic::fabs, v.tpe, v.x); });
      },
      [&](const Math::Sin &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::sin, v.tpe, v.x); }, //
      [&](const Math::Cos &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::cos, v.tpe, v.x); }, //
      [&](const Math::Tan &v) -> ValPtr { return cg.extFn1("tan", v.tpe, v.x); },               //
      [&](const Math::Asin &v) -> ValPtr { return cg.extFn1("asin", v.tpe, v.x); },             //
      [&](const Math::Acos &v) -> ValPtr { return cg.extFn1("acos", v.tpe, v.x); },             //
      [&](const Math::Atan &v) -> ValPtr { return cg.extFn1("atan", v.tpe, v.x); },             //
      [&](const Math::Sinh &v) -> ValPtr { return cg.extFn1("sinh", v.tpe, v.x); },             //
      [&](const Math::Cosh &v) -> ValPtr { return cg.extFn1("cosh", v.tpe, v.x); },             //
      [&](const Math::Tanh &v) -> ValPtr { return cg.extFn1("tanh", v.tpe, v.x); },             //
      [&](const Math::Signum &v) -> ValPtr {
        return cg.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto x) -> ValPtr {
              auto msbOffset = x->getType()->getPrimitiveSizeInBits() - 1;
              return cg.B.CreateOr(cg.B.CreateAShr(x, msbOffset), cg.B.CreateLShr(cg.B.CreateNeg(x), msbOffset));
            },
            [&](auto x) -> ValPtr {
              auto mag = [&](const Expr::Any &magnitude) {
                auto nan = cg.B.CreateFCmpUNO(x, x);
                auto zero = cg.B.CreateFCmpUNO(x, llvm::ConstantFP::get(x->getType(), 0));
                return cg.B.CreateSelect(cg.B.CreateLogicalOr(nan, zero), x, cg.intr2(llvm::Intrinsic::copysign, v.tpe, magnitude, v.x));
              };
              if (v.tpe.is<Type::Float32>())       //
                return mag(Expr::Float32Const(1)); //
              else if (v.tpe.is<Type::Float64>())  //
                return mag(Expr::Float64Const(1)); //
              else throw BackendException("unimplemented");
            });
      },                                                                                             //
      [&](const Math::Round &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::round, v.tpe, v.x); },  //
      [&](const Math::Ceil &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::ceil, v.tpe, v.x); },    //
      [&](const Math::Floor &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::floor, v.tpe, v.x); },  //
      [&](const Math::Rint &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::rint, v.tpe, v.x); },    //
      [&](const Math::Sqrt &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::sqrt, v.tpe, v.x); },    //
      [&](const Math::Cbrt &v) -> ValPtr { return cg.extFn1("cbrt", v.tpe, v.x); },                  //
      [&](const Math::Exp &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::exp, v.tpe, v.x); },      //
      [&](const Math::Expm1 &v) -> ValPtr { return cg.extFn1("expm1", v.tpe, v.x); },                //
      [&](const Math::Log &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::log, v.tpe, v.x); },      //
      [&](const Math::Log1p &v) -> ValPtr { return cg.extFn1("log1p", v.tpe, v.x); },                //
      [&](const Math::Log10 &v) -> ValPtr { return cg.intr1(llvm::Intrinsic::log10, v.tpe, v.x); },  //
      [&](const Math::Pow &v) -> ValPtr { return cg.intr2(llvm::Intrinsic::pow, v.tpe, v.x, v.y); }, //
      [&](const Math::Atan2 &v) -> ValPtr { return cg.extFn2("atan2", v.tpe, v.x, v.y); },           //
      [&](const Math::Hypot &v) -> ValPtr { return cg.extFn2("hypot", v.tpe, v.x, v.y); }            //
  );
}
