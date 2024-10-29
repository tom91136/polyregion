#include "llvm_cpu.h"

using namespace polyregion::backend;
void CPUTargetSpecificHandler::witnessEntry(LLVMBackend::AstTransformer &ctx, llvm::Module &mod, llvm::Function &fn) {}
ValPtr CPUTargetSpecificHandler::mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &expr) {
  return expr.op.match_total(                                                 //
      [&](const Spec::Assert &v) -> ValPtr { return xform.invokeAbort(fn); }, //
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuGroupSize &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuLocalSize &v) -> ValPtr { throw BackendException("unimplemented"); }   //
  );
}
ValPtr CPUTargetSpecificHandler::mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &expr) {
  return expr.op.match_total(
      [&](const Math::Abs &v) -> ValPtr {
        return xform.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto x) { return xform.intr1(fn, llvm::Intrinsic::abs, v.tpe, v.x); },
            [&](auto x) { return xform.intr1(fn, llvm::Intrinsic::fabs, v.tpe, v.x); });
      },
      [&](const Math::Sin &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::sin, v.tpe, v.x); }, //
      [&](const Math::Cos &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::cos, v.tpe, v.x); }, //
      [&](const Math::Tan &v) -> ValPtr { return xform.extFn1(fn, "tan", v.tpe, v.x); },               //
      [&](const Math::Asin &v) -> ValPtr { return xform.extFn1(fn, "asin", v.tpe, v.x); },             //
      [&](const Math::Acos &v) -> ValPtr { return xform.extFn1(fn, "acos", v.tpe, v.x); },             //
      [&](const Math::Atan &v) -> ValPtr { return xform.extFn1(fn, "atan", v.tpe, v.x); },             //
      [&](const Math::Sinh &v) -> ValPtr { return xform.extFn1(fn, "sinh", v.tpe, v.x); },             //
      [&](const Math::Cosh &v) -> ValPtr { return xform.extFn1(fn, "cosh", v.tpe, v.x); },             //
      [&](const Math::Tanh &v) -> ValPtr { return xform.extFn1(fn, "tanh", v.tpe, v.x); },             //
      [&](const Math::Signum &v) -> ValPtr {
        return xform.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto x) -> ValPtr {
              auto msbOffset = x->getType()->getPrimitiveSizeInBits() - 1;
              return xform.B.CreateOr(xform.B.CreateAShr(x, msbOffset), xform.B.CreateLShr(xform.B.CreateNeg(x), msbOffset));
            },
            [&](auto x) -> ValPtr {
              auto mag = [&](const Expr::Any &magnitude) {
                auto nan = xform.B.CreateFCmpUNO(x, x);
                auto zero = xform.B.CreateFCmpUNO(x, llvm::ConstantFP::get(x->getType(), 0));
                return xform.B.CreateSelect(xform.B.CreateLogicalOr(nan, zero), x,
                                            xform.intr2(fn, llvm::Intrinsic::copysign, v.tpe, magnitude, v.x));
              };
              if (v.tpe.is<Type::Float32>())       //
                return mag(Expr::Float32Const(1)); //
              else if (v.tpe.is<Type::Float64>())  //
                return mag(Expr::Float64Const(1)); //
              else
                throw BackendException("unimplemented");
            });
      },                                                                                                    //
      [&](const Math::Round &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::round, v.tpe, v.x); },  //
      [&](const Math::Ceil &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::ceil, v.tpe, v.x); },    //
      [&](const Math::Floor &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::floor, v.tpe, v.x); },  //
      [&](const Math::Rint &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::rint, v.tpe, v.x); },    //
      [&](const Math::Sqrt &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::sqrt, v.tpe, v.x); },    //
      [&](const Math::Cbrt &v) -> ValPtr { return xform.extFn1(fn, "cbrt", v.tpe, v.x); },                  //
      [&](const Math::Exp &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::exp, v.tpe, v.x); },      //
      [&](const Math::Expm1 &v) -> ValPtr { return xform.extFn1(fn, "expm1", v.tpe, v.x); },                //
      [&](const Math::Log &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::log, v.tpe, v.x); },      //
      [&](const Math::Log1p &v) -> ValPtr { return xform.extFn1(fn, "log1p", v.tpe, v.x); },                //
      [&](const Math::Log10 &v) -> ValPtr { return xform.intr1(fn, llvm::Intrinsic::log10, v.tpe, v.x); },  //
      [&](const Math::Pow &v) -> ValPtr { return xform.intr2(fn, llvm::Intrinsic::pow, v.tpe, v.x, v.y); }, //
      [&](const Math::Atan2 &v) -> ValPtr { return xform.extFn2(fn, "atan2", v.tpe, v.x, v.y); },           //
      [&](const Math::Hypot &v) -> ValPtr { return xform.extFn2(fn, "hypot", v.tpe, v.x, v.y); }            //
  );
}
