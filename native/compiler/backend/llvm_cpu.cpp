#include "llvm_cpu.h"

using namespace polyregion::backend;
void CPUTargetSpecificHandler::witnessEntry(LLVMBackend::AstTransformer &ctx, llvm::Module &mod, llvm::Function &fn) {}
ValPtr CPUTargetSpecificHandler::mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &expr) {
  return variants::total(
      *expr.op,                                                               //
      [&](const Spec::Assert &v) -> ValPtr { return xform.invokeAbort(fn); }, //
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return undefined(__FILE__, __LINE__); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return undefined(__FILE__, __LINE__); }, //
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return undefined(__FILE__, __LINE__); }   //
  );
}
ValPtr CPUTargetSpecificHandler::mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &expr) {
  return variants::total(
      *expr.op, //
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
            [&](auto x) {
              auto msbOffset = x->getType()->getPrimitiveSizeInBits() - 1;
              return xform.B.CreateOr(xform.B.CreateAShr(x, msbOffset), xform.B.CreateLShr(xform.B.CreateNeg(x), msbOffset));
            },
            [&](auto x) {
              auto nan = xform.B.CreateFCmpUNO(x, x);
              auto zero = xform.B.CreateFCmpUNO(x, llvm::ConstantFP::get(x->getType(), 0));
              Term::Any magnitude;
              if (holds<Type::Float32>(v.tpe))      //
                magnitude = Term::Float32Const(1);  //
              else if (holds<Type::Float64>(v.tpe)) //
                magnitude = Term::Float64Const(1);  //
              else
                error(__FILE__, __LINE__);
              return xform.B.CreateSelect(xform.B.CreateLogicalOr(nan, zero), x,
                                          xform.intr2(fn, llvm::Intrinsic::copysign, v.tpe, magnitude, v.x));
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
