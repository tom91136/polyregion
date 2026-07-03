#include "llvm_cpu.h"

using namespace polyregion::backend::details;

void CPUTargetSpecificHandler::witnessFn(CodeGen &ctx, llvm::Function &fn, const Function &source) {
  if (!source.visibility.is<FunctionVisibility::Exported>()) {
    fn.setDSOLocal(true);
  }
}
ValPtr CPUTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  const auto noop = [&] { return cg.mkTermVal(Term::Unit0Const()); };
  const auto k = [&](const auto &v, uint64_t n) -> ValPtr { return llvm::ConstantInt::get(cg.resolveType(v.tpe), n); };
  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr {
        throw BackendException("assert reached codegen; the StructuredExit pass must run before the backend");
      },                                                                //
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { return noop(); }, //
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { return noop(); },  //
      [&](const Spec::GpuBarrierAll &) -> ValPtr { return noop(); },    //
      [&](const Spec::GpuFenceGlobal &) -> ValPtr { return noop(); },   //
      [&](const Spec::GpuFenceLocal &) -> ValPtr { return noop(); },    //
      [&](const Spec::GpuFenceAll &) -> ValPtr { return noop(); },      //
      [&](const Spec::GpuGlobalIdx &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalSize &) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuGroupIdx &) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuGroupSize &) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return k(v, 0); },                           //
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return k(v, 1); }                           //
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
      [&](const Math::Signum &v) -> ValPtr { return cg.mkSignumVal(expr, v.x, v.tpe); },        //
      [&](const Math::Round &v) -> ValPtr {
        // Round may return an integral type; llvm.round preserves float, so an integral rtn rounds then fptosi
        const auto inTpe = v.x.tpe();
        if (v.tpe.is<Type::Float16>() || v.tpe.is<Type::Float32>() || v.tpe.is<Type::Float64>())
          return cg.intr1(llvm::Intrinsic::round, v.tpe, v.x);
        const auto rounded = cg.intr1(llvm::Intrinsic::round, inTpe, v.x);
        return cg.B.CreateFPToSI(rounded, cg.resolveType(v.tpe));
      },                                                                                             //
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
