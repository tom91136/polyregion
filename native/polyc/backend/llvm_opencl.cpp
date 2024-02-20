#include "llvm_opencl.h"

using namespace polyregion::backend;
/**
 * Queue a memory fence to ensure correct
 * ordering of memory operations to local memory
 */
constexpr static uint32_t CLK_LOCAL_MEM_FENCE = 0x01;

/**
 * Queue a memory fence to ensure correct
 * ordering of memory operations to global memory
 */
constexpr static uint32_t CLK_GLOBAL_MEM_FENCE = 0x02;

void OpenCLTargetSpecificHandler::witnessEntry(LLVMBackend::AstTransformer &ctx, llvm::Module &mod, llvm::Function &fn) {}

// See https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-1.2-built-in-functions
ValPtr OpenCLTargetSpecificHandler::mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &expr) {

  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return xform.extFn1(fn, "_Z13get_global_idj", v.tpe, v.dim); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return xform.extFn1(fn, "_Z15get_global_sizej", v.tpe, v.dim); },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return xform.extFn1(fn, "_Z12get_group_idj", v.tpe, v.dim); },
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return xform.extFn1(fn, "_Z14get_num_groupsj", v.tpe, v.dim); },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return xform.extFn1(fn, "_Z12get_local_idj", v.tpe, v.dim); },
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return xform.extFn1(fn, "_Z14get_local_sizej", v.tpe, v.dim); },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr {
        return xform.extFn1(fn, "_Z7barrierj", v.tpe, Term::IntS32Const(CLK_GLOBAL_MEM_FENCE));
      },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr {
        return xform.extFn1(fn, "_Z7barrierj", v.tpe, Term::IntS32Const(CLK_LOCAL_MEM_FENCE));
      },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr {
        return xform.extFn1(fn, "_Z7barrierj", v.tpe, Term::IntS32Const(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE));
      },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr {
        return xform.extFn1(fn, "_Z9mem_fencej", v.tpe, Term::IntS32Const(CLK_GLOBAL_MEM_FENCE));
      },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr {
        return xform.extFn1(fn, "_Z9mem_fencej", v.tpe, Term::IntS32Const(CLK_LOCAL_MEM_FENCE));
      },
      [&](const Spec::GpuFenceAll &v) -> ValPtr {
        return xform.extFn1(fn, "_Z9mem_fencej", v.tpe, Term::IntS32Const(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE));
      });
}
ValPtr OpenCLTargetSpecificHandler::mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &expr) {
  return expr.op.match_total( //                                                       //
      [&](const Math::Abs &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Sin &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Cos &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Tan &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Asin &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Acos &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Atan &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Sinh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Cosh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Tanh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Signum &v) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Math::Round &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Ceil &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Floor &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Rint &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Sqrt &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Cbrt &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Exp &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Expm1 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Log &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Log1p &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Log10 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Pow &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Atan2 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Hypot &v) -> ValPtr { throw BackendException("unimplemented"); }   //
  );
}
