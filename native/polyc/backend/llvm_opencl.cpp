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
      [&](const Spec::Assert &) -> ValPtr { return undefined(__FILE__, __LINE__); },
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
      [&](const Math::Abs &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Sin &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Cos &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Tan &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Asin &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Acos &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Atan &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Sinh &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Cosh &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Tanh &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Signum &v) -> ValPtr { return undefined(__FILE__, __LINE__); }, //
      [&](const Math::Round &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Ceil &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Floor &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Rint &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Sqrt &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Cbrt &v) -> ValPtr { return undefined(__FILE__, __LINE__); },   //
      [&](const Math::Exp &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Expm1 &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Log &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Log1p &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Log10 &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Pow &v) -> ValPtr { return undefined(__FILE__, __LINE__); },    //
      [&](const Math::Atan2 &v) -> ValPtr { return undefined(__FILE__, __LINE__); },  //
      [&](const Math::Hypot &v) -> ValPtr { return undefined(__FILE__, __LINE__); }   //
  );
}
