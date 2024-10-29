#include "llvm_nvptx.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace polyregion::backend::details;

void NVPTXTargetSpecificHandler::witnessEntry(CodeGen &cg, llvm::Function &fn) {
  cg.M.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDNode::get(cg.C.actual, // XXX the attribute name must be "kernel" here and not the function name!
                                     {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(cg.C.actual, "kernel"),
                                      llvm::ValueAsMetadata::get(llvm::ConstantInt::get(cg.C.i32Ty(), 1))}));
}
ValPtr NVPTXTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  // threadId =  @llvm.nvvm.read.ptx.sreg.tid.*
  // blockIdx =  @llvm.nvvm.read.ptx.sreg.ctaid.*
  // blockDim =  @llvm.nvvm.read.ptx.sreg.ntid.*
  // gridDim  =  @llvm.nvvm.read.ptx.sreg.nctaid.*
  auto globalSize = [&](const llvm::Intrinsic::ID nctaid, const llvm::Intrinsic::ID ntid) -> ValPtr {
    return cg.B.CreateMul(cg.intr0(nctaid), cg.intr0(ntid));
  };
  auto globalId = [&](const llvm::Intrinsic::ID ctaid, const llvm::Intrinsic::ID ntid, const llvm::Intrinsic::ID tid) -> ValPtr {
    return cg.B.CreateAdd(cg.B.CreateMul(cg.intr0(ctaid), cg.intr0(ntid)), cg.intr0(tid));
  };
  auto dim3OrAssert = [&](const AnyExpr &dim, ValPtr const d0, ValPtr const d1, ValPtr const d2) {
    if (dim.tpe() != Type::IntU32()) {
      throw std::logic_error("dim selector should be a " + to_string(Type::IntU32()) + " but got " + to_string(dim.tpe()));
    }
    return cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntU32Const(0))), d0,
                             cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntU32Const(1))), d0,
                                               cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntU32Const(2))),
                                                                 d0, cg.mkExprVal(Expr::IntU32Const(0)))));
  };

  return expr.op.match_total( //
      [&](const Spec::Assert &v) -> ValPtr {
        // cg.extFn1(  "__assertfail", Type::Unit0(), Expr::Unit0Const()); // TODO
        throw BackendException("unimplemented");
      },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim, //
                            globalId(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                     llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x), //
                            globalId(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
                                     llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y), //
                            globalId(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
                                     llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z));
      },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                                                                //
                            globalSize(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x), //
                            globalSize(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y), //
                            globalSize(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z));
      },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                 //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z));
      },
      [&](const Spec::GpuGroupSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                  //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z));
      },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                               //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z));
      },
      [&](const Spec::GpuLocalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y), //
                            cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z));
      });
}
ValPtr NVPTXTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total( //                                                     //
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
