#include "llvm_nvptx.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace polyregion::backend;

void NVPTXTargetSpecificHandler::witnessEntry(LLVMBackend::AstTransformer &xform, llvm::Module &mod, llvm::Function &fn) {
  mod.getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDNode::get(xform.C, // XXX the attribute name must be "kernel" here and not the function name!
                                     {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(xform.C, "kernel"),
                                      llvm::ValueAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(xform.C), 1))}));
}
ValPtr NVPTXTargetSpecificHandler::mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &expr) {
  // threadId =  @llvm.nvvm.read.ptx.sreg.tid.*
  // blockIdx =  @llvm.nvvm.read.ptx.sreg.ctaid.*
  // blockDim =  @llvm.nvvm.read.ptx.sreg.ntid.*
  // gridDim  =  @llvm.nvvm.read.ptx.sreg.nctaid.*
  auto globalSize = [&](llvm::Intrinsic::ID nctaid, llvm::Intrinsic::ID ntid) -> ValPtr {
    return xform.B.CreateMul(xform.intr0(fn, nctaid), xform.intr0(fn, ntid));
  };
  auto globalId = [&](llvm::Intrinsic::ID ctaid, llvm::Intrinsic::ID ntid, llvm::Intrinsic::ID tid) -> ValPtr {
    return xform.B.CreateAdd(xform.B.CreateMul(xform.intr0(fn, ctaid), xform.intr0(fn, ntid)), xform.intr0(fn, tid));
  };
  auto dim3OrAssert = [&](const AnyTerm &dim, ValPtr d0, ValPtr d1, ValPtr d2) {
    if (dim.tpe() != Type::IntU32()) {
      throw std::logic_error("dim selector should be a " + to_string(Type::IntU32()) + " but got " + to_string(dim.tpe()));
    }
    return xform.B.CreateSelect(
        xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntU32Const(0))), d0,
        xform.B.CreateSelect(xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntU32Const(1))), d0,
                             xform.B.CreateSelect(xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntU32Const(2))), d0,
                                                  xform.mkTermVal(Term::IntU32Const(0)))));
  };

  return expr.op.match_total( //
      [&](const Spec::Assert &v) -> ValPtr {
        // xform.extFn1(fn, "__assertfail", Type::Unit0(), Term::Unit0Const()); // TODO
        throw BackendException("unimplemented");
      },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_barrier0); },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr { return xform.intr0(fn, llvm::Intrinsic::nvvm_membar_cta); },
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
        return dim3OrAssert(v.dim,                                                        //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z));
      },
      [&](const Spec::GpuGroupSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                         //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z));
      },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                      //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z));
      },
      [&](const Spec::GpuLocalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                       //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y), //
                            xform.intr0(fn, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z));
      });
}
ValPtr NVPTXTargetSpecificHandler::mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &expr) {
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
