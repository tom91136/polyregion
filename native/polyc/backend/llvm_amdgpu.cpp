#include "llvm_amdgpu.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace polyregion::backend::details;

void AMDGPUTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  if (source.attrs.contains(FunctionAttr::Exported())) {
    fn.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  }
}
ValPtr AMDGPUTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {

  // HSA Sys Arch 1.2:  2.9.6 Kernel Dispatch Packet format:
  //  15:0    header Packet header, see 2.9.1 Packet header (on page 25).
  //  17:16   dimensions Number of dimensions specified in gridSize. Valid values are 1, 2, or 3.
  //  31:18   Reserved, must be 0.
  //  47:32   workgroup_size_x x dimension of work-group (measured in work-items).
  //  63:48   workgroup_size_y y dimension of work-group (measured in work-items).
  //  79:64   workgroup_size_z z dimension of work-group (measured in work-items).
  //  95:80   Reserved, must be 0.
  //  127:96  grid_size_x x dimension of grid (measured in work-items).
  //  159:128 grid_size_y y dimension of grid (measured in work-items).
  //  191:160 grid_size_z z dimension of grid (measured in work-items).
  //  223:192 private_segment_size_bytes Total size in bytes of private memory allocation request (per
  //          work-item).
  //  255:224 group_segment_size_bytes Total size in bytes of group memory allocation request (per
  //          work-group).
  //  319:256 kernel_object Handle for an object in memory that includes an
  //          implementation-defined executable ISA image for the kernel.
  //  383:320 kernarg_address Address of memory containing kernel arguments.
  //  447:384 Reserved, must be 0.
  //  511:448 completion_signal HSA signaling object handle used to indicate completion of the job

  // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_global_size.cl
  auto globalSizeU32 = [&](const size_t dim) -> ValPtr {
    if (dim >= 3) throw std::logic_error("Dim >= 3");
    const auto i32Ty = cg.C.i32Ty();
    const auto i32ptr = cg.B.CreatePointerCast(cg.intr0(llvm::Intrinsic::amdgcn_dispatch_ptr), llvm::PointerType::get(cg.C.actual, 0));
    // 127:96   grid_size_x;  (32*3+(0*32)==96)
    // 159:128  grid_size_y;  (32*3+(1*32)==128)
    // 191:160  grid_size_z;  (32*3+(2*32)==160)
    const auto size = cg.B.CreateInBoundsGEP(i32Ty, i32ptr, llvm::ConstantInt::get(i32Ty, 3 + dim));
    return cg.C.load(cg.B, size, i32Ty);
  };

  // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_local_size.cl
  auto localSizeU32 = [&](const size_t dim) -> ValPtr {
    if (dim >= 3) throw std::logic_error("Dim >= 3");
    const auto i16Ty = llvm::Type::getInt16Ty(cg.C.actual);
    const auto i16ptr = cg.B.CreatePointerCast(cg.intr0(llvm::Intrinsic::amdgcn_dispatch_ptr), llvm::PointerType::get(cg.C.actual, 0));
    // 47:32   workgroup_size_x (16*2+(0*16)==32)
    // 63:48   workgroup_size_y (16*2+(1*16)==48)
    // 79:64   workgroup_size_z (16*2+(2*16)==64)
    const auto size = cg.B.CreateInBoundsGEP(i16Ty, i16ptr, llvm::ConstantInt::get(i16Ty, 2 + dim));
    return cg.B.CreateIntCast(cg.C.load(cg.B, size, i16Ty), cg.C.i32Ty(), false);
  };

  auto globalIdU32 = [&](const llvm::Intrinsic::ID workgroupId, const llvm::Intrinsic::ID workitemId, const size_t dim) -> ValPtr {
    return cg.B.CreateAdd(cg.B.CreateMul(cg.intr0(workgroupId), localSizeU32(dim)), cg.intr0(workitemId));
  };

  //  see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_num_groups.cl
  auto numGroupsU32 = [&](const size_t dim) -> ValPtr {
    const auto n = globalSizeU32(dim);
    const auto d = localSizeU32(dim);
    const auto q = cg.B.CreateUDiv(globalSizeU32(dim), localSizeU32(dim));                       // q = n / d
    const auto rem = cg.B.CreateZExt(cg.B.CreateICmpUGT(n, cg.B.CreateMul(q, d)), n->getType()); // ( (uint32t) (n > q*d) )
    return cg.B.CreateAdd(q, rem);                                                               // q + rem
  };

  auto dim3OrAssert = [&](const AnyExpr &dim, const ValPtr d0, const ValPtr d1, const ValPtr d2) {
    return cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntS32Const(0))), d0,
                             cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntS32Const(1))), d1,
                                               cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkExprVal(dim), cg.mkExprVal(Expr::IntS32Const(2))),
                                                                 d2, cg.mkExprVal(Expr::IntS32Const(0)))));
  };

  return expr.op.match_total(                                                           //
      [&](const Spec::Assert &) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Expr::IntConst(0xFF));
        return cg.intr0(llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Expr::IntConst(0xFF));
        return cg.intr0(llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuBarrierAll &) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Expr::IntConst(0xFF));
        return cg.intr0(llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return cg.intr1(llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Expr::IntU32Const(0xFF));
      },
      [&](const Spec::GpuFenceLocal &) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return cg.intr1(llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Expr::IntU32Const(0xFF));
      },
      [&](const Spec::GpuFenceAll &) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return cg.intr1(llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Expr::IntU32Const(0xFF));
      },

      [&](const Spec::GpuGlobalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                                                         //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_x, llvm::Intrinsic::amdgcn_workitem_id_x, 0), //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_y, llvm::Intrinsic::amdgcn_workitem_id_y, 0), //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_z, llvm::Intrinsic::amdgcn_workitem_id_z, 0));
      },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,            //
                            globalSizeU32(0), //
                            globalSizeU32(1), //
                            globalSizeU32(2));
      },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                            //
                            cg.intr0(llvm::Intrinsic::amdgcn_workgroup_id_x), //
                            cg.intr0(llvm::Intrinsic::amdgcn_workgroup_id_y), //
                            cg.intr0(llvm::Intrinsic::amdgcn_workgroup_id_z));
      },
      [&](const Spec::GpuGroupSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,           //
                            numGroupsU32(0), //
                            numGroupsU32(1), //
                            numGroupsU32(2));
      },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                           //
                            cg.intr0(llvm::Intrinsic::amdgcn_workitem_id_x), //
                            cg.intr0(llvm::Intrinsic::amdgcn_workitem_id_y), //
                            cg.intr0(llvm::Intrinsic::amdgcn_workitem_id_z));
      },
      [&](const Spec::GpuLocalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,           //
                            localSizeU32(0), //
                            localSizeU32(1), //
                            localSizeU32(2));
      });
}
ValPtr AMDGPUTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total(                                                            //
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
