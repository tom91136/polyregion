#include "llvm_amdgpu.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace polyregion::backend;

void AMDGPUTargetSpecificHandler::witnessEntry(LLVMBackend::AstTransformer &ctx, llvm::Module &mod, llvm::Function &fn) {
  fn.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
}
ValPtr AMDGPUTargetSpecificHandler::mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &expr) {

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
  auto globalSizeU32 = [&](size_t dim) -> ValPtr {
    if (dim >= 3) throw std::logic_error("Dim >= 3");
    auto i32Ty = llvm::Type::getInt32Ty(xform.C);
    auto i32ptr = xform.B.CreatePointerCast(xform.intr0(fn, llvm::Intrinsic::amdgcn_dispatch_ptr), i32Ty->getPointerTo());
    // 127:96   grid_size_x;  (32*3+(0*32)==96)
    // 159:128  grid_size_y;  (32*3+(1*32)==128)
    // 191:160  grid_size_z;  (32*3+(2*32)==160)
    auto size = xform.B.CreateInBoundsGEP(i32Ty, i32ptr, llvm::ConstantInt::get(i32Ty, 3 + dim));
    return xform.load(size, i32Ty);
  };

  // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_local_size.cl
  auto localSizeU32 = [&](size_t dim) -> ValPtr {
    if (dim >= 3) throw std::logic_error("Dim >= 3");
    auto i16Ty = llvm::Type::getInt16Ty(xform.C);
    auto i16ptr = xform.B.CreatePointerCast(xform.intr0(fn, llvm::Intrinsic::amdgcn_dispatch_ptr), i16Ty->getPointerTo());
    // 47:32   workgroup_size_x (16*2+(0*16)==32)
    // 63:48   workgroup_size_y (16*2+(1*16)==48)
    // 79:64   workgroup_size_z (16*2+(2*16)==64)
    auto size = xform.B.CreateInBoundsGEP(i16Ty, i16ptr, llvm::ConstantInt::get(i16Ty, 2 + dim));
    return xform.B.CreateIntCast(xform.load(size, i16Ty), llvm::Type::getInt32Ty(xform.C), false);
  };

  auto globalIdU32 = [&](llvm::Intrinsic::ID workgroupId, llvm::Intrinsic::ID workitemId, size_t dim) -> ValPtr {
    return xform.B.CreateAdd(xform.B.CreateMul(xform.intr0(fn, workgroupId), localSizeU32(dim)), xform.intr0(fn, workitemId));
  };

  //            // see llvm/libclc/amdgcn-amdhsa/lib/workitem/get_num_groups.cl
  auto numGroupsU32 = [&](size_t dim) -> ValPtr {
    auto n = globalSizeU32(dim);
    auto d = localSizeU32(dim);
    auto q = xform.B.CreateUDiv(globalSizeU32(dim), localSizeU32(dim));                             // q = n / d
    auto rem = xform.B.CreateZExt(xform.B.CreateICmpUGT(n, xform.B.CreateMul(q, d)), n->getType()); // ( (uint32t) (n > q*d) )
    return xform.B.CreateAdd(q, rem);                                                               // q + rem
  };

  auto dim3OrAssert = [&](const AnyTerm &dim, ValPtr d0, ValPtr d1, ValPtr d2) {
    return xform.B.CreateSelect(
        xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntS32Const(0))), d0,
        xform.B.CreateSelect(xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntS32Const(1))), d1,
                             xform.B.CreateSelect(xform.B.CreateICmpEQ(xform.mkTermVal(dim), xform.mkTermVal(Term::IntS32Const(2))), d2,
                                                  xform.mkTermVal(Term::IntS32Const(0)))));
  };

  return expr.op.match_total(                                                            //
      [&](const Spec::Assert &v) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Term::IntConst(0xFF));
        return xform.intr0(fn, llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Term::IntConst(0xFF));
        return xform.intr0(fn, llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr {
        // work_group_barrier (__memory_scope, 1, 1)
        // FIXME
        // intr1(Intr::amdgcn_s_waitcnt,Type::Int(), Term::IntConst(0xFF));
        return xform.intr0(fn, llvm::Intrinsic::amdgcn_s_barrier);
      },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return xform.intr1(fn, llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Term::IntU32Const(0xFF));
      },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return xform.intr1(fn, llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Term::IntU32Const(0xFF));
      },
      [&](const Spec::GpuFenceAll &v) -> ValPtr {
        // atomic_work_item_fence(0, 5, 1) // FIXME
        return xform.intr1(fn, llvm::Intrinsic::amdgcn_s_waitcnt, Type::IntU32(), Term::IntU32Const(0xFF));
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
        return dim3OrAssert(v.dim,                                                   //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workgroup_id_x), //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workgroup_id_y), //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workgroup_id_z));
      },
      [&](const Spec::GpuGroupSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,           //
                            numGroupsU32(0), //
                            numGroupsU32(1), //
                            numGroupsU32(2));
      },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                  //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workitem_id_x), //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workitem_id_y), //
                            xform.intr0(fn, llvm::Intrinsic::amdgcn_workitem_id_z));
      },
      [&](const Spec::GpuLocalSize &v) -> ValPtr {
        return dim3OrAssert(v.dim,           //
                            localSizeU32(0), //
                            localSizeU32(1), //
                            localSizeU32(2));
      });
}
ValPtr AMDGPUTargetSpecificHandler::mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &expr) {
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
