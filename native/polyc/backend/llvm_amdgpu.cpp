#include "llvm_amdgpu.h"

#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace polyregion::backend::details;

void AMDGPUTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  if (source.isEntry) {
    fn.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    // without this the AMDGPUAttributor skips the multi-dim workitem/workgroup-id ABI and y/z ids read 0
    fn.addFnAttr("amdgpu-flat-work-group-size", "1,1024");
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
    const auto q = cg.B.CreateUDiv(n, d);                                                        // q = n / d
    const auto rem = cg.B.CreateZExt(cg.B.CreateICmpUGT(n, cg.B.CreateMul(q, d)), n->getType()); // ( (uint32t) (n > q*d) )
    return cg.B.CreateAdd(q, rem);                                                               // q + rem
  };

  auto dim3OrAssert = [&](const AnyTerm &dim, const ValPtr d0, const ValPtr d1, const ValPtr d2) {
    return cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntS32Const(0))), d0,
                             cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntS32Const(1))), d1,
                                               cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntS32Const(2))),
                                                                 d2, cg.mkTermVal(Term::IntS32Const(0)))));
  };

  auto callIntr = [&](llvm::Intrinsic::ID id, llvm::ArrayRef<ValPtr> args) -> ValPtr {
    return cg.B.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(&cg.M, id, {}), args);
  };
  auto laneId = [&]() -> ValPtr {
    auto negOne = llvm::ConstantInt::get(cg.C.i32Ty(), -1);
    return callIntr(llvm::Intrinsic::amdgcn_mbcnt_hi,
                    {negOne, callIntr(llvm::Intrinsic::amdgcn_mbcnt_lo, {negOne, llvm::ConstantInt::get(cg.C.i32Ty(), 0)})});
  };
  auto activeMask = [&]() -> ValPtr {
    return cg.B.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(&cg.M, llvm::Intrinsic::amdgcn_ballot, {cg.C.i32Ty()}),
                           llvm::ConstantInt::getTrue(cg.C.actual));
  };
  // ds_bpermute shuffle: out-of-range source lane selects the own word to match shfl clamp semantics
  auto shuffle = [&](char kind, const Term::Any &value, const Term::Any &arg, const Term::Any &bound, const Term::Any &mask,
                     const Type::Any &rtn) -> ValPtr {
    auto &B = cg.B;
    auto i32Ty = cg.C.i32Ty();
    auto valTy = cg.resolveType(rtn);
    const uint64_t words = (cg.M.getDataLayout().getTypeAllocSize(valTy) + 3) / 4;
    auto srcVal = cg.mkTermVal(value);
    auto lid = laneId();
    auto a = B.CreateIntCast(cg.mkTermVal(arg), i32Ty, false);
    auto clamp = B.CreateIntCast(cg.mkTermVal(bound), i32Ty, false);
    auto segmentBase = B.CreateAnd(lid, B.CreateNot(clamp));
    auto segmentLast = B.CreateOr(segmentBase, clamp);
    ValPtr srcLane, inRange;
    switch (kind) {
      case 'd':
        srcLane = B.CreateAdd(lid, a);
        inRange = B.CreateICmpULE(srcLane, segmentLast);
        break;
      case 'u':
        srcLane = B.CreateSub(lid, a);
        inRange = B.CreateICmpUGE(srcLane, segmentBase);
        break;
      case 'x':
        srcLane = B.CreateXor(lid, a);
        inRange = B.CreateAnd(B.CreateICmpUGE(srcLane, segmentBase), B.CreateICmpULE(srcLane, segmentLast));
        break;
      default:
        srcLane = B.CreateOr(segmentBase, B.CreateAnd(a, clamp));
        inRange = B.CreateICmpULE(srcLane, segmentLast);
        break;
    }
    inRange = B.CreateAnd(inRange, B.CreateICmpULT(srcLane, cg.intr0(llvm::Intrinsic::amdgcn_wavefrontsize)));
    auto *maskV = B.CreateIntCast(cg.mkTermVal(mask), i32Ty, false);
    auto *active = activeMask();
    auto *requested = B.CreateIntCast(maskV, active->getType(), false);
    auto *isAll = B.CreateICmpEQ(maskV, llvm::ConstantInt::get(i32Ty, 0xFFFFFFFFu));
    auto *members = B.CreateSelect(isAll, active, B.CreateAnd(requested, active));
    const auto member = [&](ValPtr lane) {
      const auto bits = members->getType()->getIntegerBitWidth();
      auto *bounded = B.CreateICmpULT(lane, llvm::ConstantInt::get(i32Ty, bits));
      auto *bit = B.CreateIntCast(lane, members->getType(), false);
      auto *set = B.CreateICmpNE(B.CreateAnd(B.CreateLShr(members, bit), llvm::ConstantInt::get(members->getType(), 1)),
                                 llvm::ConstantInt::get(members->getType(), 0));
      return B.CreateAnd(bounded, set);
    };
    inRange = B.CreateAnd(inRange, B.CreateAnd(member(lid), member(srcLane)));
    auto *safeSrcLane = B.CreateSelect(inRange, srcLane, lid);
    auto index = B.CreateShl(safeSrcLane, llvm::ConstantInt::get(i32Ty, 2));
    // a pointer arg (aggregate passed by address) still returns the shuffled value, not dstPtr; the caller
    // stores into a value-typed slot, so returning the pointer writes pointer bits over the aggregate
    return cg.shuffleStage(valTy, valTy, words, srcVal, "shfl", [&](llvm::Value *word) -> llvm::Value * {
      auto bp = callIntr(llvm::Intrinsic::amdgcn_ds_bpermute, {index, word});
      return B.CreateSelect(inRange, bp, word);
    });
  };

  auto amdgpuScope = [](const MemScope::Any &s) -> std::string {
    return s.match_total([](const MemScope::Subgroup &) -> std::string { return "wavefront"; },
                         [](const MemScope::Workgroup &) -> std::string { return "workgroup"; },
                         [](const MemScope::Device &) -> std::string { return "agent"; },
                         [](const MemScope::System &) -> std::string { return ""; });
  };

  // s_barrier syncs execution not memory; workgroup-scope fences make the legaliser emit the s_waitcnt
  auto wgBarrier = [&]() -> ValPtr {
    const auto ws = cg.C.actual.getOrInsertSyncScopeID("workgroup");
    cg.B.CreateFence(llvm::AtomicOrdering::Release, ws);
    auto b = cg.intr0(llvm::Intrinsic::amdgcn_s_barrier);
    cg.B.CreateFence(llvm::AtomicOrdering::Acquire, ws);
    return b;
  };
  auto ballot = [&](llvm::Value *pred) {
    return cg.B.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(&cg.M, llvm::Intrinsic::amdgcn_ballot, {cg.C.i32Ty()}), pred);
  };
  auto memberMask = [&](const Term::Any &mask, llvm::Type *ballotTy) -> llvm::Value * {
    auto *requested = cg.B.CreateIntCast(cg.mkTermVal(mask), cg.C.i32Ty(), false);
    auto *active = activeMask();
    auto *literal = cg.B.CreateIntCast(requested, ballotTy, false);
    auto *isAll = cg.B.CreateICmpEQ(requested, llvm::ConstantInt::get(cg.C.i32Ty(), 0xFFFFFFFFu));
    return cg.B.CreateSelect(isAll, active, cg.B.CreateAnd(literal, active));
  };

  return expr.op.match_total(                                                           //
      [&](const Spec::Assert &) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { return wgBarrier(); },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { return wgBarrier(); },
      [&](const Spec::GpuBarrierAll &) -> ValPtr { return wgBarrier(); },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr {
        return cg.B.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent, cg.C.actual.getOrInsertSyncScopeID("agent"));
      },
      [&](const Spec::GpuFenceLocal &) -> ValPtr {
        return cg.B.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent, cg.C.actual.getOrInsertSyncScopeID("workgroup"));
      },
      [&](const Spec::GpuFenceAll &) -> ValPtr { return cg.B.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent); },

      [&](const Spec::GpuGlobalIdx &v) -> ValPtr {
        return dim3OrAssert(v.dim,                                                                                         //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_x, llvm::Intrinsic::amdgcn_workitem_id_x, 0), //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_y, llvm::Intrinsic::amdgcn_workitem_id_y, 1), //
                            globalIdU32(llvm::Intrinsic::amdgcn_workgroup_id_z, llvm::Intrinsic::amdgcn_workitem_id_z, 2));
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
      },
      [&](const Spec::GpuLaneIdx &) -> ValPtr { return laneId(); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr { return cg.intr0(llvm::Intrinsic::amdgcn_wavefrontsize); },
      [&](const Spec::GpuShuffleDown &v) -> ValPtr { return shuffle('d', v.value, v.delta, v.width, v.mask, v.rtn); },
      [&](const Spec::GpuShuffleUp &v) -> ValPtr { return shuffle('u', v.value, v.delta, v.width, v.mask, v.rtn); },
      [&](const Spec::GpuShuffleIdx &v) -> ValPtr { return shuffle('i', v.value, v.srcLane, v.width, v.mask, v.rtn); },
      [&](const Spec::GpuShuffleXor &v) -> ValPtr { return shuffle('x', v.value, v.laneMask, v.width, v.mask, v.rtn); },
      [&](const Spec::GpuSubgroupBarrier &) -> ValPtr { return cg.intr0(llvm::Intrinsic::amdgcn_wave_barrier); },
      [&](const Spec::GpuBallot &v) -> ValPtr {
        auto *votes = ballot(cg.mkTermVal(v.pred));
        return cg.B.CreateIntCast(cg.B.CreateAnd(votes, memberMask(v.mask, votes->getType())), cg.C.i32Ty(), false);
      },
      [&](const Spec::GpuVoteAny &v) -> ValPtr {
        auto *votes = ballot(cg.mkTermVal(v.pred));
        auto *mask = memberMask(v.mask, votes->getType());
        return cg.B.CreateICmpNE(cg.B.CreateAnd(votes, mask), llvm::ConstantInt::get(mask->getType(), 0));
      },
      [&](const Spec::GpuVoteAll &v) -> ValPtr {
        auto *votes = ballot(cg.mkTermVal(v.pred));
        auto *mask = memberMask(v.mask, votes->getType());
        return cg.B.CreateICmpEQ(cg.B.CreateAnd(votes, mask), mask);
      },
      [&](const Spec::GpuAtomicRMW &v) -> ValPtr { return cg.mkAtomicRMW(v, amdgpuScope(v.scope)); },
      [&](const Spec::RemoteLaunch &) -> ValPtr { throw BackendException("Spec::RemoteLaunch is a local orchestration operation"); },
      [&](const Spec::RemoteAlloc &) -> ValPtr { throw BackendException("Spec::RemoteAlloc is a local orchestration operation"); },
      [&](const Spec::RemoteFree &) -> ValPtr { throw BackendException("Spec::RemoteFree is a local orchestration operation"); },
      [&](const Spec::RemoteMemcpy &) -> ValPtr { throw BackendException("Spec::RemoteMemcpy is a local orchestration operation"); },
      [&](const Spec::RemoteSync &) -> ValPtr { throw BackendException("Spec::RemoteSync is a local orchestration operation"); },
      [&](const Spec::GpuVolatileLoad &v) -> ValPtr { return cg.mkVolatileLoad(v); },
      [&](const Spec::GpuVolatileStore &v) -> ValPtr { return cg.mkVolatileStore(v); });
}
ValPtr AMDGPUTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  // XXX OCML: `__ocml_<name>_f32` / `__ocml_<name>_f64`.
  const auto suffix = [&](const AnyType &rtn) { return cg.resolveType(rtn)->isFloatTy() ? "_f32" : "_f64"; };
  return mkExternMathVal(
      cg, expr, //
      [&](const char *name, const AnyType &rtn, const AnyTerm &arg) {
        return cg.extFn1(std::string("__ocml_") + name + suffix(rtn), rtn, arg);
      },
      [&](const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
        return cg.extFn2(std::string("__ocml_") + name + suffix(rtn), rtn, lhs, rhs);
      },
      [&](const AnyType &tpe, const AnyTerm &x) { return cg.intr1(llvm::Intrinsic::fabs, tpe, x); });
}
