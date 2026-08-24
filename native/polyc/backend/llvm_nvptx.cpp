#include "llvm_nvptx.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "aspartame/all.hpp"
#include "aspartame/ext/llvm.hpp"

using namespace aspartame;
using namespace polyregion::backend::details;

namespace {

// during AST codegen the module still carries LLVM's default layout (i64 ABI align 4); the NVPTX layout
// with i64:64 binds only after codegen. sizing words off the default under-counts an i64-padded aggregate,
// leaving its tail words unshuffled, so resolve the target layout while the module's is still default
uint64_t nvptxShuffleWords(CodeGen &cg, llvm::Type *valTy) {
  const auto &moduleDL = cg.M.getDataLayout();
  const llvm::DataLayout dl = moduleDL.isDefault() ? cg.C.options.targetInfo().resolveDataLayout() : moduleDL;
  return (dl.getTypeAllocSize(valTy) + 3) / 4;
}

} // namespace

void NVPTXTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  if (source.convention.is<CallConvention::OffloadEntry>()) {
    // XXX as of LLVM 21, it seems that the annotation method of marking kernel entries is now standardised to normal calling conventions,
    // keeping both for compatibility reasons
    fn.setCallingConv(llvm::CallingConv::PTX_Kernel);
    cg.M.getOrInsertNamedMetadata("nvvm.annotations")
        ->addOperand(llvm::MDNode::get(cg.C.actual, // XXX the attribute name must be "kernel" here and not the function name!
                                       {llvm::ValueAsMetadata::get(&fn), llvm::MDString::get(cg.C.actual, "kernel"),
                                        llvm::ValueAsMetadata::get(llvm::ConstantInt::get(cg.C.i32Ty(), 1))}));
  } else {
    fn.setDSOLocal(true);
  }
}
ValPtr NVPTXTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  const auto archNumber = (cg.C.options.arch ^ starts_with("sm_")) ? std::stoi(cg.C.options.arch ^ drop(3)) : 0;
  const bool legacySubgroup = archNumber != 0 && archNumber < 70;

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
  auto dim3OrAssert = [&](const AnyTerm &dim, ValPtr const d0, ValPtr const d1, ValPtr const d2) {
    if (dim.tpe() != Type::IntU32()) {
      throw std::logic_error("dim selector should be a " + to_string(Type::IntU32()) + " but got " + to_string(dim.tpe()));
    }
    return cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntU32Const(0))), d0,
                             cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntU32Const(1))), d1,
                                               cg.B.CreateSelect(cg.B.CreateICmpEQ(cg.mkTermVal(dim), cg.mkTermVal(Term::IntU32Const(2))),
                                                                 d2, cg.mkTermVal(Term::IntU32Const(0)))));
  };

  auto barrier0 = [&] {
    const auto callee = llvm::Intrinsic::getOrInsertDeclaration(&cg.M, llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all, {});
    return cg.B.CreateCall(callee, cg.mkTermVal(Term::IntU32Const(0)));
  };

  auto legacyBallot = [&](llvm::Value *pred) {
    auto *fnTy = llvm::FunctionType::get(cg.C.i32Ty(), {llvm::Type::getInt1Ty(cg.C.actual)}, false);
    auto *asmFn = llvm::InlineAsm::get(fnTy, "vote.ballot.b32 $0, $1;", "=r,b", false);
    auto *call = cg.B.CreateCall(asmFn, pred);
    call->addFnAttr(llvm::Attribute::Convergent);
    return call;
  };
  auto activeMask = [&]() -> llvm::Value * {
    if (legacySubgroup) return legacyBallot(llvm::ConstantInt::getTrue(cg.C.actual));
    // LLVM exposes llvm.nvvm.activemask but the NVPTX selector does not lower it in the supported LLVM
    // toolchain. Keep the convergent register read explicit instead of failing during instruction selection.
    auto *fnTy = llvm::FunctionType::get(cg.C.i32Ty(), {}, false);
    auto *asmFn = llvm::InlineAsm::get(fnTy, "activemask.b32 $0;", "=r", false);
    auto *active = cg.B.CreateCall(asmFn);
    active->addFnAttr(llvm::Attribute::Convergent);
    return active;
  };
  auto effectiveMask = [&](const Term::Any &mask, llvm::Value *active) -> llvm::Value * {
    auto *requested = cg.B.CreateIntCast(cg.mkTermVal(mask), cg.C.i32Ty(), false);
    auto *isAll = cg.B.CreateICmpEQ(requested, llvm::ConstantInt::get(cg.C.i32Ty(), 0xFFFFFFFFu));
    // Literal masks cannot make inactive lanes valid sources; the all-lanes sentinel means every lane
    // active at this convergent operation, including a partial warp.
    return cg.B.CreateSelect(isAll, active, cg.B.CreateAnd(requested, active));
  };

  // Per-word nvvm shfl. The public clamp is an inclusive segment mask; PTX packs its inverse into
  // c[12:8] and the clamp into c[4:0]. Every currently executing lane participates in shfl.sync,
  // while the requested mask is applied logically so excluded callers/sources retain their own value.
  auto shuffle = [&](char kind, llvm::Intrinsic::ID id, const Term::Any &value, const Term::Any &delta, const Term::Any &width,
                     const Term::Any &mask, const Type::Any &rtn) -> ValPtr {
    auto &B = cg.B;
    auto valTy = cg.resolveType(rtn);
    const auto words = nvptxShuffleWords(cg, valTy);
    auto srcVal = cg.mkTermVal(value);
    auto *i32Ty = cg.C.i32Ty();
    auto *lane = cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_laneid);
    auto *a = B.CreateIntCast(cg.mkTermVal(delta), i32Ty, false);
    auto *clamp = B.CreateAnd(B.CreateIntCast(cg.mkTermVal(width), i32Ty, false), llvm::ConstantInt::get(i32Ty, 31));
    auto *segmentBase = B.CreateAnd(lane, B.CreateNot(clamp));
    auto *segmentLast = B.CreateOr(segmentBase, clamp);
    llvm::Value *srcLane = nullptr;
    llvm::Value *shuffleArg = a;
    llvm::Value *inRange = nullptr;
    switch (kind) {
      case 'd':
        srcLane = B.CreateAdd(lane, a);
        inRange = B.CreateICmpULE(a, B.CreateSub(segmentLast, lane));
        break;
      case 'u':
        srcLane = B.CreateSub(lane, a);
        inRange = B.CreateICmpULE(a, B.CreateSub(lane, segmentBase));
        break;
      case 'x':
        srcLane = B.CreateXor(lane, a);
        inRange = B.CreateAnd(B.CreateICmpUGE(srcLane, segmentBase), B.CreateICmpULE(srcLane, segmentLast));
        break;
      default:
        shuffleArg = B.CreateAnd(a, clamp);
        srcLane = B.CreateOr(segmentBase, shuffleArg);
        inRange = B.CreateICmpULE(srcLane, segmentLast);
        break;
    }
    inRange = B.CreateAnd(inRange, B.CreateICmpULT(srcLane, llvm::ConstantInt::get(i32Ty, 32)));
    auto *active = activeMask();
    auto *members = effectiveMask(mask, active);
    const auto member = [&](llvm::Value *laneV) {
      auto *bit = B.CreateAnd(laneV, llvm::ConstantInt::get(i32Ty, 31));
      auto *set =
          B.CreateICmpNE(B.CreateAnd(B.CreateLShr(members, bit), llvm::ConstantInt::get(i32Ty, 1)), llvm::ConstantInt::get(i32Ty, 0));
      return B.CreateAnd(B.CreateICmpULT(laneV, llvm::ConstantInt::get(i32Ty, 32)), set);
    };
    inRange = B.CreateAnd(inRange, B.CreateAnd(member(lane), member(srcLane)));
    auto *ownArg = kind == 'i' ? B.CreateAnd(lane, clamp) : llvm::ConstantInt::get(i32Ty, 0);
    auto *safeArg = B.CreateSelect(inRange, shuffleArg, ownArg);
    auto *segmentMask = B.CreateAnd(B.CreateNot(clamp), llvm::ConstantInt::get(i32Ty, 31));
    // PTX interprets c[4:0] as the upper bound for down/idx/bfly, but as the lower bound for up.
    // The segment mask supplies the per-segment base, so shfl.up needs a zero low clamp.
    auto *controlClamp = kind == 'u' ? llvm::ConstantInt::get(i32Ty, 0) : clamp;
    auto *control = B.CreateOr(controlClamp, B.CreateShl(segmentMask, llvm::ConstantInt::get(i32Ty, 8)));
    const auto legacyId = [&] {
      if (id == llvm::Intrinsic::nvvm_shfl_sync_down_i32) return llvm::Intrinsic::nvvm_shfl_down_i32;
      if (id == llvm::Intrinsic::nvvm_shfl_sync_up_i32) return llvm::Intrinsic::nvvm_shfl_up_i32;
      if (id == llvm::Intrinsic::nvvm_shfl_sync_bfly_i32) return llvm::Intrinsic::nvvm_shfl_bfly_i32;
      return llvm::Intrinsic::nvvm_shfl_idx_i32;
    }();
    auto shfl = llvm::Intrinsic::getOrInsertDeclaration(&cg.M, legacySubgroup ? legacyId : id, {});
    // the shuffle yields a value of the declared element type; a pointer arg (an aggregate passed by
    // address) still returns the shuffled value, not dstPtr - the caller stores the result into a
    // value-typed slot, so returning the pointer would write the pointer bits over the aggregate
    return cg.shuffleStage(valTy, valTy, words, srcVal, "shfl", [&](llvm::Value *word) {
      auto *shuffled = legacySubgroup ? B.CreateCall(shfl, {word, safeArg, control}) : B.CreateCall(shfl, {active, word, safeArg, control});
      return B.CreateSelect(inRange, shuffled, word);
    });
  };

  auto nvptxScope = [](const MemScope::Any &s) -> std::string {
    return s.match_total([](const MemScope::Subgroup &) -> std::string { return "block"; }, // no warp-scoped atomic
                         [](const MemScope::Workgroup &) -> std::string { return "block"; },
                         [](const MemScope::Device &) -> std::string { return "device"; },
                         [](const MemScope::System &) -> std::string { return ""; });
  };
  auto ballot = [&](llvm::Value *mask, const Term::Any &pred) {
    if (legacySubgroup) return legacyBallot(cg.toI1(pred));
    auto *fnTy = llvm::FunctionType::get(cg.C.i32Ty(), {llvm::Type::getInt1Ty(cg.C.actual), cg.C.i32Ty()}, false);
    auto *asmFn = llvm::InlineAsm::get(fnTy, "vote.sync.ballot.b32 $0, $1, $2;", "=r,b,r", false);
    auto *call = cg.B.CreateCall(asmFn, {cg.toI1(pred), mask});
    call->addFnAttr(llvm::Attribute::Convergent);
    return call;
  };

  return expr.op.match_total( //
      [&](const Spec::Assert &v) -> ValPtr {
        // cg.extFn1(  "__assertfail", Type::Unit0(), Term::Unit0Const()); // TODO
        throw BackendException("unimplemented");
      },
      // Migrating from nvvm_barrier0, see https://github.com/llvm/llvm-project/pull/140615
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuBarrierAll &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_gl); }, // device-scope, cross-block
      [&](const Spec::GpuFenceLocal &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceAll &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_sys); },
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
      },
      [&](const Spec::GpuLaneIdx &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_laneid); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize); },
      [&](const Spec::GpuShuffleDown &v) -> ValPtr {
        return shuffle('d', llvm::Intrinsic::nvvm_shfl_sync_down_i32, v.value, v.delta, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuShuffleUp &v) -> ValPtr {
        return shuffle('u', llvm::Intrinsic::nvvm_shfl_sync_up_i32, v.value, v.delta, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuShuffleIdx &v) -> ValPtr {
        return shuffle('i', llvm::Intrinsic::nvvm_shfl_sync_idx_i32, v.value, v.srcLane, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuShuffleXor &v) -> ValPtr {
        return shuffle('x', llvm::Intrinsic::nvvm_shfl_sync_bfly_i32, v.value, v.laneMask, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuSubgroupBarrier &v) -> ValPtr {
        (void)v;
        // Before Volta, a warp executes in lockstep and has no warp execution-barrier instruction.
        // Preserve subgroup memory ordering without strengthening this into a CTA execution barrier,
        // which could deadlock when only one warp reaches a subgroup-uniform barrier.
        if (legacySubgroup) return cg.intr0(llvm::Intrinsic::nvvm_membar_cta);
        return cg.B.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(&cg.M, llvm::Intrinsic::nvvm_bar_warp_sync, {}), activeMask());
      },
      [&](const Spec::GpuBallot &v) -> ValPtr {
        auto *active = activeMask();
        auto *mask = effectiveMask(v.mask, active);
        return cg.B.CreateAnd(ballot(active, v.pred), mask);
      },
      [&](const Spec::GpuVoteAny &v) -> ValPtr {
        auto *active = activeMask();
        auto *mask = effectiveMask(v.mask, active);
        return cg.B.CreateICmpNE(cg.B.CreateAnd(ballot(active, v.pred), mask), llvm::ConstantInt::get(mask->getType(), 0));
      },
      [&](const Spec::GpuVoteAll &v) -> ValPtr {
        auto *active = activeMask();
        auto *mask = effectiveMask(v.mask, active);
        return cg.B.CreateICmpEQ(cg.B.CreateAnd(ballot(active, v.pred), mask), mask);
      },
      [&](const Spec::GpuAtomicRMW &v) -> ValPtr { return cg.mkAtomicRMW(v, nvptxScope(v.scope)); },
      [&](const Spec::GpuAtomicCAS &v) -> ValPtr { return cg.mkAtomicCAS(v, nvptxScope(v.scope)); },
      [&](const Spec::GpuGroupReduce &) -> ValPtr { throw BackendException("Spec::GpuGroupReduce lowering not yet implemented"); },
      [&](const Spec::GpuGroupInclusiveScan &) -> ValPtr {
        throw BackendException("Spec::GpuGroupInclusiveScan lowering not yet implemented");
      },
      [&](const Spec::GpuGroupExclusiveScan &) -> ValPtr {
        throw BackendException("Spec::GpuGroupExclusiveScan lowering not yet implemented");
      },
      [&](const Spec::RemoteLaunch &) -> ValPtr { throw BackendException("Spec::RemoteLaunch is a local orchestration operation"); },
      [&](const Spec::RemoteAlloc &) -> ValPtr { throw BackendException("Spec::RemoteAlloc is a local orchestration operation"); },
      [&](const Spec::RemoteFree &) -> ValPtr { throw BackendException("Spec::RemoteFree is a local orchestration operation"); },
      [&](const Spec::RemoteMemcpy &) -> ValPtr { throw BackendException("Spec::RemoteMemcpy is a local orchestration operation"); },
      [&](const Spec::RemoteSync &) -> ValPtr { throw BackendException("Spec::RemoteSync is a local orchestration operation"); },
      [&](const Spec::GpuVolatileLoad &v) -> ValPtr { return cg.mkVolatileLoad(v); },
      [&](const Spec::GpuVolatileStore &v) -> ValPtr { return cg.mkVolatileStore(v); });
}

void NVPTXTargetSpecificHandler::postProcessModule(CodeGen &cg) {
  // XXX Lower addrspace(3) kernel params to a module-level `extern addrspace(3) global` and
  // coerce the param's AS to default; positional arg layout is preserved so the launcher slot
  // stays aligned with the OpenCL kernarg ABI.
  llvm::Module &M = cg.M;
  llvm::LLVMContext &ctx = cg.C.actual;
  llvm::GlobalVariable *sharedGlobal = nullptr;
  auto getSharedGlobal = [&]() {
    if (sharedGlobal) return sharedGlobal;
    // an `extern __shared__` decl in codegen may already have emitted the dynamic shared global; share it
    if (auto *existing = M.getNamedGlobal(PolycDynSharedGlobal)) return sharedGlobal = existing;
    auto *arrTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(ctx), 0);
    sharedGlobal = new llvm::GlobalVariable(M, arrTy, /*isConstant*/ false, llvm::GlobalValue::ExternalLinkage,
                                            /*Initializer*/ nullptr, PolycDynSharedGlobal, /*InsertBefore*/ nullptr,
                                            llvm::GlobalValue::NotThreadLocal, AddrSpace::Workgroup);
    sharedGlobal->setAlignment(llvm::Align(16));
    return sharedGlobal;
  };

  auto kernels = M                                                                                                                    //
                 | filter([](const auto &fn) { return !fn.isDeclaration() && fn.getCallingConv() == llvm::CallingConv::PTX_Kernel; }) //
                 | map([](const auto &fn) { return const_cast<llvm::Function *>(&fn); })                                              //
                 | to_vector();

  // NVPTX kernel entry points must have external (non-local) linkage; internal linkage trips
  // getFunctionParamOptimizedAlign's non-local-linkage assertion at device O0
  for (auto *fn : kernels)
    if (fn->hasLocalLinkage()) fn->setLinkage(llvm::GlobalValue::ExternalLinkage);

  for (auto *fn : kernels) {
    bool hasSharedParam = false;
    for (auto &arg : fn->args()) {
      if (auto *pty = llvm::dyn_cast<llvm::PointerType>(arg.getType()); pty && pty->getAddressSpace() == 3) {
        hasSharedParam = true;
        break;
      }
    }
    if (!hasSharedParam) continue;

    auto *sg = getSharedGlobal();
    auto *defaultPtrTy = llvm::PointerType::get(ctx, 0);
    std::vector<llvm::Type *> newParamTys;
    std::vector<bool> coerceArg;
    coerceArg.reserve(fn->arg_size());
    for (auto &arg : fn->args()) {
      auto *pty = llvm::dyn_cast<llvm::PointerType>(arg.getType());
      const bool isShared = pty && pty->getAddressSpace() == 3;
      coerceArg.push_back(isShared);
      newParamTys.push_back(isShared ? defaultPtrTy : arg.getType());
    }
    auto *newFnTy = llvm::FunctionType::get(fn->getReturnType(), newParamTys, false);
    auto *newFn = llvm::Function::Create(newFnTy, fn->getLinkage(), fn->getAddressSpace(), "", &M);
    newFn->copyAttributesFrom(fn);
    newFn->setCallingConv(fn->getCallingConv());
    newFn->takeName(fn);

    llvm::ValueToValueMapTy vmap;
    auto newArgIt = newFn->arg_begin();
    auto oldArgIt = fn->arg_begin();
    for (size_t i = 0; i < coerceArg.size(); ++i, ++oldArgIt, ++newArgIt) {
      if (coerceArg[i]) {
        vmap[&*oldArgIt] = sg;
      } else {
        newArgIt->setName(oldArgIt->getName());
        vmap[&*oldArgIt] = &*newArgIt;
      }
    }
    newFn->splice(newFn->begin(), fn);
    for (auto &bb : *newFn)
      for (auto &inst : bb)
        llvm::RemapInstruction(&inst, vmap, llvm::RF_IgnoreMissingLocals);

    // Repoint the kernel-entry annotation; without this, `nvvm.annotations` would dangle after
    // we erase the old function and the verifier rejects the module.
    if (auto *md = M.getNamedMetadata("nvvm.annotations")) {
      for (unsigned i = 0; i < md->getNumOperands(); ++i) {
        auto *node = md->getOperand(i);
        if (node->getNumOperands() < 1) continue;
        auto *first = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(node->getOperand(0).get());
        if (first && first->getValue() == fn) {
          std::vector<llvm::Metadata *> newOps;
          newOps.reserve(node->getNumOperands());
          newOps.push_back(llvm::ValueAsMetadata::get(newFn));
          for (unsigned j = 1; j < node->getNumOperands(); ++j)
            newOps.push_back(node->getOperand(j).get());
          md->setOperand(i, llvm::MDNode::get(ctx, newOps));
        }
      }
    }
    fn->eraseFromParent();
  }
}
ValPtr NVPTXTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  // XXX libdevice: `__nv_<name>` for f64, `__nv_<name>f` for f32.
  const auto suffix = [&](const AnyType &rtn) { return cg.resolveType(rtn)->isFloatTy() ? "f" : ""; };
  return mkExternMathVal(
      cg, expr, //
      [&](const char *name, const AnyType &rtn, const AnyTerm &arg) {
        return cg.extFn1(std::string("__nv_") + name + suffix(rtn), rtn, arg);
      },
      [&](const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
        return cg.extFn2(std::string("__nv_") + name + suffix(rtn), rtn, lhs, rhs);
      },
      [&](const AnyType &tpe, const AnyTerm &x) { return cg.intr1(llvm::Intrinsic::fabs, tpe, x); });
}
