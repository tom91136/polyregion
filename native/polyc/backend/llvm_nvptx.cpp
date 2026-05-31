#include "llvm_nvptx.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace polyregion::backend::details;

void NVPTXTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  if (source.isEntry) {
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

  return expr.op.match_total( //
      [&](const Spec::Assert &v) -> ValPtr {
        // cg.extFn1(  "__assertfail", Type::Unit0(), Term::Unit0Const()); // TODO
        throw BackendException("unimplemented");
      },
      // Migrating from nvvm_barrier0, see https://github.com/llvm/llvm-project/pull/140615
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuBarrierAll &) -> ValPtr { return barrier0(); },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceLocal &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
      [&](const Spec::GpuFenceAll &) -> ValPtr { return cg.intr0(llvm::Intrinsic::nvvm_membar_cta); },
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

void NVPTXTargetSpecificHandler::postProcessModule(CodeGen &cg) {
  // XXX Lower addrspace(3) kernel params to a module-level `extern addrspace(3) global` and
  // coerce the param's AS to default; positional arg layout is preserved so the launcher slot
  // stays aligned with the OpenCL kernarg ABI.
  llvm::Module &M = cg.M;
  llvm::LLVMContext &ctx = cg.C.actual;
  llvm::GlobalVariable *sharedGlobal = nullptr;
  auto getSharedGlobal = [&]() {
    if (sharedGlobal) return sharedGlobal;
    auto *arrTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(ctx), 0);
    sharedGlobal = new llvm::GlobalVariable(M, arrTy, /*isConstant*/ false, llvm::GlobalValue::ExternalLinkage,
                                            /*Initializer*/ nullptr, PolycDynSharedGlobal, /*InsertBefore*/ nullptr,
                                            llvm::GlobalValue::NotThreadLocal, AddrSpace::Workgroup);
    sharedGlobal->setAlignment(llvm::Align(16));
    return sharedGlobal;
  };

  std::vector<llvm::Function *> kernels;
  for (auto &fn : M)
    if (!fn.isDeclaration() && fn.getCallingConv() == llvm::CallingConv::PTX_Kernel) kernels.push_back(&fn);

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
  auto nv1 = [&](const char *baseName, const AnyType &rtn, const AnyTerm &arg) -> ValPtr {
    auto *fpTy = cg.resolveType(rtn);
    const char *suffix = fpTy->isFloatTy() ? "f" : "";
    return cg.extFn1(std::string("__nv_") + baseName + suffix, rtn, arg);
  };
  auto nv2 = [&](const char *baseName, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) -> ValPtr {
    auto *fpTy = cg.resolveType(rtn);
    const char *suffix = fpTy->isFloatTy() ? "f" : "";
    return cg.extFn2(std::string("__nv_") + baseName + suffix, rtn, lhs, rhs);
  };
  return expr.op.match_total( //
      [&](const Math::Abs &v) -> ValPtr {
        return cg.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto) { return cg.intr1(llvm::Intrinsic::abs, v.tpe, v.x); },
            [&](auto) { return cg.intr1(llvm::Intrinsic::fabs, v.tpe, v.x); });
      },                                                                                 //
      [&](const Math::Sin &v) -> ValPtr { return nv1("sin", v.tpe, v.x); },              //
      [&](const Math::Cos &v) -> ValPtr { return nv1("cos", v.tpe, v.x); },              //
      [&](const Math::Tan &v) -> ValPtr { return nv1("tan", v.tpe, v.x); },              //
      [&](const Math::Asin &v) -> ValPtr { return nv1("asin", v.tpe, v.x); },            //
      [&](const Math::Acos &v) -> ValPtr { return nv1("acos", v.tpe, v.x); },            //
      [&](const Math::Atan &v) -> ValPtr { return nv1("atan", v.tpe, v.x); },            //
      [&](const Math::Sinh &v) -> ValPtr { return nv1("sinh", v.tpe, v.x); },            //
      [&](const Math::Cosh &v) -> ValPtr { return nv1("cosh", v.tpe, v.x); },            //
      [&](const Math::Tanh &v) -> ValPtr { return nv1("tanh", v.tpe, v.x); },            //
      [&](const Math::Signum &v) -> ValPtr { return cg.mkSignumVal(expr, v.x, v.tpe); }, //
      [&](const Math::Round &v) -> ValPtr {
        if (v.tpe.is<Type::Float16>() || v.tpe.is<Type::Float32>() || v.tpe.is<Type::Float64>()) return nv1("round", v.tpe, v.x);
        const auto rounded = nv1("round", v.x.tpe(), v.x);
        return cg.B.CreateFPToSI(rounded, cg.resolveType(v.tpe));
      },                                                                             //
      [&](const Math::Ceil &v) -> ValPtr { return nv1("ceil", v.tpe, v.x); },        //
      [&](const Math::Floor &v) -> ValPtr { return nv1("floor", v.tpe, v.x); },      //
      [&](const Math::Rint &v) -> ValPtr { return nv1("rint", v.tpe, v.x); },        //
      [&](const Math::Sqrt &v) -> ValPtr { return nv1("sqrt", v.tpe, v.x); },        //
      [&](const Math::Cbrt &v) -> ValPtr { return nv1("cbrt", v.tpe, v.x); },        //
      [&](const Math::Exp &v) -> ValPtr { return nv1("exp", v.tpe, v.x); },          //
      [&](const Math::Expm1 &v) -> ValPtr { return nv1("expm1", v.tpe, v.x); },      //
      [&](const Math::Log &v) -> ValPtr { return nv1("log", v.tpe, v.x); },          //
      [&](const Math::Log1p &v) -> ValPtr { return nv1("log1p", v.tpe, v.x); },      //
      [&](const Math::Log10 &v) -> ValPtr { return nv1("log10", v.tpe, v.x); },      //
      [&](const Math::Pow &v) -> ValPtr { return nv2("pow", v.tpe, v.x, v.y); },     //
      [&](const Math::Atan2 &v) -> ValPtr { return nv2("atan2", v.tpe, v.x, v.y); }, //
      [&](const Math::Hypot &v) -> ValPtr { return nv2("hypot", v.tpe, v.x, v.y); }  //
  );
}
