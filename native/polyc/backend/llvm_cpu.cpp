#include "llvm_cpu.h"

#include "llvm/Analysis/ValueTracking.h"

#include "aspartame/all.hpp"

#include "polyregion/enums.h"

using namespace polyregion::backend::details;
using namespace aspartame;

static bool pointerFreeAggregate(llvm::Type *type) {
  if (type->isPointerTy()) return false;
  if (const auto *structure = llvm::dyn_cast<llvm::StructType>(type)) return llvm::all_of(structure->elements(), pointerFreeAggregate);
  if (const auto *array = llvm::dyn_cast<llvm::ArrayType>(type)) return pointerFreeAggregate(array->getElementType());
  if (const auto *vector = llvm::dyn_cast<llvm::VectorType>(type)) return pointerFreeAggregate(vector->getElementType());
  return true;
}

static llvm::Value *recoverSingleStoredPointer(llvm::Value *value) {
  auto *load = llvm::dyn_cast<llvm::LoadInst>(value);
  if (!load) return value;
  auto *slot = load->getPointerOperand()->stripPointerCasts();
  if (!llvm::isa<llvm::AllocaInst>(slot)) return value;
  llvm::StoreInst *definition = nullptr;
  for (auto *user : slot->users()) {
    if (auto *store = llvm::dyn_cast<llvm::StoreInst>(user)) {
      if (store->getPointerOperand()->stripPointerCasts() != slot || definition) return value;
      definition = store;
    } else if (const auto *read = llvm::dyn_cast<llvm::LoadInst>(user)) {
      if (read->getPointerOperand()->stripPointerCasts() != slot) return value;
    } else {
      return value;
    }
  }
  return definition ? definition->getValueOperand() : value;
}

void CPUTargetSpecificHandler::witnessFn(CodeGen &ctx, llvm::Function &fn, const Function &source) {
  if (!source.visibility.is<FunctionVisibility::Exported>()) {
    fn.setDSOLocal(true);
  }
}
ValPtr CPUTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  const auto noop = [&] { return cg.mkTermVal(Term::Unit0Const()); };
  const auto k = [&](const auto &v, uint64_t n) -> ValPtr { return llvm::ConstantInt::get(cg.resolveType(v.tpe), n); };
  auto &ctx = cg.C.actual;
  auto *i64 = llvm::Type::getInt64Ty(ctx);
  auto *i1 = llvm::Type::getInt1Ty(ctx);
  auto *i32 = llvm::Type::getInt32Ty(ctx);
  auto *i8 = llvm::Type::getInt8Ty(ctx);
  auto *ptr = llvm::PointerType::get(ctx, 0);
  auto *sizeTy = cg.M.getDataLayout().getIntPtrType(ctx);
  auto *unit = llvm::Type::getVoidTy(ctx);
  const auto external = [&](const std::string &name, llvm::Type *result, llvm::ArrayRef<llvm::Type *> args) {
    return cg.M.getOrInsertFunction(name, llvm::FunctionType::get(result, args, false));
  };
  const auto checkedRuntimeCall = [&](llvm::FunctionCallee callee, llvm::ArrayRef<llvm::Value *> args) {
    auto *success = cg.B.CreateCall(callee, args);
    auto *function = cg.B.GetInsertBlock()->getParent();
    auto *failure = llvm::BasicBlock::Create(ctx, "polyrt.failure", function);
    auto *continuation = llvm::BasicBlock::Create(ctx, "polyrt.continue", function);
    cg.B.CreateCondBr(success, continuation, failure);
    cg.B.SetInsertPoint(failure);
    cg.B.CreateCall(external("polyrt_abort", unit, {}));
    cg.B.CreateUnreachable();
    cg.B.SetInsertPoint(continuation);
  };
  const auto remoteResultSlot = [&](const char *name) {
    auto &entry = cg.B.GetInsertBlock()->getParent()->getEntryBlock();
    llvm::IRBuilder<> entryBuilder(&entry, entry.getFirstNonPHIOrDbgOrAlloca());
    return entryBuilder.CreateAlloca(sizeTy, nullptr, name);
  };
  const auto dimensionZero = [&](const Term::Any &dimension, ValPtr zero, ValPtr nonzero) -> ValPtr {
    auto *dim = cg.B.CreateZExtOrTrunc(cg.mkTermVal(dimension), i32);
    return cg.B.CreateSelect(cg.B.CreateICmpEQ(dim, llvm::ConstantInt::get(i32, 0)), zero, nonzero);
  };
  const auto threadId = [&]() -> ValPtr {
    auto *value = cg.B.CreateCall(external("__polyregion_host_thread_global_idx", i64, {}));
    return cg.B.CreateZExtOrTrunc(value, i32);
  };
  const auto globalSize = [&]() -> ValPtr {
    auto *value = cg.B.CreateCall(external("__polyregion_host_thread_global_size", i64, {}));
    return cg.B.CreateZExtOrTrunc(value, i32);
  };
  const auto asSize = [&](const Term::Any &term) -> ValPtr {
    auto *value = cg.mkTermVal(term);
    return value->getType()->isPointerTy() ? cg.B.CreatePtrToInt(value, sizeTy) : cg.B.CreateZExtOrTrunc(value, sizeTy);
  };
  const auto runtimeType = [](const Type::Any &tpe) -> uint8_t {
    using RuntimeType = polyregion::runtime::Type;
    const auto value = [](const RuntimeType x) { return static_cast<uint8_t>(x); };
    if (tpe.is<Type::Bool1>()) return value(RuntimeType::Bool1);
    if (tpe.is<Type::IntU8>()) return value(RuntimeType::IntU8);
    if (tpe.is<Type::IntU16>()) return value(RuntimeType::IntU16);
    if (tpe.is<Type::IntU32>()) return value(RuntimeType::IntU32);
    if (tpe.is<Type::IntU64>()) return value(RuntimeType::IntU64);
    if (tpe.is<Type::IntS8>()) return value(RuntimeType::IntS8);
    if (tpe.is<Type::IntS16>()) return value(RuntimeType::IntS16);
    if (tpe.is<Type::IntS32>()) return value(RuntimeType::IntS32);
    if (tpe.is<Type::IntS64>()) return value(RuntimeType::IntS64);
    if (tpe.is<Type::Float16>()) return value(RuntimeType::Float16);
    if (tpe.is<Type::Float32>()) return value(RuntimeType::Float32);
    if (tpe.is<Type::Float64>()) return value(RuntimeType::Float64);
    // Stateless callables have no runtime state, but offload kernel boundaries retain a one-byte
    // placeholder so their physical ABI remains stable while the callable body is specialised.
    if (tpe.is<Type::FnRef>()) return value(RuntimeType::IntU8);
    return value(RuntimeType::Ptr);
  };
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
      // HostThreaded models each host task as a one-work-item workgroup: execution barriers are
      // therefore no-ops, while dimension-zero global/group topology reflects the dispatch.
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return dimensionZero(v.dim, threadId(), k(v, 0)); },    //
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return dimensionZero(v.dim, globalSize(), k(v, 1)); }, //
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return dimensionZero(v.dim, threadId(), k(v, 0)); },     //
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return dimensionZero(v.dim, globalSize(), k(v, 1)); },  //
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return k(v, 0); },                                       //
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return k(v, 1); },                                      //
      [&](const Spec::GpuLaneIdx &) -> ValPtr { return llvm::ConstantInt::get(i32, 0); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr { return llvm::ConstantInt::get(i32, 1); },
      [&](const Spec::GpuShuffleDown &v) -> ValPtr { return cg.mkTermVal(v.value); },
      [&](const Spec::GpuShuffleUp &v) -> ValPtr { return cg.mkTermVal(v.value); },
      [&](const Spec::GpuShuffleIdx &v) -> ValPtr { return cg.mkTermVal(v.value); },
      [&](const Spec::GpuShuffleXor &v) -> ValPtr { return cg.mkTermVal(v.value); },
      [&](const Spec::GpuSubgroupBarrier &) -> ValPtr { return noop(); },
      [&](const Spec::GpuBallot &v) -> ValPtr {
        auto *member =
            cg.B.CreateICmpNE(cg.B.CreateAnd(cg.mkTermVal(v.mask), llvm::ConstantInt::get(i32, 1)), llvm::ConstantInt::get(i32, 0));
        return cg.B.CreateSelect(cg.B.CreateAnd(member, cg.toI1(v.pred)), llvm::ConstantInt::get(i32, 1), llvm::ConstantInt::get(i32, 0));
      },
      [&](const Spec::GpuVoteAny &v) -> ValPtr {
        auto *member =
            cg.B.CreateICmpNE(cg.B.CreateAnd(cg.mkTermVal(v.mask), llvm::ConstantInt::get(i32, 1)), llvm::ConstantInt::get(i32, 0));
        return cg.B.CreateAnd(member, cg.toI1(v.pred));
      },
      [&](const Spec::GpuVoteAll &v) -> ValPtr {
        auto *member =
            cg.B.CreateICmpNE(cg.B.CreateAnd(cg.mkTermVal(v.mask), llvm::ConstantInt::get(i32, 1)), llvm::ConstantInt::get(i32, 0));
        return cg.B.CreateOr(cg.B.CreateNot(member), cg.toI1(v.pred));
      },
      [&](const Spec::GpuAtomicRMW &v) -> ValPtr { return cg.mkAtomicRMW(v, ""); },
      [&](const Spec::GpuAtomicCAS &v) -> ValPtr { return cg.mkAtomicCAS(v, ""); },
      [&](const Spec::GpuGroupReduce &) -> ValPtr { throw BackendException("Spec::GpuGroupReduce unsupported for CPU"); },
      [&](const Spec::GpuGroupInclusiveScan &) -> ValPtr { throw BackendException("Spec::GpuGroupInclusiveScan unsupported for CPU"); },
      [&](const Spec::GpuGroupExclusiveScan &) -> ValPtr { throw BackendException("Spec::GpuGroupExclusiveScan unsupported for CPU"); },
      [&](const Spec::RemoteLaunch &v) -> ValPtr {
        const auto count = v.args.size();
        const auto zero = llvm::ConstantInt::get(i64, 0);
        auto *argPointersType = llvm::ArrayType::get(ptr, count ? count : 1);
        auto *argTypesType = llvm::ArrayType::get(i8, count ? count : 1);
        auto *mirrorSizesType = llvm::ArrayType::get(sizeTy, count ? count : 1);
        auto *mirrorKindsType = llvm::ArrayType::get(i8, count ? count : 1);
        auto *argPointers = cg.B.CreateAlloca(argPointersType, nullptr, "remote_argptrs");
        auto *argTypes = cg.B.CreateAlloca(argTypesType, nullptr, "remote_argtypes");
        auto *mirrorSizes = cg.B.CreateAlloca(mirrorSizesType, nullptr, "remote_mirror_sizes");
        auto *mirrorKinds = cg.B.CreateAlloca(mirrorKindsType, nullptr, "remote_mirror_kinds");
        for (size_t index = 0; index < v.args.size(); ++index) {
          const auto &arg = v.args[index];
          ValPtr value;
          bool directAggregate = false;
          auto *mirrorSize = llvm::ConstantInt::get(sizeTy, 0);
          auto *mirrorKind = llvm::ConstantInt::get(i8, 0);
          if (arg.tpe().template is<Type::Struct>()) {
            auto *type = cg.resolveType(arg.tpe());
            const auto allocationSize = cg.M.getDataLayout().getTypeAllocSize(type).getFixedValue();
            if (allocationSize == 0) {
              auto *local = cg.B.CreateAlloca(i8, nullptr, "remote_empty_closure");
              cg.B.CreateStore(llvm::ConstantInt::get(i8, 0), local);
              value = local;
              mirrorSize = llvm::ConstantInt::get(sizeTy, 1);
            } else {
              auto *local = cg.B.CreateAlloca(type, nullptr, "remote_closure");
              cg.B.CreateStore(cg.mkTermVal(arg), local);
              value = local;
              mirrorSize = llvm::ConstantInt::get(sizeTy, allocationSize);
            }
            directAggregate = true;
            mirrorKind = llvm::ConstantInt::get(i8, 1);
          } else {
            value = cg.mkTermVal(arg);
            if (const auto pointer = arg.tpe().template get<Type::Ptr>(); pointer && pointer->comp.template is<Type::Struct>()) {
              int64_t allocationOffset = 0;
              auto *origin = recoverSingleStoredPointer(value);
              const auto *allocation =
                  llvm::dyn_cast<llvm::AllocaInst>(llvm::GetPointerBaseWithConstantOffset(origin, allocationOffset, cg.M.getDataLayout()));
              if (allocation && allocationOffset == 0) {
                auto *pointeeType = cg.resolveType(pointer->comp);
                if (allocation->getAllocatedType() == pointeeType && pointerFreeAggregate(pointeeType)) {
                  const auto allocationSize = cg.M.getDataLayout().getTypeAllocSize(pointeeType).getFixedValue();
                  mirrorSize = llvm::ConstantInt::get(sizeTy, allocationSize == 0 ? 1 : allocationSize);
                  mirrorKind = llvm::ConstantInt::get(i8, 2);
                }
              }
            }
          }
          auto *offset = llvm::ConstantInt::get(i64, index);
          if (!directAggregate) {
            auto *slot = cg.B.CreateAlloca(value->getType(), nullptr, "remote_arg");
            cg.B.CreateStore(value, slot);
            cg.B.CreateStore(cg.B.CreatePointerCast(slot, ptr), cg.B.CreateGEP(argPointersType, argPointers, {zero, offset}));
          } else {
            cg.B.CreateStore(cg.B.CreatePointerCast(value, ptr), cg.B.CreateGEP(argPointersType, argPointers, {zero, offset}));
          }
          cg.B.CreateStore(llvm::ConstantInt::get(i8, runtimeType(arg.tpe())), cg.B.CreateGEP(argTypesType, argTypes, {zero, offset}));
          cg.B.CreateStore(mirrorSize, cg.B.CreateGEP(mirrorSizesType, mirrorSizes, {zero, offset}));
          cg.B.CreateStore(mirrorKind, cg.B.CreateGEP(mirrorKindsType, mirrorKinds, {zero, offset}));
        }
        std::string kernelName = "_kernel";
        if (const auto fn = v.kernel.tpe().get<Type::FnRef>()) {
          kernelName = normaliseSymbol(fn->name);
        }
        auto *module = cg.B.CreateGlobalString(kernelName, "remote_module", 0, &cg.M);
        auto *kernel = cg.B.CreateGlobalString(kernelName, "remote_kernel", 0, &cg.M);
        checkedRuntimeCall(
            external("polyrt_remote_launch_with_mirrors", i1,
                     {ptr, ptr, ptr, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, ptr, ptr, ptr, ptr}),
            {cg.mkTermVal(v.context), module, kernel, asSize(v.gridX), asSize(v.gridY), asSize(v.gridZ), asSize(v.blockX), asSize(v.blockY),
             asSize(v.blockZ), asSize(v.shmem), llvm::ConstantInt::get(sizeTy, count), cg.B.CreateGEP(argTypesType, argTypes, {zero, zero}),
             cg.B.CreateGEP(argPointersType, argPointers, {zero, zero}), cg.B.CreateGEP(mirrorSizesType, mirrorSizes, {zero, zero}),
             cg.B.CreateGEP(mirrorKindsType, mirrorKinds, {zero, zero})});
        return noop();
      },
      [&](const Spec::RemoteAlloc &v) -> ValPtr {
        auto *result = remoteResultSlot("remote_alloc_result");
        checkedRuntimeCall(external("polyrt_remote_malloc", i1, {ptr, sizeTy, ptr}), {cg.mkTermVal(v.context), asSize(v.bytes), result});
        auto *value = cg.B.CreateLoad(sizeTy, result);
        return cg.B.CreateIntToPtr(value, cg.resolveType(v.tpe));
      },
      [&](const Spec::RemoteTempAlloc &v) -> ValPtr {
        auto *result = remoteResultSlot("remote_temp_alloc_result");
        checkedRuntimeCall(external("polyrt_remote_temp_malloc", i1, {ptr, sizeTy, ptr}),
                           {cg.mkTermVal(v.context), asSize(v.bytes), result});
        auto *value = cg.B.CreateLoad(sizeTy, result);
        return cg.B.CreateIntToPtr(value, cg.resolveType(v.tpe));
      },
      [&](const Spec::RemoteFree &v) -> ValPtr {
        checkedRuntimeCall(external("polyrt_remote_free", i1, {ptr, sizeTy}), {cg.mkTermVal(v.context), asSize(v.ptr)});
        return noop();
      },
      [&](const Spec::RemoteMemcpy &v) -> ValPtr {
        const auto direction =
            v.direction.match_total([](const Direction::LocalToRemote &) { return 0; }, [](const Direction::RemoteToLocal &) { return 1; },
                                    [](const Direction::RemoteToRemote &) { return 2; });
        checkedRuntimeCall(
            external("polyrt_remote_memcpy", i1, {ptr, sizeTy, sizeTy, sizeTy, i32}),
            {cg.mkTermVal(v.context), asSize(v.dst), asSize(v.src), asSize(v.bytes), llvm::ConstantInt::get(i32, direction)});
        return noop();
      },
      [&](const Spec::RemoteSync &v) -> ValPtr {
        cg.B.CreateCall(external("polyrt_remote_sync", unit, {ptr}), {cg.mkTermVal(v.context)});
        return noop();
      },
      [&](const Spec::GpuVolatileLoad &v) -> ValPtr { return cg.mkVolatileLoad(v); },
      [&](const Spec::GpuVolatileStore &v) -> ValPtr { return cg.mkVolatileStore(v); } //
  );
}
ValPtr CPUTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total(
      [&](const Math::Abs &v) -> ValPtr {
        return cg.unaryNumOp(
            expr, v.x, v.tpe, //
            [&](auto) { return cg.intrAbs(v.tpe, v.x); }, [&](auto) { return cg.intr1(llvm::Intrinsic::fabs, v.tpe, v.x); });
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
