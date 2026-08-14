#include "llvm_cpu.h"

#include "aspartame/all.hpp"

#include "polyregion/enums.h"

using namespace polyregion::backend::details;
using namespace aspartame;

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
  auto *i32 = llvm::Type::getInt32Ty(ctx);
  auto *i8 = llvm::Type::getInt8Ty(ctx);
  auto *ptr = llvm::PointerType::get(ctx, 0);
  auto *sizeTy = cg.M.getDataLayout().getIntPtrType(ctx);
  auto *unit = llvm::Type::getVoidTy(ctx);
  const auto external = [&](const std::string &name, llvm::Type *result, llvm::ArrayRef<llvm::Type *> args) {
    return cg.M.getOrInsertFunction(name, llvm::FunctionType::get(result, args, false));
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
      [&](const Spec::GpuGlobalIdx &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalSize &) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Spec::GpuGroupIdx &) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Spec::GpuGroupSize &) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return k(v, 0); },                           //
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return k(v, 1); },                          //
      [&](const Spec::GpuLaneIdx &) -> ValPtr { throw BackendException("Spec::GpuLaneIdx requires SubgroupLower"); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr { throw BackendException("Spec::GpuSubgroupSize requires SubgroupLower"); },
      [&](const Spec::GpuShuffleDown &) -> ValPtr { throw BackendException("Spec::GpuShuffleDown requires SubgroupLower"); },
      [&](const Spec::GpuShuffleUp &) -> ValPtr { throw BackendException("Spec::GpuShuffleUp requires SubgroupLower"); },
      [&](const Spec::GpuShuffleIdx &) -> ValPtr { throw BackendException("Spec::GpuShuffleIdx requires SubgroupLower"); },
      [&](const Spec::GpuShuffleXor &) -> ValPtr { throw BackendException("Spec::GpuShuffleXor requires SubgroupLower"); },
      [&](const Spec::GpuSubgroupBarrier &) -> ValPtr { return noop(); },
      [&](const Spec::GpuBallot &) -> ValPtr { throw BackendException("Spec::GpuBallot requires SubgroupLower"); },
      [&](const Spec::GpuVoteAny &) -> ValPtr { throw BackendException("Spec::GpuVoteAny requires SubgroupLower"); },
      [&](const Spec::GpuVoteAll &) -> ValPtr { throw BackendException("Spec::GpuVoteAll requires SubgroupLower"); },
      [&](const Spec::GpuAtomicRMW &) -> ValPtr { throw BackendException("Spec::GpuAtomicRMW unsupported for CPU"); },
      [&](const Spec::RemoteLaunch &v) -> ValPtr {
        const auto count = v.args.size();
        const auto zero = llvm::ConstantInt::get(i64, 0);
        auto *argPointersType = llvm::ArrayType::get(ptr, count ? count : 1);
        auto *argTypesType = llvm::ArrayType::get(i8, count ? count : 1);
        auto *argPointers = cg.B.CreateAlloca(argPointersType, nullptr, "remote_argptrs");
        auto *argTypes = cg.B.CreateAlloca(argTypesType, nullptr, "remote_argtypes");
        std::vector<ValPtr> mirrored;
        const auto mirror = [&](llvm::Type *type, auto &&source) -> ValPtr {
          const auto allocationSize = cg.M.getDataLayout().getTypeAllocSize(type).getFixedValue();
          const auto bytes = llvm::ConstantInt::get(sizeTy, allocationSize == 0 ? 1 : allocationSize);
          auto *remote = cg.B.CreateCall(external("polyrt_remote_malloc", sizeTy, {ptr, sizeTy}), {cg.mkTermVal(v.context), bytes});
          if (allocationSize > 0)
            cg.B.CreateCall(external("polyrt_remote_memcpy", unit, {ptr, sizeTy, sizeTy, sizeTy, i32}),
                            {cg.mkTermVal(v.context), remote, source(), llvm::ConstantInt::get(sizeTy, allocationSize),
                             llvm::ConstantInt::get(i32, 0)});
          mirrored.emplace_back(remote);
          return cg.B.CreateIntToPtr(remote, ptr);
        };
        v.args | zip_with_index(size_t{0}) | for_each([&](const auto &arg, const auto index) {
          ValPtr value;
          if (const auto pointer = arg.tpe().template get<Type::Ptr>(); pointer && pointer->comp.template is<Type::Struct>()) {
            value = mirror(cg.resolveType(pointer->comp), [&] { return asSize(arg); });
          } else if (arg.tpe().template is<Type::Struct>()) {
            auto *type = cg.resolveType(arg.tpe());
            auto *local = cg.B.CreateAlloca(type, nullptr, "remote_closure");
            cg.B.CreateStore(cg.mkTermVal(arg), local);
            value = mirror(type, [&] { return cg.B.CreatePtrToInt(local, sizeTy); });
          } else value = cg.mkTermVal(arg);
          auto *slot = cg.B.CreateAlloca(value->getType(), nullptr, "remote_arg");
          cg.B.CreateStore(value, slot);
          auto *offset = llvm::ConstantInt::get(i64, index);
          cg.B.CreateStore(cg.B.CreatePointerCast(slot, ptr), cg.B.CreateGEP(argPointersType, argPointers, {zero, offset}));
          cg.B.CreateStore(llvm::ConstantInt::get(i8, runtimeType(arg.tpe())), cg.B.CreateGEP(argTypesType, argTypes, {zero, offset}));
        });
        std::string kernelName = "_kernel";
        if (const auto fn = v.kernel.tpe().get<Type::FnRef>()) {
          kernelName = repr(fn->name) ^ map([](const auto c) { return std::isalnum(c) || c == '_' ? c : '_'; });
        }
        auto *module = cg.B.CreateGlobalString(kernelName, "remote_module", 0, &cg.M);
        auto *kernel = cg.B.CreateGlobalString(kernelName, "remote_kernel", 0, &cg.M);
        cg.B.CreateCall(external("polyrt_remote_launch", unit,
                                 {ptr, ptr, ptr, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, sizeTy, ptr, ptr}),
                        {cg.mkTermVal(v.context), module, kernel, asSize(v.gridX), asSize(v.gridY), asSize(v.gridZ), asSize(v.blockX),
                         asSize(v.blockY), asSize(v.blockZ), asSize(v.shmem), llvm::ConstantInt::get(sizeTy, count),
                         cg.B.CreateGEP(argTypesType, argTypes, {zero, zero}), cg.B.CreateGEP(argPointersType, argPointers, {zero, zero})});
        mirrored | for_each([&](auto *remote) {
          cg.B.CreateCall(external("polyrt_remote_free", unit, {ptr, sizeTy}), {cg.mkTermVal(v.context), remote});
        });
        return noop();
      },
      [&](const Spec::RemoteAlloc &v) -> ValPtr {
        auto *value = cg.B.CreateCall(external("polyrt_remote_malloc", sizeTy, {ptr, sizeTy}), {cg.mkTermVal(v.context), asSize(v.bytes)});
        return cg.B.CreateIntToPtr(value, cg.resolveType(v.tpe));
      },
      [&](const Spec::RemoteFree &v) -> ValPtr {
        cg.B.CreateCall(external("polyrt_remote_free", unit, {ptr, sizeTy}), {cg.mkTermVal(v.context), asSize(v.ptr)});
        return noop();
      },
      [&](const Spec::RemoteMemcpy &v) -> ValPtr {
        const auto direction =
            v.direction.match_total([](const Direction::LocalToRemote &) { return 0; }, [](const Direction::RemoteToLocal &) { return 1; },
                                    [](const Direction::RemoteToRemote &) { return 2; });
        cg.B.CreateCall(external("polyrt_remote_memcpy", unit, {ptr, sizeTy, sizeTy, sizeTy, i32}),
                        {cg.mkTermVal(v.context), asSize(v.dst), asSize(v.src), asSize(v.bytes), llvm::ConstantInt::get(i32, direction)});
        return noop();
      },
      [&](const Spec::RemoteSync &v) -> ValPtr {
        cg.B.CreateCall(external("polyrt_remote_sync", unit, {ptr}), {cg.mkTermVal(v.context)});
        return noop();
      },
      [&](const Spec::GpuVolatileLoad &) -> ValPtr { throw BackendException("Spec::GpuVolatileLoad unsupported for CPU"); },
      [&](const Spec::GpuVolatileStore &) -> ValPtr { throw BackendException("Spec::GpuVolatileStore unsupported for CPU"); } //
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
