#include "llvm_spirv_cl.h"

#include "aspartame/all.hpp"

using namespace polyregion::backend::details;
using namespace aspartame;

// Queue a memory fence to ensure correct ordering of memory operations to local memory
constexpr static uint32_t CLK_LOCAL_MEM_FENCE = 0x01;

// Queue a memory fence to ensure correct ordering of memory operations to global memory
constexpr static uint32_t CLK_GLOBAL_MEM_FENCE = 0x02;

void SPIRVOpenCLTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  fn.addFnAttr(llvm::Attribute::Convergent);
  fn.addFnAttr(llvm::Attribute::NoRecurse);
  fn.addFnAttr(llvm::Attribute::NoUnwind);
  // Block FunctionAttrs from inferring memory(none); SPIRV would emit FunctionControl::Pure and
  // the optimizer folds Pure bodies to OpUnreachable. optnone+noinline keep this stable; the
  // driver / SPIRV-Tools do their own opts.
  fn.setMemoryEffects(llvm::MemoryEffects::unknown());
  fn.addFnAttr(llvm::Attribute::OptimizeNone);
  fn.addFnAttr(llvm::Attribute::NoInline);

  // SPIR_KERNEL requires void return, so internal helpers (lambdas etc.) get SPIR_FUNC instead.
  if (!source.isEntry) {
    fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
    return;
  }

  // See clang/lib/CodeGen/CodeGenModule.cpp @ CodeGenModule::GenKernelArgMetadata.
  llvm::SmallVector<llvm::Metadata *, 8> addressQuals;     // MDNode for the kernel argument address space qualifiers.
  llvm::SmallVector<llvm::Metadata *, 8> accessQuals;      // MDNode for the kernel argument access qualifiers (images only).
  llvm::SmallVector<llvm::Metadata *, 8> argTypeNames;     // MDNode for the kernel argument type names.
  llvm::SmallVector<llvm::Metadata *, 8> argBaseTypeNames; // MDNode for the kernel argument base type names.
  llvm::SmallVector<llvm::Metadata *, 8> argTypeQuals;     // MDNode for the kernel argument type qualifiers.
  llvm::SmallVector<llvm::Metadata *, 8> argNames;         // MDNode for the kernel argument names.

  for (auto arg : source.args) {
    const auto ty = cg.resolveType(arg.named.tpe, true);
    addressQuals.push_back(llvm::ConstantAsMetadata::get( //
        cg.B.getInt32(ty->isPointerTy() ? ty->getPointerAddressSpace() : 0)));
    accessQuals.push_back(llvm::MDString::get(cg.C.actual, "none")); // write_only | read_only | read_write | none

    auto typeName = [](const Type::Any &tpe) -> std::string {
      auto impl = [](const Type::Any &x, auto &thunk) -> std::string {
        return x.match_total(                                                                                               //
            [&](const Type::Float16 &) -> std::string { return "half"; },                                                   //
            [&](const Type::Float32 &) -> std::string { return "float"; },                                                  //
            [&](const Type::Float64 &) -> std::string { return "double"; },                                                 //
            [&](const Type::IntU8 &) -> std::string { return "uchar"; },                                                    //
            [&](const Type::IntU16 &) -> std::string { return "ushort"; },                                                  //
            [&](const Type::IntU32 &) -> std::string { return "uint"; },                                                    //
            [&](const Type::IntU64 &) -> std::string { return "ulong"; },                                                   //
            [&](const Type::IntS8 &) -> std::string { return "char"; },                                                     //
            [&](const Type::IntS16 &) -> std::string { return "short"; },                                                   //
            [&](const Type::IntS32 &) -> std::string { return "int"; },                                                     //
            [&](const Type::IntS64 &) -> std::string { return "long"; },                                                    //
            [&](const Type::Bool1 &) -> std::string { return "char"; },                                                     //
            [&](const Type::Unit0 &) -> std::string { return "void"; },                                                     //
            [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },                                            //
            [&](const Type::Struct &s) -> std::string { return repr(s.name); },                                             //
            [&](const Type::Ptr &p) -> std::string { return thunk(p.comp, thunk) + "*"; },                                  //
            [&](const Type::Arr &a) -> std::string { return thunk(a.comp, thunk) + "[" + std::to_string(a.length) + "]"; }, //
            [&](const Type::Var &v) -> std::string { throw std::logic_error("Type::Var should be erased"); },               //
            [&](const Type::Exec &e) -> std::string { throw std::logic_error("Type::Exec should be erased"); }              //
        );
      };
      return impl(tpe, impl);
    };

    argTypeNames.push_back(llvm::MDString::get(cg.C.actual, typeName(arg.named.tpe)));
    argBaseTypeNames.push_back(llvm::MDString::get(cg.C.actual, typeName(arg.named.tpe)));
    argTypeQuals.push_back(llvm::MDString::get(cg.C.actual, "")); // const | restrict | volatile | pipe | ""
    argNames.push_back(llvm::MDString::get(cg.C.actual, arg.named.symbol));
  }

  fn.setMetadata("kernel_arg_addr_space", llvm::MDNode::get(cg.C.actual, addressQuals));
  fn.setMetadata("kernel_arg_access_qual", llvm::MDNode::get(cg.C.actual, accessQuals));
  fn.setMetadata("kernel_arg_type", llvm::MDNode::get(cg.C.actual, argTypeNames));
  fn.setMetadata("kernel_arg_base_type", llvm::MDNode::get(cg.C.actual, argBaseTypeNames));
  fn.setMetadata("kernel_arg_type_qual", llvm::MDNode::get(cg.C.actual, argTypeQuals));
  fn.setMetadata("kernel_arg_name", llvm::MDNode::get(cg.C.actual, argNames));
  fn.setCallingConv(llvm::CallingConv::SPIR_KERNEL);
}

// OpenCL builtin signatures (OpenCL C 1.2/2.0). Itanium mangling encodes the concrete arg
// types (j=uint, m=ulong, ...); the SPIRV backend asserts the call matches the declaration,
// so we resolve the canonical signature and cast at the call site.
namespace {
struct OclBuiltin {
  const char *mangled;
  llvm::Type *(*ret)(llvm::LLVMContext &); // size_t = i64 on 64-bit SPIRV
  std::vector<llvm::Type *(*)(llvm::LLVMContext &)> args;
};
inline llvm::Type *i32(llvm::LLVMContext &c) { return llvm::Type::getInt32Ty(c); }
inline llvm::Type *i64(llvm::LLVMContext &c) { return llvm::Type::getInt64Ty(c); }
inline llvm::Type *vd(llvm::LLVMContext &c) { return llvm::Type::getVoidTy(c); }

const OclBuiltin GET_GLOBAL_ID{"_Z13get_global_idj", i64, {i32}};
const OclBuiltin GET_GLOBAL_SIZE{"_Z15get_global_sizej", i64, {i32}};
const OclBuiltin GET_GROUP_ID{"_Z12get_group_idj", i64, {i32}};
const OclBuiltin GET_NUM_GROUPS{"_Z14get_num_groupsj", i64, {i32}};
const OclBuiltin GET_LOCAL_ID{"_Z12get_local_idj", i64, {i32}};
const OclBuiltin GET_LOCAL_SIZE{"_Z14get_local_sizej", i64, {i32}};
const OclBuiltin BARRIER{"_Z7barrierj", vd, {i32}};
const OclBuiltin MEM_FENCE{"_Z9mem_fencej", vd, {i32}};
} // namespace

static ValPtr callOcl(CodeGen &cg, const OclBuiltin &b, const AnyType &requestedRtn, llvm::ArrayRef<ValPtr> args) {
  auto &ctx = cg.C.actual;
  std::vector<llvm::Type *> paramTys;
  paramTys.reserve(b.args.size());
  for (auto &mk : b.args)
    paramTys.push_back(mk(ctx));
  auto *fnTy = llvm::FunctionType::get(b.ret(ctx), paramTys, /*isVarArg*/ false);
  auto fnCallee = cg.M.getOrInsertFunction(b.mangled, fnTy);
  auto *fn = llvm::cast<llvm::Function>(fnCallee.getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  fn->addFnAttr(llvm::Attribute::Convergent);
  fn->addFnAttr(llvm::Attribute::NoUnwind);

  std::vector<llvm::Value *> coerced;
  coerced.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto *src = args[i];
    auto *dst = paramTys[i];
    if (src->getType() == dst) coerced.push_back(src);
    else if (src->getType()->isIntegerTy() && dst->isIntegerTy()) coerced.push_back(cg.B.CreateIntCast(src, dst, /*isSigned*/ false));
    else throw polyregion::backend::BackendException(std::string("cannot coerce arg to OCL builtin ") + b.mangled);
  }
  auto *call = cg.B.CreateCall(fn, coerced);
  call->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  if (b.ret(ctx)->isVoidTy()) return call;
  auto *want = cg.resolveType(requestedRtn, true);
  if (call->getType() == want) return call;
  if (call->getType()->isIntegerTy() && want->isIntegerTy()) return cg.B.CreateIntCast(call, want, /*isSigned*/ false);
  throw polyregion::backend::BackendException(std::string("cannot coerce OCL builtin ") + b.mangled + " result to requested type");
}

// See https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-1.2-built-in-functions
ValPtr SPIRVOpenCLTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  auto &ctx = cg.C.actual;
  auto u32 = [&](uint32_t k) { return llvm::ConstantInt::get(i32(ctx), k); };

  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return callOcl(cg, GET_GLOBAL_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return callOcl(cg, GET_GLOBAL_SIZE, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return callOcl(cg, GET_GROUP_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return callOcl(cg, GET_NUM_GROUPS, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return callOcl(cg, GET_LOCAL_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return callOcl(cg, GET_LOCAL_SIZE, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return callOcl(cg, BARRIER, v.tpe, {u32(CLK_GLOBAL_MEM_FENCE)}); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return callOcl(cg, BARRIER, v.tpe, {u32(CLK_LOCAL_MEM_FENCE)}); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr {
        return callOcl(cg, BARRIER, v.tpe, {u32(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)});
      },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return callOcl(cg, MEM_FENCE, v.tpe, {u32(CLK_GLOBAL_MEM_FENCE)}); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return callOcl(cg, MEM_FENCE, v.tpe, {u32(CLK_LOCAL_MEM_FENCE)}); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr {
        return callOcl(cg, MEM_FENCE, v.tpe, {u32(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)});
      });
}
ValPtr SPIRVOpenCLTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total( //                                                       //
      [&](const Math::Abs &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Sin &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Cos &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Tan &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Asin &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Acos &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Atan &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Sinh &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Cosh &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Tanh &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Signum &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); }, //
      [&](const Math::Round &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Ceil &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Floor &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Rint &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Sqrt &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Cbrt &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },   //
      [&](const Math::Exp &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Expm1 &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Log &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Log1p &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Log10 &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Pow &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },    //
      [&](const Math::Atan2 &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },  //
      [&](const Math::Hypot &v) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); }   //
  );
}
