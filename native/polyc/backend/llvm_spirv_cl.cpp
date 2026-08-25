#include "llvm_spirv_cl.h"

#include <cstring>
#include <string_view>
#include <unordered_map>

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#include "aspartame/all.hpp"
#include "aspartame/ext/llvm.hpp"

#include "polyregion/types.h"

using namespace polyregion::backend;
using namespace polyregion::backend::details;
using namespace aspartame;

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
  if (!source.convention.is<CallConvention::OffloadEntry>()) {
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

  for (auto arg : source.decl.args) {
    const auto ty = cg.resolveType(arg.named.tpe, true, /*kernelEntryArg*/ true);
    addressQuals.push_back(llvm::ConstantAsMetadata::get( //
        cg.B.getInt32(ty->isPointerTy() ? ty->getPointerAddressSpace() : 0)));
    accessQuals.push_back(llvm::MDString::get(cg.C.actual, "none")); // write_only | read_only | read_write | none

    auto typeName = [](const Type::Any &tpe) -> std::string {
      auto impl = [](const Type::Any &x, const auto &thunk) -> std::string {
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
            [&](const Type::Struct &s) -> std::string { return fqcn(s.name); },                                             //
            [&](const Type::Ptr &p) -> std::string { return thunk(p.comp, thunk) + "*"; },                                  //
            [&](const Type::Arr &a) -> std::string { return thunk(a.comp, thunk) + "[" + std::to_string(a.length) + "]"; }, //
            [&](const Type::Var &v) -> std::string { throw std::logic_error("Type::Var should be erased"); },               //
            [&](const Type::Exec &e) -> std::string { throw std::logic_error("Type::Exec should be erased"); },             //
            [&](const Type::FnRef &f) -> std::string { throw std::logic_error("Type::FnRef should be erased"); }            //
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
const OclBuiltin GET_SUB_GROUP_SIZE{"_Z18get_sub_group_sizev", i32, {}};
const OclBuiltin GET_SUB_GROUP_LOCAL_ID{"_Z22get_sub_group_local_idv", i32, {}};
// XXX Use the `__spirv_*` form: OpenCL `barrier(flags)` routes through SPIRVBuiltins.cpp::
// buildBarrierInst which ORs SequentiallyConsistent into the semantics; OpenCL drivers reject
// that and silently drop the barrier. The __spirv_* path bypasses that and emits the op with
// our chosen semantics.
const OclBuiltin CONTROL_BARRIER{"_Z22__spirv_ControlBarrierjjj", vd, {i32, i32, i32}};
const OclBuiltin MEMORY_BARRIER{"_Z21__spirv_MemoryBarrierjj", vd, {i32, i32}};

// Mirrors SPIRV::MemorySemantics / SPIRV::Scope; the upstream header is private to LLVM,
// but the constants are part of the SPIR-V ABI.
namespace SpvMemSem {
constexpr uint32_t AcquireRelease = 0x008;
constexpr uint32_t WorkgroupMemory = 0x100;
constexpr uint32_t CrossWorkgroupMemory = 0x200;
} // namespace SpvMemSem
namespace SpvScope {
constexpr uint32_t Workgroup = 2;
constexpr uint32_t Subgroup = 3;
} // namespace SpvScope

struct OclMangledMath {
  CodeGen &cg;
  static const char *typeCode(const AnyType &t) {
    if (t.is<polyregion::polyast::Type::Float32>()) return "f";
    if (t.is<polyregion::polyast::Type::Float64>()) return "d";
    if (t.is<polyregion::polyast::Type::Float16>()) return "Dh";
    throw polyregion::backend::BackendException("unsupported fp type for OpenCL math builtin");
  }
  static std::string mangle(const char *name, const AnyType &tpe, int arity) {
    std::string out = "_Z";
    out += std::to_string(std::strlen(name));
    out += name;
    const char *tc = typeCode(tpe);
    for (int i = 0; i < arity; ++i)
      out += tc;
    return out;
  }
  ValPtr unary(const char *name, const AnyType &rtn, const AnyTerm &arg) { return cg.extFn1(mangle(name, rtn, 1), rtn, arg); }
  ValPtr binary(const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
    return cg.extFn2(mangle(name, rtn, 2), rtn, lhs, rhs);
  }
};
} // namespace

static ValPtr callOcl(CodeGen &cg, const OclBuiltin &b, const AnyType &requestedRtn, llvm::ArrayRef<ValPtr> args) {
  auto &ctx = cg.C.actual;
  auto paramTys = b.args ^ map([&](const auto &mk) { return mk(ctx); });
  auto *fnTy = llvm::FunctionType::get(b.ret(ctx), paramTys, /*isVarArg*/ false);
  auto fnCallee = cg.M.getOrInsertFunction(b.mangled, fnTy);
  auto *fn = llvm::cast<llvm::Function>(fnCallee.getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  fn->addFnAttr(llvm::Attribute::Convergent);
  fn->addFnAttr(llvm::Attribute::NoUnwind);

  auto coerced = args | zip_with_index<size_t>() | map([&](const auto &src, const auto &i) -> llvm::Value * {
                   auto *dst = paramTys[i];
                   if (src->getType() == dst) return src;
                   if (src->getType()->isIntegerTy() && dst->isIntegerTy()) return cg.B.CreateIntCast(src, dst, /*isSigned*/ false);
                   throw polyregion::backend::BackendException(std::string("cannot coerce arg to OCL builtin ") + b.mangled);
                 }) //
                 | to_vector();
  auto *call = cg.B.CreateCall(fn, coerced);
  call->setCallingConv(llvm::CallingConv::SPIR_FUNC);

  if (b.ret(ctx)->isVoidTy()) return call;
  auto *want = cg.resolveType(requestedRtn, true);
  if (call->getType() == want) return call;
  if (call->getType()->isIntegerTy() && want->isIntegerTy()) return cg.B.CreateIntCast(call, want, /*isSigned*/ false);
  throw polyregion::backend::BackendException(std::string("cannot coerce OCL builtin ") + b.mangled + " result to requested type");
}

// call a mangled OpenCL builtin: SPIR_FUNC on both the declaration and the call site, and convergent so the
// optimiser cannot sink a collective out of uniform control flow
static llvm::CallInst *callSpirFunc(CodeGen &cg, const std::string &mangled, llvm::Type *rtn, llvm::ArrayRef<ValPtr> args) {
  std::vector<llvm::Type *> argTys;
  argTys.reserve(args.size());
  for (auto *a : args)
    argTys.push_back(a->getType());
  auto callee = cg.M.getOrInsertFunction(mangled, llvm::FunctionType::get(rtn, argTys, false));
  auto *fn = llvm::cast<llvm::Function>(callee.getCallee());
  fn->setCallingConv(llvm::CallingConv::SPIR_FUNC), fn->addFnAttr(llvm::Attribute::Convergent), fn->addFnAttr(llvm::Attribute::NoUnwind);
  auto *call = cg.B.CreateCall(fn, args);
  call->setCallingConv(llvm::CallingConv::SPIR_FUNC);
  return call;
}

// Itanium mangling of an OpenCL builtin taking `suffix`-typed arguments
static std::string mangleOcl(const std::string &name, const std::string &suffix) {
  return "_Z" + std::to_string(name.size()) + name + suffix;
}

static ValPtr emitGroupCollective(CodeGen &cg, const std::string &base, const AtomicOp::Any &op, const Term::Any &value) {
  const std::string suffix = op.is<AtomicOp::Add>()   ? "add"
                             : op.is<AtomicOp::Min>() ? "min"
                             : op.is<AtomicOp::Max>() ? "max"
                                                      : throw BackendException("SPIR-V group reduce/scan supports only add/min/max");
  const auto valueType = value.tpe();
  const std::string mangle = valueType.is<Type::Float32>()   ? "f"
                             : valueType.is<Type::Float64>() ? "d"
                             : valueType.is<Type::IntS32>()  ? "i"
                             : valueType.is<Type::IntU32>()  ? "j"
                             : valueType.is<Type::IntS64>()  ? "l"
                             : valueType.is<Type::IntU64>()  ? "m"
                                                             : throw BackendException("unsupported SPIR-V group collective element type");
  auto *valueV = cg.mkTermVal(value);
  return callSpirFunc(cg, mangleOcl(base + "_" + suffix, mangle), valueV->getType(), {valueV});
}

// OpenCL work_group_any/all take and return int, while the source predicate may be bool/i1.
static ValPtr emitGroupPredicate(CodeGen &cg, const std::string &name, const Term::Any &value) {
  auto *i32Ty = cg.C.i32Ty();
  auto *valueV = cg.mkTermVal(value);
  auto *predicateV = valueV->getType() == i32Ty ? valueV : cg.B.CreateIntCast(valueV, i32Ty, /*isSigned*/ false);
  auto *call = callSpirFunc(cg, mangleOcl(name, "i"), i32Ty, {predicateV});
  return call->getType() == valueV->getType() ? call : cg.B.CreateIntCast(call, valueV->getType(), /*isSigned*/ false);
}

// OpenCL Itanium mangle suffix for a scalar shuffle element from the polyast type; nullptr if not a scalar
static const char *scalarShuffleMangle(const Type::Any &rtn) {
  return rtn.is<Type::Float32>()   ? "f"
         : rtn.is<Type::Float64>() ? "d"
         : rtn.is<Type::Float16>() ? "Dh"
         : rtn.is<Type::IntS8>()   ? "c"
         : rtn.is<Type::IntU8>()   ? "h"
         : rtn.is<Type::IntS16>()  ? "s"
         : rtn.is<Type::IntU16>()  ? "t"
         : rtn.is<Type::IntS32>()  ? "i"
         : rtn.is<Type::IntU32>()  ? "j"
         : rtn.is<Type::IntS64>()  ? "l"
         : rtn.is<Type::IntU64>()  ? "m"
                                   : nullptr;
}

// width-derived mangle for an LLVM scalar leaf; shuffle is a bit-exact lane permutation so signedness is irrelevant
static std::string leafShuffleMangle(llvm::Type *ty) {
  if (ty->isFloatTy()) return "f";
  if (ty->isDoubleTy()) return "d";
  if (ty->isHalfTy()) return "Dh";
  if (ty->isIntegerTy()) switch (ty->getIntegerBitWidth()) {
      case 8: return "h";
      case 16: return "t";
      case 32: return "j";
      case 64: return "m";
      default: break;
    }
  throw BackendException("unsupported subgroup shuffle leaf type on SPIRV-CL");
}

static ValPtr emitShuffleCall(CodeGen &cg, const std::string &base, const std::string &m, ValPtr valV, ValPtr idxV) {
  return callSpirFunc(cg, mangleOcl(base, m + "j"), valV->getType(), {valV, idxV});
}

// permute an aggregate by shuffling each scalar leaf with the same lane index and reassembling
static ValPtr emitShuffleAgg(CodeGen &cg, const std::string &base, ValPtr valV, ValPtr idxV) {
  auto *ty = valV->getType();
  if (ty->isStructTy() || ty->isArrayTy()) {
    const unsigned n = ty->isStructTy() ? ty->getStructNumElements() : static_cast<unsigned>(ty->getArrayNumElements());
    ValPtr agg = llvm::UndefValue::get(ty);
    for (unsigned i = 0; i < n; ++i)
      agg = cg.B.CreateInsertValue(agg, emitShuffleAgg(cg, base, cg.B.CreateExtractValue(valV, {i}), idxV), {i});
    return agg;
  }
  return emitShuffleCall(cg, base, leafShuffleMangle(ty), valV, idxV);
}

// shuffle an aggregate carried by pointer: walk the leaves via GEP, shuffling each scalar from src into dst. this
// keeps only scalar loads/shuffles/stores in the IR - the SPIR-V backend selects an aggregate-SSA
// ExtractValue/InsertValue chain poorly (it asserts in ISel), so never form a whole-aggregate value here
static void emitShuffleAggPtr(CodeGen &cg, const std::string &base, llvm::Type *ty, ValPtr srcPtr, ValPtr dstPtr, ValPtr idxV) {
  auto *i32 = cg.C.i32Ty();
  if (ty->isStructTy() || ty->isArrayTy()) {
    const unsigned n = ty->isStructTy() ? ty->getStructNumElements() : static_cast<unsigned>(ty->getArrayNumElements());
    for (unsigned i = 0; i < n; ++i) {
      auto *elemTy = ty->isStructTy() ? ty->getStructElementType(i) : ty->getArrayElementType();
      llvm::Value *idx[] = {llvm::ConstantInt::get(i32, 0), llvm::ConstantInt::get(i32, i)};
      emitShuffleAggPtr(cg, base, elemTy, cg.B.CreateInBoundsGEP(ty, srcPtr, idx), cg.B.CreateInBoundsGEP(ty, dstPtr, idx), idxV);
    }
  } else {
    cg.B.CreateStore(emitShuffleCall(cg, base, leafShuffleMangle(ty), cg.B.CreateLoad(ty, srcPtr), idxV), dstPtr);
  }
}

// sub_group_shuffle{,_up,_down,_xor} lower to OpGroupNonUniformShuffle{,Up,Down,Xor} (subgroup scope); the SPIRV
// backend infers the scope from the `sub_group` name prefix. a scalar element shuffles in one call; an aggregate
// (by-segment scan carries a tuple<value,key> through the subgroup scan) permutes each scalar leaf the same way
static ValPtr emitSubgroupShuffle(CodeGen &cg, const std::string &base, const Term::Any &value, ValPtr idxV, const Type::Any &rtn) {
  idxV = cg.B.CreateIntCast(idxV, cg.C.i32Ty(), /*isSigned*/ false);
  auto *valV = cg.mkTermVal(value);
  if (const char *m = scalarShuffleMangle(rtn)) return emitShuffleCall(cg, base, m, valV, idxV);
  if (valV->getType()->isStructTy() || valV->getType()->isArrayTy()) return emitShuffleAgg(cg, base, valV, idxV);
  // structByPtr backends carry an aggregate (the by-segment scan's tuple<value,key>) by pointer; load it, shuffle
  // each scalar leaf, then hand back a fresh slot so the result stays in the by-pointer form the caller expects
  if (valV->getType()->isPointerTy() && (rtn.is<Type::Struct>() || rtn.is<Type::Arr>())) {
    auto *aggTy = cg.resolveType(rtn);
    auto *slot = cg.B.CreateAlloca(aggTy, cg.C.AllocaAS, nullptr, "sg_shuffle_agg");
    emitShuffleAggPtr(cg, base, aggTy, valV, slot, idxV);
    return slot;
  }
  throw BackendException("unsupported subgroup shuffle element type on SPIRV-CL");
}

// OpenCL subgroup shuffle builtins do not carry CUDA's clamp/member-mask operands. Compute a segment-relative
// source lane, execute the collective convergently, then retain the caller's value for lanes excluded by the
// clamp or mask. A clamp of 31 therefore describes independent 32-lane segments even on a 64-lane subgroup.
static ValPtr emitClampedSubgroupShuffle(CodeGen &cg, const char kind, const Term::Any &value, const Term::Any &arg, const Term::Any &bound,
                                         const Term::Any &mask, const Type::Any &rtn) {
  auto *i32 = cg.C.i32Ty();
  auto *lane = callOcl(cg, GET_SUB_GROUP_LOCAL_ID, Type::IntU32(), {});
  auto *subgroupSize = callOcl(cg, GET_SUB_GROUP_SIZE, Type::IntU32(), {});
  auto *a = cg.B.CreateIntCast(cg.mkTermVal(arg), i32, false);
  auto *clamp = cg.B.CreateIntCast(cg.mkTermVal(bound), i32, false);
  auto *segmentBase = cg.B.CreateAnd(lane, cg.B.CreateNot(clamp));
  auto *segmentLast = cg.B.CreateOr(segmentBase, clamp);
  ValPtr srcLane;
  ValPtr inRange;
  switch (kind) {
    case 'd':
      srcLane = cg.B.CreateAdd(lane, a);
      inRange = cg.B.CreateICmpULE(a, cg.B.CreateSub(segmentLast, lane));
      break;
    case 'u':
      srcLane = cg.B.CreateSub(lane, a);
      inRange = cg.B.CreateICmpULE(a, cg.B.CreateSub(lane, segmentBase));
      break;
    case 'x':
      srcLane = cg.B.CreateXor(lane, a);
      inRange = cg.B.CreateAnd(cg.B.CreateICmpUGE(srcLane, segmentBase), cg.B.CreateICmpULE(srcLane, segmentLast));
      break;
    default:
      srcLane = cg.B.CreateOr(segmentBase, cg.B.CreateAnd(a, clamp));
      inRange = cg.B.CreateICmpULE(srcLane, segmentLast);
      break;
  }
  inRange = cg.B.CreateAnd(inRange, cg.B.CreateICmpULT(srcLane, subgroupSize));

  auto *maskV = cg.B.CreateIntCast(cg.mkTermVal(mask), i32, false);
  // Match NVPTX's maskless convention: -1 means all currently active subgroup lanes; other values are
  // 32-bit masks relative to each aligned 32-lane segment on a wider subgroup.
  auto *isAll = cg.B.CreateICmpEQ(maskV, llvm::ConstantInt::get(i32, 0xFFFFFFFFu));
  const auto member = [&](ValPtr laneV) {
    auto *bit = cg.B.CreateAnd(laneV, llvm::ConstantInt::get(i32, 31));
    auto *set =
        cg.B.CreateICmpNE(cg.B.CreateAnd(cg.B.CreateLShr(maskV, bit), llvm::ConstantInt::get(i32, 1)), llvm::ConstantInt::get(i32, 0));
    return cg.B.CreateOr(isAll, set);
  };
  inRange = cg.B.CreateAnd(inRange, cg.B.CreateAnd(member(lane), member(srcLane)));
  // The collective still executes for every lane, so its operand itself must be valid; selecting the result
  // afterwards is too late for an out-of-range subgroup source. Selecting the caller lane here also makes the
  // shuffled result the original value when excluded, including aggregate values carried by pointer.
  auto *safeSrcLane = cg.B.CreateSelect(inRange, srcLane, lane);
  return emitSubgroupShuffle(cg, "sub_group_shuffle", value, safeSrcLane, rtn);
}

// See https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-1.2-built-in-functions
ValPtr SPIRVOpenCLTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  auto &ctx = cg.C.actual;
  auto u32 = [&](uint32_t k) { return llvm::ConstantInt::get(i32(ctx), k); };
  auto barrier = [&](const AnyType &tpe, uint32_t memSem) -> ValPtr {
    const auto scope = u32(SpvScope::Workgroup);
    return callOcl(cg, CONTROL_BARRIER, tpe, {scope, scope, u32(memSem | SpvMemSem::AcquireRelease)});
  };
  auto fence = [&](const AnyType &tpe, uint32_t memSem) -> ValPtr {
    return callOcl(cg, MEMORY_BARRIER, tpe, {u32(SpvScope::Workgroup), u32(memSem | SpvMemSem::AcquireRelease)});
  };
  auto subgroupBarrier = [&]() -> ValPtr {
    const auto scope = u32(SpvScope::Subgroup);
    return callOcl(cg, CONTROL_BARRIER, Type::Unit0(), {scope, scope, u32(SpvMemSem::WorkgroupMemory | SpvMemSem::AcquireRelease)});
  };

  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return callOcl(cg, GET_GLOBAL_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return callOcl(cg, GET_GLOBAL_SIZE, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return callOcl(cg, GET_GROUP_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return callOcl(cg, GET_NUM_GROUPS, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return callOcl(cg, GET_LOCAL_ID, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return callOcl(cg, GET_LOCAL_SIZE, v.tpe, {cg.mkTermVal(v.dim)}); },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return barrier(v.tpe, SpvMemSem::CrossWorkgroupMemory); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return barrier(v.tpe, SpvMemSem::WorkgroupMemory); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr { return barrier(v.tpe, SpvMemSem::WorkgroupMemory | SpvMemSem::CrossWorkgroupMemory); },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return fence(v.tpe, SpvMemSem::CrossWorkgroupMemory); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return fence(v.tpe, SpvMemSem::WorkgroupMemory); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr { return fence(v.tpe, SpvMemSem::WorkgroupMemory | SpvMemSem::CrossWorkgroupMemory); },
      [&](const Spec::GpuLaneIdx &) -> ValPtr { return callOcl(cg, GET_SUB_GROUP_LOCAL_ID, Type::IntU32(), {}); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr { return callOcl(cg, GET_SUB_GROUP_SIZE, Type::IntU32(), {}); },
      [&](const Spec::GpuShuffleDown &v) -> ValPtr {
        return emitClampedSubgroupShuffle(cg, 'd', v.value, v.delta, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuShuffleUp &v) -> ValPtr { return emitClampedSubgroupShuffle(cg, 'u', v.value, v.delta, v.width, v.mask, v.rtn); },
      [&](const Spec::GpuShuffleIdx &v) -> ValPtr {
        return emitClampedSubgroupShuffle(cg, 'i', v.value, v.srcLane, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuShuffleXor &v) -> ValPtr {
        return emitClampedSubgroupShuffle(cg, 'x', v.value, v.laneMask, v.width, v.mask, v.rtn);
      },
      [&](const Spec::GpuSubgroupBarrier &v) -> ValPtr {
        const auto literal = v.mask.get<Term::IntU32Const>();
        if (!literal || literal->value != -1) throw BackendException("Masked subgroup barriers are unsupported for SPIRV-OpenCL");
        return subgroupBarrier();
      },
      [&](const Spec::GpuBallot &) -> ValPtr { throw BackendException("Spec::GpuBallot requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuVoteAny &) -> ValPtr { throw BackendException("Spec::GpuVoteAny requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuVoteAll &) -> ValPtr { throw BackendException("Spec::GpuVoteAll requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuAtomicRMW &v) -> ValPtr { return cg.mkAtomicRMW(v, ""); },
      [&](const Spec::GpuAtomicCAS &v) -> ValPtr { return cg.mkAtomicCAS(v, ""); },
      [&](const Spec::GpuGroupReduce &v) -> ValPtr {
        if (v.value.tpe().is<Type::Bool1>() && v.op.is<AtomicOp::Or>()) return emitGroupPredicate(cg, "work_group_any", v.value);
        if (v.value.tpe().is<Type::Bool1>() && v.op.is<AtomicOp::And>()) return emitGroupPredicate(cg, "work_group_all", v.value);
        return emitGroupCollective(cg, "work_group_reduce", v.op, v.value);
      },
      [&](const Spec::GpuGroupInclusiveScan &v) -> ValPtr { return emitGroupCollective(cg, "work_group_scan_inclusive", v.op, v.value); },
      [&](const Spec::GpuGroupExclusiveScan &v) -> ValPtr { return emitGroupCollective(cg, "work_group_scan_exclusive", v.op, v.value); },
      [&](const Spec::RemoteLaunch &) -> ValPtr { throw BackendException("Spec::RemoteLaunch is a local orchestration operation"); },
      [&](const Spec::RemoteAlloc &) -> ValPtr { throw BackendException("Spec::RemoteAlloc is a local orchestration operation"); },
      [&](const Spec::RemoteFree &) -> ValPtr { throw BackendException("Spec::RemoteFree is a local orchestration operation"); },
      [&](const Spec::RemoteMemcpy &) -> ValPtr { throw BackendException("Spec::RemoteMemcpy is a local orchestration operation"); },
      [&](const Spec::RemoteSync &) -> ValPtr { throw BackendException("Spec::RemoteSync is a local orchestration operation"); },
      [&](const Spec::GpuVolatileLoad &v) -> ValPtr { return cg.mkVolatileLoad(v); },
      [&](const Spec::GpuVolatileStore &v) -> ValPtr { return cg.mkVolatileStore(v); });
}
ValPtr SPIRVOpenCLTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  OclMangledMath m{cg};
  const auto unary = [&](const char *name, const AnyType &rtn, const AnyTerm &arg) { return m.unary(name, rtn, arg); };
  const auto binary = [&](const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
    return m.binary(name, rtn, lhs, rhs);
  };
  return mkExternMathVal(cg, expr, unary, binary, [&](const AnyType &tpe, const AnyTerm &x) { return unary("fabs", tpe, x); });
}

void SPIRVVulkanTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  fn.addFnAttr(llvm::Attribute::Convergent);
  fn.addFnAttr(llvm::Attribute::NoRecurse);
  fn.addFnAttr(llvm::Attribute::NoUnwind);
  // block FunctionAttrs inferring memory(none): SPIRV would emit FunctionControl::Pure -> OpUnreachable
  fn.setMemoryEffects(llvm::MemoryEffects::unknown());

  // entry points use the hlsl.shader attribute, not SPIR_KERNEL; no optnone (it forces an unsupported capability)
  if (source.convention.is<CallConvention::OffloadEntry>()) {
    fn.addFnAttr("hlsl.shader", "compute");
    fn.addFnAttr("hlsl.numthreads", std::to_string(program_meta::VkWorkgroupSizeXValue) + ",1,1");
  } else fn.setCallingConv(llvm::CallingConv::SPIR_FUNC);
}

ValPtr SPIRVVulkanTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {
  auto &ctx = cg.C.actual;
  auto &B = cg.B;
  auto *i32t = cg.C.i32Ty();
  auto dimI32 = [&](const AnyTerm &dim) { return B.CreateIntCast(cg.mkTermVal(dim), i32t, /*isSigned*/ false); };
  auto coerce = [&](llvm::Value *v, const AnyType &tpe) -> ValPtr {
    auto *want = cg.resolveType(tpe, true);
    return v->getType() == want ? v : B.CreateIntCast(v, want, /*isSigned*/ false);
  };
  auto builtin = [&](llvm::Intrinsic::ID id, const AnyTerm &dim, const AnyType &tpe) -> ValPtr {
    return coerce(B.CreateIntrinsic(i32t, id, {dimI32(dim)}), tpe);
  };
  auto localSize = [&](const AnyTerm &dim) -> llvm::Value * {
    return B.CreateIntrinsic(i32t, llvm::Intrinsic::spv_workgroup_size, {dimI32(dim)});
  };
  auto groupBarrier = [&]() -> ValPtr {
    B.CreateIntrinsic(llvm::Type::getVoidTy(ctx), llvm::Intrinsic::spv_group_memory_barrier_with_group_sync, {});
    return cg.mkTermVal(Term::Unit0Const());
  };
  auto atomicScope = [](const MemScope::Any &scope) -> std::string {
    return scope.match_total([](const MemScope::Subgroup &) -> std::string { return "subgroup"; },
                             [](const MemScope::Workgroup &) -> std::string { return "workgroup"; },
                             [](const MemScope::Device &) -> std::string { return "device"; },
                             [](const MemScope::System &) -> std::string { return ""; });
  };
  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr { throw polyregion::backend::BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return builtin(llvm::Intrinsic::spv_thread_id, v.dim, v.tpe); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr {
        auto *nw = B.CreateIntrinsic(i32t, llvm::Intrinsic::spv_num_workgroups, {dimI32(v.dim)});
        return coerce(B.CreateMul(nw, localSize(v.dim)), v.tpe);
      },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return builtin(llvm::Intrinsic::spv_group_id, v.dim, v.tpe); },
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return builtin(llvm::Intrinsic::spv_num_workgroups, v.dim, v.tpe); },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return builtin(llvm::Intrinsic::spv_thread_id_in_group, v.dim, v.tpe); },
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return coerce(localSize(v.dim), v.tpe); },
      [&](const Spec::GpuBarrierGlobal &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuBarrierLocal &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuBarrierAll &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuFenceGlobal &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuFenceLocal &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuFenceAll &) -> ValPtr { return groupBarrier(); },
      [&](const Spec::GpuLaneIdx &) -> ValPtr { throw BackendException("Spec::GpuLaneIdx requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuSubgroupSize &) -> ValPtr {
        throw BackendException("Spec::GpuSubgroupSize requires native lowering or SubgroupLower");
      },
      [&](const Spec::GpuShuffleDown &) -> ValPtr {
        throw BackendException("Spec::GpuShuffleDown requires native lowering or SubgroupLower");
      },
      [&](const Spec::GpuShuffleUp &) -> ValPtr { throw BackendException("Spec::GpuShuffleUp requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuShuffleIdx &) -> ValPtr {
        throw BackendException("Spec::GpuShuffleIdx requires native lowering or SubgroupLower");
      },
      [&](const Spec::GpuShuffleXor &) -> ValPtr {
        throw BackendException("Spec::GpuShuffleXor requires native lowering or SubgroupLower");
      },
      [&](const Spec::GpuSubgroupBarrier &) -> ValPtr {
        throw BackendException("Spec::GpuSubgroupBarrier is unsupported for SPIRV-Vulkan");
      },
      [&](const Spec::GpuBallot &) -> ValPtr { throw BackendException("Spec::GpuBallot requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuVoteAny &) -> ValPtr { throw BackendException("Spec::GpuVoteAny requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuVoteAll &) -> ValPtr { throw BackendException("Spec::GpuVoteAll requires native lowering or SubgroupLower"); },
      [&](const Spec::GpuAtomicRMW &v) -> ValPtr { return cg.mkAtomicRMW(v, atomicScope(v.scope)); },
      [&](const Spec::GpuAtomicCAS &v) -> ValPtr { return cg.mkAtomicCAS(v, atomicScope(v.scope)); },
      [&](const Spec::GpuGroupReduce &) -> ValPtr {
        throw BackendException("Spec::GpuGroupReduce requires SubgroupLower for SPIRV-Vulkan");
      },
      [&](const Spec::GpuGroupInclusiveScan &) -> ValPtr {
        throw BackendException("Spec::GpuGroupInclusiveScan requires SubgroupLower for SPIRV-Vulkan");
      },
      [&](const Spec::GpuGroupExclusiveScan &) -> ValPtr {
        throw BackendException("Spec::GpuGroupExclusiveScan requires SubgroupLower for SPIRV-Vulkan");
      },
      [&](const Spec::RemoteLaunch &) -> ValPtr { throw BackendException("Spec::RemoteLaunch is a local orchestration operation"); },
      [&](const Spec::RemoteAlloc &) -> ValPtr { throw BackendException("Spec::RemoteAlloc is a local orchestration operation"); },
      [&](const Spec::RemoteFree &) -> ValPtr { throw BackendException("Spec::RemoteFree is a local orchestration operation"); },
      [&](const Spec::RemoteMemcpy &) -> ValPtr { throw BackendException("Spec::RemoteMemcpy is a local orchestration operation"); },
      [&](const Spec::RemoteSync &) -> ValPtr { throw BackendException("Spec::RemoteSync is a local orchestration operation"); },
      [&](const Spec::GpuVolatileLoad &v) -> ValPtr { return cg.mkVolatileLoad(v); },
      [&](const Spec::GpuVolatileStore &v) -> ValPtr { return cg.mkVolatileStore(v); });
}

// XXX Vulkan float math uses LLVM intrinsics (GLSL.std.450), the OpenCL.std mangled libcalls crash the Intel driver
ValPtr SPIRVVulkanTargetSpecificHandler::isNaN(CodeGen &cg, llvm::Value *from) {
  return cg.B.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {from->getType()}, {from, cg.B.getInt32(llvm::fcNan)});
}

ValPtr SPIRVVulkanTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  OclMangledMath m{cg};
  const auto unary = [&](const char *name, const AnyType &rtn, const AnyTerm &arg) { return m.unary(name, rtn, arg); };
  const auto binary = [&](const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) {
    return m.binary(name, rtn, lhs, rhs);
  };
  static const std::unordered_map<std::string_view, llvm::Intrinsic::ID> uIntr = {
      {"sin", llvm::Intrinsic::sin},     {"cos", llvm::Intrinsic::cos},   {"tan", llvm::Intrinsic::tan},
      {"asin", llvm::Intrinsic::asin},   {"acos", llvm::Intrinsic::acos}, {"atan", llvm::Intrinsic::atan},
      {"sinh", llvm::Intrinsic::sinh},   {"cosh", llvm::Intrinsic::cosh}, {"tanh", llvm::Intrinsic::tanh},
      {"sqrt", llvm::Intrinsic::sqrt},   {"exp", llvm::Intrinsic::exp},   {"exp2", llvm::Intrinsic::exp2},
      {"log", llvm::Intrinsic::log},     {"log2", llvm::Intrinsic::log2}, {"log10", llvm::Intrinsic::log10},
      {"floor", llvm::Intrinsic::floor}, {"ceil", llvm::Intrinsic::ceil}, {"round", llvm::Intrinsic::round},
      {"rint", llvm::Intrinsic::rint},   {"fabs", llvm::Intrinsic::fabs}};
  static const std::unordered_map<std::string_view, llvm::Intrinsic::ID> bIntr = {{"pow", llvm::Intrinsic::pow}};
  const auto vkUnary = [&](const char *name, const AnyType &rtn, const AnyTerm &arg) -> ValPtr {
    if (const auto id = uIntr ^ get_maybe(std::string_view(name))) return cg.intr1(*id, rtn, arg);
    return unary(name, rtn, arg);
  };
  const auto vkBinary = [&](const char *name, const AnyType &rtn, const AnyTerm &lhs, const AnyTerm &rhs) -> ValPtr {
    if (const auto id = bIntr ^ get_maybe(std::string_view(name))) return cg.intr2(*id, rtn, lhs, rhs);
    return binary(name, rtn, lhs, rhs);
  };
  return mkExternMathVal(cg, expr, vkUnary, vkBinary, [&](const AnyType &tpe, const AnyTerm &x) { return vkUnary("fabs", tpe, x); });
}
