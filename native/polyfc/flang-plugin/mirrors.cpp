#include "mirrors.h"
#include "ftypes.h"

using namespace polyregion::polyfc;

CharStarMirror::CharStarMirror(ModuleOp &M) : AggregateMirror(M), ptr(M) {}
const char *CharStarMirror::typeName() const { return "CharStar"; }
std::array<Type, 1> CharStarMirror::types() const { return {ptr.widen()}; }
AggregateMemberMirror::AggregateMemberMirror(ModuleOp &M)
    : AggregateMirror(M), //
      name(M), offsetInBytes(M, 64), sizeInBytes(M, 64), ptrIndirection(M, 64), componentSize(M, 64), type(M), resolvePtrSizeInBytes(M) {
  validateMirrorSize<runtime::AggregateMember>();
}
const char *AggregateMemberMirror::typeName() const { return "AggregateMember"; }
std::array<Type, 7> AggregateMemberMirror::types() const {
  return {name.widen(),           //
          offsetInBytes.widen(),  //
          sizeInBytes.widen(),    //
          ptrIndirection.widen(), //
          componentSize.widen(),  //
          type.widen(),           //
          resolvePtrSizeInBytes.widen()};
}
TypeLayoutMirror::TypeLayoutMirror(ModuleOp &M)
    : AggregateMirror(M), //
      name(M), sizeInBytes(M, 64), alignmentInBytes(M, 64), memberCount(M, 64), members(M) {
  validateMirrorSize<runtime::TypeLayout>();
}
const char *TypeLayoutMirror::typeName() const { return "TypeLayout"; }
std::array<Type, 5> TypeLayoutMirror::types() const {
  return {name.widen(),             //
          sizeInBytes.widen(),      //
          alignmentInBytes.widen(), //
          memberCount.widen(),      //
          members.widen()};
}
KernelObjectMirror::KernelObjectMirror(ModuleOp &M)
    : AggregateMirror(M), kind(M, 8), structCount(M, 8), featureCount(M, 64), features(M), imageLength(M, 64), image(M) {
  validateMirrorSize<runtime::KernelObject>();
}
const char *KernelObjectMirror::typeName() const { return "KernelObject"; }
std::array<Type, 6> KernelObjectMirror::types() const {
  return {kind.widen(),         //
          structCount.widen(),  //
          featureCount.widen(), //
          features.widen(),     //
          imageLength.widen(),  //
          image.widen()};
}
KernelBundleMirror::KernelBundleMirror(ModuleOp &M)
    : AggregateMirror(M), //
      moduleName(M), objectCount(M, 64), objects(M), structCount(M, 64), structs(M), interfaceLayoutIdx(M, 64), metadata(M) {
  validateMirrorSize<runtime::KernelBundle>();
}
const char *KernelBundleMirror::typeName() const { return "KernelBundle"; }
std::array<Type, 7> KernelBundleMirror::types() const {
  return {moduleName.widen(),         //
          objectCount.widen(),        //
          objects.widen(),            //
          structCount.widen(),        //
          structs.widen(),            //
          interfaceLayoutIdx.widen(), //
          metadata.widen()};
}
FReductionMirror::FReductionMirror(ModuleOp &M) : AggregateMirror(M), kind(M, 8), type(M, 8), dest(M) {
  validateMirrorSize<polydco::FReduction>();
}
const char *FReductionMirror::typeName() const { return "FReduction"; }
std::array<Type, 3> FReductionMirror::types() const { return {kind.widen(), type.widen(), dest.widen()}; }
Value PolyDCOMirror::valueOf(OpBuilder &B, const runtime::PlatformKind kind) { return intConst(B, TLB.getI8Type(), value_of(kind)); }
Value PolyDCOMirror::convertIfNeeded(OpBuilder &B, Value value, Type required) {
  return value.getType() != required ? B.create<fir::ConvertOp>(TLB.getUnknownLoc(), required, value) : value;
}
PolyDCOMirror::PolyDCOMirror(ModuleOp &m)
    : TLB(m), ptrTy(LLVM::LLVMPointerType::get(TLB.getContext())), voidTy(LLVM::LLVMVoidType::get(TLB.getContext())), //
      recordFn(defineFunc(m, "polydco_record", voidTy, {ptrTy, TLB.getI64Type()})),
      releaseFn(defineFunc(m, "polydco_release", voidTy, {ptrTy})),
      debugLayoutFn(defineFunc(m, "polydco_debug_typelayout", voidTy, {ptrTy})),
      isPlatformKindFn(defineFunc(m, "polydco_is_platformkind", TLB.getI1Type(), {TLB.getI8Type()})),
      dispatchFn(defineFunc(m, "polydco_dispatch", TLB.getI1Type(),
                            {/* lowerBound     */ TLB.getI64Type(),
                             /* upperBound     */ TLB.getI64Type(),
                             /* step           */ TLB.getI64Type(),
                             /* platformKind   */ TLB.getI8Type(),
                             /* bundle         */ ptrTy,
                             /* reductionCount */ TLB.getI64Type(),
                             /* reductions     */ ptrTy,
                             /* captures       */ ptrTy})) {}
void PolyDCOMirror::record(OpBuilder &B, const Value ptr, const Value sizeInBytes) {
  B.create<LLVM::CallOp>(B.getUnknownLoc(), recordFn, ValueRange{ptr, convertIfNeeded(B, sizeInBytes, B.getI64Type())});
}
void PolyDCOMirror::release(OpBuilder &B, const Value ptr) { B.create<LLVM::CallOp>(B.getUnknownLoc(), releaseFn, ValueRange{ptr}); }
Value PolyDCOMirror::isPlatformKind(OpBuilder &B, runtime::PlatformKind kind) {
  return B.create<LLVM::CallOp>(B.getUnknownLoc(), isPlatformKindFn, ValueRange{valueOf(B, kind)}).getResult();
}
Value PolyDCOMirror::dispatch(OpBuilder &B, const Value lowerBound, const Value upperBound, const Value step,
                              const runtime::PlatformKind kind,                    //
                              const Value bundle,                                  //
                              const Value reductionsCount, const Value reductions, //
                              const Value captures) {
  return B
      .create<LLVM::CallOp>(B.getUnknownLoc(), dispatchFn,
                            ValueRange{convertIfNeeded(B, lowerBound, B.getI64Type()), //
                                       convertIfNeeded(B, upperBound, B.getI64Type()), //
                                       convertIfNeeded(B, step, B.getI64Type()),       //
                                       valueOf(B, kind),                               //
                                       bundle,                                         //
                                       reductionsCount,                                //
                                       reductions,                                     //
                                       captures})
      .getResult();
}