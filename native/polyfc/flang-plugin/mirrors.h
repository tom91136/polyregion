#pragma once

#include "mlir_utils.h"
#include "polyregion/types.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace polyregion::polyfc {

using namespace mlir;

struct CharStarMirror final : AggregateMirror<1> {
  Field<LLVM::LLVMPointerType, 0> ptr;
  explicit CharStarMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 1> types() const override;
};

struct AggregateMemberMirror final : AggregateMirror<7> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> offsetInBytes;
  Field<IntegerType, 2> sizeInBytes;
  Field<IntegerType, 3> ptrIndirection;
  Field<IntegerType, 4> componentSize;
  Field<LLVM::LLVMPointerType, 5> type;
  Field<LLVM::LLVMPointerType, 6> resolvePtrSizeInBytes;

  explicit AggregateMemberMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 7> types() const override;
};

struct TypeLayoutMirror final : AggregateMirror<6> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> sizeInBytes;
  Field<IntegerType, 2> alignmentInBytes;
  Field<IntegerType, 3> attrs;
  Field<IntegerType, 4> memberCount;
  Field<LLVM::LLVMPointerType, 5> members;
  explicit TypeLayoutMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 6> types() const override;
};

struct KernelObjectMirror final : AggregateMirror<6> {
  Field<IntegerType, 0> kind;
  Field<IntegerType, 1> structCount;
  Field<IntegerType, 2> featureCount;
  Field<LLVM::LLVMPointerType, 3> features;
  Field<IntegerType, 4> imageLength;
  Field<LLVM::LLVMPointerType, 5> image;
  explicit KernelObjectMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 6> types() const override;
};

struct KernelBundleMirror final : AggregateMirror<7> {
  Field<LLVM::LLVMPointerType, 0> moduleName;
  Field<IntegerType, 1> objectCount;
  Field<LLVM::LLVMPointerType, 2> objects;
  Field<IntegerType, 3> structCount;
  Field<LLVM::LLVMPointerType, 4> structs;
  Field<IntegerType, 5> interfaceLayoutIdx;
  Field<LLVM::LLVMPointerType, 6> metadata;
  explicit KernelBundleMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 7> types() const override;
};

struct FReductionMirror final : AggregateMirror<3> {
  Field<IntegerType, 0> kind;
  Field<IntegerType, 1> type;
  Field<LLVM::LLVMPointerType, 2> dest;
  explicit FReductionMirror(ModuleOp &M);
  const char *typeName() const override;
  std::array<Type, 3> types() const override;
};

class PolyDCOMirror {
  OpBuilder TLB;
  LLVM::LLVMPointerType ptrTy;
  LLVM::LLVMVoidType voidTy;
  LLVM::LLVMFuncOp recordFn, releaseFn, debugLayoutFn, isPlatformKindFn, dispatchFn;

  Value valueOf(OpBuilder &B, runtime::PlatformKind kind);

  Value convertIfNeeded(OpBuilder &B, Value value, Type required);

public:
  explicit PolyDCOMirror(ModuleOp &m) //
      ;

  void record(OpBuilder &B, Value ptr, Value sizeInBytes);

  void release(OpBuilder &B, Value ptr);

  Value isPlatformKind(OpBuilder &B, runtime::PlatformKind kind);

  Value dispatch(OpBuilder &B, Value lowerBound, Value upperBound, Value step, //
                 runtime::PlatformKind kind,                                   //
                 Value bundle,                                                 //
                 Value reductionsCount, Value reductions,                      //
                 Value captures);
};

} // namespace polyregion::polyfc
