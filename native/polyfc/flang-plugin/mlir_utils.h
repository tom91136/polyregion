#pragma once

#include <string>

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

namespace polyregion::polyfc {

using namespace mlir;
using namespace aspartame;

std::optional<func::FuncOp> resolveDefiningFunction(Operation *op);
std::vector<Value> findCapturesInOrder(Block *block);
Location uLoc(OpBuilder &B);
Type i64Ty(OpBuilder &B);
Type i32Ty(OpBuilder &B);
Type i8Ty(OpBuilder &B);
LLVM::LLVMPointerType ptrTy(MLIRContext *C);
LLVM::LLVMPointerType ptrTy(const OpBuilder &B);
Value nullConst(OpBuilder &B);
Value intConst(OpBuilder &B, Type ty, int64_t value);
Value boolConst(OpBuilder &B, bool value);
Value strConst(OpBuilder &B, ModuleOp &m, const std::string &value, bool nullTerminate = true);
LLVM::LLVMFuncOp defineFunc(ModuleOp &m, const std::string &name, Type rtnTy, const std::vector<Type> &argTys,
                            LLVM::Linkage linkage = LLVM::Linkage::External,
                            const std::function<void(OpBuilder &, LLVM::LLVMFuncOp &)> &f = {});

void defineGlobalCtor(ModuleOp &m, const std::string &name, const std::function<void(OpBuilder &, LLVM::LLVMFuncOp &)> &f);

struct DynamicAggregateMirror {

  MLIRContext *C;
  LLVM::LLVMStructType ty;

  explicit DynamicAggregateMirror(MLIRContext *C, const std::string &name, const std::vector<Type> &types);

  Value local(OpBuilder &B, const std::vector<std::vector<Value>> &fieldGroups) const;
};

template <size_t N> struct AggregateMirror {

  using Init = std::array<Value, N>;
  using Group = std::vector<Init>;

  template <typename T, size_t Idx> struct Field {
    static size_t index() { return Idx; }
    T type;
    explicit Field(MLIRContext *C)
        : type([&] {
            if constexpr (std::is_same_v<T, LLVM::LLVMPointerType>) {
              return LLVM::LLVMPointerType::get(C);
            } else {
              static_assert(sizeof(T) == 0, "Unsupported field type");
            }
          }()) {}

    Field(MLIRContext *C, const size_t width)
        : type([&] {
            if constexpr (std::is_same_v<T, IntegerType>) {
              return IntegerType::get(C, width);
            } else {
              static_assert(sizeof(T) == 0, "Unsupported field type");
            }
          }()) {}
    Type widen() const { return type; }
  };

  struct Global {
    static Value gepUnsafe(OpBuilder &B0, LLVM::GlobalOp &global, const size_t index, const std::optional<size_t> field = {}) {
      const auto baseIdx = intConst(B0, i64Ty(B0), 0);
      const auto indexIdx = intConst(B0, i64Ty(B0), index);
      return B0.create<LLVM::GEPOp>(uLoc(B0), LLVM::LLVMPointerType::get(B0.getContext()), global.getType(),
                                    B0.create<LLVM::AddressOfOp>(uLoc(B0), global),
                                    field ? ValueRange{baseIdx, indexIdx, intConst(B0, i64Ty(B0), *field)} : ValueRange{baseIdx, indexIdx});
    }
    LLVM::GlobalOp global;
    Value gep(OpBuilder &B0, const size_t index = 0) { return gepUnsafe(B0, global, index); }
    template <typename T, size_t I> //
    Value gep(OpBuilder &B0, const size_t index, const Field<T, I> &field) {
      return gepUnsafe(B0, global, index, field.index());
    }
  };

  MLIRContext *C;

  explicit AggregateMirror(MLIRContext *C) : C(C) {}

  virtual const char *typeName() const = 0;
  virtual std::array<Type, N> types() const = 0;
  virtual ~AggregateMirror() = default;

  mutable std::optional<LLVM::LLVMStructType> cachedStructTy{};

  LLVM::LLVMStructType structTy() const {
    if (!cachedStructTy) cachedStructTy = LLVM::LLVMStructType::getNewIdentified(C, typeName(), types());
    return *cachedStructTy;
  }

  Value local(OpBuilder &B, const std::vector<std::array<Value, N>> &fieldGroups) const {
    const auto ty = structTy();
    auto alloca = B.create<LLVM::AllocaOp>(uLoc(B), ptrTy(B), intConst(B, i64Ty(B), fieldGroups.size()), B.getI64IntegerAttr(1), ty);
    for (auto &fields : fieldGroups) {
      for (size_t i = 0; i < N; ++i) {
        B.create<LLVM::StoreOp>(
            uLoc(B), fields[i],
            B.create<LLVM::GEPOp>(uLoc(B), ptrTy(B), ty, alloca, llvm::ArrayRef{intConst(B, i64Ty(B), 0), intConst(B, i64Ty(B), i)}));
      }
    }
    return alloca;
  }

  Global global(ModuleOp &m, const std::function<std::vector<std::array<Value, N>>(OpBuilder &)> &f) const {
    static size_t id = 0;
    const auto ty = structTy();
    OpBuilder B(m);
    B.setInsertionPointToStart(m.getBody());
    // use placeholder size before we know the actual size
    auto global = B.create<LLVM::GlobalOp>(uLoc(B), LLVM::LLVMArrayType::get(ty, 1), false, LLVM::Linkage::Private,
                                           fmt::format("{}_arr_{}", typeName(), ++id), Attribute{});
    {
      OpBuilder::InsertionGuard initGuard(B);
      B.createBlock(&global.getRegion(), global.getRegion().end(), {}, {});
      B.setInsertionPointToEnd(&global.getRegion().back());
      const auto groups = f(B);
      const auto arrayTy = LLVM::LLVMArrayType::get(ty, groups.size());
      B.create<LLVM::ReturnOp>(
          uLoc(B), groups                          //
                       | zip_with_index<int64_t>() //
                       | fold_left<Value>(B.create<LLVM::UndefOp>(uLoc(B), arrayTy), [&](auto acc, auto &group) {
                           Value structInit = group.first                 //
                                              | zip_with_index<int64_t>() //
                                              | fold_left<Value>(B.create<LLVM::UndefOp>(uLoc(B), ty), [&](auto acc, auto &value) {
                                                  return B.create<LLVM::InsertValueOp>(uLoc(B), ty, acc, value.first,
                                                                                       llvm::ArrayRef<int64_t>{value.second});
                                                });
                           return B.create<LLVM::InsertValueOp>(uLoc(B), arrayTy, acc, structInit, llvm::ArrayRef<int64_t>{group.second});
                         }));
      global.setGlobalType(arrayTy);
    }
    return Global{std::move(global)};
  }
};
} // namespace polyregion::polyfc
