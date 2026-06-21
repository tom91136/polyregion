#include "mlir_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "fmt/format.h"

#include "polyregion/conventions.h"

#include "utils.h"

mlir::Location polyregion::polyfc::uLoc(OpBuilder &B) { return B.getUnknownLoc(); }
mlir::Type polyregion::polyfc::i64Ty(OpBuilder &B) { return B.getI64Type(); }
mlir::Type polyregion::polyfc::i32Ty(OpBuilder &B) { return B.getI32Type(); }
mlir::Type polyregion::polyfc::i8Ty(OpBuilder &B) { return B.getI8Type(); }
mlir::LLVM::LLVMPointerType polyregion::polyfc::ptrTy(MLIRContext *C) { return LLVM::LLVMPointerType::get(C); }
mlir::LLVM::LLVMPointerType polyregion::polyfc::ptrTy(const OpBuilder &B) { return ptrTy(B.getContext()); }
mlir::Value polyregion::polyfc::nullConst(OpBuilder &B) { return LLVM::ZeroOp::create(B, uLoc(B), ptrTy(B)); }
mlir::Value polyregion::polyfc::intConst(OpBuilder &B, Type ty, const int64_t value) {
  return LLVM::ConstantOp::create(B, uLoc(B), ty, B.getIntegerAttr(ty, value));
}
mlir::Value polyregion::polyfc::boolConst(OpBuilder &B, const bool value) { return intConst(B, B.getI1Type(), value); }
mlir::Value polyregion::polyfc::idxConst(OpBuilder &B, const int64_t value) {
  return arith::ConstantIndexOp::create(B, uLoc(B), value).getResult();
}
mlir::Value polyregion::polyfc::zeroConst(OpBuilder &B, Type ty) {
  return arith::ConstantOp::create(B, uLoc(B), B.getZeroAttr(ty)).getResult();
}
mlir::Value polyregion::polyfc::strConst(OpBuilder &B, ModuleOp &m, const std::string &value, const bool nullTerminate) {
  static size_t id = 0;
  const auto saved = B.saveInsertionPoint();
  B.setInsertionPointToStart(m.getBody());
  const StringRef ref(value.c_str(), value.size() + (nullTerminate ? 1 : 0));
  auto var = LLVM::GlobalOp::create(B, uLoc(B),                                          //
                                    LLVM::LLVMArrayType::get(B.getI8Type(), ref.size()), //
                                    true, LLVM::Linkage::Private, fmt::format("str_const_{}", ++id), B.getStringAttr(ref));
  B.restoreInsertionPoint(saved);
  return LLVM::GEPOp::create(B, uLoc(B), ptrTy(B), B.getI8Type(), LLVM::AddressOfOp::create(B, uLoc(B), var),
                             ValueRange{intConst(B, i32Ty(B), 0)});
}

void polyregion::polyfc::emitMirrorBcGlobal(ModuleOp &m, const std::string &bitcode) {
  static size_t id = 0;
  OpBuilder B(m);
  B.setInsertionPointToStart(m.getBody());
  const StringRef ref(bitcode.data(), bitcode.size());
  LLVM::GlobalOp::create(B, uLoc(B), LLVM::LLVMArrayType::get(B.getI8Type(), ref.size()), true, LLVM::Linkage::External,
                         fmt::format("{}_{}", conventions::reflect::MirrorBitcodeGlobal, ++id), B.getStringAttr(ref));
}

mlir::LLVM::LLVMFuncOp polyregion::polyfc::defineFunc(ModuleOp &m, const std::string &name, const Type rtnTy,
                                                      const std::vector<Type> &argTys, LLVM::Linkage linkage,
                                                      const std::function<void(OpBuilder &, LLVM::LLVMFuncOp &)> &f) {
  if (auto existing = m.lookupSymbol<LLVM::LLVMFuncOp>(name)) return existing;
  OpBuilder B(m);
  B.setInsertionPointToStart(m.getBody());
  auto func = LLVM::LLVMFuncOp::create(B, uLoc(B), name, LLVM::LLVMFunctionType::get(rtnTy, argTys), linkage);
  if (func.empty() && f) {
    OpBuilder FB(func);
    FB.setInsertionPointToStart(func.addEntryBlock(FB));
    f(FB, func);
  }
  return func;
}
void polyregion::polyfc::defineGlobalCtor(ModuleOp &m, const std::string &name,
                                          const std::function<void(OpBuilder &, LLVM::LLVMFuncOp &)> &f) {
  const auto ctor = defineFunc(m, name, LLVM::LLVMVoidType::get(m.getContext()), {}, LLVM::Linkage::Internal, f);
  OpBuilder B(m);
  B.setInsertionPointToStart(m.getBody());
  LLVM::GlobalCtorsOp::create(B, uLoc(B), B.getArrayAttr({SymbolRefAttr::get(ctor)}), B.getArrayAttr({B.getIntegerAttr(i32Ty(B), 1)}),
                              B.getArrayAttr({LLVM::ZeroAttr::get(B.getContext())}));
}

polyregion::polyfc::DynamicAggregateMirror::DynamicAggregateMirror(MLIRContext *C, const std::string &name, const std::vector<Type> &types)
    : C(C), ty(LLVM::LLVMStructType::getNewIdentified(C, name, types)) {}

mlir::Value polyregion::polyfc::DynamicAggregateMirror::local(OpBuilder &B, const std::vector<std::vector<Value>> &fieldGroups) const {
  auto alloca = LLVM::AllocaOp::create(B, uLoc(B), ptrTy(B), intConst(B, i64Ty(B), fieldGroups.size()), B.getI64IntegerAttr(1), ty);
  fieldGroups | zip_with_index() | for_each([&](auto &fields, auto group) {
    if (ty.getBody().size() != fields.size()) {
      raise(fmt::format("Cannot initialise LLVM struct {} with mismatching ({}) field counts", show(static_cast<Type>(ty)), fields.size()));
    }
    fields | zip_with_index() | for_each([&](auto &field, auto idx) {
      LLVM::StoreOp::create(
          B, uLoc(B), field,
          LLVM::GEPOp::create(B, uLoc(B), ptrTy(B), ty, alloca, llvm::ArrayRef{intConst(B, i64Ty(B), group), intConst(B, i64Ty(B), idx)}));
    });
  });
  return alloca;
}