#include "mlir_utils.h"
#include "utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

std::optional<mlir::func::FuncOp> polyregion::polyfc::resolveDefiningFunction(Operation *op) {
  while (op) {
    if (const auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) return funcOp;
    op = op->getParentOp();
  }
  return {};
}

std::vector<mlir::Value> polyregion::polyfc::findCapturesInOrder(Block *block) {
  std::vector<Value> captures;
  const auto args = block->getArguments();
  for (Operation &op : *block) {
    for (Value arg : op.getOperands()) {
      if (const Block *origin = arg.getParentBlock()) {
        if (origin != block                   //
            && !llvm::is_contained(args, arg) //
            && !llvm::isa<arith::ConstantOp>(arg.getDefiningOp())) {
          captures.emplace_back(arg);
        }
      }
    }
  }
  return captures ^ distinct_by([](const Value &op) { return op.getAsOpaquePointer(); });
}
mlir::Location polyregion::polyfc::uLoc(OpBuilder &B) { return B.getUnknownLoc(); }
mlir::Type polyregion::polyfc::i64Ty(OpBuilder &B) { return B.getI64Type(); }
mlir::Type polyregion::polyfc::i32Ty(OpBuilder &B) { return B.getI32Type(); }
mlir::Type polyregion::polyfc::i8Ty(OpBuilder &B) { return B.getI8Type(); }
mlir::LLVM::LLVMPointerType polyregion::polyfc::ptrTy(MLIRContext *C) { return LLVM::LLVMPointerType::get(C); }
mlir::LLVM::LLVMPointerType polyregion::polyfc::ptrTy(const OpBuilder &B) { return ptrTy(B.getContext()); }
mlir::Value polyregion::polyfc::nullConst(OpBuilder &B) { return B.create<LLVM::ZeroOp>(uLoc(B), ptrTy(B)); }
mlir::Value polyregion::polyfc::intConst(OpBuilder &B, Type ty, const int64_t value) {
  return B.create<arith::ConstantOp>(uLoc(B), ty, B.getIntegerAttr(ty, value));
}
mlir::Value polyregion::polyfc::boolConst(OpBuilder &B, const bool value) { return intConst(B, B.getI1Type(), value); }
mlir::Value polyregion::polyfc::strConst(OpBuilder &B, ModuleOp &m, const std::string &value, const bool nullTerminate) {
  static size_t id = 0;
  const auto saved = B.saveInsertionPoint();
  B.setInsertionPointToStart(m.getBody());
  const StringRef ref(value.c_str(), value.size() + (nullTerminate ? 1 : 0));
  auto var = B.create<LLVM::GlobalOp>(uLoc(B),                                             //
                                      LLVM::LLVMArrayType::get(B.getI8Type(), ref.size()), //
                                      true, LLVM::Linkage::Private, fmt::format("str_const_{}", ++id), B.getStringAttr(ref));
  B.restoreInsertionPoint(saved);
  return B.create<LLVM::GEPOp>(uLoc(B), ptrTy(B), B.getI8Type(), B.create<LLVM::AddressOfOp>(uLoc(B), var),
                               ValueRange{intConst(B, i32Ty(B), 0)});
}

mlir::LLVM::LLVMFuncOp polyregion::polyfc::defineFunc(ModuleOp &m, const std::string &name, const Type rtnTy,
                                                      const std::vector<Type> &argTys, LLVM::Linkage linkage,
                                                      const std::function<void(OpBuilder &, LLVM::LLVMFuncOp &)> &f) {
  OpBuilder B(m);
  B.setInsertionPointToStart(m.getBody());
  auto func = B.create<LLVM::LLVMFuncOp>(uLoc(B), name, LLVM::LLVMFunctionType::get(rtnTy, argTys), linkage);
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
  B.create<LLVM::GlobalCtorsOp>(uLoc(B), B.getArrayAttr({SymbolRefAttr::get(ctor)}), B.getArrayAttr({B.getIntegerAttr(i32Ty(B), 1)}));
}

polyregion::polyfc::DynamicAggregateMirror::DynamicAggregateMirror(MLIRContext *C, const std::string &name, const std::vector<Type> &types)
    : C(C), ty(LLVM::LLVMStructType::getNewIdentified(C, name, types)) {}

mlir::Value polyregion::polyfc::DynamicAggregateMirror::local(OpBuilder &B, const std::vector<std::vector<Value>> &fieldGroups) const {
  auto alloca = B.create<LLVM::AllocaOp>(uLoc(B), ptrTy(B), intConst(B, i64Ty(B), fieldGroups.size()), B.getI64IntegerAttr(1), ty);
  for (auto &fields : fieldGroups) {
    if (ty.getBody().size() != fields.size()) {
      raise(fmt::format("Cannot initialise LLVM struct {} with mismatching ({}) field counts", show(static_cast<Type>(ty)), fields.size()));
    }
    for (size_t i = 0; i < ty.getBody().size(); ++i) {
      auto fieldPtr =
          B.create<LLVM::GEPOp>(uLoc(B), ptrTy(B), ty, alloca, llvm::ArrayRef{intConst(B, i64Ty(B), 0), intConst(B, i64Ty(B), i)});
      B.create<LLVM::StoreOp>(uLoc(B), fields[i], fieldPtr.getResult());
    }
  }
  return alloca;
}