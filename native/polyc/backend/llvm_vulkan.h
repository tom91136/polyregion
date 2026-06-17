#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "llvm/IR/Value.h"

#include "ast.h"
#include "llvm.h"

namespace polyregion::backend::details {

struct VulkanLowering : PointerModel {
  CodeGen &cg;
  explicit VulkanLowering(CodeGen &cg) : cg(cg) {}

  Map<std::string, std::pair<AnyType, llvm::Value *>> bufferHandles{};
  llvm::Value *scalarBlock = nullptr;
  Map<std::string, std::pair<unsigned, AnyType>> scalarSlots{};
  Map<std::string, std::tuple<AnyType, llvm::Type *, llvm::Value *>> localBases{};

  void reset() override;

  [[nodiscard]] llvm::Value *i64Zero() const;

  [[nodiscard]] llvm::Value *bufferHandle(llvm::Type *elemTy, unsigned binding, const std::string &name);
  [[nodiscard]] llvm::Value *uniformBlockHandle(llvm::Type *blockTy, unsigned binding, const std::string &name);
  [[nodiscard]] llvm::Value *bufferElementPtr(const AnyType &ptrTpe, llvm::Value *handle, llvm::Value *index);

  [[nodiscard]] llvm::Value *handleOf(const Term::Select &select);
  [[nodiscard]] llvm::Value *elementPtr(const Term::Select &lhs, llvm::Value *idx);
  [[nodiscard]] llvm::Value *scalarValueOf(const Term::Select &select);
  [[nodiscard]] llvm::Value *localElementPtr(const Term::Select &select, llvm::Value *index);
  [[nodiscard]] std::optional<std::tuple<llvm::Value *, llvm::Type *, std::vector<llvm::Value *>>> localOrigin(const Term::Select &select);
  [[nodiscard]] llvm::Value *localChainPtr(llvm::Value *base, llvm::Type *arrTy, const std::vector<llvm::Value *> &indices);
  [[nodiscard]] std::vector<llvm::Value *> extendIndices(const std::vector<llvm::Value *> &base, const AnyType &lhsTpe, llvm::Value *idx);

  [[nodiscard]] static std::pair<llvm::ArrayType *, std::vector<uint64_t>> flattenArray(llvm::Type *arrTy);

  void structFieldCopy(llvm::Value *dst, llvm::Value *src, llvm::Type *rootTy, const AnyType &tpe, std::vector<llvm::Value *> idxs);

  [[nodiscard]] std::optional<ValPtr> termSelectVal(CodeGen &, const Term::Select &select) override;
  [[nodiscard]] std::optional<ValPtr> mkIndex(const Term::Select &lhs, const Term::Any &idx);
  [[nodiscard]] std::optional<ValPtr> mkRefTo(const Term::Select &lhs, const Opt<Term::Any> &idx);
  [[nodiscard]] bool mkUpdate(const Term::Select &lhs, const Term::Any &idx, const Term::Any &value);

  [[nodiscard]] llvm::Type *localAllocType(CodeGen &, const Type::Any &nameTpe, llvm::Type *tpe) override;

  bool bindEntryArgs(llvm::Function &llvmFn, const std::vector<Arg> &argsNoUnit, const Function &fn) override;
};

} // namespace polyregion::backend::details
