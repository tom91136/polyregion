#pragma once

#include "ast.h"
#include "codegen.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"


namespace polyregion::codegen {

class LLVMCodeGen : public CodeGen {

private:
  using LUT = std::unordered_map<std::string, llvm::Value *>;
  llvm::LLVMContext ctx;
  llvm::Module module;

  llvm::Type *mkTpe(const Types_Type &tpe);
  llvm::Value *mkSelect(const Refs_Select &select, llvm::IRBuilder<> &B, const LUT &lut);
  llvm::Value *mkRef(const Refs_Ref &ref, llvm::IRBuilder<> &builder, const LUT &lut);

  llvm::Value *mkExpr(const Tree_Expr &expr, const std::string &key, llvm::IRBuilder<> &B, LUT &lut);
  void mkStmt(const Tree_Stmt &stmt, llvm::IRBuilder<> &B, llvm::Function *fn, LUT &lut);

public:
  explicit LLVMCodeGen(const std::string &moduleName) : ctx(), module(moduleName, ctx) {}

  void run(const Tree_Function &arg) override;
};

} // namespace polyregion::codegen