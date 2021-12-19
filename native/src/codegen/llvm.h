#pragma once

#include "ast.h"
#include "codegen.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace polyregion::codegen {

class LLVMCodeGen : public polyregion::codegen::CodeGen {

private:
  using LUT = std::unordered_map<std::string, llvm::Value *>;
  llvm::LLVMContext ctx;
  llvm::Module module;

  llvm::Type *mkTpe(const Types_Type &tpe);
  llvm::Value *mkRef(const Refs_Ref &ref, const LUT &lut);

  llvm::Value *mkExpr(const Tree_Expr &expr, const std::string &key, llvm::IRBuilder<> &builder, LUT &lut);
  void mkStmt(const Tree_Stmt &stmt, llvm::IRBuilder<> &builder, LUT &lut);

public:
  explicit LLVMCodeGen(const std::string &moduleName) : ctx(), module(moduleName, ctx) {}

  void run(const Tree_Function &arg) override;
};

} // namespace polyregion::codegen