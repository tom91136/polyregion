#include "llvm.h"
#include "ast.h"
#include "utils.hpp"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

using namespace llvm;
using namespace polyregion;

Type *codegen::LLVMCodeGen::mkTpe(const Types_Type &tpe) {
  if (tpe.has_booltpe()) return Type::getInt1Ty(ctx);
  if (tpe.has_bytetpe()) return Type::getInt8Ty(ctx);
  if (tpe.has_chartpe()) return Type::getInt8Ty(ctx);
  if (tpe.has_shorttpe()) return Type::getInt16Ty(ctx);
  if (tpe.has_inttpe()) return Type::getInt32Ty(ctx);
  if (tpe.has_longtpe()) return Type::getInt64Ty(ctx);
  if (tpe.has_doubletpe()) return Type::getDoubleTy(ctx);
  if (tpe.has_floattpe()) return Type::getFloatTy(ctx);
  if (tpe.has_stringtpe()) return undefined();
  if (auto arrtpe = POLY_OPT(tpe, arraytpe); arrtpe) {
    return mkTpe(arrtpe->tpe())->getPointerTo();
  }
  if (auto reftpe = POLY_OPT(tpe, reftpe); reftpe) {
    return undefined();
  }
  return Type::getVoidTy(ctx);
}

Value *codegen::LLVMCodeGen::mkSelect(const Refs_Select &select, IRBuilder<> &B, const LUT &lut) {
  if (auto x = lut.find(ast::qualified(select)); x != lut.end()) {
    return ast::selectLast(select).tpe().has_arraytpe()                      //
               ? x->second                                                   //
               : B.CreateLoad(x->second, ast::qualified(select) + "_value"); //
  } else {
    return undefined("Unseen select: " + select.DebugString());
  }
}

Value *codegen::LLVMCodeGen::mkRef(const Refs_Ref &ref, IRBuilder<> &builder, const LUT &lut) {
  if (auto c = POLY_OPT(ref, select); c) return mkSelect(*c, builder, lut);
  if (auto c = POLY_OPT(ref, boolconst); c) return ConstantInt::get(Type::getInt1Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, byteconst); c) return ConstantInt::get(Type::getInt8Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, charconst); c) return ConstantInt::get(Type::getInt8Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, shortconst); c) return ConstantInt::get(Type::getInt16Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, intconst); c) return ConstantInt::get(Type::getInt32Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, longconst); c) return ConstantInt::get(Type::getInt64Ty(ctx), c->value());
  if (auto c = POLY_OPT(ref, doubleconst); c) return ConstantFP::get(Type::getDoubleTy(ctx), c->value());
  if (auto c = POLY_OPT(ref, floatconst); c) return ConstantFP::get(Type::getFloatTy(ctx), c->value());
  if (auto c = POLY_OPT(ref, stringconst); c) return undefined();
  return undefined("Unimplemented ref:" + ref.DebugString());
}

Value *codegen::LLVMCodeGen::mkExpr(const Tree_Expr &expr, const std::string &key, IRBuilder<> &B, LUT &lut) {

  std::optional<std::array<int, 3>> a = {};

  if (auto alias = POLY_OPT(expr, alias); alias) {
    return mkRef(alias->ref(), B, lut);
  }
  if (auto invoke = POLY_OPT(expr, invoke); invoke) {

    auto name = invoke->name();
    auto lhs = mkRef(invoke->lhs(), B, lut);

    if (invoke->args_size() == 1) {
      auto rhs = mkRef(invoke->args(0), B, lut);
      auto lkind = ast::numKind(invoke->lhs());
      auto rkind = ast::numKind(invoke->args(0));
      if (lkind.has_value() && rkind.has_value()) {
        switch (*lkind) {
        case ast::NumKind::Integral:
          switch (hash(name)) {
          case "+"_:
            return B.CreateAdd(lhs, rhs, key + "_+");
          case "-"_:
            return B.CreateSub(lhs, rhs, key + "_-");
          case "*"_:
            return B.CreateMul(lhs, rhs, key + "_*");
          case "/"_:
            return B.CreateSDiv(lhs, rhs, key + "_/");
          case "%"_:
            return B.CreateSRem(lhs, rhs, key + "_%");
          case "<"_:
            return B.CreateICmpSLT(lhs, rhs, key + "_<");
          case ">"_:
            return B.CreateICmpSGT(lhs, rhs, key + "_>");
          default:
            undefined("Unimplemented imath op:" + name);
          }
        case ast::NumKind::Fractional:
          switch (hash(name)) {
          case "+"_:
            return B.CreateFAdd(lhs, rhs, key + "_+");
          case "-"_:
            return B.CreateFSub(lhs, rhs, key + "_-");
          case "*"_:
            return B.CreateFMul(lhs, rhs, key + "_*");
          case "/"_:
            return B.CreateFDiv(lhs, rhs, key + "_/");
          case "%"_:
            return B.CreateFRem(lhs, rhs, key + "_%");
          default:
            undefined("Unimplemented fmath op:" + name);
          }
        }
      }
    }
    return undefined("Unimplemented invoke:`" + invoke->name() + "`");
  }
  if (auto index = POLY_OPT(expr, index); index) {
    auto ptr = B.CreateInBoundsGEP(mkSelect(index->lhs(), B, lut), {mkRef(index->idx(), B, lut)}, key + "_ptr");
    return B.CreateLoad(ptr, key + "_value");
  }
  return undefined("Unimplemented expr: " + expr.DebugString());
}

void codegen::LLVMCodeGen::mkStmt(const Tree_Stmt &stmt, IRBuilder<> &B, Function *fn, LUT &lut) {
  if (auto comment = POLY_OPT(stmt, comment); comment) {
    return; // discard comments
  }

  if (auto var = POLY_OPT(stmt, var); var) {
    if (var->name().tpe().has_arraytpe()) {
      lut[var->name().symbol()] = mkExpr(var->rhs(), var->name().symbol(), B, lut);
    } else {
      auto stack = B.CreateAlloca(mkTpe(var->name().tpe()), nullptr, var->name().symbol() + "_stack_ptr");
      auto val = mkExpr(var->rhs(), var->name().symbol() + "_var_rhs", B, lut);
      B.CreateStore(val, stack);
      lut[var->name().symbol()] = stack;
    }
  }

  if (auto mut = POLY_OPT(stmt, mut); mut) {
    auto expr = mkExpr(mut->expr(), ast::qualified(mut->name()) + "_mut", B, lut);
    auto select = lut[ast::qualified(mut->name())]; // XXX do NOT allocate (mkSelect) here, we're mutating!
    B.CreateStore(expr, select);
  }

  if (auto update = POLY_OPT(stmt, update); update) {
    auto select = update->lhs();
    auto ptr =
        B.CreateInBoundsGEP(mkSelect(select, B, lut), {mkRef(update->idx(), B, lut)}, ast::qualified(select) + "_ptr");
    B.CreateStore(mkRef(update->value(), B, lut), ptr);
  }

  if (auto effect = POLY_OPT(stmt, effect); effect) {

    if (effect->args_size() == 1) {
      auto name = effect->name();
      auto rhs = effect->args(0);
      undefined("effect not implemented");
      //      if (auto select = POLY_OPTM(effect, lhs, select); select) {
      //        if (auto tpe = POLY_OPTM(select, head, tpe); tpe) {
      //          if (auto arr = POLY_OPTM(tpe, arraytpe, tpe); arr) {
      //            if (auto v = lut.find(name); v != lut.end()) {
      //              B.CreateInBoundsGEP(v->second, {mkRef(rhs, lut)}, name + "_ptr");
      //            } else {
      //              throw std::logic_error("name `" + name + "` not defined previously");
      //            }
      //          }
      //        }
      //      }
    }
  }

  if (auto while_ = POLY_OPT(stmt, while_); while_) {
    auto loopTest = BasicBlock::Create(ctx, "loop_test", fn);
    auto loopBody = BasicBlock::Create(ctx, "loop_body", fn);
    auto loopExit = BasicBlock::Create(ctx, "loop_exit", fn);
    B.CreateBr(loopTest);
    {
      B.SetInsertPoint(loopTest);
      auto continue_ = mkExpr(while_->cond(), "loop", B, lut);
      B.CreateCondBr(continue_, loopBody, loopExit);
    }
    {
      B.SetInsertPoint(loopBody);
      for (auto &body : while_->body())
        mkStmt(body, B, fn, lut);
      B.CreateBr(loopTest);
    }
    B.SetInsertPoint(loopExit);
  }

  if (auto break_ = POLY_OPT(stmt, break_); break_) {
    undefined("break");
  }

  if (auto cond = POLY_OPT(stmt, cond); cond) {
    auto condTrue = BasicBlock::Create(ctx, "cond_true", fn);
    auto condFalse = BasicBlock::Create(ctx, "cond_false", fn);
    auto condExit = BasicBlock::Create(ctx, "cond_exit", fn);
    B.CreateCondBr(mkExpr(cond->cond(), "cond", B, lut), condTrue, condFalse);
    {
      B.SetInsertPoint(condTrue);
      for (auto &body : cond->truebr())
        mkStmt(body, B, fn, lut);
      B.CreateBr(condExit);
    }
    {
      B.SetInsertPoint(condFalse);
      for (auto &body : cond->falsebr())
        mkStmt(body, B, fn, lut);
      B.CreateBr(condExit);
    }
    B.SetInsertPoint(condExit);
  }
}

void codegen::LLVMCodeGen::run(const Tree_Function &fnTree) {

  auto paramTpes = map_vec<Named, Type *>(fnTree.args(), [&](auto &&named) { return mkTpe(named.tpe()); });
  auto fnTpe = FunctionType::get(mkTpe(fnTree.returntpe()), {paramTpes}, false);

  auto *fn = Function::Create(fnTpe, Function::ExternalLinkage, fnTree.name(), module);

  LUT lut = {};

  auto *BB = BasicBlock::Create(ctx, "entry", fn);
  IRBuilder<> B(BB);

  // add function params to the lut first as function body will need these at some point
  std::transform(                                            //
      fn->arg_begin(), fn->arg_end(), fnTree.args().begin(), //
      std::inserter(lut, lut.end()),                         //
      [&](auto &arg, const auto &named) -> std::pair<std::string, Value *> {
        arg.setName(named.symbol());
        if (named.tpe().has_arraytpe()) {
          return {named.symbol(), &arg};
        } else {
          auto stack = B.CreateAlloca(mkTpe(named.tpe()), nullptr, named.symbol() + "_stack_ptr");
          B.CreateStore(&arg, stack);
          return {named.symbol(), stack};
        }
      });

  for (auto &stmt : fnTree.statements()) {
    std::cout << "[LLVM]" << ast::repr(stmt) << std::endl;
    mkStmt(stmt, B, fn, lut);
  }
  B.CreateRetVoid();
  module.print(llvm::errs(), nullptr);
  llvm::verifyModule(module, &llvm::errs());
  std::cout << "Pre-opt verify OK!" << std::endl;

  llvm::PassManagerBuilder builder;
  //  builder.OptLevel = 3;
  llvm::legacy::PassManager m;
  builder.populateModulePassManager(m);
  m.add(llvm::createInstructionCombiningPass());
  m.run(module);

  llvm::verifyModule(module, &llvm::errs());
  module.print(llvm::errs(), nullptr);
}
