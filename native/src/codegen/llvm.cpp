#include "llvm.h"
#include "utils.hpp"

using namespace llvm;

llvm::Type *polyregion::codegen::LLVMCodeGen::mkTpe(const polyregion::Types_Type &tpe) {
  if (tpe.has_booltpe()) return undefined();
  if (tpe.has_bytetpe()) return Type::getInt8Ty(ctx);
  if (tpe.has_chartpe()) return Type::getInt8Ty(ctx);
  if (tpe.has_shorttpe()) return Type::getInt16Ty(ctx);
  if (tpe.has_inttpe()) return Type::getInt32Ty(ctx);
  if (tpe.has_longtpe()) return Type::getInt64Ty(ctx);
  if (tpe.has_doubletpe()) return Type::getDoubleTy(ctx);
  if (tpe.has_floattpe()) return Type::getFloatTy(ctx);
  if (tpe.has_stringtpe()) return undefined();
  if (auto arrtpe = POLY_OPT(tpe, arraytpe); arrtpe) {
    auto comp = arrtpe->tpe();
    if (comp.has_booltpe()) return undefined();
    if (comp.has_bytetpe()) return Type::getInt8Ty(ctx);
    if (comp.has_chartpe()) return Type::getInt8Ty(ctx);
    if (comp.has_shorttpe()) return Type::getInt16Ty(ctx);
    if (comp.has_inttpe()) return Type::getInt32Ty(ctx);
    if (comp.has_longtpe()) return Type::getInt64Ty(ctx);
    if (comp.has_doubletpe()) return Type::getDoubleTy(ctx);
    if (comp.has_floattpe()) return Type::getFloatTy(ctx);
    if (comp.has_stringtpe()) return undefined();
    return undefined();
  }
  if (auto reftpe = POLY_OPT(tpe, reftpe); reftpe) {
                return undefined();
  }

  return Type::getVoidTy(ctx);
}

llvm::Value *polyregion::codegen::LLVMCodeGen::mkRef(const polyregion::Refs_Ref &ref, const LUT &lut) {
  if (auto select = POLY_OPT(ref, select); select) {
    if (auto x = lut.find(select->head().name()); x != lut.end()) {
      return x->second;
    } else {
      return undefined("Unseen select: " + select->DebugString());
    }
  }

  if (auto c = POLY_OPT(ref, boolconst); c) return undefined();
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

llvm::Value *polyregion::codegen::LLVMCodeGen::mkExpr(const polyregion::Tree_Expr &expr, //
                                                      const std::string &key,            //
                                                      llvm::IRBuilder<> &builder, LUT &lut) {
  if (auto alias = POLY_OPT(expr, alias); alias) {
    return mkRef(alias->ref(), lut);
  }
  if (auto invoke = POLY_OPT(expr, invoke); invoke) {

    auto name = invoke->name();
    auto lhs = mkRef(invoke->lhs(), lut);

    if (invoke->args_size() == 1) {
      auto rhs = mkRef(invoke->args(0), lut);
      switch (hash(name)) {
      case "+"_:
        return builder.CreateFAdd(lhs, rhs, key + "_+");
      case "-"_:
        return builder.CreateFSub(lhs, rhs, key + "_-");
      case "*"_:
        return builder.CreateFMul(lhs, rhs, key + "_*");
      case "/"_:
        return builder.CreateFDiv(lhs, rhs, key + "_/");
      case "%"_:
        return builder.CreateFRem(lhs, rhs, key + "_%");
      }
    }
  }
  return undefined();
}

void polyregion::codegen::LLVMCodeGen::mkStmt(const polyregion::Tree_Stmt &stmt, IRBuilder<> &builder, LUT &lut) {
  if (auto comment = POLY_OPT(stmt, comment); comment) {
    return; // discard comments
  }
  if (auto var = POLY_OPT(stmt, var); var) {
    lut[var->key()] = mkExpr(var->rhs(), var->key(), builder, lut);
  }

  //
  if (auto effect = POLY_OPT(stmt, effect); effect) {

    if (effect->args_size() == 1) {
      auto name = effect->name();
      auto rhs = effect->args(0);
      if (auto select = POLY_OPTM(effect, lhs, select); select) {
        if (auto tpe = POLY_OPTM(select, head, tpe); tpe) {
          if (auto arr = POLY_OPTM(tpe, arraytpe, tpe); arr) {
            if (auto v = lut.find(name); v != lut.end()) {
              builder.CreateInBoundsGEP(v->second, {mkRef(rhs, lut)}, name + "_ptr");
            } else {
              throw std::logic_error("name `" + name + "` not defined previously");
            }
          }
        }
      }
    }
  }
  if (auto mut = POLY_OPT(stmt, mut); mut) {
  }
  if (auto while_ = POLY_OPT(stmt, while_); while_) {
  }
}

void polyregion::codegen::LLVMCodeGen::run(const polyregion::Tree_Function &fnTree) {

  auto paramTpes = map_vec<Named, Type *>(fnTree.args(), [&](auto &&named) { return mkTpe(named.tpe()); });
  auto fnTpe = FunctionType::get(mkTpe(fnTree.returntpe()), {paramTpes}, false);




  auto *fn = Function::Create(fnTpe, Function::ExternalLinkage, fnTree.name(), module);

  LUT lut = {};

  // add function params to the lut first as function body will need these at some point
  std::transform(                                            //
      fn->arg_begin(), fn->arg_end(), fnTree.args().begin(), //
      std::inserter(lut, lut.end()),                         //
      [](auto &arg, const auto &named) {
        arg.setName(named.name());
        return std::make_pair(named.name(), (&arg));
      });

  auto *BB = BasicBlock::Create(ctx, "Entry", fn);
  IRBuilder<> B(BB);

  for (const auto &stmt : fnTree.statements()) {
    mkStmt(stmt, B, lut);
  }
  B.CreateRetVoid();

  module.print(llvm::errs(), nullptr);
}
