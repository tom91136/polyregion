#include <iostream>

#include "ast.h"
#include "llvm.h"
#include "llvmc.h"
#include "utils.hpp"
#include "variants.hpp"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace polyregion::backend;

static llvm::ExitOnError ExitOnErr;

template <typename T> static std::string llvm_tostring(const T *t) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  t->print(rso);
  return rso.str();
}

std::pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable>
LLVMAstTransformer::mkStruct(const StructDef &def) {
  std::vector<llvm::Type *> types(def.members.size());
  std::transform(def.members.begin(), def.members.end(), types.begin(),
                 [&](const polyast::Named &n) { return mkTpe(n.tpe); });
  LLVMAstTransformer::StructMemberTable table;
  for (size_t i = 0; i < def.members.size(); ++i) {
    table[def.members[i].symbol] = i;
  }
  return {llvm::StructType::create(C, types, qualified(def.name)), table};
}

llvm::Type *LLVMAstTransformer::mkTpe(const Type::Any &tpe) {                                    //
  return variants::total(                                                                        //
      *tpe,                                                                                      //
      [&](const Type::Float &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },           //
      [&](const Type::Double &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },         //
      [&](const Type::Bool &x) -> llvm::Type * { return llvm::Type::getInt1Ty(C); },             //
      [&](const Type::Byte &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },             //
      [&](const Type::Char &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },            //
      [&](const Type::Short &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },           //
      [&](const Type::Int &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); },             //
      [&](const Type::Long &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); },            //
      [&](const Type::String &x) -> llvm::Type * { return undefined(__FILE_NAME__, __LINE__); }, //
      [&](const Type::Unit &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); },             //
      [&](const Type::Struct &x) -> llvm::Type * {
        if (auto def = polyregion::get_opt(structTypes, x.name); def) {
          return def->first;
        } else {
          return undefined(__FILE_NAME__, __LINE__, "Unseen struct def : " + to_string(x));
        }
      }, //
      [&](const Type::Array &x) -> llvm::Type * {
        if (x.length) {
          return llvm::ArrayType::get(mkTpe(x.component), *x.length);
        } else {
          return mkTpe(x.component)->getPointerTo();
        }
      }

  );
}

llvm::Value *LLVMAstTransformer::mkSelect(const Term::Select &select, bool load) {

  auto fail = [&]() { return " (part of the select expression " + to_string(select) + ")"; };

  auto structTypeOf = [&](const Type::Any &tpe) -> std::pair<llvm::StructType *, StructMemberTable> {
    if (auto s = polyast::get_opt<Type::Struct>(tpe); s) {
      if (auto def = polyregion::get_opt(structTypes, s->name); def) return *def;
      else
        error(__FILE_NAME__, __LINE__, "Unseen struct type " + to_string(s->name) + " in select path" + fail());
    } else
      error(__FILE_NAME__, __LINE__, "Illegal select path involving non-struct type " + to_string(tpe) + fail());
  };

  auto selectNamed = [&](const Named &named) -> llvm::Value * {
    //  check the LUT table for known variables brought in scope by parameters
    if (auto x = polyregion::get_opt(lut, named.symbol); x) {
      if (std::holds_alternative<Type::Array>(*named.tpe) || std::holds_alternative<Type::Struct>(*named.tpe)) {
        return *x;
      } else {
        return B.CreateLoad(mkTpe(named.tpe), *x, named.symbol + "_value");
      } //
    } else {
      auto pool = mk_string2<std::string, llvm::Value *>(
          lut, [](auto &&p) { return p.first + " = " + llvm_tostring(p.second); }, "\n->");
      return undefined(__FILE_NAME__, __LINE__, "Unseen select: " + to_string(select) + ", LUT=\n->" + pool);
    }
  };

  if (select.init.empty()) {
    // plain path lookup
    return selectNamed(select.last);
  } else {
    // we're in a select chain, init elements must return struct type; the head must come from LUT

    auto head = selectNamed(select.init.front());
    auto [structTy, _] = structTypeOf(select.init.front().tpe);

    auto selectors = tail(select);

    std::vector<llvm::Value *> selectorValues;
    selectorValues.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), 0));

    auto tpe = select.init.front().tpe;
    for (auto &path : selectors) {
      auto [ignore, table] = structTypeOf(tpe);
      if (auto idx = get_opt(table, path.symbol); idx) {
        selectorValues.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), *idx));

        tpe = path.tpe;
      } else {
        return undefined(__FILE_NAME__, __LINE__,
                         "Illegal select path with unknown struct member index of name `" + to_string(path) + "`" +
                             fail());
      }
    }

    auto ptr = B.CreateInBoundsGEP(structTy, head, selectorValues, qualified(select) + "_ptr");
    if (load) return B.CreateLoad(mkTpe(select.tpe), ptr, qualified(select) + "_value");
    else
      return ptr;
  }
}

llvm::Value *LLVMAstTransformer::mkRef(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> llvm::Value * { return mkSelect(x); },
      [&](const Term::UnitConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), 0); },
      [&](const Term::BoolConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), x.value); },
      [&](const Term::ByteConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::CharConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::ShortConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::LongConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },
      [&](const Term::FloatConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getFloatTy(C), x.value); },
      [&](const Term::DoubleConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getDoubleTy(C), x.value); },
      [&](const Term::StringConst &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__); });
}

llvm::Value *LLVMAstTransformer::mkExpr(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

  const auto binaryExpr = [&](const Term::Any &l, const Term::Any &r) { return std::make_tuple(mkRef(l), mkRef(r)); };

  const auto binaryNumOp =
      [&](const Term::Any &l, const Term::Any &r, const Type::Any &promoteTo,
          const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &integralFn,
          const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &fractionalFn) -> llvm::Value * {
    auto [lhs, rhs] = binaryExpr(l, r);
    if (std::holds_alternative<TypeKind::Integral>(*kind(promoteTo))) {
      return integralFn(lhs, rhs);
    } else if (std::holds_alternative<TypeKind::Fractional>(*kind(promoteTo))) {
      return fractionalFn(lhs, rhs);
    } else {
      //    B.CreateSIToFP()
      return undefined(__FILE_NAME__, __LINE__);
    }
  };

  const auto unaryIntrinsic = [&](llvm::Intrinsic::ID id, const Type::Any &tpe, const Term::Any &arg) {
    auto cos = llvm::Intrinsic::getDeclaration(fn->getParent(), id, {mkTpe(tpe)});
    return B.CreateCall(cos, mkRef(arg));
  };

  const auto externUnaryCall = [&](const std::string &name, const Type::Any &tpe, const Term::Any &arg) {
    auto t = mkTpe(tpe);
    auto ft = llvm::FunctionType::get(t, {t}, false);
    auto f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, fn->getParent());
    return B.CreateCall(f, mkRef(arg));
  };

  return variants::total(
      *expr, //
      [&](const Expr::UnaryIntrinsic &x) {
        return variants::total(
            *x.kind, //
            [&](const UnaryIntrinsicKind::Sin &) -> llvm::Value * {
              return unaryIntrinsic(llvm::Intrinsic::sin, x.rtn, x.lhs);
            },
            [&](const UnaryIntrinsicKind::Cos &) -> llvm::Value * {
              return unaryIntrinsic(llvm::Intrinsic::cos, x.rtn, x.lhs);
            },
            [&](const UnaryIntrinsicKind::Tan &) -> llvm::Value * {
              // XXX apparently there isn't a tan in LLVM so we just do an external call
              return externUnaryCall("tan", x.rtn, x.lhs);
            },
            [&](const UnaryIntrinsicKind::Abs &) -> llvm::Value * {
              return unaryIntrinsic(llvm::Intrinsic::abs, x.rtn, x.lhs);
            },
            [&](const UnaryIntrinsicKind::BNot &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BNot");
            });
      },
      [&](const Expr::BinaryIntrinsic &x) {
        return variants::total(
            *x.kind, //
            [&](const BinaryIntrinsicKind::Add &) -> llvm::Value * {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateAdd(l, r, key + "_+"); },
                  [&](auto l, auto r) { return B.CreateFAdd(l, r, key + "_+"); });
            },
            [&](const BinaryIntrinsicKind::Sub &) -> llvm::Value * {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSub(l, r, key + "_-"); },
                  [&](auto l, auto r) { return B.CreateFSub(l, r, key + "_-"); });
            },
            [&](const BinaryIntrinsicKind::Div &) -> llvm::Value * {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSDiv(l, r, key + "_*"); },
                  [&](auto l, auto r) { return B.CreateFDiv(l, r, key + "_*"); });
            },
            [&](const BinaryIntrinsicKind::Mul &) -> llvm::Value * {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateMul(l, r, key + "_/"); },
                  [&](auto l, auto r) { return B.CreateFMul(l, r, key + "_/"); });
            },
            [&](const BinaryIntrinsicKind::Rem &) -> llvm::Value * {
              return binaryNumOp(
                  x.lhs, x.rhs, x.rtn, //
                  [&](auto l, auto r) { return B.CreateSRem(l, r, key + "_%"); },
                  [&](auto l, auto r) { return B.CreateFRem(l, r, key + "_%"); });
            },
            [&](const BinaryIntrinsicKind::Pow &) -> llvm::Value * {
              return unaryIntrinsic(llvm::Intrinsic::pow, x.rtn, x.lhs);
            },

            [&](const BinaryIntrinsicKind::BAnd &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BAnd");
            },
            [&](const BinaryIntrinsicKind::BOr &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BOr");
            },
            [&](const BinaryIntrinsicKind::BXor &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BXor");
            },
            [&](const BinaryIntrinsicKind::BSL &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BSL");
            },
            [&](const BinaryIntrinsicKind::BSR &) -> llvm::Value * {
              return undefined(__FILE_NAME__, __LINE__, "BSR");
            });
      },

      [&](const Expr::Not &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Inv"); },
      [&](const Expr::Eq &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Eq"); },
      [](const Expr::Neq &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Neq"); },
      [](const Expr::And &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "And"); },
      [](const Expr::Or &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Or"); },
      [&](const Expr::Lte &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Lte"); },
      [&](const Expr::Gte &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "Gte"); },
      [&](const Expr::Lt &x) -> llvm::Value * {
        auto [lhs, rhs] = binaryExpr(x.lhs, x.rhs);
        return B.CreateICmpSLT(lhs, rhs, key + "_<");
      },
      [&](const Expr::Gt &x) -> llvm::Value * {
        auto [lhs, rhs] = binaryExpr(x.lhs, x.rhs);
        return B.CreateICmpSGT(lhs, rhs, key + "_>");
      },

      [&](const Expr::Alias &x) -> llvm::Value * { return mkRef(x.ref); },
      [&](const Expr::Invoke &x) -> llvm::Value * {
        //        auto lhs = mkRef(x.lhs );
        return undefined(__FILE_NAME__, __LINE__, "Unimplemented invoke:`" + repr(x) + "`");
      },
      [&](const Expr::Index &x) -> llvm::Value * {
        auto tpe = mkTpe(x.tpe);
        auto ptr = B.CreateInBoundsGEP(tpe, mkSelect(x.lhs), mkRef(x.idx), key + "_ptr");
        return B.CreateLoad(tpe, ptr, key + "_value");
      });
}

void LLVMAstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn) {
  return variants::total(
      *stmt,
      [&](const Stmt::Comment &x) { /* discard comments */
                                    return;
      },
      [&](const Stmt::Var &x) {
        if (std::holds_alternative<Type::Array>(*x.name.tpe)) {
          if (x.expr) {
            lut[x.name.symbol] = mkExpr(*x.expr, fn, x.name.symbol);
          } else {
            undefined(__FILE_NAME__, __LINE__, "var array with no expr?");
          }

        } else if (std::holds_alternative<Type::Struct>(*x.name.tpe)) {
          std::cout << "var to " << (x.expr ? repr(*x.expr) : "_") << std::endl;

          if (x.expr) {
            lut[x.name.symbol] = mkExpr(*x.expr, fn, x.name.symbol);
          } else {
            auto stack = B.CreateAlloca(mkTpe(x.name.tpe), nullptr, x.name.symbol + "_stack_ptr");
            lut[x.name.symbol] = stack;
          }

        } else {
          auto stack = B.CreateAlloca(mkTpe(x.name.tpe), nullptr, x.name.symbol + "_stack_ptr");
          if (x.expr) {
            auto val = mkExpr(*x.expr, fn, x.name.symbol + "_var_rhs");
            B.CreateStore(val, stack);
          }
          lut[x.name.symbol] = stack;
        }
      },
      [&](const Stmt::Mut &x) {
        auto expr = mkExpr(x.expr, fn, qualified(x.name) + "_mut");
        auto select = mkSelect(x.name, false); // XXX do NOT allocate (mkSelect) here, we're mutating!
        std::cout << "storing to " << select << std::endl;
        B.CreateStore(expr, select);
      },
      [&](const Stmt::Update &x) {
        auto select = x.lhs;
        auto ptr = B.CreateInBoundsGEP(mkSelect(select), mkRef(x.idx), qualified(select) + "_ptr");
        B.CreateStore(mkRef(x.value), ptr);
      },
      [&](const Stmt::While &x) {
        auto loopTest = llvm::BasicBlock::Create(C, "loop_test", fn);
        auto loopBody = llvm::BasicBlock::Create(C, "loop_body", fn);
        auto loopExit = llvm::BasicBlock::Create(C, "loop_exit", fn);
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          auto continue_ = mkExpr(x.cond, fn, "loop");
          B.CreateCondBr(continue_, loopBody, loopExit);
        }
        {
          B.SetInsertPoint(loopBody);
          for (auto &body : x.body)
            mkStmt(body, fn);
          B.CreateBr(loopTest);
        }
        B.SetInsertPoint(loopExit);
      },
      [&](const Stmt::Break &x) { undefined(__FILE_NAME__, __LINE__, "break"); }, //
      [&](const Stmt::Cont &x) { undefined(__FILE_NAME__, __LINE__, "cont"); },   //
      [&](const Stmt::Cond &x) {
        auto condTrue = llvm::BasicBlock::Create(C, "cond_true", fn);
        auto condFalse = llvm::BasicBlock::Create(C, "cond_false", fn);
        auto condExit = llvm::BasicBlock::Create(C, "cond_exit", fn);
        B.CreateCondBr(mkExpr(x.cond, fn, "cond"), condTrue, condFalse);
        {
          B.SetInsertPoint(condTrue);
          for (auto &body : x.trueBr)
            mkStmt(body, fn);
          B.CreateBr(condExit);
        }
        {
          B.SetInsertPoint(condFalse);
          for (auto &body : x.falseBr)
            mkStmt(body, fn);
          B.CreateBr(condExit);
        }
        B.SetInsertPoint(condExit);
      },
      [&](const Stmt::Return &x) {
        if (std::holds_alternative<Type::Unit>(*tpe(x.value))) {
          B.CreateRetVoid();
        } else {
          B.CreateRet(mkExpr(x.value, fn, "return"));
        }
      } //

  );
}

void LLVMAstTransformer::transform(const std::unique_ptr<llvm::Module> &module, const Program &program) {

  auto fnTree = program.entry;

  // setup the struct defs first so that structs in params work
  std::transform(                                    //
      program.defs.begin(), program.defs.end(),      //
      std::inserter(structTypes, structTypes.end()), //
      [&](auto &x) -> std::pair<Sym, std::pair<llvm::StructType *, LLVMAstTransformer::StructMemberTable>> {
        return {x.name, mkStruct(x)};
      });

  auto paramTpes = map_vec<Named, llvm::Type *>(fnTree.args, [&](auto &&named) { return mkTpe(named.tpe); });
  auto fnTpe = llvm::FunctionType::get(mkTpe(fnTree.rtn), {paramTpes}, false);

  auto *fn = llvm::Function::Create(fnTpe, llvm::Function::ExternalLinkage, "lambda", *module);

  auto *entry = llvm::BasicBlock::Create(C, "entry", fn);
  B.SetInsertPoint(entry);

  // add function params to the lut first as function body will need these at some point
  std::transform(                                          //
      fn->arg_begin(), fn->arg_end(), fnTree.args.begin(), //
      std::inserter(lut, lut.end()),                       //
      [&](auto &arg, const auto &named) -> std::pair<std::string, llvm::Value *> {
        arg.setName(named.symbol);
        if (std::holds_alternative<Type::Array>(*named.tpe)) {
          return {named.symbol, &arg};
        } else {
          auto stack = B.CreateAlloca(mkTpe(named.tpe), nullptr, named.symbol + "_stack_ptr");
          B.CreateStore(&arg, stack);
          return {named.symbol, stack};
        }
      });

  for (auto &stmt : fnTree.body) {
    //    std::cout << "[LLVM]" << repr(stmt) << std::endl;
    mkStmt(stmt, fn);
  }
  module->print(llvm::errs(), nullptr);
  llvm::verifyModule(*module, &llvm::errs());
  //  std::cout << "Pre-opt verify OK!" << std::endl;

  llvm::PassManagerBuilder builder;
  //  builder.OptLevel = 3;
  llvm::legacy::PassManager m;
  builder.populateModulePassManager(m);
  m.add(llvm::createInstructionCombiningPass());
  m.run(*module);

  llvm::verifyModule(*module, &llvm::errs());
  module->print(llvm::errs(), nullptr);
}
std::optional<llvm::StructType *> LLVMAstTransformer::lookup(const Sym &s) {
  if (auto x = get_opt(structTypes, s); x) return x->first;
  else
    return {};
}

backend::LLVM::LLVM() = default; // cache(), jit(mkJit(cache)) {}

compiler::Compilation backend::LLVM::run(const Program &program) {
  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("test", *ctx);

  auto astXform = compiler::nowMono();

  LLVMAstTransformer xform(*ctx);
  xform.transform(mod, program);

  auto elapsed = compiler::elapsedNs(astXform);

  auto c = llvmc::compileModule(true, std::move(mod), *ctx);

  // at this point we know the target machine, so we derive the struct layout here
  for (auto def : program.defs) {
    auto x = xform.lookup(def.name);
    if (!x) {
      throw std::logic_error("Missing struct def:" + repr(def));
    } else {
      // FIXME this needs to use the same LLVM target machine context as the compiler
      c.layouts.emplace_back(compiler::layoutOf(def));
    }
  }

  c.events.emplace_back(compiler::nowMs(), "ast_to_llvm_ir", elapsed);

  return c;
}
