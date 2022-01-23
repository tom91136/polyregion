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

static llvm::ExitOnError ExitOnErr;

llvm::Type *backend::AstTransformer::mkTpe(const Type::Any &tpe) {
  return variants::total(
      *tpe,                                                                                      //
      [&](const Type::Float &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },           //
      [&](const Type::Double &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },         //
      [&](const Type::Bool &x) -> llvm::Type * { return llvm::Type::getInt1Ty(C); },             //
      [&](const Type::Byte &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },             //
      [&](const Type::Char &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },             //
      [&](const Type::Short &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },           //
      [&](const Type::Int &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); },             //
      [&](const Type::Long &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); },            //
      [&](const Type::String &x) -> llvm::Type * { return undefined(__FILE_NAME__, __LINE__); }, //
      [&](const Type::Unit &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); },             //
      [&](const Type::Struct &x) -> llvm::Type * { return undefined(__FILE_NAME__, __LINE__); }, //
      [&](const Type::Array &x) -> llvm::Type * { return mkTpe(x.component)->getPointerTo(); }   //
  );
}

llvm::Value *backend::AstTransformer::mkSelect(const Term::Select &select) {

  if (auto x = lut.find(qualified(select)); x != lut.end()) {
    return std::holds_alternative<Type::Array>(*select.last.tpe)                           //
               ? x->second                                                                 //
               : B.CreateLoad(mkTpe(select.tpe), x->second, qualified(select) + "_value"); //
  } else {
    return undefined(__FILE_NAME__, __LINE__, "Unseen select: " + to_string(select));
  }
}

llvm::Value *backend::AstTransformer::mkRef(const Term::Any &ref) {
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

llvm::Value *backend::AstTransformer::mkExpr(const Expr::Any &expr, llvm::Function *fn, const std::string &key) {

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

      [&](const Expr::Sin &x) -> llvm::Value * { return unaryIntrinsic(llvm::Intrinsic::sin, x.rtn, x.lhs); },
      [&](const Expr::Cos &x) -> llvm::Value * { return unaryIntrinsic(llvm::Intrinsic::cos, x.rtn, x.lhs); },
      [&](const Expr::Tan &x) -> llvm::Value * {
        // XXX apparently there isn't a tan in LLVM so we just do an external call
        return externUnaryCall("tan", x.rtn, x.lhs);
      },
      [&](const Expr::Abs &x) -> llvm::Value * { return unaryIntrinsic(llvm::Intrinsic::abs, x.rtn, x.lhs); },

      [&](const Expr::Add &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateAdd(l, r, key + "_+"); },
            [&](auto l, auto r) { return B.CreateFAdd(l, r, key + "_+"); });
      },
      [&](const Expr::Sub &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateSub(l, r, key + "_-"); },
            [&](auto l, auto r) { return B.CreateFSub(l, r, key + "_-"); });
      },
      [&](const Expr::Div &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateSDiv(l, r, key + "_*"); },
            [&](auto l, auto r) { return B.CreateFDiv(l, r, key + "_*"); });
      },
      [&](const Expr::Mul &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateMul(l, r, key + "_/"); },
            [&](auto l, auto r) { return B.CreateFMul(l, r, key + "_/"); });
      },
      [&](const Expr::Rem &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateSRem(l, r, key + "_%"); },
            [&](auto l, auto r) { return B.CreateFRem(l, r, key + "_%"); });
      },
      [&](const Expr::Pow &x) -> llvm::Value * { return unaryIntrinsic(llvm::Intrinsic::pow, x.rtn, x.lhs); },

      [](const Expr::BNot &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BNot"); },
      [](const Expr::BAnd &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BAnd"); },
      [](const Expr::BOr &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BOr"); },
      [](const Expr::BXor &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BXor"); },
      [](const Expr::BSL &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BSL"); },
      [](const Expr::BSR &x) -> llvm::Value * { return undefined(__FILE_NAME__, __LINE__, "BSR"); },

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
        return undefined(__FILE_NAME__, __LINE__, "Unimplemented invoke:`" + x.name + "`");
      },
      [&](const Expr::Index &x) -> llvm::Value * {
        auto tpe = mkTpe(x.tpe);
        auto ptr = B.CreateInBoundsGEP(tpe, mkSelect(x.lhs), mkRef(x.idx), key + "_ptr");
        return B.CreateLoad(tpe, ptr, key + "_value");
      });
}

void backend::AstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn) {
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
        auto select = lut[qualified(x.name)]; // XXX do NOT allocate (mkSelect) here, we're mutating!
        B.CreateStore(expr, select);
      },
      [&](const Stmt::Update &x) {
        auto select = x.lhs;
        auto ptr = B.CreateInBoundsGEP(mkSelect(select), mkRef(x.idx), qualified(select) + "_ptr");
        B.CreateStore(mkRef(x.value), ptr);
      },
      [&](const Stmt::Effect &x) {
        if (x.args.size() == 1) {
          auto name = x.name;
          auto rhs = x.args[0];
          undefined(__FILE_NAME__, __LINE__, "effect not implemented");
        }
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

void backend::AstTransformer::transform(const std::unique_ptr<llvm::Module> &module, const Function &fnTree) {

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

// void polyregion::backend::JitObjectCache::notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef
// ObjBuffer) {
//   llvm::dbgs() << "Compiled object for " << M->getModuleIdentifier() << "\n";
//
//   auto x = ExitOnErr(llvm::object::createBinary(ObjBuffer));
//
//   std::ofstream outfile("obj.o", std::ofstream::binary);
//   outfile.write(ObjBuffer.getBufferStart(), ObjBuffer.getBufferSize());
//   outfile.close();
//
//   std::cout << "S=" << ObjBuffer.getBufferSize() << std::endl;
//
//   if (auto *file = llvm::dyn_cast<llvm::object::ObjectFile>(&*x)) {
//     llvm::dbgs() << "Yes!\n";
//     auto sections = dis::disassembleCodeSections(*file);
//     //    polyregion::dis::dump(std::cerr, sections);
//     std::cerr << std::endl;
//   }
//
//   CachedObjects[M->getModuleIdentifier()] =
//       llvm::MemoryBuffer::getMemBufferCopy(ObjBuffer.getBuffer(), ObjBuffer.getBufferIdentifier());
// }
//
// std::unique_ptr<llvm::MemoryBuffer> polyregion::backend::JitObjectCache::getObject(const llvm::Module *M) {
//   auto I = CachedObjects.find(M->getModuleIdentifier());
//   if (I == CachedObjects.end()) {
//     llvm::dbgs() << "No object for " << M->getModuleIdentifier() << " in cache. Compiling.\n";
//     return nullptr;
//   }
//
//   llvm::dbgs() << "Object for " << M->getModuleIdentifier() << " loaded from cache.\n";
//   return llvm::MemoryBuffer::getMemBuffer(I->second->getMemBufferRef());
// }
// backend::JitObjectCache::~JitObjectCache() = default;
// void backend::JitObjectCache::anchor() {}
// backend::JitObjectCache::JitObjectCache() = default;
//
// static std::unique_ptr<llvm::orc::LLJIT> mkJit(llvm::ObjectCache &cache) {
//   using namespace llvm;
//   orc::LLJITBuilder builder = orc::LLJITBuilder();
//   builder.setCompileFunctionCreator(
//       [&](orc::JITTargetMachineBuilder JTMB) -> Expected<std::unique_ptr<orc::IRCompileLayer::IRCompiler>> {
//         auto TM = JTMB.createTargetMachine();
//         if (!TM) return TM.takeError();
//         return std::make_unique<orc::TMOwningSimpleCompiler>(orc::TMOwningSimpleCompiler(std::move(*TM), &cache));
//       });
//   return ExitOnErr(builder.create());
// }

backend::LLVM::LLVM() = default; // cache(), jit(mkJit(cache)) {}

compiler::Compilation backend::LLVM::run(const Function &fn) {
  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("test", *ctx);

  auto astXform = compiler::nowMono();

  AstTransformer xform(*ctx);
  xform.transform(mod, fn);

  auto elapsed = compiler::elapsedNs(astXform);

  auto c = llvmc::compileModule(true, std::move(mod), *ctx);

  c.events.emplace_back(compiler::nowMs(), "ast_to_llvm_ir", elapsed);

  return c;

  //  orc::ThreadSafeModule tsm(std::move(mod), std::move(ctx));
  //  ExitOnErr(jit->addIRModule(std::move(tsm)));
  //  JITEvaluatedSymbol symbol = ExitOnErr(jit->lookup("lambda"));
  //  std::cout << "S= "
  //            << " " << symbol.getAddress() << "  " << std::hex << symbol.getAddress() << std::endl;

  //  std::cout << "Prep for DL" << std::endl;
  //
  //  void *client_hndl = dlopen("/home/tom/Desktop/prime.so", RTLD_LAZY);
  //
  //  std::cout << "Prep for DL =     " << client_hndl << std::endl;
  //
  //  if (!client_hndl) {
  //    std::cerr << "DL failed=" << dlerror() << std::endl;
  //  } else {
  //    std::cout << "Handle="
  //              << " " << client_hndl << std::endl;
  //    void *ptr = dlsym(client_hndl, "isPrime");
  //    void *ptr2 = dlsym(client_hndl, "doit");
  //
  //    typedef int (*FF)(int);
  //    typedef char *(*FF2)();
  //
  //    FF f = (FF)ptr;
  //    FF2 f2 = (FF2)ptr2;
  //
  //    std::cout << "ptr1="
  //              << " " << ptr << " ptr2=" << ptr2 << std::endl;
  //    std::cout << "res="
  //              << " " << std::to_string(f(100)) << " " << f2() << std::endl;
  //  }
}
