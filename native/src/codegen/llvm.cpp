
#include "llvm.h"
#include "ast.h"
#include "utils.hpp"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include <dis.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Object/ObjectFile.h>

using namespace polyregion;
using namespace polyregion::polyast;

static llvm::ExitOnError ExitOnErr;

llvm::Type *codegen::AstTransformer::mkTpe(const Type::Any &tpe) {
  return variants::total(
      *tpe,                                                                                    //
      [&](const Type::Float &x) -> llvm::Type * { return llvm::Type::getFloatTy(C); },        //
      [&](const Type::Double &x) -> llvm::Type * { return llvm::Type::getDoubleTy(C); },      //
      [&](const Type::Bool &x) -> llvm::Type * { return llvm::Type::getInt1Ty(C); },          //
      [&](const Type::Byte &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },          //
      [&](const Type::Char &x) -> llvm::Type * { return llvm::Type::getInt8Ty(C); },          //
      [&](const Type::Short &x) -> llvm::Type * { return llvm::Type::getInt16Ty(C); },        //
      [&](const Type::Int &x) -> llvm::Type * { return llvm::Type::getInt32Ty(C); },          //
      [&](const Type::Long &x) -> llvm::Type * { return llvm::Type::getInt64Ty(C); },         //
      [&](const Type::String &x) -> llvm::Type * { return undefined(); },                      //
      [&](const Type::Unit &x) -> llvm::Type * { return llvm::Type::getVoidTy(C); },          //
      [&](const Type::Struct &x) -> llvm::Type * { return undefined(); },                      //
      [&](const Type::Array &x) -> llvm::Type * { return mkTpe(x.component)->getPointerTo(); } //
  );
}

llvm::Value *codegen::AstTransformer::mkSelect(const Term::Select &select) {

  if (auto x = lut.find(qualified(select)); x != lut.end()) {
    return std::holds_alternative<Type::Array>(*select.last.tpe)        //
               ? x->second                                              //
               : B.CreateLoad(x->second, qualified(select) + "_value"); //
  } else {
    return undefined("Unseen select: " + to_string(select));
  }
}

llvm::Value *codegen::AstTransformer::mkRef(const Term::Any &ref) {
  using llvm::ConstantFP;
  using llvm::ConstantInt;
  return variants::total(
      *ref, //
      [&](const Term::Select &x) -> llvm::Value * { return mkSelect(x); },
      [&](const Term::BoolConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt1Ty(C), x.value); },
      [&](const Term::ByteConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::CharConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt8Ty(C), x.value); },
      [&](const Term::ShortConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt16Ty(C), x.value); },
      [&](const Term::IntConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt32Ty(C), x.value); },
      [&](const Term::LongConst &x) -> llvm::Value * { return ConstantInt::get(llvm::Type::getInt64Ty(C), x.value); },
      [&](const Term::FloatConst &x) -> llvm::Value * { return ConstantFP::get(llvm::Type::getFloatTy(C), x.value); },
      [&](const Term::DoubleConst &x) -> llvm::Value * {
        return ConstantFP::get(llvm::Type::getDoubleTy(C), x.value);
      },
      [&](const Term::StringConst &x) -> llvm::Value * { return undefined(); });
}

llvm::Value *codegen::AstTransformer::mkExpr(const Expr::Any &expr, const std::string &key) {

  const auto binaryExpr = [&](const Term::Any &l, const Term::Any &r) { return std::make_tuple(mkRef(l), mkRef(r)); };

  const auto binaryNumOp =
      [&](const Term::Any &l, const Term::Any &r, const Type::Any &promoteTo,
          const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &integralFn,
          const std::function<llvm::Value *(llvm::Value *, llvm::Value *)> &fractionalFn) -> llvm::Value * {
    auto [lhs, rhs] = binaryExpr(l, r);
    std::cout << "-> " << kind(promoteTo) << promoteTo << std::endl;
    if (std::holds_alternative<TypeKind::Integral>(*kind(promoteTo))) {
      return integralFn(lhs, rhs);
    } else if (std::holds_alternative<TypeKind::Fractional>(*kind(promoteTo))) {
      return fractionalFn(lhs, rhs);
    } else {
      //    B.CreateSIToFP()
      return undefined();
    }
  };

  return variants::total(
      *expr, //

      [&](const Expr::Sin &x) -> llvm::Value * { return undefined(); },
      [&](const Expr::Cos &x) -> llvm::Value * { return undefined(); },
      [&](const Expr::Tan &x) -> llvm::Value * { return undefined(); },

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
      [&](const Expr::Mod &x) -> llvm::Value * {
        return binaryNumOp(
            x.lhs, x.rhs, x.rtn, //
            [&](auto l, auto r) { return B.CreateSRem(l, r, key + "_%"); },
            [&](auto l, auto r) { return B.CreateFRem(l, r, key + "_%"); });
      },
      [&](const Expr::Pow &x) -> llvm::Value * { return undefined(); },

      [&](const Expr::Inv &x) -> llvm::Value * { return undefined(); },
      [&](const Expr::Eq &x) -> llvm::Value * { return undefined(); },
      [&](const Expr::Lte &x) -> llvm::Value * { return undefined(); },
      [&](const Expr::Gte &x) -> llvm::Value * { return undefined(); },
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
        return undefined("Unimplemented invoke:`" + x.name + "`");
      },
      [&](const Expr::Index &x) -> llvm::Value * {
        auto ptr = B.CreateInBoundsGEP(mkSelect(x.lhs), {mkRef(x.idx)}, key + "_ptr");
        return B.CreateLoad(ptr, key + "_value");
      });
}

void codegen::AstTransformer::mkStmt(const Stmt::Any &stmt, llvm::Function *fn) {
  return variants::total(
      *stmt,
      [&](const Stmt::Comment &x) { /* discard comments */
                                    return;
      },
      [&](const Stmt::Var &x) {
        if (std::holds_alternative<Type::Array>(*x.name.tpe)) {
          lut[x.name.symbol] = mkExpr(x.expr, x.name.symbol);
        } else {
          auto stack = B.CreateAlloca(mkTpe(x.name.tpe), nullptr, x.name.symbol + "_stack_ptr");
          auto val = mkExpr(x.expr, x.name.symbol + "_var_rhs");
          B.CreateStore(val, stack);
          lut[x.name.symbol] = stack;
        }
      },
      [&](const Stmt::Mut &x) {
        auto expr = mkExpr(x.expr, qualified(x.name) + "_mut");
        auto select = lut[qualified(x.name)]; // XXX do NOT allocate (mkSelect) here, we're mutating!
        B.CreateStore(expr, select);
      },
      [&](const Stmt::Update &x) {
        auto select = x.lhs;
        auto ptr = B.CreateInBoundsGEP(mkSelect(select), {mkRef(x.idx)}, qualified(select) + "_ptr");
        B.CreateStore(mkRef(x.value), ptr);
      },
      [&](const Stmt::Effect &x) {
        if (x.args.size() == 1) {
          auto name = x.name;
          auto rhs = x.args[0];
          undefined("effect not implemented");
        }
      },
      [&](const Stmt::While &x) {
        auto loopTest = llvm::BasicBlock::Create(C, "loop_test", fn);
        auto loopBody = llvm::BasicBlock::Create(C, "loop_body", fn);
        auto loopExit = llvm::BasicBlock::Create(C, "loop_exit", fn);
        B.CreateBr(loopTest);
        {
          B.SetInsertPoint(loopTest);
          auto continue_ = mkExpr(x.cond, "loop");
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
      [&](const Stmt::Break &x) { undefined("break"); }, [&](const Stmt::Cont &x) { undefined("break"); },
      [&](const Stmt::Cond &x) {
        auto condTrue = llvm::BasicBlock::Create(C, "cond_true", fn);
        auto condFalse = llvm::BasicBlock::Create(C, "cond_false", fn);
        auto condExit = llvm::BasicBlock::Create(C, "cond_exit", fn);
        B.CreateCondBr(mkExpr(x.cond, "cond"), condTrue, condFalse);
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
      [&](const Stmt::Return &x) { undefined("break"); }

  );
}

void codegen::AstTransformer::transform(const std::unique_ptr<llvm::Module> &module, const Function &fnTree) {

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
    std::cout << "[LLVM]" << repr(stmt) << std::endl;
    mkStmt(stmt, fn);
  }
  B.CreateRetVoid();
  module->print(llvm::errs(), nullptr);
  llvm::verifyModule(*module, &llvm::errs());
  std::cout << "Pre-opt verify OK!" << std::endl;

  llvm::PassManagerBuilder builder;
  //  builder.OptLevel = 3;
  llvm::legacy::PassManager m;
  builder.populateModulePassManager(m);
  m.add(llvm::createInstructionCombiningPass());
  m.run(*module);

  llvm::verifyModule(*module, &llvm::errs());
  module->print(llvm::errs(), nullptr);
}

void polyregion::codegen::JitObjectCache::notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef ObjBuffer) {
  llvm::dbgs() << "Compiled object for " << M->getModuleIdentifier() << "\n";

  auto x = ExitOnErr(llvm::object::createBinary(ObjBuffer));

  std::ofstream outfile("obj.so", std::ofstream::binary);
  outfile.write(ObjBuffer.getBufferStart(), ObjBuffer.getBufferSize());
  outfile.close();

  std::cout << "S=" << ObjBuffer.getBufferSize() << std::endl;

  if (auto *file = llvm::dyn_cast<llvm::object::ObjectFile>(&*x)) {
    llvm::dbgs() << "Yes!\n";
    auto sections = dis::disassembleCodeSections(*file);
    polyregion::dis::dump(std::cerr, sections);
    std::cerr << std::endl;
  }

  CachedObjects[M->getModuleIdentifier()] =
      llvm::MemoryBuffer::getMemBufferCopy(ObjBuffer.getBuffer(), ObjBuffer.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer> polyregion::codegen::JitObjectCache::getObject(const llvm::Module *M) {
  auto I = CachedObjects.find(M->getModuleIdentifier());
  if (I == CachedObjects.end()) {
    llvm::dbgs() << "No object for " << M->getModuleIdentifier() << " in cache. Compiling.\n";
    return nullptr;
  }

  llvm::dbgs() << "Object for " << M->getModuleIdentifier() << " loaded from cache.\n";
  return llvm::MemoryBuffer::getMemBuffer(I->second->getMemBufferRef());
}

static std::unique_ptr<llvm::orc::LLJIT> mkJit(llvm::ObjectCache &cache) {
  using namespace llvm;
  orc::LLJITBuilder builder = orc::LLJITBuilder();
  builder.setCompileFunctionCreator(
      [&](orc::JITTargetMachineBuilder JTMB) -> Expected<std::unique_ptr<orc::IRCompileLayer::IRCompiler>> {
        auto TM = JTMB.createTargetMachine();
        if (!TM) return TM.takeError();
        return std::make_unique<orc::TMOwningSimpleCompiler>(orc::TMOwningSimpleCompiler(std::move(*TM), &cache));
      });
  return ExitOnErr(builder.create());
}

codegen::LLVMCodeGen::LLVMCodeGen() : cache(), jit(mkJit(cache)) {}

#include <unistd.h>
#include <dlfcn.h>
void codegen::LLVMCodeGen::run(const Function &fn) {
  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto mod = std::make_unique<llvm::Module>("test", *ctx);

  AstTransformer xform(*ctx);
  xform.transform(mod, fn);

  orc::ThreadSafeModule tsm(std::move(mod), std::move(ctx));
  ExitOnErr(jit->addIRModule(std::move(tsm)));
  JITEvaluatedSymbol symbol = ExitOnErr(jit->lookup("lambda"));
  std::cout << "S="<< " " <<symbol.getAddress()  << " "  << std::hex << symbol.getAddress()   << std::endl;


  void* client_hndl = dlopen("/home/tom/polyregion/native/obj.so",  RTLD_LAZY);
  if(!client_hndl){
    std::cerr << "DL failed=" <<dlerror() <<std::endl;
  }else{
    std::cout << "DL="<< " " << client_hndl<< std::endl;
  }


}
