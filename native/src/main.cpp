#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#include "capstone/capstone.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "PolyAst.pb.h"
#include "ast.h"
#include "dis.h"
#include "utils.hpp"
#include "codegen/llvm.h"
#include "codegen/opencl.h"


using namespace llvm;
using namespace llvm::orc;
using namespace polyregion;

// ThreadSafeModule createDemoModule() {
//   auto ctx = std::make_unique<LLVMContext>();
//   auto mod = std::make_unique<Module>("test", *ctx);
//
//   // Create the add1 function entry and insert this entry into module mod.  The
//   // function will have a return type of "int" and take an argument of "int".
//   Function *Add1F = Function::Create(FunctionType::get(Type::getInt32Ty(*ctx), {Type::getInt32Ty(*ctx)}, false),
//                                      Function::ExternalLinkage, "add1", mod.get());
//
//   // Add a basic block to the function. As before, it automatically inserts
//   // because of the last argument.
//   BasicBlock *BB = BasicBlock::Create(*ctx, "EntryBlock", Add1F);
//
//   // Create a basic block builder with default parameters.  The builder will
//   // automatically append instructions to the basic block `BB'.
//   IRBuilder<> builder(BB);
//
//   // Get pointers to the constant `1'.
//   Value *One = builder.getInt32(42);
//
//   // Get pointers to the integer argument of the add1 function...
//   assert(Add1F->arg_begin() != Add1F->arg_end()); // Make sure there's an arg
//   Argument *ArgX = &*Add1F->arg_begin();          // Get the arg
//   ArgX->setName("AnArg");                         // Give it a nice symbolic name for fun.
//
//   // Create the add instruction, inserting it into the end of BB.
//   Value *Add = builder.CreateAdd(One, ArgX);
//
//   // Create the return instruction and add it to the basic block
//   builder.CreateRet(Add);
//
//   return {std::move(mod), std::move(ctx)};
// }

ExitOnError ExitOnErr;

class MyObjectCache : public ObjectCache {
private:
  StringMap<std::unique_ptr<MemoryBuffer>> CachedObjects;

public:
  void notifyObjectCompiled(const Module *M, MemoryBufferRef ObjBuffer) override {

    dbgs() << "Compiled object for " << M->getModuleIdentifier() << "\n";

    auto x = ExitOnErr(object::createBinary(ObjBuffer));

    std::ofstream outfile("obj.o", std::ofstream::binary);
    outfile.write(ObjBuffer.getBufferStart(), ObjBuffer.getBufferSize());
    outfile.close();

    std::cout << "S=" << ObjBuffer.getBufferSize() << std::endl;

    if (auto *file = dyn_cast<object::ObjectFile>(&*x)) {
      dbgs() << "Yes!\n";
      auto sections = dis::disassembleCodeSections(*file);
      polyregion::dis::dump(std::cerr, sections);
      std::cerr << std::endl;
    }

    CachedObjects[M->getModuleIdentifier()] =
        MemoryBuffer::getMemBufferCopy(ObjBuffer.getBuffer(), ObjBuffer.getBufferIdentifier());
  }

  std::unique_ptr<MemoryBuffer> getObject(const Module *M) override {
    auto I = CachedObjects.find(M->getModuleIdentifier());
    if (I == CachedObjects.end()) {
      dbgs() << "No object for " << M->getModuleIdentifier() << " in cache. Compiling.\n";
      return nullptr;
    }

    dbgs() << "Object for " << M->getModuleIdentifier() << " loaded from cache.\n";
    return MemoryBuffer::getMemBuffer(I->second->getMemBufferRef());
  }
};




void doIt(const Tree_Function &fnTree) {
  auto ctx = std::make_unique<LLVMContext>();
  auto mod = std::make_unique<Module>("test", *ctx);
  IRBuilder<> B(*ctx);

  auto mkTpe = [&](const Types_Type &type) -> Type * {
    if (type.has_booltpe()) return undefined();
    if (type.has_bytetpe()) return B.getInt8Ty();
    if (type.has_chartpe()) return B.getInt8Ty();
    if (type.has_shorttpe()) return B.getInt16Ty();
    if (type.has_inttpe()) return B.getInt32Ty();
    if (type.has_longtpe()) return B.getInt64Ty();
    if (type.has_doubletpe()) return B.getDoubleTy();
    if (type.has_floattpe()) return B.getFloatTy();
    if (type.has_stringtpe()) return undefined();
    if (auto arrtpe = POLY_OPT(type, arraytpe); arrtpe) {
      auto comp = arrtpe->tpe();
      if (comp.has_booltpe()) return undefined();
      if (comp.has_bytetpe()) return B.getInt8Ty();
      if (comp.has_chartpe()) return B.getInt8Ty();
      if (comp.has_shorttpe()) return B.getInt16Ty();
      if (comp.has_inttpe()) return B.getInt32Ty();
      if (comp.has_longtpe()) return B.getInt64Ty();
      if (comp.has_doubletpe()) return B.getDoubleTy();
      if (comp.has_floattpe()) return B.getFloatTy();
      if (comp.has_stringtpe()) return undefined();
      return undefined();
    }
    if (auto reftpe = POLY_OPT(type, reftpe); reftpe) {
      //            return undefined();
    }
    return undefined("Unimplemented type:" + type.DebugString());
  };

  auto mkRef = [&](const Refs_Ref &ref, //
                   const std::unordered_map<std::string, Value *> &lut) -> Value * {
    if (auto select = POLY_OPT(ref, select); select) {
      if (auto x = lut.find(select->head().name()); x != lut.end()) {
        return x->second;
      } else {
        return undefined("Unseen select: " + select->DebugString());
      }
    }
    if (auto c = POLY_OPT(ref, boolconst); c) return undefined();
    if (auto c = POLY_OPT(ref, byteconst); c) return B.getInt8(c->value());
    if (auto c = POLY_OPT(ref, charconst); c) return B.getInt8(c->value());
    if (auto c = POLY_OPT(ref, shortconst); c) return B.getInt16(c->value());
    if (auto c = POLY_OPT(ref, intconst); c) return B.getInt32(c->value());
    if (auto c = POLY_OPT(ref, longconst); c) return B.getInt64(c->value());
    if (auto c = POLY_OPT(ref, doubleconst); c) return ConstantFP::get(B.getDoubleTy(), c->value());
    if (auto c = POLY_OPT(ref, floatconst); c) return ConstantFP::get(B.getFloatTy(), c->value());
    if (auto c = POLY_OPT(ref, stringconst); c) return undefined();
    return undefined();
  };



  auto mkExpr = [&](const Tree_Expr &expr, //
                    const std::unordered_map<std::string, Value *> &lut) -> Value * {
    if (auto alias = POLY_OPT(expr, alias); alias) {
      return mkRef(alias->ref(), lut);
    }
    if (auto invoke = POLY_OPT(expr, invoke); invoke) {

      auto name = invoke->name();

      switch (hash(name.data(), name.size())) {
      case ""_: {
      }
      }
    }
    return undefined();
  };

  auto mkStmt = [&](const Tree_Stmt &stmt) -> Value * {
    if (auto comment = POLY_OPT(stmt, comment); comment) {
    }
    if (auto var = POLY_OPT(stmt, var); var) {
    }
    if (auto effect = POLY_OPT(stmt, effect); effect) {
    }
    if (auto mut = POLY_OPT(stmt, mut); mut) {
    }
    if (auto while_ = POLY_OPT(stmt, while_); while_) {
    }
    return undefined();
  };

  auto fnTpe = FunctionType::get(                                                                //
      mkTpe(fnTree.returntpe()),                                                                 //
      {                                                                                          //
       map_vec<Named, Type *>(fnTree.args(), [&](auto &&named) { return mkTpe(named.tpe()); })}, //
      false);
  auto *fn = Function::Create(fnTpe, Function::ExternalLinkage, fnTree.name(), *mod);

  auto *BB = BasicBlock::Create(*ctx, "EntryBlock", fn);






  // Get pointers to the constant `1'.
  Value *One = B.getInt32(42);

  // Get pointers to the integer argument of the add1 fnTree...
  assert(fn->arg_begin() != fn->arg_end()); // Make sure there's an arg
  Argument *ArgX = &*fn->arg_begin();       // Get the arg
  ArgX->setName("AnArg");                   // Give it a nice symbolic name for fun.

  // Create the add instruction, inserting it into the end of BB.
  Value *Add = B.CreateAdd(One, ArgX);

  // Create the return instruction and add it to the basic block
  B.CreateRet(Add);
}

int main(int argc, char *argv[]) {
  // Initialize LLVM.
  InitLLVM X(argc, argv);

  MyObjectCache cache;

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  InitializeAllTargets();
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "HowToUseLLJIT");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  // Create an LLJIT instance.
  LLJITBuilder builder = LLJITBuilder();

  builder.setCompileFunctionCreator(
      [&](JITTargetMachineBuilder JTMB) -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
        auto TM = JTMB.createTargetMachine();
        if (!TM) return TM.takeError();
        return std::make_unique<TMOwningSimpleCompiler>(TMOwningSimpleCompiler(std::move(*TM), &cache));
      });

  auto jit = ExitOnErr(builder.create());
  //  auto module = createDemoModule();
  //  module.getModuleUnlocked()->print(llvm::errs(), nullptr);

//  ExitOnErr(jit->addIRModule(std::move(module)));
//  // Look up the JIT'd function, cast it to a function pointer, then call it.
//  auto Add1Sym = ExitOnErr(jit->lookup("add1"));
//  std::cout << "Addr=" << Add1Sym.getAddress() << "\n";
//  int (*Add1)(int) = (int (*)(int))Add1Sym.getAddress();
//
//  int Result = Add1(42);
//  outs() << "add1(42) = " << Result << "\n";

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  //    std::ifstream input("/home/tom/polyregion/ast.bin", std::ios::in | std::ios::binary);

  std::vector<uint8_t> xs = readNStruct<uint8_t>("/home/tom/polyregion/ast.bin");

  std::cout << mk_string<uint8_t>(
                   xs, [](auto x) { return std::to_string(x); }, " ")
            << "\n";

  Tree_Function p;
  std::cout << "s=" << xs.size() << "\n";
  auto ok = p.ParseFromArray(xs.data(), xs.size());
  ast::DebugPrinter printer;
  std::cout << ">> " << ok << " " << xs.size() << "\n" << printer.repr(p) << std::endl;

  codegen::LLVMCodeGen gen("a");
  gen.run(p);

  codegen::OpenCLCodeGen oclGen ;
  oclGen.run(p);


  return 0;
}
