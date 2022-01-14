#include "llvmc.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <fstream>
#include <iostream>

using namespace polyregion::backend;

void llvmc::initialise() {

  llvm::LLVMContext Context;

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  auto *r = llvm::PassRegistry::getPassRegistry();
  initializeCore(*r);
  initializeCodeGen(*r);
  initializeLoopStrengthReducePass(*r);
  initializeLowerIntrinsicsPass(*r);
  initializeEntryExitInstrumenterPass(*r);
  initializePostInlineEntryExitInstrumenterPass(*r);
  initializeUnreachableBlockElimLegacyPassPass(*r);
  initializeConstantHoistingLegacyPassPass(*r);
  initializeScalarOpts(*r);
  initializeVectorization(*r);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*r);
  initializeExpandReductionsPass(*r);
  initializeExpandVectorPredicationPass(*r);
  initializeHardwareLoopsPass(*r);
  initializeTransformUtils(*r);
  initializeReplaceWithVeclibLegacyPass(*r);
}

polyregion::compiler::Compilation llvmc::compileModule(bool emitDisassembly,            //
                                                       std::unique_ptr<llvm::Module> M, //
                                                       llvm::LLVMContext &Context) {
  using namespace llvm;

  auto start = compiler::nowMono();

  SMDiagnostic Err;

  std::string CPUStr = sys::getHostCPUName().str();

  StringMap<bool> HostFeatures;
  SubtargetFeatures Features;
  if (sys::getHostCPUFeatures(HostFeatures))
    for (auto &F : HostFeatures)
      Features.AddFeature(F.first(), F.second);
  std::string FeaturesStr = Features.getString();

  std::cout << "F=" << FeaturesStr << std::endl;
  CodeGenOpt::Level OLvl = CodeGenOpt::Aggressive;

  std::string IRTargetTriple = M->getDataLayoutStr();
  std::string WantedTriple = Triple::normalize("x86_64-pc-linux-gnu");
  std::string SysDefaultTriple = sys::getDefaultTargetTriple();
  Triple TheTriple = Triple(SysDefaultTriple);

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);

  if (!Error.empty()) {
    return polyregion::compiler::Compilation{""};
  }

  std::cout << "E=" << Error << "; " << SysDefaultTriple << std::endl;

  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine( //
      TheTriple.getTriple(),                                        //
      CPUStr,                                                       //
      FeaturesStr,                                                  //
      Options, Reloc::Model::PIC_, llvm::None, OLvl));
  M->setDataLayout(TM->createDataLayout());

  verifyModule(*M, &errs());

  auto mkPass = [&](const std::function<void(LLVMTargetMachine &, legacy::PassManager &, MCContext &)> &f) {
    legacy::PassManager PM;
    // XXX we have no rtti here so no dynamic cast
    auto &LLVMTM = static_cast<LLVMTargetMachine &>(*TM);    // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
    auto *MMIWP = new MachineModuleInfoWrapperPass(&LLVMTM); // pass manager takes owner of this

    TargetPassConfig *PassConfig = LLVMTM.createPassConfig(PM);
    // Set PassConfig options provided by TargetMachine.
    PassConfig->setDisableVerify(true);
    PM.add(PassConfig);
    PM.add(MMIWP);

    if (PassConfig->addISelPasses()) throw std::logic_error("No ISEL");
    PassConfig->addMachinePasses();
    PassConfig->setInitialized();

    if (TargetPassConfig::willCompleteCodeGenPipeline()) {
      f(LLVMTM, PM, MMIWP->getMMI().getContext());
    }
    return PM;
  };

  llvm::SmallVector<char, 0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  auto asmPM = mkPass([&](auto &tm, auto &pm, auto &ctx) {
    tm.addAsmPrinter(pm, asmStream, nullptr, CodeGenFileType::CGFT_AssemblyFile, ctx);
  });
  asmPM.run(*llvm::CloneModule(*M));

  llvm::SmallVector<char, 0> objBuffer;
  llvm::raw_svector_ostream objStream(objBuffer);
  auto objPM = mkPass([&](auto &tm, auto &pm, auto &ctx) {
    tm.addAsmPrinter(pm, objStream, nullptr, CodeGenFileType::CGFT_ObjectFile, ctx);
  });
  objPM.run(*llvm::CloneModule(*M));

  //  for (auto &&f : MMIWP->getMMI().getModule()->functions()) {
  //    std::cout << f.getName().str() << std::endl;
  //    auto mf = MMIWP->getMMI().getMachineFunction(f);
  //    std::for_each(mf->begin(), mf->end(), [&](MachineBasicBlock &b) {
  //      std::cout << "[F] BB=" << b.getName().str() << "\n";
  //      std::for_each(b.begin(), b.end(), [&](MachineInstr &ins) {
  //        auto mnem = TM->getMCInstrInfo()->getName(ins.getOpcode());
  //        std::cout << "    " << mnem.str() << "\n";
  //      });
  //    });
  //  }

  std::cout << "Done = "
            << " b=" << asmBuffer.size() << std::endl;

  std::ofstream file("the_obj2.o", std::ios::binary);
  file.write(objBuffer.data(), ssize_t(objBuffer.size_in_bytes()));
  file.flush();

  std::ofstream file2("the_obj2.s", std::ios::binary);
  file2.write(asmBuffer.data(), ssize_t(asmBuffer.size_in_bytes()));
  file2.flush();

  //  auto b =
  //      llvm::object::createBinary(*llvm::MemoryBuffer::getMemBuffer(StringRef(data.data(), data.size()), "", false));
  //  if (auto e = b.takeError()) {
  //    std::cout << "E=" << toString(std::move(e)) << std::endl;
  //  }

  auto elapsed = compiler::elapsedNs(start);
  polyregion::compiler::Compilation c(                          //
      std::vector<uint8_t>(objBuffer.begin(), objBuffer.end()), //
      std::string(asmBuffer.begin(), asmBuffer.end()),          //
      {{compiler::nowMs(), "llvm_to_obj", elapsed}},            //
      ""                                                        //
  );

  return c;
}
