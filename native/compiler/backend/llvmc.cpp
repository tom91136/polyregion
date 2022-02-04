#include "llvmc.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
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
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <iostream>

using namespace polyregion::backend;

// FIXME this is horrible, NOT THREAD SAFE
static std::unique_ptr<llvm::TargetMachine> LLVMCurrentMachine;
static std::string LLVMFeaturesStr;
static std::string LLVMCPUStr;

static void setupTargetMachine() {
  using namespace llvm;

  std::string CPUStr = sys::getHostCPUName().str();

  StringMap<bool> HostFeatures;
  SubtargetFeatures Features;
  if (sys::getHostCPUFeatures(HostFeatures))
    for (auto &F : HostFeatures)
      Features.AddFeature(F.first(), F.second);
  std::string FeaturesStr = Features.getString();

  CodeGenOpt::Level OLvl = CodeGenOpt::Aggressive;

  //  std::string WantedTriple = Triple::normalize("x86_64-pc-linux-gnu"); // TODO fix this
  std::string SysDefaultTriple = sys::getDefaultTargetTriple();
  Triple TheTriple = Triple(SysDefaultTriple);

  std::string targetError;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple, targetError);
  if (!targetError.empty()) {
    throw std::logic_error("Target lookup failed: " + targetError);
  }

  TargetOptions Options;
  Options.AllowFPOpFusion = FPOpFusion::Fast;
  Options.UnsafeFPMath = true;

  std::cout << TheTriple.getTriple() << " : " << CPUStr << " = " << FeaturesStr << std::endl;
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine( //
      TheTriple.getTriple(),                                        //
      CPUStr,                                                       //
      FeaturesStr,                                                  //
      Options, Reloc::Model::PIC_, llvm::CodeModel::Large, OLvl));

  LLVMCurrentMachine = std::move(TM);
  LLVMFeaturesStr = FeaturesStr;
  LLVMCPUStr = CPUStr;
}

void llvmc::initialise() {

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

  setupTargetMachine();
}

const llvm::TargetMachine &llvmc::targetMachine() { return *LLVMCurrentMachine; }

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
static void setFunctionAttributes(llvm::StringRef CPU, llvm::StringRef Features, llvm::Function &F) {
  auto &Ctx = F.getContext();
  llvm::AttributeList Attrs = F.getAttributes();
  llvm::AttrBuilder NewAttrs;

  if (!CPU.empty() && !F.hasFnAttribute("target-cpu")) NewAttrs.addAttribute("target-cpu", CPU);
  if (!Features.empty()) {
    // Append the command line features to any that are already on the function.
    llvm::StringRef OldFeatures = F.getFnAttribute("target-features").getValueAsString();
    if (OldFeatures.empty()) NewAttrs.addAttribute("target-features", Features);
    else {
      llvm::SmallString<256> Appended(OldFeatures);
      Appended.push_back(',');
      Appended.append(Features);
      NewAttrs.addAttribute("target-features", Appended);
    }
  }

  //  if (FramePointerUsageView->getNumOccurrences() > 0 &&
  //      !F.hasFnAttribute("frame-pointer")) {
  //    if (getFramePointerUsage() == FramePointerKind::All)
  //      NewAttrs.addAttribute("frame-pointer", "all");
  //    else if (getFramePointerUsage() == FramePointerKind::NonLeaf)
  //      NewAttrs.addAttribute("frame-pointer", "non-leaf");
  //    else if (getFramePointerUsage() == FramePointerKind::None)
  //      NewAttrs.addAttribute("frame-pointer", "none");
  //  }
  //  if (DisableTailCallsView->getNumOccurrences() > 0)
  //    NewAttrs.addAttribute("disable-tail-calls",
  //                          toStringRef(getDisableTailCalls()));
  //  if (getStackRealign())
  //    NewAttrs.addAttribute("stackrealign");
  //
  //  HANDLE_BOOL_ATTR(EnableUnsafeFPMathView, "unsafe-fp-math");
  //  HANDLE_BOOL_ATTR(EnableNoInfsFPMathView, "no-infs-fp-math");
  //  HANDLE_BOOL_ATTR(EnableNoNaNsFPMathView, "no-nans-fp-math");
  //  HANDLE_BOOL_ATTR(EnableNoSignedZerosFPMathView, "no-signed-zeros-fp-math");

  //  if (DenormalFPMathView->getNumOccurrences() > 0 &&
  //      !F.hasFnAttribute("denormal-fp-math")) {
  //    llvm::DenormalMode::DenormalModeKind DenormKind = getDenormalFPMath();
  //
  //    // FIXME: Command line flag should expose separate input/output modes.
  //    NewAttrs.addAttribute("denormal-fp-math",
  //                          llvm::DenormalMode(DenormKind, DenormKind).str());
  //  }
  //
  //  if (DenormalFP32MathView->getNumOccurrences() > 0 &&
  //      !F.hasFnAttribute("denormal-fp-math-f32")) {
  //    // FIXME: Command line flag should expose separate input/output modes.
  //    llvm::DenormalMode::DenormalModeKind DenormKind = getDenormalFP32Math();
  //
  //    NewAttrs.addAttribute(
  //        "denormal-fp-math-f32",
  //        llvm::DenormalMode(DenormKind, DenormKind).str());
  //  }
  //
  //  if (TrapFuncNameView->getNumOccurrences() > 0)
  //    for (auto &B : F)
  //      for (auto &I : B)
  //        if (auto *Call = dyn_cast<llvm::CallInst>(&I))
  //          if (const auto *F = Call->getCalledFunction())
  //            if (F->getIntrinsicID() == llvm::Intrinsic::debugtrap ||
  //                F->getIntrinsicID() == llvm::Intrinsic::trap)
  //              Call->addAttribute(
  //                  llvm::AttributeList::FunctionIndex,
  //                  llvm::Attribute::get(Ctx, "trap-func-name", getTrapFuncName()));

  // Let NewAttrs override Attrs.
  F.setAttributes(Attrs.addAttributes(Ctx, llvm::AttributeList::FunctionIndex, NewAttrs));
}

polyregion::compiler::Compilation llvmc::compileModule(bool emitDisassembly,            //
                                                       std::unique_ptr<llvm::Module> M, //
                                                       llvm::LLVMContext &Context) {
  auto start = compiler::nowMono();

  M->setDataLayout(LLVMCurrentMachine->createDataLayout());

  for (llvm::Function &F : M->functions())
    setFunctionAttributes(LLVMCPUStr, LLVMFeaturesStr, F);

  // AddOptimizationPasses
  llvm::PassManagerBuilder B;
  B.NewGVN = true;
  B.SizeLevel = 0;
  B.OptLevel = 3;
  B.LoopVectorize = true;
  B.SLPVectorize = true;
  //  B.RerollLoops = true;
  //  B.LoopsInterleaved = true;

  auto doCodegen =
      [&](llvm::Module &m,
          const std::function<void(llvm::LLVMTargetMachine &, llvm::legacy::PassManager &, llvm::MCContext &)> &f) {
        llvm::legacy::PassManager PM;
        // XXX we have no rtti here so no dynamic cast
        auto &LLVMTM =
            static_cast<llvm::LLVMTargetMachine &>( // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
                *LLVMCurrentMachine);
        auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM); // pass manager takes owner of this
        PM.add(MMIWP);
        PM.add(createTargetTransformInfoWrapperPass(LLVMCurrentMachine->getTargetIRAnalysis()));
        llvm::TargetPassConfig *PassConfig = LLVMTM.createPassConfig(PM);
        // Set PassConfig options provided by TargetMachine.
        PassConfig->setDisableVerify(true);
        PM.add(PassConfig);

        // PM done

        llvm::legacy::FunctionPassManager FNP(&m);
        FNP.add(createTargetTransformInfoWrapperPass(LLVMCurrentMachine->getTargetIRAnalysis()));

        LLVMCurrentMachine->adjustPassManager(B);
        B.populateFunctionPassManager(FNP);
        B.populateModulePassManager(PM);

        FNP.doInitialization();
        for (llvm::Function &func : *M) {
          FNP.run(func);
        }
        FNP.doFinalization();

        if (PassConfig->addISelPasses()) throw std::logic_error("No ISEL");
        PassConfig->addMachinePasses();
        PassConfig->setInitialized();

        if (llvm::TargetPassConfig::willCompleteCodeGenPipeline()) {
          f(LLVMTM, PM, MMIWP->getMMI().getContext());
        }
        PM.run(m);
      };

  llvm::SmallVector<char, 0> asmBuffer;
  llvm::raw_svector_ostream asmStream(asmBuffer);
  doCodegen(*llvm::CloneModule(*M), [&](auto &tm, auto &pm, auto &ctx) {
    tm.addAsmPrinter(pm, asmStream, nullptr, llvm::CodeGenFileType::CGFT_AssemblyFile, ctx);
  });

  llvm::SmallVector<char, 0> objBuffer;
  llvm::raw_svector_ostream objStream(objBuffer);
  doCodegen(*llvm::CloneModule(*M), [&](auto &tm, auto &pm, auto &ctx) {
    tm.addAsmPrinter(pm, objStream, nullptr, llvm::CodeGenFileType::CGFT_ObjectFile, ctx);
  });

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

  //  std::cout << "Done = "
  //            << " b=" << asmBuffer.size() << std::endl;

  auto elapsed = compiler::elapsedNs(start);
  polyregion::compiler::Compilation c(                                                                //
      std::vector<uint8_t>(objBuffer.begin(), objBuffer.end()),                                       //
      {{compiler::nowMs(), elapsed, "llvm_to_obj", std::string(asmBuffer.begin(), asmBuffer.end())}}, //
      ""                                                                                              //
  );

  return c;
}
