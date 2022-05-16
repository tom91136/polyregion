#include "llvmc.h"

#include "compiler.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

// /home/tom/polyregion/native/cmake-build-debug-clang/_deps/llvm-src/llvm-14.0.3.src/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h

#include <iostream>

using namespace polyregion::backend;

llvm::Triple llvmc::defaultHostTriple() { return llvm::Triple(llvm::sys::getProcessTriple()); }

const llvm::Target &llvmc::targetFromTriple(const llvm::Triple &triple) {
  llvm::Triple triple0 = triple; // XXX lookup might modify triple (!?)
  std::string targetError;
  auto TheTarget = llvm::TargetRegistry::lookupTarget("", triple0, targetError);
  if (!targetError.empty()) {
    throw std::logic_error("Target lookup failed: " + targetError);
  }
  assert(TheTarget && "NULL target");
  return *TheTarget;
}

const llvmc::CpuInfo &llvmc::hostCpuInfo() {
  // XXX This is cached because host CPU can never change (hopefully!) at runtime.
  static std::optional<llvmc::CpuInfo> hostCpuInfo = {};
  if (!hostCpuInfo) {
    llvm::StringMap<bool> HostFeatures;
    llvm::SubtargetFeatures Features;
    if (llvm::sys::getHostCPUFeatures(HostFeatures))
      for (auto &F : HostFeatures)
        Features.AddFeature(F.first(), F.second);
    hostCpuInfo = {.uArch = llvm::sys::getHostCPUName().str(), .features = Features.getString()};
  }
  assert(hostCpuInfo && "Host CPU info not valid");
  return *hostCpuInfo;
}

std::unique_ptr<llvm::TargetMachine> llvmc::targetMachineFromTarget(const TargetInfo &info) {
  return std::unique_ptr<llvm::TargetMachine>(info.target.createTargetMachine( //
      info.triple.str(),                                                       //
      info.cpu.uArch,                                                          //
      info.cpu.features,                                                       //
      {}, {}));
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
}

// const llvm::TargetMachine &llvmc::targetMachine() { return *LLVMCurrentMachine; }

/// Set function attributes of function \p F based on CPU, Features, and command
/// line flags.
static void setFunctionAttributes(llvm::StringRef CPU, llvm::StringRef Features, llvm::Function &F) {
  auto &Ctx = F.getContext();
  llvm::AttributeList Attrs = F.getAttributes();
  llvm::AttrBuilder NewAttrs(Ctx);

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

  llvm::DenormalMode::DenormalModeKind DenormKind = llvm::DenormalMode::DenormalModeKind::IEEE;
  NewAttrs.addAttribute("denormal-fp-math", llvm::DenormalMode(DenormKind, DenormKind).str());
  llvm::DenormalMode::DenormalModeKind DenormKindF32 = llvm::DenormalMode::DenormalModeKind::Invalid;
  NewAttrs.addAttribute("denormal-fp-math-f32", llvm::DenormalMode(DenormKind, DenormKind).str());

  NewAttrs.addAttribute("unsafe-fp-math", "true");
  NewAttrs.addAttribute("no-infs-fp-math", "true");
  NewAttrs.addAttribute("no-signed-zeros-fp-math", "true");
  NewAttrs.addAttribute("amdgpu-early-inline-all", "true");
  NewAttrs.addAttribute("amdgpu-function-calls", "true");

  // Let NewAttrs override Attrs.
  F.setAttributes(Attrs.addFnAttributes(Ctx, NewAttrs));
}

polyregion::compiler::Compilation llvmc::compileModule(const TargetInfo &info, bool emitDisassembly,
                                                       std::unique_ptr<llvm::Module> M, llvm::LLVMContext &Context) {
  auto start = compiler::nowMono();

  llvm::TargetOptions Options;
  Options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  Options.UnsafeFPMath = true;

  llvm::CodeGenOpt::Level OLvl = llvm::CodeGenOpt::Aggressive;

  std::unique_ptr<llvm::TargetMachine> TM(info.target.createTargetMachine( //
      info.triple.str(),                                                   //
      info.cpu.uArch,                                                      //
      info.cpu.features,                                                   //
      Options, llvm::Reloc::Model::PIC_, llvm::CodeModel::Large, OLvl));

  if (M->getDataLayout().isDefault()) {
    M->setDataLayout(TM->createDataLayout());
  }

  for (llvm::Function &F : M->functions())
    setFunctionAttributes(info.cpu.uArch, info.cpu.features, F);

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
                *TM);
        auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM); // pass manager takes owner of this
        PM.add(MMIWP);
        PM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
        llvm::TargetPassConfig *PassConfig = LLVMTM.createPassConfig(PM);
        // Set PassConfig options provided by TargetMachine.
        PassConfig->setDisableVerify(true);
        PM.add(PassConfig);
        // PM done

        llvm::legacy::FunctionPassManager FNP(&m);
        FNP.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

        TM->adjustPassManager(B);
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
    //    tm.addAsmPrinter(pm, objStream, nullptr, llvm::CodeGenFileType::CGFT_ObjectFile, ctx);
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
