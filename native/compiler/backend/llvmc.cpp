#include "llvmc.h"

#include "compiler.h"
#include "lld_lite.h"
#include "utils.hpp"

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
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Pass.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm_utils.hpp"

#include <iostream>

using namespace polyregion;
using namespace ::backend;

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
//  initializeEntryExitInstrumenterPass(*r);
//  initializePostInlineEntryExitInstrumenterPass(*r);
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
  //  NewAttrs.addAttribute("amdgpu-early-inline-all", "true");
  //  NewAttrs.addAttribute("amdgpu-function-calls", "true");

  // Let NewAttrs override Attrs.
  F.setAttributes(Attrs.addFnAttributes(Ctx, NewAttrs));
}

compiler::Compilation llvmc::compileModule(const TargetInfo &info, const compiler::Opt &opt, bool emitDisassembly,
                                           std::unique_ptr<llvm::Module> M, llvm::LLVMContext &Context) {
  auto start = compiler::nowMono();

  auto useUnsafeMath = opt == compiler::Opt::Ofast;
  llvm::TargetOptions options;
  options.AllowFPOpFusion = useUnsafeMath ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Standard;
  options.UnsafeFPMath = useUnsafeMath;
  options.NoInfsFPMath = useUnsafeMath;
  options.NoNaNsFPMath = useUnsafeMath;
  options.NoTrappingFPMath = useUnsafeMath;
  options.NoSignedZerosFPMath = useUnsafeMath;

  llvm::CodeGenOpt::Level genOpt;
  switch (opt) {
    case compiler::Opt::O0: genOpt = llvm::CodeGenOpt::None; break;
    case compiler::Opt::O1: genOpt = llvm::CodeGenOpt::Less; break;
    case compiler::Opt::O2: genOpt = llvm::CodeGenOpt::Default; break;
    case compiler::Opt::O3: // fallthrough
    case compiler::Opt::Ofast: genOpt = llvm::CodeGenOpt::Aggressive; break;
  }

  // XXX We *MUST* use the large code model as we will be ingesting the object later with RuntimeDyld
  // The code model here has nothing to do with the actual object code size, it's about controlling the relocation.
  // See https://stackoverflow.com/questions/40493448/what-does-the-codemodel-in-clang-llvm-refer-to
  std::unique_ptr<llvm::TargetMachine> TM(info.target.createTargetMachine( //
      info.triple.str(),                                                   //
      info.cpu.uArch,                                                      //
      info.cpu.features,                                                   //
      options, llvm::Reloc::Model::PIC_, llvm::CodeModel::Large, genOpt));

  if (M->getDataLayout().isDefault()) {
    M->setDataLayout(TM->createDataLayout());
  }

  for (llvm::Function &F : M->functions())
    setFunctionAttributes(info.cpu.uArch, info.cpu.features, F);

  // AddOptimizationPasses
  llvm::PassManagerBuilder B;

  switch (opt) {
    case compiler::Opt::O0: B.OptLevel = 0; break;
    case compiler::Opt::O1: B.OptLevel = 1; break;
    case compiler::Opt::O2: B.OptLevel = 2; break;
    case compiler::Opt::O3: // fallthrough
    case compiler::Opt::Ofast: B.OptLevel = 3; break;
  }

  B.NewGVN = true;
  B.SizeLevel = 0;
  B.LoopVectorize = B.OptLevel >= 2;
  B.SLPVectorize = B.OptLevel >= 2;
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

  auto mkArtefact = [&](const llvm::CodeGenFileType &tpe) {
    auto start = compiler::nowMono();
    auto timestamp = compiler::nowMs();
    llvm::SmallVector<char, 0> objBuffer;
    llvm::raw_svector_ostream objStream(objBuffer);
    doCodegen(*llvm::CloneModule(*M),
              [&](auto &tm, auto &pm, auto &ctx) { tm.addAsmPrinter(pm, objStream, nullptr, tpe, ctx); });
    return std::make_tuple(objBuffer, timestamp, compiler::elapsedNs(start));
  };

  //  llvm::legacy::PassManager pass;
  //  llvm::SmallString<8> dataObj;
  //  llvm::raw_svector_ostream destObj(dataObj);
  //  TM->addPassesToEmitFile(pass, destObj, nullptr, llvm::CGFT_ObjectFile);
  //  pass.run(*llvm::CloneModule(*M));
  //  std::string obj(dataObj.begin(), dataObj.end());

  auto objectSize = [](const llvm::SmallVector<char, 0> &xs) {
    return std::to_string(static_cast<float>(xs.size_in_bytes()) / 1024) + "KiB";
  };

  switch (info.triple.getOS()) {
    case llvm::Triple::AMDHSA: {
      // We need to link the object file for AMDGPU at this stage to get a working ELF binary.
      // This can only be done with LLD so just do it here after compiling.

      std::vector<compiler::Event> events;

      auto [object, objectStart, objectElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_ObjectFile);
      events.emplace_back(objectStart, objectElapsed, "llvm_to_obj", objectSize(object));
      if (emitDisassembly) {
        auto [assembly, assemblyStart, assemblyElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_AssemblyFile);
        events.emplace_back(assemblyStart, assemblyElapsed, "llvm_to_asm",
                            std::string(assembly.begin(), assembly.end()));
      }

      llvm::StringRef objectString(object.begin(), object.size());
      lld::elf::ObjFile<llvm::object::ELF64LE> kernelObject(llvm::MemoryBufferRef(objectString, ""), "kernel.hsaco");
      // XXX Don't strip AMDGCN ELFs as hipModuleLoad will report "Invalid ptx". GC and optimisation is fine.
      auto linkerStart = compiler::nowMono();
      auto [err, result] = backend::lld_lite::link({"-shared", "--gc-sections", "-O3"}, {&kernelObject});
      auto linkerElapsed = compiler::elapsedNs(linkerStart);
      events.emplace_back(compiler::nowMs(), linkerElapsed, "lld_link_amdgpu", "");
      if (!result) { // linker failed
        return {
            {}, {info.cpu.uArch}, events, "Linker did not complete normally: " + err.value_or("(no message reported)")};
      } else { // linker succeeded, still report any stdout to as message
        return {std::vector<char>(result->begin(), result->end()), {info.cpu.uArch}, events, err.value_or("")};
      }
    }
    case llvm::Triple::CUDA: {
      // NVIDIA's documentation only supports up-to PTX generation and ingestion via the CUDA driver API, so we can't
      // assemble the PTX to a CUBIN (SASS). Given that PTX ingestion is supported, we just generate that for now.
      // XXX ignore emitDisassembly here as PTX *is* the binary
      auto [ptx, ptxStart, ptxElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_AssemblyFile);
      return {std::vector<char>(ptx.begin(), ptx.end()),
              {info.cpu.uArch},
              {{ptxStart, ptxElapsed, "llvm_to_ptx", std::string(ptx.begin(), ptx.end())}}};
    }
    default:


      auto features = polyregion::split(info.cpu.features, ',');
      polyregion::llvm_shared::collectCPUFeatures(info.cpu.uArch, info.triple.getArch(), features);

      auto [object, objectStart, objectElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_ObjectFile);
      std::vector<char> binary(object.begin(), object.end());
      if (emitDisassembly) {
        auto [assembly, assemblyStart, assemblyElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_AssemblyFile);
        return {binary,
                features,
                {{objectStart, objectElapsed, "llvm_to_obj", objectSize(object)},
                 {assemblyStart, assemblyElapsed, "llvm_to_asm", std::string(assembly.begin(), assembly.end())}}};
      } else
        return {binary, features, {{objectStart, objectElapsed, "llvm_to_obj", objectSize(object)}}};
  }
}
