#include "llvmc.h"

#include "clspv.h"
#include "compiler.h"
#include "lld_lite.h"
#include "llvm_utils.hpp"
#include "utils.hpp"

#include "spirv-tools/libspirv.h"
#include "spirv-tools/libspirv.hpp"

#include "spirv-tools/optimizer.hpp"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
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
#include "llvm/Object/ELF.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

// #include "LLVMSPIRVLib.h"

#include <fstream>
#include <iostream>

using namespace polyregion;
using namespace ::backend;

llvm::Triple llvmc::defaultHostTriple() { return llvm::Triple(llvm::sys::getProcessTriple()); }

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

const llvm::Target *llvmc::targetFromTriple(const llvm::Triple &triple) {
  llvm::Triple triple0 = triple; // XXX lookup might modify triple (!?)
  std::string targetError;
  auto TheTarget = llvm::TargetRegistry::lookupTarget("", triple0, targetError);
  if (!targetError.empty() || !TheTarget) {
    throw std::logic_error("Target lookup failed: " + targetError);
  }
  return TheTarget;
}

llvm::DataLayout llvmc::TargetInfo::resolveDataLayout() const {
  if (layout) return *layout;
  else if (target) {
    return target
        ->createTargetMachine( //
            triple.str(),      //
            cpu.uArch,         //
            cpu.features,      //
            {}, {})
        ->createDataLayout();
  } else {
    throw std::logic_error(triple.str() + " does not have a known layout or a registered LLVM target.");
  }
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

  //    initializeLLVMToSPIRVLegacyPass(*r);
  //    initializeOCLToSPIRVLegacyPass(*r);
  //    initializeOCLTypeToSPIRVLegacyPass(*r);
  //    initializeSPIRVLowerBoolLegacyPass(*r);
  //    initializeSPIRVLowerConstExprLegacyPass(*r);
  //    initializeSPIRVLowerOCLBlocksLegacyPass(*r);
  //    initializeSPIRVLowerMemmoveLegacyPass(*r);
  //    initializeSPIRVLowerSaddWithOverflowLegacyPass(*r);
  //    initializeSPIRVRegularizeLLVMLegacyPass(*r);
  //    initializeSPIRVToOCL12LegacyPass(*r);
  //    initializeSPIRVToOCL20LegacyPass(*r);
  //    initializePreprocessMetadataLegacyPass(*r);
  //    initializeSPIRVLowerBitCastToNonStandardTypeLegacyPass(*r);
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

polyast::Pair<polyast::Opt<std::string>, std::string> llvmc::verifyModule(llvm::Module &mod) {
  std::string err;
  llvm::raw_string_ostream errOut(err);
  if (llvm::verifyModule(mod, &errOut)) {
    return {errOut.str(), "(module failed verification)"};
  } else {
    return {{}, "(module passed verification)"};
  }
}

static std::string module2Ir(const llvm::Module &m) {
  std::string ir;
  llvm::raw_string_ostream irOut(ir);
  m.print(irOut, nullptr);
  return ir;
}

// See
// https://github.com/pytorch/pytorch/blob/6d4d9840cd4f18232e201cbcd843ea4f6cb4aabb/torch/csrc/jit/tensorexpr/llvm_codegen.cpp#L2466
static void optimise(llvm::TargetMachine &TM, llvm::Module &M, llvm::OptimizationLevel &level) {
  // Create the analysis managers.
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  llvm::PassBuilder PB(&TM);

  TM.registerPassBuilderCallbacks(PB);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(level);
  llvm::FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(level, llvm::ThinOrFullLTOPhase::None);

  FAM.registerPass([&] { return TM.getTargetIRAnalysis(); });

  FPM.addPass(llvm::LoopVectorizePass());
  FPM.addPass(llvm::SLPVectorizerPass());

  FPM.addPass(llvm::DCEPass());
  MPM.addPass(llvm::AlwaysInlinerPass());

  MPM.run(M, MAM);
  for (auto &FF : M) {
    if (!FF.empty()) {
      FPM.run(FF, FAM);
    }
  }
}

polyast::CompileResult llvmc::compileModule(const TargetInfo &info, const polyast::OptLevel &opt, bool emitDisassembly,
                                          std::unique_ptr<llvm::Module> M) {
  auto start = compiler::nowMono();

  auto useUnsafeMath = opt == polyast::OptLevel::Ofast;
  llvm::TargetOptions options;
  options.AllowFPOpFusion = useUnsafeMath ? llvm::FPOpFusion::Fast : llvm::FPOpFusion::Standard;
  options.UnsafeFPMath = useUnsafeMath;
  options.NoInfsFPMath = useUnsafeMath;
  options.NoNaNsFPMath = useUnsafeMath;
  options.NoTrappingFPMath = useUnsafeMath;
  options.NoSignedZerosFPMath = useUnsafeMath;

  llvm::CodeGenOpt::Level genOpt;
  switch (opt) {
    case polyast::OptLevel::O0: genOpt = llvm::CodeGenOpt::None; break;
    case polyast::OptLevel::O1: genOpt = llvm::CodeGenOpt::Less; break;
    case polyast::OptLevel::O2: genOpt = llvm::CodeGenOpt::Default; break;
    case polyast::OptLevel::O3: // fallthrough
    case polyast::OptLevel::Ofast: genOpt = llvm::CodeGenOpt::Aggressive; break;
  }

  // We have two groups of targets:
  //  * Ones that are enabled via LLVM_TARGETS_TO_BUILD, these will have a llvm::Target and we can create a TargetMachine from it
  //  * Targets that aren't registered like SPIRV, we know the data layout of these but nothing else.

  if (!info.target && !info.layout) {
    throw std::logic_error(info.triple.str() + " has no known data layout or registered LLVM target.");
  }

  for (llvm::Function &F : M->functions())
    setFunctionAttributes(info.cpu.uArch, info.cpu.features, F);

  llvm::OptimizationLevel optLevel;
  switch (opt) {
    case polyast::OptLevel::O0: optLevel = llvm::OptimizationLevel::O0; break;
    case polyast::OptLevel::O1: optLevel = llvm::OptimizationLevel::O1; break;
    case polyast::OptLevel::O2: optLevel = llvm::OptimizationLevel::O2; break;
    case polyast::OptLevel::O3: // fallthrough
    case polyast::OptLevel::Ofast: optLevel = llvm::OptimizationLevel::O3; break;
  }

  auto mkLLVMTargetMachine = [](const TargetInfo &info, const llvm::TargetOptions &options, const llvm::CodeGenOpt::Level &level) {
    // XXX We *MUST* use the large code model as we will be ingesting the object later with RuntimeDyld
    // The code model here has nothing to do with the actual object code size, it's about controlling the relocation.
    // See https://stackoverflow.com/questions/40493448/what-does-the-codemodel-in-clang-llvm-refer-to
    auto tm = static_cast<llvm::LLVMTargetMachine *>(info.target->createTargetMachine( //
        info.triple.str(),                                                             //
        info.cpu.uArch,                                                                //
        info.cpu.features,                                                             //
        options, llvm::Reloc::Model::PIC_, llvm::CodeModel::Large, level));
    return std::unique_ptr<llvm::LLVMTargetMachine>(tm);
  };

  auto bindLLVMTargetMachineDataLayout = [&](llvm::LLVMTargetMachine &TM, llvm::Module &M) {
    if (M.getDataLayout().isDefault()) {
      M.setDataLayout(TM.createDataLayout());
    }
  };

  auto mkLLVMTargetMachineArtefact = [&](llvm::LLVMTargetMachine &TM,                     //
                                         const std::optional<llvm::CodeGenFileType> &tpe, //
                                         const llvm::Module &m0,                          //
                                         std::vector<polyast::CompileEvent> &events, bool emplaceEvent) {
    auto m = llvm::CloneModule(m0);
    auto optPassStart = compiler::nowMono();

    optimise(TM, *m, optLevel);

    llvm::legacy::PassManager PM;
    auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&TM); // pass manager takes owner of this
    PM.add(MMIWP);
    PM.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
    llvm::TargetPassConfig *PassConfig = TM.createPassConfig(PM);
    // Set PassConfig options provided by TargetMachine.
    PassConfig->setDisableVerify(true);
    PM.add(PassConfig);
    // PM done

    auto iselPassStart = compiler::nowMono();
    if (PassConfig->addISelPasses()) throw std::logic_error("No ISEL");
    PassConfig->addMachinePasses();
    PassConfig->setInitialized();

    llvm::SmallVector<char, 0> objBuffer;
    llvm::raw_svector_ostream objStream(objBuffer);

    if (tpe) {
      if (llvm::TargetPassConfig::willCompleteCodeGenPipeline()) {
        TM.addAsmPrinter(PM, objStream, nullptr, *tpe, MMIWP->getMMI().getContext());
      }
    }
    PM.run(*m);
    if (emplaceEvent) {
      events.emplace_back(compiler::nowMs(), compiler::elapsedNs(optPassStart), "llvm_to_obj_opt", module2Ir(*m));
    }
    return std::make_tuple(std::move(m), objBuffer, compiler::nowMs(), compiler::elapsedNs(iselPassStart));
  };

  auto objectSize = [](const llvm::SmallVector<char, 0> &xs) {
    return std::to_string(static_cast<float>(xs.size_in_bytes()) / 1024) + "KiB";
  };

  std::vector<polyast::CompileEvent> events;

  switch (info.triple.getOS()) {
    case llvm::Triple::AMDHSA: {
      auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
      bindLLVMTargetMachineDataLayout(*llvmTM, *M);
      // We need to link the object file for AMDGPU at this stage to get a working ELF binary.
      // This can only be done with LLD so just do it here after compiling.
      auto [_, object, objectStart, objectElapsed] = //
          mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::CGFT_ObjectFile, *M, events, true);
      events.emplace_back(objectStart, objectElapsed, "llvm_to_obj", objectSize(object));
      if (emitDisassembly) {
        auto [m, assembly, assemblyStart, assemblyElapsed] = //
            mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::CGFT_AssemblyFile, *M, events, false);
        events.emplace_back(assemblyStart, assemblyElapsed, "llvm_to_asm", std::string(assembly.begin(), assembly.end()));
      }
      llvm::StringRef objectString(object.begin(), object.size());
      llvm::MemoryBufferRef kernelObject(objectString, "kernel.hsaco");
      // XXX Don't strip AMDGCN ELFs as hipModuleLoad will report "Invalid ptx". GC and optimisation is fine.
      auto linkerStart = compiler::nowMono();
      auto [err, result] = backend::lld_lite::linkElf({"-shared", "--gc-sections", "-O3"}, {kernelObject});
      auto linkerElapsed = compiler::elapsedNs(linkerStart);
      events.emplace_back(compiler::nowMs(), linkerElapsed, "lld_link_amdgpu", "");
      if (!result) { // linker failed
        return {{}, {info.cpu.uArch}, events, {}, "Linker did not complete normally: " + err.value_or("(no message reported)")};
      } else { // linker succeeded, still report any stdout to as message
        return {std::vector<int8_t>(result->begin(), result->end()), {info.cpu.uArch}, events, {}, err.value_or("")};
      }
    }
    case llvm::Triple::CUDA: {
      auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
      bindLLVMTargetMachineDataLayout(*llvmTM, *M);
      // NVIDIA's documentation only supports up-to PTX generation and ingestion via the CUDA driver API, so we can't
      // assemble the PTX to a CUBIN (SASS). Given that PTX ingestion is supported, we just generate that for now.
      // XXX ignore emitDisassembly here as PTX *is* the binary
      auto [_, ptx, ptxStart, ptxElapsed] = //
          mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::CGFT_AssemblyFile, *M, events, true);
      events.emplace_back(ptxStart, ptxElapsed, "llvm_to_ptx", std::string(ptx.begin(), ptx.end()));
      return {std::vector<int8_t>(ptx.begin(), ptx.end()), {info.cpu.uArch}, events, {}, ""};
    }
    default:
      switch (info.triple.getArch()) {
        case llvm::Triple::ArchType::spirv32:
        case llvm::Triple::ArchType::spirv64: {

          auto opt0Start = compiler::nowMono();

          events.emplace_back(compiler::nowMs(), compiler::elapsedNs(opt0Start), "opt0", module2Ir(*M));

          auto clspvStart = compiler::nowMono();

          llvm::SmallVector<char, 0> objBuffer;
          llvm::raw_svector_ostream objStream(objBuffer);
          clspv::RunPassPipeline(*M, '3', &objStream);

          events.emplace_back(compiler::nowMs(), compiler::elapsedNs(clspvStart), "clspv", module2Ir(*M));

          auto validateStart = compiler::nowMono();
          auto [maybeVerifyErr, verifyMsg] = llvmc::verifyModule(*M);
          events.emplace_back(compiler::nowMs(), compiler::elapsedNs(validateStart), "clspv_validate", verifyMsg);

          auto optPassStart = compiler::nowMono();
          std::vector<char> binary(objBuffer.begin(), objBuffer.end());

          std::vector<uint32_t> spvRaw((binary.size() + 3) / 4, 0);
          std::memcpy(spvRaw.data(), binary.data(), binary.size());

          spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
          optimizer.RegisterPass(spvtools::CreateStripNonSemanticInfoPass());
          optimizer.RegisterPerformancePasses();
          std::vector<uint32_t> nonSemanticSpv;
          optimizer.Run(spvRaw.data(), spvRaw.size(), &nonSemanticSpv);

          //          std::ofstream fileS("foo.spv", std::ios::binary);
          //          fileS.write(reinterpret_cast<const char *>(nonSemanticSpv.data()), nonSemanticSpv.size() * sizeof(uint32_t));
          //          fileS.close();

          //        spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_5);
          //        std::string out;
          //        auto error = tools.Disassemble(disBuffer.data(), disBuffer.size(), &out, disOptions);
          spv_text text;
          spv_diagnostic diagnostic = nullptr;
          spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_1);
          spv_result_t error = spvBinaryToText(context, nonSemanticSpv.data(), nonSemanticSpv.size(),
                                               SPV_BINARY_TO_TEXT_OPTION_NONE |               //
                                                   SPV_BINARY_TO_TEXT_OPTION_INDENT |         //
                                                   SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | //
                                                   SPV_BINARY_TO_TEXT_OPTION_COMMENT,
                                               &text, &diagnostic);
          spvContextDestroy(context);
          if (error) {
            events.emplace_back(compiler::nowMs(), compiler::elapsedNs(optPassStart), "spirv-opt",
                                "[Error " +                                                                             //
                                    std::to_string(diagnostic->position.index) + ":" +                                  //
                                    std::to_string(diagnostic->position.line) + ":" +                                   //
                                    std::to_string(diagnostic->position.column) + "] " + std::string(diagnostic->error) //
            );
            spvDiagnosticDestroy(diagnostic);
          } else {
            std::string out(text->str, text->length);
            events.emplace_back(compiler::nowMs(), compiler::elapsedNs(optPassStart), "spirv-opt", out);
            spvTextDestroy(text);
          }

          //        auto [ptx, ptxStart, ptxElapsed] = mkArtefact(llvm::CodeGenFileType::CGFT_ObjectFile, *M, events, true);
          //        events.emplace_back(ptxStart, ptxElapsed, "llvm_to_ptx", std::string(ptx.begin(), ptx.end()));
          //        return {std::vector<char>(ptx.begin(), ptx.end()), {info.cpu.uArch}, events};

          //        SPIRV::TranslatorOpts opts;
          //        opts.shouldReplaceLLVMFmulAddWithOpenCLMad()
          //        auto spirvConvertStart = compiler::nowMono();
          //        std::string errors;
          //        std::stringstream ss;
          //        llvm::writeSpirv(M.get(), opts, ss, errors);
          //        std::string spirvBin = ss.str();
          //        events.emplace_back(compiler::nowMs(), compiler::elapsedNs(spirvConvertStart), "llvm_to_spirv",
          //                            "(" + std::to_string(spirvBin.size()) + " bytes)");
          //        std::vector<char> binary(spirvBin.begin(), spirvBin.end());

          std::vector<int8_t> convertedSpv(nonSemanticSpv.size() * sizeof(uint32_t));
          std::memcpy(convertedSpv.data(), nonSemanticSpv.data(), convertedSpv.size());

          return {convertedSpv, {info.cpu.uArch}, events, {}, ""};
        }
        default: {

          auto features = polyregion::split(info.cpu.features, ',');
          polyregion::llvm_shared::collectCPUFeatures(info.cpu.uArch, info.triple.getArch(), features);

          auto llvmTM = mkLLVMTargetMachine(info, options, genOpt);
          bindLLVMTargetMachineDataLayout(*llvmTM, *M);
          auto [_, object, objectStart, objectElapsed] =
              mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::CGFT_ObjectFile, *M, events, true);
          events.emplace_back(objectStart, objectElapsed, "llvm_to_obj", objectSize(object));

          std::vector<int8_t> binary(object.begin(), object.end());
          if (emitDisassembly) {
            auto [_, assembly, assemblyStart, assemblyElapsed] =
                mkLLVMTargetMachineArtefact(*llvmTM, llvm::CodeGenFileType::CGFT_AssemblyFile, *M, events, false);
            events.emplace_back(assemblyStart, assemblyElapsed, "llvm_to_asm", std::string(assembly.begin(), assembly.end()));
          }
          return {binary, features, events, {}, ""};
        }
      }
  }
}
