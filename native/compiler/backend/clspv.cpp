#include "clspv.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "clspv/AddressSpace.h"
#include "clspv/Option.h"
#include "clspv/Passes.h"
#include "clspv/Sampler.h"
//#include "clspv/clspv64_builtin_library.h"
//#include "clspv/clspv_builtin_library.h"
//#include "clspv/opencl_builtins_header.h"

#include "Builtins.h"
#include "Constants.h"
#include "Passes.h"
#include "Types.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>

int polyregion::backend::clspv::RunPassPipeline(llvm::Module &M, char OptimizationLevel, llvm::raw_svector_ostream *binaryStream) {




  llvm::cl::ResetAllOptionOccurrences();

  std::vector<const char *> args = {
      "",
      "-show-producer-ir",
//      "-print-after-all",
      "-pod-ubo=true",
      "-cl-kernel-arg-info=true"
  };

  llvm::cl::ParseCommandLineOptions(args.size(), args.data());



  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassInstrumentationCallbacks PIC;
  llvm::StandardInstrumentations si(M.getContext(), false /*DebugLogging*/);
  ::clspv::RegisterClspvPasses(&PIC);
  si.registerCallbacks(PIC, &fam);
  llvm::PassBuilder pb(nullptr, llvm::PipelineTuningOptions(), std::nullopt,
                       &PIC);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager pm;
  llvm::FunctionPassManager fpm;

  switch (OptimizationLevel) {
    case '0':
    case '1':
    case '2':
    case '3':
    case 's':
    case 'z':
      break;
    default:
      llvm::errs() << "Unknown optimization level -O" << OptimizationLevel
                   << " specified!\n";
      return -1;
  }

  llvm::OptimizationLevel level;
  switch (OptimizationLevel) {
    case '0':
      level = llvm::OptimizationLevel::O0;
      break;
    case '1':
      level = llvm::OptimizationLevel::O1;
      break;
    case '2':
      level = llvm::OptimizationLevel::O2;
      break;
    case '3':
      level = llvm::OptimizationLevel::O3;
      break;
    case 's':
      level = llvm::OptimizationLevel::Os;
      break;
    case 'z':
      level = llvm::OptimizationLevel::Oz;
      break;
    default:
      break;
  }

  // Run the following optimizations prior to the standard LLVM pass pipeline.
  pb.registerPipelineStartEPCallback([](llvm::ModulePassManager &pm,
                                        llvm::OptimizationLevel level) {
    pm.addPass(::clspv::NativeMathPass());
    pm.addPass(::clspv::ZeroInitializeAllocasPass());
    pm.addPass(::clspv::AddFunctionAttributesPass());
    pm.addPass(::clspv::AutoPodArgsPass());
    pm.addPass(::clspv::DeclarePushConstantsPass());
    pm.addPass(::clspv::DefineOpenCLWorkItemBuiltinsPass());

    // RewritePackedStructsPass will rewrite packed struct types, and
    // ReplacePointerBitcastPass will lower the new packed struct type. So,
    // RewritePackedStructsPass must come before ReplacePointerBitcastPass.
    if (::clspv::Option::RewritePackedStructs()) {
      pm.addPass(::clspv::RewritePackedStructs());
    }

    if (level.getSpeedupLevel() > 0) {
      pm.addPass(::clspv::OpenCLInlinerPass());
    }

    pm.addPass(::clspv::UndoByvalPass());
    pm.addPass(::clspv::UndoSRetPass());

    // Handle physical pointer arguments by converting them to POD integers,
    // and update all uses to bitcast them to a pointer first. This allows these
    // arguments to be handled in later passes as if they were regular PODs.
    if (::clspv::Option::PhysicalStorageBuffers()) {
      pm.addPass(::clspv::PhysicalPointerArgsPass());
    }

    pm.addPass(llvm::createModuleToFunctionPassAdaptor(
        llvm::InferAddressSpacesPass(::clspv::AddressSpace::Generic)));

    // We need to run mem2reg and inst combine early because some of our passes
    // (e.g. ThreeElementVectorLowering and InlineFuncWithBitCastArgsPass)
    // cannot handle the pattern:
    //
    //   %1 = alloca i32 1
    //        store <something> %1
    //   %2 = bitcast float* %1
    //   %3 = load float %2
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass()));
    pm.addPass(::clspv::ClusterPodKernelArgumentsPass());
    // ReplaceOpenCLBuiltinPass can generate vec8 and vec16 elements. It needs
    // to be before the potential LongVectorLoweringPass pass.
    pm.addPass(::clspv::ReplaceOpenCLBuiltinPass());
    pm.addPass(::clspv::FixupBuiltinsPass());
    pm.addPass(::clspv::ThreeElementVectorLoweringPass());

    if (::clspv::Option::HackLogicalPtrtoint()) {
      pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass()));
      pm.addPass(::clspv::LogicalPointerToIntPass());
    }

    // Lower longer vectors when requested. Note that this pass depends on
    // ReplaceOpenCLBuiltinPass and expects DeadCodeEliminationPass to be run
    // afterwards.
    if (::clspv::Option::LongVectorSupport()) {
      pm.addPass(::clspv::LongVectorLoweringPass());
    }

    // Try to deal with pointer bitcasts early. This can prevent problems like
    // issue #409 where LLVM is looser about access chain addressing than
    // SPIR-V. This needs to happen before instcombine and after replacing
    // OpenCL builtins.  This run of the pass will not handle all pointer
    // bitcasts that could be handled. It should be run again after other
    // optimizations (e.g InlineFuncWithPointerBitCastArgPass).
    pm.addPass(::clspv::SimplifyPointerBitcastPass());
    pm.addPass(::clspv::ReplacePointerBitcastPass());
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::DCEPass()));

    // Hide loads from __constant address space away from instcombine.
    // This prevents us from generating select between pointers-to-__constant.
    // See https://github.com/google/clspv/issues/71
    pm.addPass(::clspv::HideConstantLoadsPass());

    pm.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::InstCombinePass()));

    pm.addPass(::clspv::InlineEntryPointsPass());
    pm.addPass(::clspv::InlineFuncWithImageMetadataGetterPass());
    pm.addPass(::clspv::InlineFuncWithPointerBitCastArgPass());
    pm.addPass(::clspv::InlineFuncWithPointerToFunctionArgPass());
    pm.addPass(::clspv::InlineFuncWithSingleCallSitePass());

    // This pass needs to be after every inlining to make sure we are capable of
    // removing every addrspacecast. It only needs to run if generic addrspace
    // is used.
    if (::clspv::Option::LanguageUsesGenericAddressSpace()) {
      pm.addPass(::clspv::LowerAddrSpaceCastPass());
    }

    // Mem2Reg pass should be run early because O0 level optimization leaves
    // redundant alloca, load and store instructions from function arguments.
    // clspv needs to remove them ahead of transformation.
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass()));

    // SROA pass is run because it will fold structs/unions that are
    // problematic on Vulkan SPIR-V away.
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(
        llvm::SROAPass(llvm::SROAOptions::PreserveCFG)));

    // InstructionCombining pass folds bitcast and gep instructions which are
    // not supported by Vulkan SPIR-V.
    pm.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::InstCombinePass()));

    pm.addPass(llvm::createModuleToFunctionPassAdaptor(
        llvm::InferAddressSpacesPass(::clspv::AddressSpace::Generic)));
  });

  // Run the following passes after the default LLVM pass pipeline.
  pb.registerOptimizerLastEPCallback([binaryStream](llvm::ModulePassManager &pm,
                                                    llvm::OptimizationLevel) {
    // No point attempting to handle freeze currently so strip them from the
    // IR.
    pm.addPass(::clspv::StripFreezePass());

    // Unhide loads from __constant address space.  Undoes the action of
    // HideConstantLoadsPass.
    pm.addPass(::clspv::UnhideConstantLoadsPass());

    pm.addPass(::clspv::UndoInstCombinePass());
    pm.addPass(::clspv::FunctionInternalizerPass());
    pm.addPass(::clspv::ReplaceLLVMIntrinsicsPass());
    // Replace LLVM intrinsics can leave dead code around.
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::DCEPass()));
    pm.addPass(::clspv::UndoBoolPass());
    pm.addPass(::clspv::UndoTruncateToOddIntegerPass());
    // StructurizeCFG requires LowerSwitch to run first.
    pm.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::LowerSwitchPass()));
    pm.addPass(
        llvm::createModuleToFunctionPassAdaptor(llvm::StructurizeCFGPass()));
    // Must be run after structurize cfg.
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(
        ::clspv::FixupStructuredCFGPass()));
    // Must be run after structured cfg fixup.
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(
        ::clspv::ReorderBasicBlocksPass()));
    pm.addPass(::clspv::UndoGetElementPtrConstantExprPass());
    pm.addPass(::clspv::SplatArgPass());
    pm.addPass(::clspv::SimplifyPointerBitcastPass());
    pm.addPass(::clspv::ReplacePointerBitcastPass());
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::DCEPass()));

    pm.addPass(::clspv::UndoTranslateSamplerFoldPass());

    if (::clspv::Option::ModuleConstantsInStorageBuffer()) {
      pm.addPass(::clspv::ClusterModuleScopeConstantVars());
    }

    pm.addPass(::clspv::ShareModuleScopeVariablesPass());
    // Specialize images before assigning descriptors to disambiguate the
    // various types.
    pm.addPass(::clspv::SpecializeImageTypesPass());
    // This should be run after LLVM and OpenCL intrinsics are replaced.
    pm.addPass(::clspv::AllocateDescriptorsPass());
    pm.addPass(llvm::VerifierPass());
    pm.addPass(::clspv::DirectResourceAccessPass());
    // Replacing pointer bitcasts can leave some trivial GEPs
    // that are easy to remove.  Also replace GEPs of GEPS
    // left by replacing indirect buffer accesses.
    pm.addPass(::clspv::SimplifyPointerBitcastPass());
    // Run after DRA to clean up parameters and help reduce the need for
    // variable pointers.
    pm.addPass(::clspv::RemoveUnusedArguments());
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::DCEPass()));

    // SPIR-V 1.4 and higher do not need to splat scalar conditions for vector
    // data.
    if (::clspv::Option::SpvVersion() < ::clspv::Option::SPIRVVersion::SPIRV_1_4) {
      pm.addPass(::clspv::SplatSelectConditionPass());
    }
    pm.addPass(::clspv::SignedCompareFixupPass());
    // This pass generates insertions that need to be rewritten.
    pm.addPass(::clspv::ScalarizePass());
    pm.addPass(::clspv::RewriteInsertsPass());
    // UBO Transformations
    if (::clspv::Option::ConstantArgsInUniformBuffer() &&
        !::clspv::Option::InlineEntryPoints()) {
      // MultiVersionUBOFunctionsPass will examine non-kernel functions with
      // UBO arguments and either multi-version them as necessary or inline
      // them if multi-versioning cannot be accomplished.
      pm.addPass(::clspv::MultiVersionUBOFunctionsPass());
      // Cleanup passes.
      // Specialization can blindly generate GEP chains that are easily
      // cleaned up by SimplifyPointerBitcastPass.
      pm.addPass(::clspv::SimplifyPointerBitcastPass());
      // RemoveUnusedArgumentsPass removes the actual UBO arguments that were
      // problematic to begin with now that they have no uses.
      pm.addPass(::clspv::RemoveUnusedArguments());
      // DCE cleans up callers of the specialized functions.
      pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::DCEPass()));
    }

    // This pass needs to run before an interation of
    // SimplifyPointerBitcastPass/ReplacePointerBitcastPass.
    pm.addPass(::clspv::LowerPrivatePointerPHIPass());

    // Last minute pointer simplification. With opaque pointers, we can often
    // end up in a situation where LLVM has simplified GEPs by removing zero
    // indices where an equivalent address would be computed. These lead to
    // situations that are awkward for clspv. The following passes canonicalize
    // GEPs into forms easier to codegen in SPIR-V, including those more likely
    // to avoid extra functionality (e.g. VariablePointers).
    pm.addPass(::clspv::SimplifyPointerBitcastPass());
    pm.addPass(::clspv::ReplacePointerBitcastPass());
    pm.addPass(::clspv::SimplifyPointerBitcastPass());

    // This pass mucks with types to point where you shouldn't rely on
    // DataLayout anymore so leave this right before SPIR-V generation.
    pm.addPass(::clspv::UBOTypeTransformPass());

    // This pass depends on the inlining of the image metadata getter from
    // InlineFuncWithImageMetadataGetterPass
    pm.addPass(::clspv::SetImageChannelMetadataPass());

    // This is needed to remove long vectors created by SROA passes. Especially
    // with vstore_half, which tends to always recreate long vectors after the
    // first iteration of the longvectorlowering pass
    if (::clspv::Option::LongVectorSupport()) {
      pm.addPass(::clspv::LongVectorLoweringPass());
    }

    pm.addPass(
        ::clspv::SPIRVProducerPass(binaryStream, false));
  });

  // Add the default optimizations for the requested optimization level.
  if (level.getSpeedupLevel() > 0) {
    auto mpm = pb.buildPerModuleDefaultPipeline(level);
    mpm.run(M, mam);
  } else {
    auto mpm = pb.buildO0DefaultPipeline(level);
    mpm.run(M, mam);
  }

  return 0;
}

