package polyregion

object LLVM_ {

  import org.bytedeco.javacpp._
  import org.bytedeco.llvm.LLVM._
  import org.bytedeco.llvm.global.LLVM._

  LLVMInitializeNativeAsmPrinter()
  LLVMInitializeNativeAsmParser()
  LLVMInitializeNativeDisassembler()
  LLVMInitializeNativeTarget()

  class Module(name: String) {
    val threadContext                   = LLVMOrcCreateNewThreadSafeContext()
    private val context: LLVMContextRef = LLVMOrcThreadSafeContextGetContext(threadContext)
    val module: LLVMModuleRef           = LLVMModuleCreateWithNameInContext(name, context)

    def i1           = LLVMInt1TypeInContext(context)
    def i8           = LLVMInt8TypeInContext(context)
    def i16          = LLVMInt16TypeInContext(context)
    def i32          = LLVMInt32TypeInContext(context)
    def i64          = LLVMInt64TypeInContext(context)
    def i128         = LLVMInt128TypeInContext(context)
    def i(bits: Int) = LLVMIntTypeInContext(context, bits)

    def constInt(tpe: LLVMTypeRef, value: Long): LLVMValueRef = LLVMConstInt(tpe, value, 0)

    def half                 = LLVMHalfTypeInContext(context)
    def bfloat               = LLVMBFloatTypeInContext(context)
    def float: LLVMTypeRef   = LLVMFloatTypeInContext(context)
    def double               = LLVMDoubleTypeInContext(context)
    def x86FP80: LLVMTypeRef = LLVMX86FP80TypeInContext(context)
    def fp128                = LLVMFP128TypeInContext(context)
    def ppcfp128             = LLVMPPCFP128TypeInContext(context)

    def constReal(tpe: LLVMTypeRef, value: Double): LLVMValueRef = LLVMConstReal(tpe, value)

    def void: LLVMTypeRef = LLVMVoidTypeInContext(context)
    def label             = LLVMLabelTypeInContext(context)
    def x86MMX            = LLVMX86MMXTypeInContext(context)
    def x86AMX            = LLVMX86AMXTypeInContext(context)
    def token             = LLVMTokenTypeInContext(context)
    def metadata          = LLVMMetadataTypeInContext(context)

    def ptr(tpe: LLVMTypeRef) = LLVMPointerType(tpe, 0)

    def gepInbound(builder: LLVMBuilderRef, name: String)(ref: LLVMValueRef, offsets: LLVMValueRef*) =
      LLVMBuildInBoundsGEP(
        builder,
        ref,
        new PointerPointer[Pointer](offsets: _*),
        offsets.length,
        name
      )

    def validate(): Unit = LLVMVerifyModule(module, LLVMPrintMessageAction, new BytePointer())

    def load(array: Array[Byte]) = {
      val data =
        LLVMCreateMemoryBufferWithMemoryRange(new BytePointer(array: _*), array.length, new BytePointer("a"), 1)
      LLVMParseBitcodeInContext2(context, data, module)
    }

    def dump(): Unit = LLVMDumpModule(module)

    def i32loop(
        builder: LLVMBuilderRef,
        fn: LLVMValueRef
    )(from: LLVMValueRef, lim: LLVMValueRef, inc: Int, induction: String)(f: LLVMValueRef => Unit) = {

      val loopBB = LLVMAppendBasicBlock(fn, s"loop_$induction") // loop:
      LLVMBuildBr(builder, loopBB) // goto loop:
      LLVMPositionBuilderAtEnd(builder, loopBB)

      val i = LLVMBuildPhi(builder, i32, induction) // var i
      LLVMAddIncoming(i, from, LLVMGetPreviousBasicBlock(loopBB), 1) // i = from

      f(i)

      val nextI = LLVMBuildAdd(builder, i, constInt(i32, inc), s"loop_$induction(+$inc)") // var nextI = i + inc

      val continue = LLVMBuildICmp(builder, LLVMIntSLT, nextI, lim, s"looptest_$induction")

      val endBB       = LLVMGetInsertBlock(builder)
      val afterLoopBB = LLVMAppendBasicBlock(fn, "after_loop")
      LLVMBuildCondBr(builder, continue, loopBB, afterLoopBB)
      LLVMPositionBuilderAtEnd(builder, afterLoopBB)
      LLVMAddIncoming(i, nextI, endBB, 1) // i = nextI
    }

    def function(name: String, returnTpe: LLVMTypeRef, params: (String, LLVMTypeRef)*)(
        body: (Map[String, LLVMValueRef], LLVMValueRef, LLVMBuilderRef) => Unit
    ): Boolean = {
      val paramTpes = params.map(_._2).toArray
      val fnType    = LLVMFunctionType(returnTpe, new PointerPointer[Pointer](paramTpes: _*), paramTpes.length, 0)
      val fn        = LLVMAddFunction(module, name, fnType)
      LLVMSetFunctionCallConv(fn, LLVMCCallConv)

      val builder = LLVMCreateBuilderInContext(context)

      val entry = LLVMAppendBasicBlockInContext(context, fn, "entry")
      LLVMPositionBuilderAtEnd(builder, entry)

      val paramsVals = params.zipWithIndex.map { case ((name, ref), idx) =>
        val param = LLVMGetParam(fn, idx)
        LLVMSetValueName(param, name)
        name -> param
      }.toMap
      body(paramsVals, fn, builder)
      LLVMVerifyFunction(fn, LLVMPrintMessageAction) == 1
    }

    def optimise() = {
      val pm = LLVMCreatePassManager()
//      LLVMAddAggressiveDCEPass(pm)
//      LLVMAddPromoteMemoryToRegisterPass(pm)
      //          LLVMAddDCEPass(pm)
      //          LLVMAddBitTrackingDCEPass(pm)
      //          LLVMAddAlignmentFromAssumptionsPass(pm)
      //          LLVMAddCFGSimplificationPass(pm)
      //          LLVMAddDeadStoreEliminationPass(pm)
      //          LLVMAddScalarizerPass(pm)
      //          LLVMAddMergedLoadStoreMotionPass(pm)
      //          LLVMAddGVNPass(pm)
      //          LLVMAddNewGVNPass(pm)
      //          LLVMAddIndVarSimplifyPass(pm)
//      LLVMAddInstructionSimplifyPass(pm)
      //          LLVMAddJumpThreadingPass(pm)
      //          LLVMAddLICMPass(pm)
      //          LLVMAddLoopDeletionPass(pm)
      //          LLVMAddLoopIdiomPass(pm)
      //          LLVMAddLoopRotatePass(pm)
      //          LLVMAddLoopRerollPass(pm)
      //          LLVMAddLoopUnrollPass(pm)
      //          LLVMAddLoopUnrollAndJamPass(pm)
      //          LLVMAddLoopUnswitchPass(pm)
      //          LLVMAddLowerAtomicPass(pm)
      //          LLVMAddMemCpyOptPass(pm)
      //          LLVMAddPartiallyInlineLibCallsPass(pm)
      //          LLVMAddReassociatePass(pm)
      //          LLVMAddSCCPPass(pm)
      //          LLVMAddScalarReplAggregatesPass(pm)
      //          LLVMAddScalarReplAggregatesPassSSA(pm)
      //          LLVMAddSimplifyLibCallsPass(pm)
      //          LLVMAddTailCallEliminationPass(pm)
      ////          LLVMAddDemoteMemoryToRegisterPass(pm)
      //          LLVMAddVerifierPass(pm)
      //          LLVMAddCorrelatedValuePropagationPass(pm)
      //          LLVMAddEarlyCSEPass(pm)
      //          LLVMAddEarlyCSEMemSSAPass(pm)
      //          LLVMAddLowerExpectIntrinsicPass(pm)
      //          LLVMAddLowerConstantIntrinsicsPass(pm)
      //          LLVMAddTypeBasedAliasAnalysisPass(pm)
      //          LLVMAddScopedNoAliasAAPass(pm)
      //          LLVMAddBasicAliasAnalysisPass(pm)
      //          LLVMAddUnifyFunctionExitNodesPass(pm)
      LLVMRunPassManager(pm, module)
      optimizeModule(module, LLVMGetHostCPUName(), 3, 0)
    }
  }

}
