package polyregion

import polyregion.Runtime.LibFfi

class OrcJIT_(mod: LLVM_.Module) {
  import org.bytedeco.javacpp.{LongPointer, Pointer}
  import org.bytedeco.llvm.global.LLVM._
  import org.bytedeco.llvm.LLVM._
  val threadModule = LLVMOrcCreateNewThreadSafeModule(mod.module, mod.threadContext)

  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeTarget();

  // setup ORC
  var err: LLVMErrorRef = _
  val jit               = new LLVMOrcLLJITRef
  val jitBuilder        = LLVMOrcCreateLLJITBuilder()

  val sa = LLVMOrcJITTargetMachineBuilderRef()
  LLVMOrcJITTargetMachineBuilderDetectHost(sa)
  LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jitBuilder, sa)

  err = LLVMOrcCreateLLJIT(jit, jitBuilder)
  if (err != null) {
    System.err.println("Failed to create LLJIT: " + LLVMGetErrorMessage(err).getString)
    LLVMConsumeError(err)
  }

  println("ORC Triplet:" + LLVMOrcLLJITGetTripleString(jit).getString)

  val mainDylib = LLVMOrcLLJITGetMainJITDylib(jit)
  err = LLVMOrcLLJITAddLLVMIRModule(jit, mainDylib, threadModule)
  if (err != null) {
    System.err.println("Failed to add LLVM IR module: " + LLVMGetErrorMessage(err))
    LLVMConsumeError(err)
  }

  val ES       = LLVMOrcLLJITGetExecutionSession(jit)
  val objLayer = LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(ES)
  LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(objLayer, LLVMCreatePerfJITEventListener())



  def invokeORC(name: String, rtnTpe: (Pointer, LibFfi.Type), in: (Pointer, LibFfi.Type)*) = {
    val fnAddress = new LongPointer(1)
    err = LLVMOrcLLJITLookup(jit, fnAddress, name)
    if (err != null) {
      System.err.println(s"Failed to look up $name symbol: " + LLVMGetErrorMessage(err))
      LLVMConsumeError(err)
    }
    LibFfi.invoke(fnAddress.get, rtnTpe, in: _*)
  }

  def dispose() = {
    LLVMOrcDisposeLLJIT(jit)
    LLVMShutdown()
  }

}
