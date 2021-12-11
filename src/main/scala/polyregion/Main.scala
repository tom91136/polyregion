package hps

import net.openhft.affinity.{AffinityLock, AffinityStrategies, AffinityStrategy, AffinityThreadFactory}
import org.bytedeco.javacpp.{BytePointer, IntPointer, PointerPointer}
import org.bytedeco.libffi.ffi_cif
import org.bytedeco.libffi.global.ffi.*
import org.bytedeco.libffi.presets.ffi
import org.bytedeco.llvm.LLVM.{LLVMBasicBlockRef, LLVMBuilderRef, LLVMValueRef}
import org.openjdk.jol.info.GraphLayout

import java.nio.ByteBuffer
import java.util.concurrent.{CountDownLatch, Executors}

object Main:

  object AA {

    trait B {
      def Else(f: => Any): Unit
      def ElseIf(a: => Any)(f: => Any): B
    }

    object A {
      def If(a: => Any)(f: => Any): B = ???
    }

    import A._

//    val xs = (If(1) {}.ElseIf()) {}.Else {}

  }

  class LLVMBuilder(ref: LLVMBuilderRef) {
    import org.bytedeco.javacpp._
    import org.bytedeco.llvm.LLVM._
    import org.bytedeco.llvm.global.LLVM._
    def positionBuilder(Block: LLVMBasicBlockRef, Instr: LLVMValueRef): Unit = LLVMPositionBuilder(ref, Block, Instr)
    def positionBuilderBefore(Instr: LLVMValueRef): Unit                     = LLVMPositionBuilderBefore(ref, Instr)
    def positionBuilderAtEnd(Block: LLVMBasicBlockRef): Unit                 = LLVMPositionBuilderAtEnd(ref, Block)
    def getInsertBlock(Builder: LLVMBuilderRef): LLVMBasicBlockRef           = LLVMGetInsertBlock(ref)
    def clearInsertionPosition(Builder: LLVMBuilderRef): Unit                = LLVMClearInsertionPosition(ref)
    def insertIntoBuilder(Instr: LLVMValueRef): Unit                         = LLVMInsertIntoBuilder(ref, Instr)
    def insertIntoBuilderWithName(Instr: LLVMValueRef, Name: BytePointer): Unit =
      LLVMInsertIntoBuilderWithName(ref, Instr, Name)
    def insertIntoBuilderWithName(Instr: LLVMValueRef, Name: String): Unit =
      LLVMInsertIntoBuilderWithName(ref, Instr, Name)
    def disposeBuilder(Builder: LLVMBuilderRef): Unit = LLVMDisposeBuilder(ref)

  }

  def main(args: Array[String]): Unit = {

//    case class Index(value : Int)
//
//    case class Memory(value : Int)
//
//    trait HPS{
//
//      def parallel_for(f : Index => Unit) : Unit
//
//    }
//
//
//    trait Repr[A]
//
//    implicit class X[A]()
//
//    val a = Memory(1)
//    val b = Memory(1)
//    val c = Memory(1)
//
//    val h : HPS = ???
//        h.parallel_for(i => )
//

//    case class Else(a : Any)
//    case class ElseIf(a : Any)
//    case class If(a : Any)

    import org.openjdk.jol.info.ClassLayout
    import org.openjdk.jol.vm.VM

    println(VM.current.details)
    import scala.collection.immutable.ArraySeq

    // 4 4 4 4 = 16
    case class Atom(x: Float, y: Float, z: Float, tpe: Int)
    case class Atom2(x: Float)


    println(GraphLayout.parseInstance(Atom(42, 43, 44, 120)).toPrintable)
    println(GraphLayout.parseInstance(Atom2(42)).toPrintable)

    println("Hey!")
    import org.bytedeco.javacpp._
    import org.bytedeco.llvm.LLVM._
    import org.bytedeco.llvm.global.LLVM._
    // General stuff

    val error = new BytePointer(null.asInstanceOf[Pointer]) // Used to retrieve messages from functions
//    LLVMLinkInMCJIT()
    LLVMInitializeNativeAsmPrinter()
    LLVMInitializeNativeAsmParser()
    LLVMInitializeNativeDisassembler()
    LLVMInitializeNativeTarget()

    val threadContext           = LLVMOrcCreateNewThreadSafeContext()
    val context: LLVMContextRef = LLVMOrcThreadSafeContextGetContext(threadContext)

    object LLVM_ {

      class Module(name: String) {
        private val threadContext           = LLVMOrcCreateNewThreadSafeContext()
        private val context: LLVMContextRef = LLVMOrcThreadSafeContextGetContext(threadContext)
        val module                          = LLVMModuleCreateWithNameInContext(name, context)

        def i1           = LLVMInt1TypeInContext(context)
        def i8           = LLVMInt8TypeInContext(context)
        def i16          = LLVMInt16TypeInContext(context)
        def i32          = LLVMInt32TypeInContext(context)
        def i64          = LLVMInt64TypeInContext(context)
        def i128         = LLVMInt128TypeInContext(context)
        def i(bits: Int) = LLVMIntTypeInContext(context, bits)

        def constInt(tpe: LLVMTypeRef, value: Int) = LLVMConstInt(tpe, value, 0)

        def half               = LLVMHalfTypeInContext(context)
        def bfloat             = LLVMBFloatTypeInContext(context)
        def float: LLVMTypeRef = LLVMFloatTypeInContext(context)
        def double             = LLVMDoubleTypeInContext(context)
        def x86FP80            = LLVMX86FP80TypeInContext(context)
        def fp128              = LLVMFP128TypeInContext(context)
        def ppcfp128           = LLVMPPCFP128TypeInContext(context)

        def constReal(tpe: LLVMTypeRef, value: Double) = LLVMConstReal(tpe, value)

        def void     = LLVMVoidTypeInContext(context)
        def label    = LLVMLabelTypeInContext(context)
        def x86MMX   = LLVMX86MMXTypeInContext(context)
        def x86AMX   = LLVMX86AMXTypeInContext(context)
        def token    = LLVMTokenTypeInContext(context)
        def metadata = LLVMMetadataTypeInContext(context)

        def ptr(tpe: LLVMTypeRef) = LLVMPointerType(tpe, 0)

        def gepInbound(builder: LLVMBuilderRef)(ref: LLVMValueRef, offsets: LLVMValueRef*) =
          LLVMBuildInBoundsGEP(
            builder,
            ref,
            new PointerPointer[Pointer](offsets: _*),
            offsets.length,
            s"${LLVMGetValueName(ref).getString}[${offsets.map(LLVMGetValueName(_).getString).mkString(",")}]"
          )

        def validate(): Unit = LLVMVerifyModule(module, LLVMPrintMessageAction, new BytePointer())

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
          LLVMAddAggressiveDCEPass(pm)
          LLVMAddDCEPass(pm)
          LLVMAddBitTrackingDCEPass(pm)
          LLVMAddAlignmentFromAssumptionsPass(pm)
          LLVMAddCFGSimplificationPass(pm)
          LLVMAddDeadStoreEliminationPass(pm)
          LLVMAddScalarizerPass(pm)
          LLVMAddMergedLoadStoreMotionPass(pm)
          LLVMAddGVNPass(pm)
          LLVMAddNewGVNPass(pm)
          LLVMAddIndVarSimplifyPass(pm)
          LLVMAddInstructionSimplifyPass(pm)
          LLVMAddJumpThreadingPass(pm)
          LLVMAddLICMPass(pm)
          LLVMAddLoopDeletionPass(pm)
          LLVMAddLoopIdiomPass(pm)
          LLVMAddLoopRotatePass(pm)
          LLVMAddLoopRerollPass(pm)
          LLVMAddLoopUnrollPass(pm)
          LLVMAddLoopUnrollAndJamPass(pm)
          LLVMAddLoopUnswitchPass(pm)
          LLVMAddLowerAtomicPass(pm)
          LLVMAddMemCpyOptPass(pm)
          LLVMAddPartiallyInlineLibCallsPass(pm)
          LLVMAddReassociatePass(pm)
          LLVMAddSCCPPass(pm)
          LLVMAddScalarReplAggregatesPass(pm)
          LLVMAddScalarReplAggregatesPassSSA(pm)
            LLVMAddSimplifyLibCallsPass(pm)
          LLVMAddTailCallEliminationPass(pm)
//          LLVMAddDemoteMemoryToRegisterPass(pm)
          LLVMAddVerifierPass(pm)
          LLVMAddCorrelatedValuePropagationPass(pm)
          LLVMAddEarlyCSEPass(pm)
          LLVMAddEarlyCSEMemSSAPass(pm)
          LLVMAddLowerExpectIntrinsicPass(pm)
          LLVMAddLowerConstantIntrinsicsPass(pm)
          LLVMAddTypeBasedAliasAnalysisPass(pm)
          LLVMAddScopedNoAliasAAPass(pm)
          LLVMAddBasicAliasAnalysisPass(pm)
          LLVMAddUnifyFunctionExitNodesPass(pm)
          LLVMRunPassManager(pm, module)

          optimizeModule(module, LLVMGetHostCPUName(), 3, 0)


        }

      }

    }

    val mod = new LLVM_.Module("sum")
    mod.function("times2AndSumAll", mod.i32, ("a", mod.ptr(mod.i32))) { case (params, fn, builder) =>
      val a0Ptr   = mod.gepInbound(builder)(params("a"), mod.constInt(mod.i32, 0))
      val a1Ref   = LLVMBuildLoad(builder, a0Ptr, "&a[0]")
      val a1x2Ref = LLVMBuildMul(builder, a1Ref, mod.constInt(mod.i32, 2), "*2")
      LLVMBuildStore(builder, a1x2Ref, a0Ptr)
      LLVMBuildRet(builder, a1x2Ref)
    }

    mod.function("mulDD", mod.double, ("a", (mod.double)), ("b", mod.double)) { case (params, fn, builder) =>
      val res = LLVMBuildFMul(builder, params("a"), params("b"), "c")
      LLVMBuildRet(builder, res)
    }

    mod.function("mulDI", mod.double, ("a", (mod.double)), ("b", mod.i32)) { case (params, fn, builder) =>
      val u   = LLVMBuildSIToFP(builder, params("b"), mod.double, "b:double")
      val res = LLVMBuildFMul(builder, params("a"), u, "c")
      LLVMBuildRet(builder, res)
    }

    mod.function("twiceFN", mod.double, ("a", (mod.double)), ("N", mod.i32)) { case (params, fn, builder) =>
//      val u = LLVMBuildSIToFP(builder, params("N"), mod.double, "N:double")
      val u = mod.constReal(mod.double, 4.0)

      val res = LLVMBuildFMul(builder, params("a"), u, "*N")
      LLVMBuildRet(builder, res)
    }

    mod.function("copyAF", mod.void, ("a", mod.ptr(mod.double)), ("N", mod.i32)) { case (params, fn, builder) =>
      //      val a0Ptr   = mod.gepInbound(builder)(params("a"), 0)
      //      val a1Ref   = LLVMBuildLoad(builder, a0Ptr, "&a[0]")
      //      val a1x2Ref = LLVMBuildMul(builder, a1Ref, mod.constInt(mod.double, 2), "*2")

      //      LLVMBuildStore(builder, a1x2Ref, a0Ptr)

//        mod.i32loop(builder, fn)(mod.constInt(mod.i32, 0), params("N"), 1, "i") { i =>
//          val aPtr = mod.gepInbound(builder)(params("a"), i)
//          val aRef = LLVMBuildLoad(builder, aPtr, "a[i]")
//          val res  = LLVMBuildFMul(builder, aRef,  mod.constReal(mod.double, 2.0), "a*2")
//          LLVMBuildStore(builder, res, aPtr)
//        }

      LLVMBuildRetVoid(builder)

    //      LLVMBuildRet(builder, a1x2Ref)
    }

    mod.function("copy", mod.void, ("a", mod.ptr(mod.double)), ("b", mod.ptr(mod.double)), ("FROM", mod.i32), ("N", mod.i32)) {
      case (params, fn, builder) =>
//      val a0Ptr   = mod.gepInbound(builder)(params("a"), 0)
//      val a1Ref   = LLVMBuildLoad(builder, a0Ptr, "&a[0]")
//      val a1x2Ref = LLVMBuildMul(builder, a1Ref, mod.constInt(mod.double, 2), "*2")

//      LLVMBuildStore(builder, a1x2Ref, a0Ptr)



//        LLVMCreateTypeAttribute(context, 0,mod.double )
        val id = LLVMGetEnumAttributeKindForName("noalias", "noalias".length)
        val attr = LLVMCreateEnumAttribute(context, id, 0)

        LLVMAddAttributeAtIndex(fn, 1,  attr)
        LLVMAddAttributeAtIndex(fn, 2,  attr)

//        val s = LLVMBuildPhi(builder, mod.i32, "FIXED S")
//        val e = LLVMBuildPhi(builder, mod.i32, "FIXED E")

//        LLVMAddIncoming(s, mod.constInt(mod.i32, 0), LLVMGetLastBasicBlock(fn), 1);
//        LLVMAddIncoming(e, mod.constInt(mod.i32, 100000000), LLVMGetLastBasicBlock(fn), 1);


        val s = params("FROM")
        val e = params("N")

        mod.i32loop(builder, fn)(s, e, 1, "i") { i =>
          val aPtr = mod.gepInbound(builder)(params("a"), i)
          val bPtr = mod.gepInbound(builder)(params("b"), i)
          val aRef = LLVMBuildLoad(builder, aPtr, "a[i]")
          val bRef = LLVMBuildLoad(builder, bPtr, "b[i]")
          val aa  = LLVMBuildFMul(builder, aRef, aRef, "a*a")
          val aab  = LLVMBuildFMul(builder, aa, bRef, "aa*b")
          val aabb  = LLVMBuildFMul(builder, aab, bRef, "aa*bb")


          LLVMBuildStore(builder, aabb, aPtr)
        }

        LLVMBuildRetVoid(builder)

//      LLVMBuildRet(builder, a1x2Ref)
    }


    mod.validate()
    mod.optimise()
    mod.dump()

    enum Type {
      case UInt8
      case SInt8
      case UInt16
      case SInt16
      case UInt32
      case SInt32
      case UInt64
      case SInt64
      case Float
      case Double
      case Ptr
      case Void
    }

    class OrcJIT_ {
      val threadModule = LLVMOrcCreateNewThreadSafeModule(mod.module, threadContext)

      val sa = LLVMOrcJITTargetMachineBuilderRef()
      LLVMOrcJITTargetMachineBuilderDetectHost(sa)


      // setup ORC
      var err: LLVMErrorRef = null
      val jit               = new LLVMOrcLLJITRef
      val jitBuilder        = LLVMOrcCreateLLJITBuilder()

      LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jitBuilder, sa)

      err = LLVMOrcCreateLLJIT(jit, jitBuilder)
      if (err != null) {
        System.err.println("Failed to create LLJIT: " + LLVMGetErrorMessage(err))
        LLVMConsumeError(err)
      }

      println("ORC Triplet:" + LLVMOrcLLJITGetTripleString(jit).getString)



      val mainDylib = LLVMOrcLLJITGetMainJITDylib(jit)
      err = LLVMOrcLLJITAddLLVMIRModule(jit, mainDylib, threadModule)
      if (err != null) {
        System.err.println("Failed to add LLVM IR module: " + LLVMGetErrorMessage(err))
        LLVMConsumeError(err)
      }

      val ES = LLVMOrcLLJITGetExecutionSession(jit)
      val objLayer = LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(ES)
      LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(objLayer,LLVMCreatePerfJITEventListener() )


      LLVMInitializeNativeAsmPrinter();
      LLVMInitializeNativeAsmParser();
      LLVMInitializeNativeDisassembler();
      LLVMInitializeNativeTarget();

      def invoke(name: String, rtnTpe: (Type, Pointer), in: (Type, Pointer)*) = {
        val fnAddress = new LongPointer(1)
        err = LLVMOrcLLJITLookup(jit, fnAddress, name)
        if (err != null) {
          System.err.println(s"Failed to look up $name symbol: " + LLVMGetErrorMessage(err))
          LLVMConsumeError(err)
        }

        def decodeTpe(f: Type) = f match {
          case Type.UInt8  => ffi_type_uint8()
          case Type.SInt8  => ffi_type_sint8()
          case Type.UInt16 => ffi_type_uint16()
          case Type.SInt16 => ffi_type_sint16()
          case Type.UInt32 => ffi_type_sint32()
          case Type.SInt32 => ffi_type_sint32()
          case Type.UInt64 => ffi_type_uint64()
          case Type.SInt64 => ffi_type_sint64()
          case Type.Float  => ffi_type_float()
          case Type.Double => ffi_type_double()
          case Type.Ptr    => ffi_type_pointer()
          case Type.Void   => ffi_type_void()
        }

        val argTypes = new PointerPointer[Pointer](in.size)
        in.zipWithIndex.foreach { case ((t, _), i) => argTypes.put(i, decodeTpe(t)) }

        val argValues = new PointerPointer[Pointer](in.size) // .put(0, nativeArray)
        in.zipWithIndex.foreach {
          case ((Type.Ptr, p), i) => argValues.put(i, new LongPointer(Array(p.address()): _*))
          case ((_, p)       , i) => argValues.put(i, p)
        }

        val cif = new ffi_cif()
        val rr  = ffi_prep_cif(cif, ffi.FFI_DEFAULT_ABI(), in.size, decodeTpe(rtnTpe._1), argTypes)
        if (rr != FFI_OK) {
          System.err.println(s"Failed to prepare the libffi cif, code=$rr")
        }
        val fnPtr = new Pointer() { address = fnAddress.get() }

        ffi_call(cif, fnPtr, rtnTpe._2, argValues)

      }

      def dispose() = {
        LLVMOrcDisposeLLJIT(jit)
        LLVMShutdown()
      }

    }

//    val engine = new LLVMExecutionEngineRef
//    if (LLVMCreateJITCompilerForModule(engine, mod.module, 2, error) != 0) {
//      Console.err.println(error.getString)
//      LLVMDisposeMessage(error)
//      sys.exit(-1)
//    }

//    println("===")
//    val pass = LLVMCreatePassManager
////    LLVMAddConstantPropagationPass(pass)
//    LLVMAddInstructionCombiningPass(pass)
//    LLVMAddPromoteMemoryToRegisterPass(pass)
//    // LLVMAddDemoteMemoryToRegisterPass(pass) // Demotes every possible value to memory
//    LLVMAddGVNPass(pass)
//    LLVMAddCFGSimplificationPass(pass)
//    LLVMRunPassManager(pass, module)
//    LLVMDumpModule(module)

    val nioBuffer = ByteBuffer.allocateDirect(Integer.BYTES * 100).asIntBuffer()
//    for (i <- 0 until 10) nioBuffer.put (i)
//    nioBuffer.rewind()

    val view = Vector.tabulate(10)(i => nioBuffer.asReadOnlyBuffer().get(i))
    println(">> " + view)

    //
////    val ptr    = new Pointer(nioBuffer)
////
////    LLVMCreateGenericValueOfPointer()
//
////    val exec_args  = LLVMCreateGenericValueOfInt(LLVMInt32Type, 12, 0)
//    val exec_args2 = LLVMCreateGenericValueOfPointer(new Pointer(nioBuffer))
////    val exec_res  = LLVMRunFunction(engine, times2, 1, exec_args2)
//
//
//    val addr = LLVMGetFunctionAddress(engine, "times2")
//    Console.err.println()
//    Console.err.println(" Running fac(10) with JIT..." + addr.toHexString)
//
//

    // Stage 4: Call the function with libffi// Stage 4: Call the function with libffi

    val N = 200000000

    def read(x: IntPointer) = {
      val xs = Array.ofDim[Int](N)
      x.get(xs)
      xs.toVector
    }

    def readF(x: DoublePointer) = {
      val xs = Array.ofDim[Double](N)
      x.get(xs)
      xs
    }

    val jit = new OrcJIT_()

//    val actual =
//      new IntPointer(N.toLong).fill[IntPointer](0).put(0L, 50).put(5L, 1).position(0)
//    val returnV = new IntPointer(1L)
//    println(s"Before=${read(actual)}")
//
//    jit.invoke("times2AndSumAll", (Type.UInt32, returnV), (Type.Ptr, actual))
//
//    println(s"Ret=${returnV.get()}")
//    println(s"After=${read(actual)}")
//
    println("-===-")


    val ex = Executors.newCachedThreadPool(new AffinityThreadFactory("af-%s", AffinityStrategies.DIFFERENT_CORE))

    def time[R](op: String )(block: => R): R = {
      val t0 = System.nanoTime()
      val result = block    // call-by-name
      val t1 = System.nanoTime()
      println(s"$op: ${(t1 - t0) / 1e6}ms")
      result
    }
    for(_ <- 0 until 10){


    val resultF = new DoublePointer(1L)
    jit.invoke(
      "mulDI",
      (Type.Double, resultF),
      (Type.Double, new DoublePointer(1.toLong).put(42.0)),
//      (Type.Double, new DoublePointer(1.toLong).put(42.0)),
//
      (Type.SInt32, new IntPointer(1.toLong).put(3))
    )
    println(s"Ret=${resultF.get()}")

    val a = new DoublePointer(N.toLong)
    val b = new DoublePointer(N.toLong)

      val ax = Array.ofDim[Double](N)
      val bx = Array.ofDim[Double](N)
    time("native fill"){
      for (i <- 0 until N) {
        a.put(i.toLong, math.Pi)
        b.put(i.toLong, i)
        ax(i) = math.Pi
        bx(i) = i
      }
    }




    val void = new IntPointer(1L)
//    println(s"Before=${readF(a)}")

    time("JIT warmup")(jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(0)), (Type.UInt32, new IntPointer(1.toLong).put((0 )  ))))


    val Core = 1


    val slice = N / Core

import scala.jdk.CollectionConverters._


      time("JIT all"){
        ex.invokeAll(
        (0 until Core).map(S => {
           ({() =>

            time(s"  JIT[${S}]")(jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(S * slice)), (Type.UInt32, new IntPointer(1.toLong).put((S +1 ) * slice))))


          }) : java.util.concurrent.Callable[Unit]
        }).asJavaCollection
        )
      }

//    time("JIT all"){
//      (0 until Core).map(S => {
//        new Thread({() =>
//
//          time(s"  JIT[${S}]")(jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(S * slice)), (Type.UInt32, new IntPointer(1.toLong).put((S +1 ) * slice))))
//
//
//        })
//      }).tapEach(t => t.start()).foreach(_.join())
//    }


      time("JVM all"){
        ex.invokeAll(
          (0 until Core).map(S => {
            ({() =>

              for (idx <- (S* slice) until ((S+1) * slice)){
                ax(idx) = ax(idx)* ax(idx) * bx(idx) * bx(idx)
              }

            }) : java.util.concurrent.Callable[Unit]
          }).asJavaCollection
        )
      }




//    jit.invoke("copyAF", (Type.Void, void), (Type.Ptr, a), (Type.UInt32, new IntPointer(1.toLong).put(N) ))

//    jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(0)), (Type.UInt32, new IntPointer(1.toLong).put(N/2)))
//    jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(N/2)), (Type.UInt32, new IntPointer(1.toLong).put(N)))



    import scala.collection.parallel.CollectionConverters._



//    val og = time("JVM")((0 until N).map(2 * _).toVector)

    println(s"Ret=${void.get()}")

//    println(s"${readF(a).zip (ax).forall((l,r) => math.abs(l-r) < 0.001)}")


      a.deallocate()
      b.deallocate()
    }


    //    println(s"After=${readF(a)}")




//    val cif       = new ffi_cif()
//    val arguments = new PointerPointer[Pointer](1).put(0, ffi_type_pointer())
//    val actual =
//      new IntPointer(N.toLong).fill[IntPointer](0).put(0L, 50).put(5L, 1).position(0) // new IntPointer(nioBuffer)
//    val nativeArray       = new LongPointer(Array(actual.address()): _*)
//    val values            = new PointerPointer[Pointer](1).put(0, nativeArray)
//    val voidReturnIgnored = new IntPointer(1L)
//
//    val fff = ffi.FFI_DEFAULT_ABI()
//    println(s"fff = $fff")
//    val rr = ffi_prep_cif(cif, fff, 1, ffi_type_sint(), arguments)
//    if (rr != FFI_OK) {
//      System.err.println(s"Failed to prepare the libffi cif, code=$rr")
//      return
//    }
//    println("ok, go!")
//    val function = new Pointer() {
//      address = res.get()
//    }
//
//    println(s"Before=${read(actual)}")
//
//    ffi_call(cif, function, voidReturnIgnored, values)
//
////    Console.err.println(" Result: " + LLVMGenericValueToInt(exec_res, 0))
//
//    println(s"Ret=${voidReturnIgnored.get()}")
//    println(s"After=${read(actual)}")
//    LLVMOrcDisposeLLJIT(jit)
//    LLVMShutdown()

//    LLVMGetFunctionAddress(engine, exec_args2)
//    LLVMDisposePassManager(pass)
//    LLVMDisposeBuilder(builder)
//    LLVMDisposeExecutionEngine(engine)

    println("End!")
  }
