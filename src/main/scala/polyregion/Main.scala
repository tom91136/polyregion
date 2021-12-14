package polyregion

import net.openhft.affinity.{AffinityLock, AffinityStrategies, AffinityStrategy, AffinityThreadFactory}
import org.bytedeco.javacpp.*
import org.openjdk.jol.info.GraphLayout
import polyregion.Runtime.{Buffer, LibFfi}

import java.nio.ByteBuffer
import java.util.concurrent.{CountDownLatch, Executors}

object Main:

  def time[R](op: String)(block: => R): R = {
    val t0     = System.nanoTime()
    val result = block // call-by-name
    val t1     = System.nanoTime()
    println(s"$op: ${(t1 - t0) / 1e6}ms")
    result
  }

//  class LLVMBuilder(ref: LLVMBuilderRef) {
//    import org.bytedeco.javacpp.*
//    import org.bytedeco.llvm.LLVM.*
//    import org.bytedeco.llvm.global.LLVM.*
//    def positionBuilder(Block: LLVMBasicBlockRef, Instr: LLVMValueRef): Unit = LLVMPositionBuilder(ref, Block, Instr)
//    def positionBuilderBefore(Instr: LLVMValueRef): Unit                     = LLVMPositionBuilderBefore(ref, Instr)
//    def positionBuilderAtEnd(Block: LLVMBasicBlockRef): Unit                 = LLVMPositionBuilderAtEnd(ref, Block)
//    def getInsertBlock(Builder: LLVMBuilderRef): LLVMBasicBlockRef           = LLVMGetInsertBlock(ref)
//    def clearInsertionPosition(Builder: LLVMBuilderRef): Unit                = LLVMClearInsertionPosition(ref)
//    def insertIntoBuilder(Instr: LLVMValueRef): Unit                         = LLVMInsertIntoBuilder(ref, Instr)
//    def insertIntoBuilderWithName(Instr: LLVMValueRef, Name: BytePointer): Unit =
//      LLVMInsertIntoBuilderWithName(ref, Instr, Name)
//    def insertIntoBuilderWithName(Instr: LLVMValueRef, Name: String): Unit =
//      LLVMInsertIntoBuilderWithName(ref, Instr, Name)
//    def disposeBuilder(Builder: LLVMBuilderRef): Unit = LLVMDisposeBuilder(ref)
//
//  }

  def main(args: Array[String]): Unit = {

    import org.openjdk.jol.info.ClassLayout
    import org.openjdk.jol.vm.VM
    println(VM.current.details)
    import scala.collection.immutable.ArraySeq

    // 4 4 4 4 = 16
    case class Atom(x: Float, y: Float, z: Float, tpe: Int)
    case class Atom2(x: Float)

    println(GraphLayout.parseInstance(Atom(42, 43, 44, 120)).toPrintable)
    println(GraphLayout.parseInstance(Atom2(42)).toPrintable)

    val mod = new LLVM_.Module("sum")
    mod.function("times2AndSumAll", mod.i32, ("a", mod.ptr(mod.i32))) { case (params, fn, builder) =>
      import org.bytedeco.llvm.global.LLVM.*
      val a0Ptr   = mod.gepInbound(builder, "a[]")(params("a"), mod.constInt(mod.i32, 0))
      val a1Ref   = LLVMBuildLoad(builder, a0Ptr, "&a[0]")
      val a1x2Ref = LLVMBuildMul(builder, a1Ref, mod.constInt(mod.i32, 2), "*2")
      LLVMBuildStore(builder, a1x2Ref, a0Ptr)
      LLVMBuildRet(builder, a1x2Ref)
    }

    mod.function("mulDD", mod.double, ("a", (mod.double)), ("b", mod.double)) { case (params, fn, builder) =>
      import org.bytedeco.llvm.global.LLVM.*
      val res = LLVMBuildFMul(builder, params("a"), params("b"), "c")
      LLVMBuildRet(builder, res)
    }

    mod.function("mulDI", mod.double, ("a", (mod.double)), ("b", mod.i32)) { case (params, fn, builder) =>
      import org.bytedeco.llvm.global.LLVM.*
      val u   = LLVMBuildSIToFP(builder, params("b"), mod.double, "b:double")
      val res = LLVMBuildFMul(builder, params("a"), u, "c")
      LLVMBuildRet(builder, res)
    }

    mod.function("twiceFN", mod.double, ("a", (mod.double)), ("N", mod.i32)) { case (params, fn, builder) =>
//      val u = LLVMBuildSIToFP(builder, params("N"), mod.double, "N:double")
      val u = mod.constReal(mod.double, 4.0)
      import org.bytedeco.llvm.global.LLVM.*
      val res = LLVMBuildFMul(builder, params("a"), u, "*N")
      LLVMBuildRet(builder, res)
    }

    mod.function("copyAF", mod.void, ("a", mod.ptr(mod.double)), ("N", mod.i32)) { case (params, fn, builder) =>
      import org.bytedeco.llvm.global.LLVM.*
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

    mod.function(
      "copy",
      mod.void,
      ("a", mod.ptr(mod.double)),
      ("b", mod.ptr(mod.double)),
      ("FROM", mod.i32),
      ("N", mod.i32)
    ) { case (params, fn, builder) =>
      import org.bytedeco.llvm.global.LLVM.*
//      val id   = LLVMGetEnumAttributeKindForName("noalias", "noalias".length)
//      val attr = LLVMCreateEnumAttribute(context, id, 0)
//      LLVMAddAttributeAtIndex(fn, 1, attr)
//      LLVMAddAttributeAtIndex(fn, 2, attr)

//        val s = LLVMBuildPhi(builder, mod.i32, "FIXED S")
//        val e = LLVMBuildPhi(builder, mod.i32, "FIXED E")
//        LLVMAddIncoming(s, mod.constInt(mod.i32, 0), LLVMGetLastBasicBlock(fn), 1);
//        LLVMAddIncoming(e, mod.constInt(mod.i32, 100000000), LLVMGetLastBasicBlock(fn), 1);

      val s = params("FROM")
      val e = params("N")

      mod.i32loop(builder, fn)(s, e, 1, "i") { i =>

        val aPtr = mod.gepInbound(builder, "a[]")(params("a"), i)
        val bPtr = mod.gepInbound(builder, "b[]")(params("b"), i)
        val aRef = LLVMBuildLoad(builder, aPtr, "a[i]")
        val bRef = LLVMBuildLoad(builder, bPtr, "b[i]")
        val aa   = LLVMBuildFMul(builder, aRef, aRef, "a*a")
        val aab  = LLVMBuildFMul(builder, aa, bRef, "aa*b")
        val aabb = LLVMBuildFMul(builder, aab, bRef, "aa*bb")

        LLVMBuildStore(builder, aabb, aPtr)
      }

      LLVMBuildRetVoid(builder)

//      LLVMBuildRet(builder, a1x2Ref)
    }

    mod.validate()
    mod.optimise()
    mod.dump()

    val nioBuffer = ByteBuffer.allocateDirect(Integer.BYTES * 100).asIntBuffer()
    val view      = Vector.tabulate(10)(i => nioBuffer.asReadOnlyBuffer().get(i))
    println(">> " + view)

    val N = 20000000

    def read(x: IntPointer) = {
      val xs = Array.ofDim[Int](N)
      x.get(xs)
      xs.toVector
    }

    val jit = new OrcJIT_(mod)

    println("-===- ")

    val ex = Executors.newCachedThreadPool(new AffinityThreadFactory("af-%s", AffinityStrategies.DIFFERENT_CORE))

    for (_ <- 0 until 10) {

      val resultF = new DoublePointer(1L)
      jit.invokeORC(
        "mulDI",
        (resultF, LibFfi.Type.Double),
        (new DoublePointer(1.toLong).put(42.0), LibFfi.Type.Double),
//      (LibFFI.Type.Double, new DoublePointer(1.toLong).put(42.0)),
//
        (new IntPointer(1.toLong).put(3), LibFfi.Type.SInt32)
      )
      println(s"Ret=${resultF.get()}")

//      val x : EitherT[Eval, Exception, String] = ???
//      x.map

      val a = Buffer.ofDim[Double](N)
      val b = Buffer.ofDim[Double](N)

      val ax = Array.ofDim[Double](N)
      val bx = Array.ofDim[Double](N)
      time("native fill") {
        for (i <- 0 until N) {
          a(i) = math.Pi
          b(i) = i
          ax(i) = math.Pi
          bx(i) = i
        }
      }

//    println(s"Before=${readF(a)}")

      time("JIT warmup")(
        jit.invokeORC(
          "copy",
          (new Pointer(), LibFfi.Type.Void),
          (a.pointer, LibFfi.Type.Ptr),
          (b.pointer, LibFfi.Type.Ptr),
          (new IntPointer(1.toLong).put(0), LibFfi.Type.UInt32),
          (new IntPointer(1.toLong).put(0), LibFfi.Type.UInt32)
        )
      )

      val Core = 1

      val slice = N / Core

      import scala.jdk.CollectionConverters.*

      time("JIT all") {
        ex.invokeAll(
          (0 until Core).map { S =>
            ({ () =>
              time(s"  JIT[${S}]")(
                jit.invokeORC(
                  "copy",
                  (new Pointer(), LibFfi.Type.Void),
                  (a.pointer, LibFfi.Type.Ptr),
                  (b.pointer, LibFfi.Type.Ptr),
                  (new IntPointer(1.toLong).put(S * slice), LibFfi.Type.UInt32),
                  (new IntPointer(1.toLong).put((S + 1) * slice), LibFfi.Type.UInt32)
                )
              )

            }): java.util.concurrent.Callable[Unit]
          }.asJavaCollection
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

      time("JVM all") {
        ex.invokeAll(
          (0 until Core).map { S =>
            ({ () =>
              for (idx <- (S * slice) until ((S + 1) * slice))
                ax(idx) = ax(idx) * ax(idx) * bx(idx) * bx(idx)

            }): java.util.concurrent.Callable[Unit]
          }.asJavaCollection
        )
      }

//    jit.invoke("copyAF", (Type.Void, void), (Type.Ptr, a), (Type.UInt32, new IntPointer(1.toLong).put(N) ))

//    jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(0)), (Type.UInt32, new IntPointer(1.toLong).put(N/2)))
//    jit.invoke("copy", (Type.Void, void), (Type.Ptr, a), (Type.Ptr, b), (Type.UInt32, new IntPointer(1.toLong).put(N/2)), (Type.UInt32, new IntPointer(1.toLong).put(N)))

      import scala.collection.parallel.CollectionConverters.*

//    val og = time("JVM")((0 until N).map(2 * _).toVector)

      println(s"Ret=${}")

      println(s"${a.zip(ax).forall((l, r) => math.abs(l - r) < 0.001)}")

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
