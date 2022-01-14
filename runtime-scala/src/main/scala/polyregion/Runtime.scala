package polyregion

// import org.bytedeco.javacpp.LongPointer
// import org.bytedeco.llvm.global.LLVM.{LLVMConsumeError, LLVMGetErrorMessage, LLVMOrcLLJITLookup}
// import polyregion.Runtime.LibFfi.Type

import scala.collection.mutable.ArrayBuffer
//import polyregion.Runtime.PolyAst.{Expr, Intrinsics, Path, Primitives, Ref, Stmt, Type}

import java.lang.reflect.Modifier
import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.annotation.tailrec
import scala.reflect.ClassTag
import scala.collection.mutable
import scala.quoted.ToExpr

object Runtime {

  import org.bytedeco.javacpp.{Pointer, PointerPointer}

  private[polyregion] val NullPtr                  = new Pointer()
  private[polyregion] def ptrToAddress(addr: Long) = new Pointer() { address = addr }

  trait Buffer[T] extends mutable.IndexedSeq[T] {
    def pointer: Option[Long]
    def buffer: java.nio.Buffer
    def putAll(xs: T*): this.type
  }

  object Buffer {

//    inline private def unsafe = {
//      import java.lang.reflect.Field
//      val f = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe")
//      f.setAccessible(true)
//      val unsafe = f.get(null).asInstanceOf[sun.misc.Unsafe]
//    }

    inline private def ptr(b: java.nio.Buffer): Option[Long] =
      if (b.isDirect) Some(b.asInstanceOf[sun.nio.ch.DirectBuffer].address) else None

    inline private def alloc(size: Int): java.nio.ByteBuffer =
      java.nio.ByteBuffer.allocateDirect(size).order(java.nio.ByteOrder.nativeOrder())

    class DoubleBuffer(val buffer: java.nio.DoubleBuffer) extends Buffer[Double] {
      override def update(idx: Int, elem: Double): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Double                = buffer.get(i)
      override def length: Int                          = buffer.capacity()
      override def putAll(xs: Double*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]                = ptr(buffer)
    }
    class FloatBuffer(val buffer: java.nio.FloatBuffer) extends Buffer[Float] {
      override def update(idx: Int, elem: Float): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Float                = buffer.get(i)
      override def length: Int                         = buffer.capacity()
      override def putAll(xs: Float*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]               = ptr(buffer)
    }
    class LongBuffer(val buffer: java.nio.LongBuffer) extends Buffer[Long] {
      override def update(idx: Int, elem: Long): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Long                = buffer.get(i)
      override def length: Int                        = buffer.capacity()
      override def putAll(xs: Long*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]              = ptr(buffer)
    }
    class IntBuffer(val buffer: java.nio.IntBuffer) extends Buffer[Int] {
      override def update(idx: Int, elem: Int): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Int                = buffer.get(i)
      override def length: Int                       = buffer.capacity()
      override def putAll(xs: Int*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]             = ptr(buffer)
    }
    class ShortBuffer(val buffer: java.nio.ShortBuffer) extends Buffer[Short] {
      override def update(idx: Int, elem: Short): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Short                = buffer.get(i)
      override def length: Int                         = buffer.capacity()
      override def putAll(xs: Short*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]               = ptr(buffer)
    }
    class ByteBuffer(val buffer: java.nio.ByteBuffer) extends Buffer[Byte] {
      override def update(idx: Int, elem: Byte): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Byte                = buffer.get(i)
      override def length: Int                        = buffer.capacity()
      override def putAll(xs: Byte*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]              = ptr(buffer)
    }
    class CharBuffer(val buffer: java.nio.CharBuffer) extends Buffer[Char] {
      override def update(idx: Int, elem: Char): Unit = buffer.put(idx, elem)
      override def apply(i: Int): Char                = buffer.get(i)
      override def length: Int                        = buffer.capacity()
      override def putAll(xs: Char*): this.type       = { buffer.put(xs.toArray); this }
      override def pointer: Option[Long]              = ptr(buffer)
    }

    def ofDim[T <: AnyVal](dim: Int)(using tag: ClassTag[T]): Buffer[T] = (tag.runtimeClass match {
      case java.lang.Double.TYPE    => DoubleBuffer(alloc(java.lang.Double.BYTES * dim).asDoubleBuffer())
      case java.lang.Float.TYPE     => FloatBuffer(alloc(java.lang.Float.BYTES * dim).asFloatBuffer())
      case java.lang.Long.TYPE      => LongBuffer(alloc(java.lang.Long.BYTES * dim).asLongBuffer())
      case java.lang.Integer.TYPE   => IntBuffer(alloc(java.lang.Integer.BYTES * dim).asIntBuffer())
      case java.lang.Short.TYPE     => ShortBuffer(alloc(java.lang.Short.BYTES * dim).asShortBuffer())
      case java.lang.Character.TYPE => CharBuffer(alloc(java.lang.Character.BYTES * dim).asCharBuffer())
      case java.lang.Byte.TYPE      => ByteBuffer(alloc(java.lang.Byte.BYTES * dim))
    }).asInstanceOf[Buffer[T]]

    def ref[T <: AnyVal] (using tag: ClassTag[T]): Buffer[T] = ofDim[T](1)
    def nil[T <: AnyVal] (using tag: ClassTag[T]): Buffer[T] = ofDim[T](0)


    def apply[T <: AnyVal](xs: T*)(using tag: ClassTag[T]): Buffer[T] = ofDim[T](xs.size).putAll(xs*)

  }

  // def ingest[U](bitcode: Array[Byte], invoke: Long => U) = {

  //   PolyregionRuntime.invoke()

  //   println(s"Ingesting ${bitcode.length} bytes...")
  //   val mod = new LLVM_.Module("a")
  //   mod.load(bitcode)
  //   mod.optimise()
  //   mod.dump()

  //   val orc = new OrcJIT_(mod)

  //   val fnAddress = new LongPointer(1)
  //   val err       = LLVMOrcLLJITLookup(orc.jit, fnAddress, "lambda")
  //   if (err != null) {
  //     System.err.println(s"Failed to look up lambda symbol: " + LLVMGetErrorMessage(err))
  //     LLVMConsumeError(err)
  //   }
  //   invoke(fnAddress.get)
  // }

  // class FFIInvocationBuilder {
  //   import com.kenai.jffi.Invoker
  //   import com.kenai.jffi.Function
  //   import com.kenai.jffi.HeapInvocationBuffer
  //   import com.kenai.jffi.{Type => JType}

  //   private val argTpes = ArrayBuffer[JType]()
  //   private val argVals = ArrayBuffer[HeapInvocationBuffer => Unit]()
  //   private var rtnTpe  = JType.VOID

  //   inline def trn(tpe: JType): Unit = rtnTpe = tpe

  //   inline def arg(v: Byte): Unit = {
  //     argTpes += JType.SINT8
  //     argVals += (x => x.putByte(v))
  //   }
  //   inline def arg(v: Short): Unit = {
  //     argTpes += JType.SINT16
  //     argVals += (x => x.putShort(v))
  //   }
  //   inline def arg(v: Int): Unit = {
  //     argTpes += JType.SINT32
  //     argVals += (x => x.putInt(v))
  //   }
  //   inline def arg(v: Long): Unit = {
  //     argTpes += JType.SINT64
  //     argVals += (x => x.putLong(v))
  //   }
  //   inline def arg(v: Float): Unit = {
  //     argTpes += JType.FLOAT
  //     argVals += (x => x.putFloat(v))
  //   }
  //   inline def arg(v: Double): Unit = {
  //     argTpes += JType.DOUBLE
  //     argVals += (x => x.putDouble(v))
  //   }

  //   inline def arg(v: java.nio.Buffer): Unit = {
  //     argTpes += JType.POINTER
  //     argVals += (x => x.putDirectBuffer(v, 0, v.capacity()))
  //   }

  //   inline def invoke(addr: Long): Either[Exception, Unit] = {
  //     val f = new Function(addr, rtnTpe, argTpes.toArray*)
  //     val b = new HeapInvocationBuffer(f)
  //     argVals.foreach(_(b))
  //     try {
  //       Invoker.getInstance().invokeAddress(f, b)
  //       Right(())
  //     } catch {
  //       case e: Exception => Left(e)
  //     }
  //   }
  // }

  // private[polyregion] object LibFfi {
  //   import org.bytedeco.libffi.ffi_cif
  //   import org.bytedeco.libffi.ffi_type
  //   import org.bytedeco.libffi.global.ffi.*
  //   import org.bytedeco.libffi.presets.ffi

  //   enum Type(private[LibFfi] val actual: ffi_type) {
  //     case UInt8 extends Type(ffi_type_uint8())
  //     case SInt8 extends Type(ffi_type_sint8())
  //     case UInt16 extends Type(ffi_type_uint16())
  //     case SInt16 extends Type(ffi_type_sint16())
  //     case UInt32 extends Type(ffi_type_sint32())
  //     case SInt32 extends Type(ffi_type_sint32())
  //     case UInt64 extends Type(ffi_type_uint64())
  //     case SInt64 extends Type(ffi_type_sint64())
  //     case Float extends Type(ffi_type_float())
  //     case Double extends Type(ffi_type_double())
  //     case Ptr extends Type(ffi_type_pointer())
  //     case Void extends Type(ffi_type_void())
  //   }

  //   inline def invoke(addr: Long, rtn: (Pointer, Type), in: (Pointer, Type)*): Either[Exception, Unit] = {
  //     import org.bytedeco.javacpp.LongPointer
  //     val argTypes = new PointerPointer[Pointer](in.size)
  //     in.zipWithIndex.foreach { case ((_, tpe), i) => argTypes.put(i, tpe.actual) }

  //     val argValues = new PointerPointer[Pointer](in.size)
  //     in.zipWithIndex.foreach {
  //       case ((p, Type.Ptr), i) => argValues.put(i, new LongPointer(Array(p.address()): _*))
  //       case ((p, _), i)        => argValues.put(i, p)
  //     }

  //     val (rtnPtr, rtnTpe) = rtn
  //     def raise(s: String) = Left(new Exception(s"ffi_prep_cif failed with: ${s}"))

  //     val cif = new ffi_cif()
  //     ffi_prep_cif(cif, ffi.FFI_DEFAULT_ABI(), in.size, rtnTpe.actual, argTypes) match {
  //       case FFI_OK          => Right(ffi_call(cif, ptrToAddress(addr), rtnPtr, argValues))
  //       case FFI_BAD_TYPEDEF => raise("FFI_BAD_TYPEDEF")
  //       case FFI_BAD_ARGTYPE => raise("FFI_BAD_ARGTYPE")
  //       case FFI_BAD_ABI     => raise("FFI_BAD_ABI")
  //       case unknown         => raise(s"$unknown (unknown ffi_status)")
  //     }
  //   }

  //   inline def invokeJFFI(addr: Long, rtn: (Pointer, Type), in: (Pointer, Type)*): Either[Exception, Unit] = {

  //     import com.kenai.jffi.Invoker
  //     import com.kenai.jffi.Function
  //     import com.kenai.jffi.HeapInvocationBuffer
  //     import com.kenai.jffi.{Type => JType}

  //     def t2t(t: Type): JType =
  //       t match {
  //         case Type.UInt8  => JType.UINT8
  //         case Type.SInt8  => JType.SINT8
  //         case Type.UInt16 => JType.UINT16
  //         case Type.SInt16 => JType.SINT16
  //         case Type.UInt32 => JType.UINT32
  //         case Type.SInt32 => JType.SINT32
  //         case Type.UInt64 => JType.UINT64
  //         case Type.SInt64 => JType.SINT64
  //         case Type.Float  => JType.FLOAT
  //         case Type.Double => JType.DOUBLE
  //         case Type.Ptr    => JType.POINTER
  //         case Type.Void   => JType.VOID
  //       }

  //     val f = new Function(addr, t2t(rtn._2), in.map(x => t2t(x._2))*)
  //     val b = new HeapInvocationBuffer(f)

  //     in.foreach { (ptr, tpe) =>
  //       f

  //     }
  //     val returnDataPtr = Invoker.getInstance().invokeAddress(f, b)

  //     ???
  //   }

  // }

}
