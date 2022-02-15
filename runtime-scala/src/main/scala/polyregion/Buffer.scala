package polyregion

import scala.reflect.ClassTag
import scala.math.Integral
import scala.collection.immutable.ArraySeq
import scala.collection.mutable

trait Buffer[A] extends mutable.IndexedSeq[A] {
  def name: String
  def pointer: Option[Long]
  def buffer: java.nio.Buffer
  def putAll(xs: A*): this.type
  def foreach2(f: Int => Int): Int = {
    var i   = 0
    var out = 0
    while (i < 10) {
      out = f(i)
      i += 1
    }
    out
  }
  def copyToArray: Array[A]
  override def toString: String = s"Buffer[$name](${mkString(", ")})"
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

  inline private def alloc(elemSize: Int, n: Int): java.nio.ByteBuffer = {
    val bytes =
      try math.multiplyExact(elemSize, n)
      catch {
        case e => throw new IllegalArgumentException(s"Cannot allocated buffer of size ${elemSize * n} (overflow?)", e)
      }
    java.nio.ByteBuffer.allocateDirect(bytes).order(java.nio.ByteOrder.nativeOrder())
  }

  class DoubleBuffer(val buffer: java.nio.DoubleBuffer) extends Buffer[Double] {
    override val name                                 = "Double"
    override def update(idx: Int, elem: Double): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Double                = buffer.get(i)
    override def length: Int                          = buffer.capacity()
    override def putAll(xs: Double*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]                = ptr(buffer)
    override def copyToArray: Array[Double] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Double](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class FloatBuffer(val buffer: java.nio.FloatBuffer) extends Buffer[Float] {
    override val name                                = "Float"
    override def update(idx: Int, elem: Float): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Float                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Float*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]               = ptr(buffer)
    override def copyToArray: Array[Float] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Float](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class LongBuffer(val buffer: java.nio.LongBuffer) extends Buffer[Long] {
    override val name                               = "Long"
    override def update(idx: Int, elem: Long): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Long                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Long*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]              = ptr(buffer)
    override def copyToArray: Array[Long] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Long](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class IntBuffer(val buffer: java.nio.IntBuffer) extends Buffer[Int] {
    override val name                              = "Int"
    override def update(idx: Int, elem: Int): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Int                = buffer.get(i)
    override def length: Int                       = buffer.capacity()
    override def putAll(xs: Int*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]             = ptr(buffer)
    override def copyToArray: Array[Int] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Int](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class ShortBuffer(val buffer: java.nio.ShortBuffer) extends Buffer[Short] {
    override val name                                = "Short"
    override def update(idx: Int, elem: Short): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Short                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Short*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]               = ptr(buffer)
    override def copyToArray: Array[Short] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Short](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class ByteBuffer(val buffer: java.nio.ByteBuffer) extends Buffer[Byte] {
    override val name                               = "Byte"
    override def update(idx: Int, elem: Byte): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Byte                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Byte*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]              = ptr(buffer)
    override def copyToArray: Array[Byte] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Byte](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class CharBuffer(val buffer: java.nio.CharBuffer) extends Buffer[Char] {
    override val name                               = "Char"
    override def update(idx: Int, elem: Char): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Char                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Char*): this.type       = { buffer.put(xs.toArray); this }
    override def pointer: Option[Long]              = ptr(buffer)
    override def copyToArray: Array[Char] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Char](buffer.capacity)
        buffer.get(xs)
        xs
      }
  }
  class BoolBuffer(val buffer: java.nio.ByteBuffer) extends Buffer[Boolean] {
    override val name                                  = "Boolean"
    override def update(idx: Int, elem: Boolean): Unit = buffer.put(idx, (if (elem) 1 else 0).toByte)
    override def apply(i: Int): Boolean                = buffer.get(i) != 0
    override def length: Int                           = buffer.capacity()
    override def putAll(xs: Boolean*): this.type = {
      var i = 0
      val n = length
      while (i < n) { update(i, xs(i)); i += 1 }
      this
    }
    override def pointer: Option[Long]       = ptr(buffer)
    override def copyToArray: Array[Boolean] = this.toArray

  }
  class StructBuffer[A](val buffer: java.nio.ByteBuffer)(using S: NativeStruct[A]) extends Buffer[A] {
    override val name                            = "Struct"
    override def update(idx: Int, elem: A): Unit = S.encode(buffer, idx, elem)
    override def apply(i: Int): A                = S.decode(buffer, i)
    override def length: Int                     = buffer.capacity() / S.sizeInBytes
    override def putAll(xs: A*): this.type = {
      var i = 0
      val n = length
      while (i < n) { update(i, xs(i)); i += 1 }
      this
    }
    override def pointer: Option[Long] = ptr(buffer)
    override def copyToArray: Array[A] = ???

  }

  def view[A <: AnyVal](actual: java.nio.ByteBuffer)(using tag: ClassTag[A]): Buffer[A] = {
    actual.order(java.nio.ByteOrder.nativeOrder())
    (tag.runtimeClass match {
      case java.lang.Double.TYPE    => DoubleBuffer(actual.asDoubleBuffer())
      case java.lang.Float.TYPE     => FloatBuffer(actual.asFloatBuffer())
      case java.lang.Long.TYPE      => LongBuffer(actual.asLongBuffer())
      case java.lang.Integer.TYPE   => IntBuffer(actual.asIntBuffer())
      case java.lang.Short.TYPE     => ShortBuffer(actual.asShortBuffer())
      case java.lang.Character.TYPE => CharBuffer(actual.asCharBuffer())
      case java.lang.Boolean.TYPE   => BoolBuffer(actual)
      case java.lang.Byte.TYPE      => ByteBuffer(actual)
      case java.lang.Void.TYPE      => ByteBuffer(actual)
    }).asInstanceOf[Buffer[A]]
  }

  def ofDim[A <: AnyVal](dim: Int)(using tag: ClassTag[A]): Buffer[A] = (tag.runtimeClass match {
    case java.lang.Double.TYPE    => DoubleBuffer(alloc(java.lang.Double.BYTES, dim).asDoubleBuffer())
    case java.lang.Float.TYPE     => FloatBuffer(alloc(java.lang.Float.BYTES, dim).asFloatBuffer())
    case java.lang.Long.TYPE      => LongBuffer(alloc(java.lang.Long.BYTES, dim).asLongBuffer())
    case java.lang.Integer.TYPE   => IntBuffer(alloc(java.lang.Integer.BYTES, dim).asIntBuffer())
    case java.lang.Short.TYPE     => ShortBuffer(alloc(java.lang.Short.BYTES, dim).asShortBuffer())
    case java.lang.Character.TYPE => CharBuffer(alloc(java.lang.Character.BYTES, dim).asCharBuffer())
    case java.lang.Boolean.TYPE   => BoolBuffer(alloc(java.lang.Byte.BYTES, dim))
    case java.lang.Byte.TYPE      => ByteBuffer(alloc(java.lang.Byte.BYTES, dim))
    case java.lang.Void.TYPE      => ByteBuffer(alloc(0, dim))
  }).asInstanceOf[Buffer[A]]

  // AnyVals

  def apply[A <: AnyVal: ClassTag](xs: A*): Buffer[A] = ofDim[A](xs.size).putAll(xs*)
  def ref[A <: AnyVal: ClassTag]: Buffer[A]           = ofDim[A](1)
  def empty[A <: AnyVal: ClassTag]: Buffer[A]         = ofDim[A](0)
  def fill[A <: AnyVal: ClassTag](n: Int)(elem: => A): Buffer[A] =
    ofDim[A](n).putAll(ArraySeq.fill(n)(elem)*)
  def range[A <: AnyVal: Integral: ClassTag](start: A, end: A, step: A): Buffer[A] = {
    val xs = ArraySeq.range[A](start, end, step)
    ofDim[A](xs.size).putAll(xs*)
  }
  def range[A <: AnyVal: Integral: ClassTag](start: A, end: A): Buffer[A] = {
    val xs = ArraySeq.range[A](start, end)
    ofDim[A](xs.size).putAll(xs*)
  }
  def tabulate[A <: AnyVal: ClassTag](n: Int)(f: Int => A): Buffer[A] =
    ofDim[A](n).putAll(ArraySeq.tabulate(n)(f)*)

  // AnyRefs

  def ofZeroed[A <: AnyRef](dim: Int)(using S: NativeStruct[A]): Buffer[A] =
    StructBuffer[A](alloc(S.sizeInBytes, dim)) // zeros by default

  def ofZeroedAny(dim: Int)(using S: NativeStruct[Any]): Buffer[Any] =
    StructBuffer[Any](alloc(S.sizeInBytes, dim)) // zeros by default

  def refAny(x: Any)(using S: NativeStruct[Any]): Buffer[Any] =
    ofZeroedAny(1).putAll(x)

  def apply[A <: AnyRef](xs: A*)(using S: NativeStruct[A]): Buffer[A] =
    ofZeroed(xs.size).putAll(xs*)

  def empty[A <: AnyRef: NativeStruct]: Buffer[A] = //
    apply[A]()
  def fill[A <: AnyRef: ClassTag: NativeStruct](n: Int)(elem: => A): Buffer[A] = apply[A](ArraySeq.fill(n)(elem)*)
  def range[A <: AnyRef: Integral: ClassTag: NativeStruct](start: A, end: A, step: A): Buffer[A] =
    apply[A](ArraySeq.range[A](start, end, step)*)
  def range[A <: AnyRef: Integral: ClassTag: NativeStruct](start: A, end: A): Buffer[A] =
    apply[A](ArraySeq.range[A](start, end)*)
  def tabulate[A <: AnyRef: ClassTag: NativeStruct](n: Int)(f: Int => A): Buffer[A] =
    apply[A](ArraySeq.tabulate(n)(f)*)

}
