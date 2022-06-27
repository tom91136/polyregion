package polyregion.scala

import java.util.Objects
import java.util.concurrent.atomic.AtomicLong
import java.{lang as jl, nio}
import scala.collection.immutable.NumericRange
import scala.collection.mutable
import scala.math.Integral
import scala.reflect.ClassTag

sealed trait Buffer[A] extends mutable.IndexedSeq[A] {
  def name: String
  def buffer: nio.Buffer
  def backingBuffer: nio.ByteBuffer
  def putAll(xs: A*): this.type
  def copyToArray: Array[A]
  override def toString: String = s"Buffer[$name](${mkString(", ")})"
}

object Buffer {

  class DoubleBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.DoubleBuffer, val id: Long)
      extends Buffer[Double] {
    override val name                                 = "Double"
    override def update(idx: Int, elem: Double): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Double                = buffer.get(i)
    override def length: Int                          = buffer.capacity()
    override def putAll(xs: Double*): this.type       = { buffer.put(xs.toArray); this }

    override def copyToArray: Array[Double] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Double](buffer.capacity)
        buffer.get(xs)
        xs
      }

    override def equals(other: Any): Boolean = other match {
      case that: DoubleBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)

  }
  class FloatBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.FloatBuffer, val id: Long)
      extends Buffer[Float] {
    override val name                                = "Float"
    override def update(idx: Int, elem: Float): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Float                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Float*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Float] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Float](buffer.capacity)
        buffer.get(xs)
        xs
      }

    override def equals(other: Any): Boolean = other match {
      case that: FloatBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)
  }
  class LongBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.LongBuffer, val id: Long) extends Buffer[Long] {
    override val name                               = "Long"
    override def update(idx: Int, elem: Long): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Long                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Long*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Long] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Long](buffer.capacity)
        buffer.get(xs)
        xs
      }

    override def equals(other: Any): Boolean = other match {
      case that: LongBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)
  }
  class IntBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.IntBuffer, val id: Long) extends Buffer[Int] {
    override val name                              = "Int"
    override def update(idx: Int, elem: Int): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Int                = buffer.get(i)
    override def length: Int                       = buffer.capacity()
    override def putAll(xs: Int*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Int] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Int](buffer.capacity)
        buffer.get(xs)
        xs
      }
    override def equals(other: Any): Boolean = other match {
      case that: DoubleBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)

  }
  class ShortBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.ShortBuffer, val id: Long)
      extends Buffer[Short] {
    override val name                                = "Short"
    override def update(idx: Int, elem: Short): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Short                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Short*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Short] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Short](buffer.capacity)
        buffer.get(xs)
        xs
      }
    override def equals(other: Any): Boolean = other match {
      case that: ShortBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)

  }
  class ByteBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long) extends Buffer[Byte] {
    override val name                               = "Byte"
    override def update(idx: Int, elem: Byte): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Byte                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Byte*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Byte] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Byte](buffer.capacity)
        buffer.get(xs)
        xs
      }
    override def equals(other: Any): Boolean = other match {
      case that: ByteBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)

  }
  class CharBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.CharBuffer, val id: Long) extends Buffer[Char] {
    override val name                               = "Char"
    override def update(idx: Int, elem: Char): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Char                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Char*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Char] =
      if (buffer.hasArray) buffer.array
      else {
        val xs = new Array[Char](buffer.capacity)
        buffer.get(xs)
        xs
      }
    override def equals(other: Any): Boolean = other match {
      case that: CharBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)

  }
  class BoolBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long)
      extends Buffer[Boolean] {
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
    override def copyToArray: Array[Boolean] = this.toArray
    override def equals(other: Any): Boolean = other match {
      case that: BoolBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)
  }

  class UnitBuffer(val backingBuffer: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long) extends Buffer[Unit] {
    override val name                               = "Unit"
    override def update(idx: Int, elem: Unit): Unit = buffer.put(idx, 0.toByte)
    override def apply(i: Int): Unit                = { buffer.get(i); () }
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Unit*): this.type = {
      var i = 0
      val n = length
      while (i < n) { update(i, xs(i)); i += 1 }
      this
    }
    override def copyToArray: Array[Unit] = this.toArray
    override def equals(other: Any): Boolean = other match {
      case that: UnitBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)
  }

  class StructBuffer[A](val backingBuffer: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long)(using
      S: NativeStruct[A]
  ) extends Buffer[A] {
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
    override def copyToArray: Array[A] = ???
    override def equals(other: Any): Boolean = other match {
      case that: StructBuffer[_] => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
    override def hashCode(): Int = Objects.hashCode(id)
  }

  private final val Counter = AtomicLong(0)

  inline def view[A <: AnyVal](inline actual: nio.ByteBuffer)(using inline tag: ClassTag[A]): Buffer[A] = {
    actual.order(nio.ByteOrder.nativeOrder())
    val id = Counter.getAndIncrement()
    (tag.runtimeClass match {
      case x if x.equals(java.lang.Double.TYPE)    => DoubleBuffer(actual, actual.asDoubleBuffer(), id)
      case x if x.equals(java.lang.Float.TYPE)     => FloatBuffer(actual, actual.asFloatBuffer(), id)
      case x if x.equals(java.lang.Long.TYPE)      => LongBuffer(actual, actual.asLongBuffer(), id)
      case x if x.equals(java.lang.Integer.TYPE)   => IntBuffer(actual, actual.asIntBuffer(), id)
      case x if x.equals(java.lang.Short.TYPE)     => ShortBuffer(actual, actual.asShortBuffer(), id)
      case x if x.equals(java.lang.Character.TYPE) => CharBuffer(actual, actual.asCharBuffer(), id)
      case x if x.equals(java.lang.Boolean.TYPE)   => BoolBuffer(actual, actual, id)
      case x if x.equals(java.lang.Byte.TYPE)      => ByteBuffer(actual, actual, id)
      case x if x.equals(java.lang.Void.TYPE)      => UnitBuffer(actual, actual, id)
    }).asInstanceOf[Buffer[A]]
  }

  inline def view[A <: AnyRef](inline actual: nio.ByteBuffer)(using inline S: NativeStruct[A]): Buffer[A] = {
    actual.order(nio.ByteOrder.nativeOrder())
    StructBuffer[A](actual, actual, Counter.getAndIncrement())
  }

  private inline def checkedAlloc(inline elemSize: Int, inline n: Int): nio.ByteBuffer = {
    val bytes =
      try math.multiplyExact(elemSize, n)
      catch {
        case e: Throwable =>
          throw new IllegalArgumentException(
            s"Cannot allocated buffer of size ${elemSize * n} (overflow or negative size?)",
            e
          )
      }
    nio.ByteBuffer.allocateDirect(bytes).order(nio.ByteOrder.nativeOrder())
  }

  private inline def mkBuffer[A, B](
      inline size: Int,
      inline dim: Int,
      inline mkSpecific: nio.ByteBuffer => A,
      inline mkCol: (nio.ByteBuffer, A) => B
  ) = {
    val buffer = checkedAlloc(size, dim)
    mkCol(buffer, mkSpecific(buffer))
  }

  inline def ofDim[A <: AnyVal](inline dim: Int)(using inline tag: ClassTag[A]): Buffer[A] = {
    val id = Counter.getAndIncrement()
    (tag.runtimeClass match {
      case x if x.equals(jl.Double.TYPE)  => mkBuffer(jl.Double.BYTES, dim, _.asDoubleBuffer(), DoubleBuffer(_, _, id))
      case x if x.equals(jl.Float.TYPE)   => mkBuffer(jl.Float.BYTES, dim, _.asFloatBuffer(), FloatBuffer(_, _, id))
      case x if x.equals(jl.Long.TYPE)    => mkBuffer(jl.Long.BYTES, dim, _.asLongBuffer(), LongBuffer(_, _, id))
      case x if x.equals(jl.Integer.TYPE) => mkBuffer(jl.Integer.BYTES, dim, _.asIntBuffer(), IntBuffer(_, _, id))
      case x if x.equals(jl.Short.TYPE)   => mkBuffer(jl.Short.BYTES, dim, _.asShortBuffer(), ShortBuffer(_, _, id))
      case x if x.equals(jl.Character.TYPE) => mkBuffer(jl.Character.BYTES, dim, _.asCharBuffer(), CharBuffer(_, _, id))
      case x if x.equals(jl.Boolean.TYPE)   => mkBuffer(jl.Byte.BYTES, dim, identity, BoolBuffer(_, _, id))
      case x if x.equals(jl.Byte.TYPE)      => mkBuffer(jl.Byte.BYTES, dim, identity, ByteBuffer(_, _, id))
      case x if x.equals(jl.Void.TYPE)      => mkBuffer(jl.Byte.BYTES, dim, identity, UnitBuffer(_, _, id))
    }).asInstanceOf[Buffer[A]]
  }

  // zeros by default
  inline def ofDim[A <: AnyRef](inline dim: Int)(using inline S: NativeStruct[A]): Buffer[A] =
    mkBuffer(S.sizeInBytes, dim, identity, StructBuffer(_, _, Counter.getAndIncrement()))

  private inline def eachN[A](inline n: Int, a: A)(inline f: (A, Int) => Unit): A = {
    var i = 0
    while (i < n) { f(a, i); i += 1 }
    a
  }

  // == AnyVals ==

  inline def apply[A <: AnyVal: ClassTag](inline ys: A*): Buffer[A] = ofDim[A](ys.size).putAll(ys*)
  inline def empty[A <: AnyVal: ClassTag]: Buffer[A]                = ofDim[A](0)
  inline def from[A <: AnyVal: ClassTag](inline ys: scala.collection.Seq[A]): Buffer[A] =
    eachN(ys.size, ofDim[A](ys.size))((xs, i) => xs(i) = ys(i))
  inline def fill[A <: AnyVal: ClassTag](inline n: Int)(a: => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = a)
  inline def tabulate[A <: AnyVal: ClassTag](inline n: Int)(inline f: Int => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = f(i))

  inline def range[A <: AnyVal: ClassTag](using inline num: Integral[A])(start: A, end: A, step: A): Buffer[A] = {
    val range = NumericRange[A](start, end, step)
    eachN(range.size, ofDim[A](range.size))((xs, i) => xs(i) = range(i))
  }
  inline def range[A <: AnyVal: ClassTag](using inline num: Integral[A])(start: A, end: A): Buffer[A] =
    range[A](start, end, num.one)

  // == AnyRefs ==

  inline def apply[A <: AnyRef: NativeStruct](inline xs: A*): Buffer[A] = ofDim[A](xs.size).putAll(xs*)
  inline def empty[A <: AnyRef: NativeStruct]: Buffer[A]                = ofDim[A](0)
  inline def from[A <: AnyRef: NativeStruct](inline ys: scala.collection.Seq[A]): Buffer[A] =
    eachN(ys.size, ofDim[A](ys.size))((xs, i) => xs(i) = ys(i))
  inline def fill[A <: AnyRef: NativeStruct](inline n: Int)(inline a: => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = a)
  inline def tabulate[A <: AnyRef: NativeStruct](inline n: Int)(inline f: Int => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = f(i))

}
