package polyregion.scala

import java.util.Objects
import java.util.concurrent.atomic.AtomicLong
import java.{lang as jl, nio}
import scala.collection.immutable.NumericRange
import scala.collection.mutable
import scala.compiletime.*
import scala.math.Integral
import scala.reflect.ClassTag

sealed trait Buffer[A] extends mutable.IndexedSeq[A] {
  def name: String
  def buffer: nio.Buffer
  def backing: nio.ByteBuffer
  def putAll(xs: A*): this.type
  def copyToArray: Array[A]
  override def toString: String = s"Buffer[$name](${mkString(", ")})"
}

object Buffer {

  private inline def drainArray[A: ClassTag, B <: nio.Buffer](
      inline buffer: B,
      inline f: (B, Array[A]) => Unit,
      inline g: B => Array[A]
  ) = if (buffer.hasArray) g(buffer) else { val xs = new Array[A](buffer.capacity); f(buffer, xs); xs }

  private inline def updateAll[A, B <: Buffer[A]](inline buffer: B, inline xs: A*) = {
    var i = 0
    val n = buffer.length
    while (i < n) { buffer.update(i, xs(i)); i += 1 }
    buffer
  }

  private inline def mkBuffer[A, B](
      inline elementSize: Int,
      inline dim: Int,
      inline mkSpecific: nio.ByteBuffer => A,
      inline mkCol: (nio.ByteBuffer, A) => B
  ) = {
    val bytes =
      try math.multiplyExact(dim, elementSize)
      catch {
        case e: Throwable =>
          throw new IllegalArgumentException(
            s"Cannot allocated buffer of size ${dim * elementSize} bytes (overflow or negative size?)",
            e
          )
      }
    val buffer = nio.ByteBuffer.allocateDirect(bytes).order(nio.ByteOrder.nativeOrder())
    mkCol(buffer, mkSpecific(buffer))
  }

  private final val IdCounter = AtomicLong(0)

  object DoubleBuffer {
    def apply(dim: Int): DoubleBuffer =
      mkBuffer(jl.Double.BYTES, dim, _.asDoubleBuffer(), new DoubleBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class DoubleBuffer(val backing: nio.ByteBuffer, val buffer: nio.DoubleBuffer, val id: Long) //
      extends Buffer[Double] {
    override val name                                 = "Double"
    override def update(idx: Int, elem: Double): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Double                = buffer.get(i)
    override def length: Int                          = buffer.capacity()
    override def putAll(xs: Double*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Double]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                      = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: DoubleBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                  => false
    }
  }
  object FloatBuffer {
    def apply(dim: Int): FloatBuffer =
      mkBuffer(jl.Float.BYTES, dim, _.asFloatBuffer(), new FloatBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class FloatBuffer(val backing: nio.ByteBuffer, val buffer: nio.FloatBuffer, val id: Long) //
      extends Buffer[Float] {
    override val name                                = "Float"
    override def update(idx: Int, elem: Float): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Float                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Float*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Float]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                     = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: FloatBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                 => false
    }
  }
  object LongBuffer {
    def apply(dim: Int): LongBuffer =
      mkBuffer(jl.Long.BYTES, dim, _.asLongBuffer(), new LongBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class LongBuffer(val backing: nio.ByteBuffer, val buffer: nio.LongBuffer, val id: Long) //
      extends Buffer[Long] {
    override val name                               = "Long"
    override def update(idx: Int, elem: Long): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Long                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Long*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Long]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                    = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: LongBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                => false
    }
  }
  object IntBuffer {
    def apply(dim: Int): IntBuffer =
      mkBuffer(jl.Integer.BYTES, dim, _.asIntBuffer(), new IntBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class IntBuffer(val backing: nio.ByteBuffer, val buffer: nio.IntBuffer, val id: Long) //
      extends Buffer[Int] {
    override val name                              = "Int"
    override def update(idx: Int, elem: Int): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Int                = buffer.get(i)
    override def length: Int                       = buffer.capacity()
    override def putAll(xs: Int*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Int]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                   = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: IntBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _               => false
    }
  }
  object ShortBuffer {
    def apply(dim: Int): ShortBuffer =
      mkBuffer(jl.Short.BYTES, dim, _.asShortBuffer(), new ShortBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class ShortBuffer(val backing: nio.ByteBuffer, val buffer: nio.ShortBuffer, val id: Long) //
      extends Buffer[Short] {
    override val name                                = "Short"
    override def update(idx: Int, elem: Short): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Short                = buffer.get(i)
    override def length: Int                         = buffer.capacity()
    override def putAll(xs: Short*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Short]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                     = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: ShortBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                 => false
    }

  }
  object ByteBuffer {
    def apply(dim: Int): ByteBuffer =
      mkBuffer(jl.Byte.BYTES, dim, identity, new ByteBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class ByteBuffer(val backing: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long) //
      extends Buffer[Byte] {
    override val name                               = "Byte"
    override def update(idx: Int, elem: Byte): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Byte                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Byte*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Byte]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                    = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: ByteBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                => false
    }

  }
  object CharBuffer {
    def apply(dim: Int): CharBuffer =
      mkBuffer(jl.Character.BYTES, dim, _.asCharBuffer(), new CharBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class CharBuffer(val backing: nio.ByteBuffer, val buffer: nio.CharBuffer, val id: Long) //
      extends Buffer[Char] {
    override val name                               = "Char"
    override def update(idx: Int, elem: Char): Unit = buffer.put(idx, elem)
    override def apply(i: Int): Char                = buffer.get(i)
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Char*): this.type       = { buffer.put(xs.toArray); this }
    override def copyToArray: Array[Char]           = drainArray(buffer, _.get(_), _.array)
    override def hashCode(): Int                    = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: CharBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                => false
    }

  }
  object BoolBuffer {
    def apply(dim: Int): BoolBuffer =
      mkBuffer(jl.Byte.BYTES, dim, identity, new BoolBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class BoolBuffer(val backing: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long) //
      extends Buffer[Boolean] {
    override val name                                  = "Boolean"
    override def update(idx: Int, elem: Boolean): Unit = buffer.put(idx, (if (elem) 1 else 0).toByte)
    override def apply(i: Int): Boolean                = buffer.get(i) != 0
    override def length: Int                           = buffer.capacity()
    override def putAll(xs: Boolean*): this.type       = updateAll(this, xs: _*)
    override def copyToArray: Array[Boolean]           = this.toArray
    override def hashCode(): Int                       = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: BoolBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                => false
    }
  }
  object UnitBuffer {
    def apply(dim: Int): UnitBuffer =
      mkBuffer(jl.Byte.BYTES, dim, identity, new UnitBuffer(_, _, IdCounter.getAndIncrement()))
  }
  class UnitBuffer(val backing: nio.ByteBuffer, val buffer: nio.ByteBuffer, val id: Long) //
      extends Buffer[Unit] {
    override val name                               = "Unit"
    override def update(idx: Int, elem: Unit): Unit = buffer.put(idx, 0.toByte)
    override def apply(i: Int): Unit                = { buffer.get(i); () }
    override def length: Int                        = buffer.capacity()
    override def putAll(xs: Unit*): this.type       = updateAll(this, xs: _*)
    override def copyToArray: Array[Unit]           = this.toArray
    override def hashCode(): Int                    = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: UnitBuffer => super.equals(that) && that.canEqual(this) && id == that.id
      case _                => false
    }
  }

  object StructBuffer {
    inline def apply[A](inline dim: Int): StructBuffer[A] = {
      val ns = compiletime.deriveNativeStruct[A]
      mkBuffer(
        ns.sizeInBytes,
        dim,
        identity,
        new StructBuffer[A](_, _, IdCounter.getAndIncrement())(using ns, summonInline[ClassTag[A]])
      )
    }
  }
  class StructBuffer[A](
      val backing: nio.ByteBuffer,
      val buffer: nio.ByteBuffer,
      val id: Long
  )(using S: NativeStruct[A], C: ClassTag[A])
      extends Buffer[A] {
    override val name                            = C.runtimeClass.getName
    override def update(idx: Int, elem: A): Unit = S.encode(buffer, idx, elem)
    override def apply(i: Int): A                = S.decode(buffer, i)
    override def length: Int                     = buffer.capacity() / S.sizeInBytes
    override def putAll(xs: A*): this.type       = updateAll(this, xs: _*)
    override def copyToArray: Array[A]           = this.toArray
    override def hashCode(): Int                 = Objects.hashCode(id)
    override def equals(other: Any): Boolean = other match {
      case that: StructBuffer[_] => super.equals(that) && that.canEqual(this) && id == that.id
      case _                     => false
    }
  }

  inline def ofDim[A](inline dim: Int): Buffer[A] = inline erasedValue[A] match {
    case _: Double /* */ | _: jl.Double    => DoubleBuffer(dim).asInstanceOf
    case _: Float /*  */ | _: jl.Float     => FloatBuffer(dim).asInstanceOf
    case _: Long /*   */ | _: jl.Long      => LongBuffer(dim).asInstanceOf
    case _: Int /*    */ | _: jl.Integer   => IntBuffer(dim).asInstanceOf
    case _: Short /*  */ | _: jl.Short     => ShortBuffer(dim).asInstanceOf
    case _: Char /*   */ | _: jl.Character => CharBuffer(dim).asInstanceOf
    case _: Boolean /**/ | _: jl.Boolean   => BoolBuffer(dim).asInstanceOf
    case _: Byte /*   */ | _: jl.Byte      => ByteBuffer(dim).asInstanceOf
    case _: Unit /*   */ | _: jl.Void      => UnitBuffer(dim).asInstanceOf
    case _: AnyRef                         => StructBuffer[A](dim).asInstanceOf
    case _ => // should not happen
      error("Unexpected element type: cannot instantiate Buffer with a type that isn't <: AnyVal or <: AnyRef)")
  }

  private inline def eachN[A](inline n: Int, inline a: A)(inline f: (A, Int) => Unit): A = {
    var i = 0
    while (i < n) { f(a, i); i += 1 }
    a
  }

  inline def apply[A](inline ys: A*): Buffer[A] = ofDim[A](ys.size).putAll(ys*)
  inline def empty[A]: Buffer[A]                = ofDim[A](0)
  inline def from[A](inline ys: scala.collection.Seq[A]): Buffer[A] =
    eachN(ys.size, ofDim[A](ys.size))((xs, i) => xs(i) = ys(i))
  inline def fill[A](inline n: Int)(a: => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = a)
  inline def tabulate[A](inline n: Int)(inline f: Int => A): Buffer[A] =
    eachN(n, ofDim[A](n))((xs, i) => xs(i) = f(i))

  inline def range[A](using inline num: Integral[A])(start: A, end: A, step: A): Buffer[A] = {
    val range = NumericRange[A](start, end, step)
    eachN(range.size, ofDim[A](range.size))((xs, i) => xs(i) = range(i))
  }
  inline def range[A](using inline num: Integral[A])(start: A, end: A): Buffer[A] = range[A](start, end, num.one)

}
