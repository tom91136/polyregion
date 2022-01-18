package polyregion

import scala.reflect.ClassTag
import scala.collection.mutable

object Runtime {

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
    class StructBuffer[A](val buffer: java.nio.ByteBuffer)(using struct: NativeStruct[A]) extends Buffer[A] {
      override def update(idx: Int, elem: A): Unit = ???
      override def apply(i: Int): A                = ???
      override def length: Int                     = buffer.capacity() / struct.sizeInBytes
      override def putAll(xs: A*): this.type       = ???
      override def pointer: Option[Long]           = ptr(buffer)
    }

    // case class Z(a : Int, b: String)
    //

    enum Type {
      case Double, Float, Long, Int, Short, Char, Byte // String
    }

    trait NativeStruct[A] {
      def sizeInBytes: Int
      def members: IndexedSeq[(Type, String)]
      def encode(offset: Int, buffer: ByteBuffer, a: A): Unit
      def decode(offset: Int, buffer: ByteBuffer): A
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

    def ref[T <: AnyVal](using tag: ClassTag[T]): Buffer[T] = ofDim[T](1)
    def nil[T <: AnyVal](using tag: ClassTag[T]): Buffer[T] = ofDim[T](0)

    def apply[T <: AnyVal](xs: T*)(using tag: ClassTag[T]): Buffer[T] = ofDim[T](xs.size).putAll(xs*)
    def apply[T <: AnyRef](xs: T*)(using struct: NativeStruct[T]): Buffer[T] =
      StructBuffer[T](alloc(struct.sizeInBytes * xs.size))

  }

}
