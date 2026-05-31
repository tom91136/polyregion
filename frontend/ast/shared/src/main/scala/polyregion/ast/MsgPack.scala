package polyregion.ast

import java.nio.charset.StandardCharsets

import scala.collection.immutable.ArraySeq
import scala.collection.mutable
import scala.compiletime.{constValue, erasedValue, summonFrom}
import scala.deriving.Mirror
import scala.reflect.ClassTag

object MsgPack {

  trait Encoder[A] {
    def encode(w: Writer, a: A): Unit
    final def encode(a: A): Array[Byte] = {
      val w = new Writer
      encode(w, a)
      w.toByteArray
    }
  }
  trait Decoder[A] { def decode(r: Reader): A }
  trait Codec[A] extends Encoder[A], Decoder[A]

  final class MsgPackException(message: String) extends RuntimeException(message)

  trait ByteInput {
    def length: Int
    def unsignedByteAt(index: Int): Int
    def unsignedShortBE(index: Int): Int =
      (unsignedByteAt(index) << 8) | unsignedByteAt(index + 1)
    def int32BE(index: Int): Int =
      (unsignedByteAt(index) << 24) |
        (unsignedByteAt(index + 1) << 16) |
        (unsignedByteAt(index + 2) << 8) |
        unsignedByteAt(index + 3)
    def int64BE(index: Int): Long =
      (unsignedByteAt(index).toLong << 56) |
        (unsignedByteAt(index + 1).toLong << 48) |
        (unsignedByteAt(index + 2).toLong << 40) |
        (unsignedByteAt(index + 3).toLong << 32) |
        (unsignedByteAt(index + 4).toLong << 24) |
        (unsignedByteAt(index + 5).toLong << 16) |
        (unsignedByteAt(index + 6).toLong << 8) |
        unsignedByteAt(index + 7).toLong
    def copyToArray(srcPos: Int, dest: Array[Byte], destPos: Int, length: Int): Unit = {
      var i = 0
      while (i < length) {
        dest(destPos + i) = unsignedByteAt(srcPos + i).toByte
        i += 1
      }
    }
    def utf8String(srcPos: Int, length: Int): String = {
      val out = new Array[Byte](length)
      copyToArray(srcPos, out, 0, length)
      String(out, StandardCharsets.UTF_8)
    }
  }

  final class ArrayByteInput(private val bytes: Array[Byte]) extends ByteInput {
    def length: Int                     = bytes.length
    def unsignedByteAt(index: Int): Int = bytes(index) & 0xff
    override def copyToArray(srcPos: Int, dest: Array[Byte], destPos: Int, length: Int): Unit =
      Array.copy(bytes, srcPos, dest, destPos, length)
    override def utf8String(srcPos: Int, length: Int): String =
      String(bytes, srcPos, length, StandardCharsets.UTF_8)
  }

  trait ByteOutput {
    def size: Int
    def writeByte(x: Int): Unit
    final def writeBytes(xs: Array[Byte]): Unit = writeBytes(xs, 0, xs.length)
    def writeBytes(xs: Array[Byte], srcPos: Int, length: Int): Unit = {
      var i = 0
      while (i < length) {
        writeByte(xs(srcPos + i))
        i += 1
      }
    }
    def writeShortBE(x: Int): Unit = {
      writeByte(x >>> 8)
      writeByte(x)
    }
    def writeIntBE(x: Int): Unit = {
      writeByte(x >>> 24)
      writeByte(x >>> 16)
      writeByte(x >>> 8)
      writeByte(x)
    }
    def writeLongBE(x: Long): Unit = {
      writeByte((x >>> 56).toInt)
      writeByte((x >>> 48).toInt)
      writeByte((x >>> 40).toInt)
      writeByte((x >>> 32).toInt)
      writeByte((x >>> 24).toInt)
      writeByte((x >>> 16).toInt)
      writeByte((x >>> 8).toInt)
      writeByte(x.toInt)
    }
    def toByteArray: Array[Byte]
  }

  object NullByteOutput extends ByteOutput {
    def size: Int                                                            = 0
    def writeByte(x: Int): Unit                                              = ()
    override def writeBytes(xs: Array[Byte], srcPos: Int, length: Int): Unit = ()
    override def writeShortBE(x: Int): Unit                                  = ()
    override def writeIntBE(x: Int): Unit                                    = ()
    override def writeLongBE(x: Long): Unit                                  = ()
    def toByteArray: Array[Byte]                                             = Array.emptyByteArray
  }

  final class ArrayByteOutput(initialSize: Int = 256) extends ByteOutput {
    private var bytes  = new Array[Byte](math.max(16, initialSize))
    private var cursor = 0

    def size: Int = cursor

    def toByteArray: Array[Byte] = {
      val out = new Array[Byte](cursor)
      Array.copy(bytes, 0, out, 0, cursor)
      out
    }

    private def ensure(extra: Int): Unit = {
      val needed = cursor + extra
      if (needed > bytes.length) {
        var next = bytes.length
        while (next < needed) next = next << 1
        val resized = new Array[Byte](next)
        Array.copy(bytes, 0, resized, 0, cursor)
        bytes = resized
      }
    }

    def writeByte(x: Int): Unit = {
      ensure(1)
      bytes(cursor) = x.toByte
      cursor += 1
    }

    override def writeBytes(xs: Array[Byte], srcPos: Int, length: Int): Unit = {
      ensure(length)
      Array.copy(xs, srcPos, bytes, cursor, length)
      cursor += length
    }

    override def writeShortBE(x: Int): Unit = {
      ensure(2)
      bytes(cursor) = (x >>> 8).toByte
      bytes(cursor + 1) = x.toByte
      cursor += 2
    }

    override def writeIntBE(x: Int): Unit = {
      ensure(4)
      bytes(cursor) = (x >>> 24).toByte
      bytes(cursor + 1) = (x >>> 16).toByte
      bytes(cursor + 2) = (x >>> 8).toByte
      bytes(cursor + 3) = x.toByte
      cursor += 4
    }

    override def writeLongBE(x: Long): Unit = {
      ensure(8)
      bytes(cursor) = (x >>> 56).toByte
      bytes(cursor + 1) = (x >>> 48).toByte
      bytes(cursor + 2) = (x >>> 40).toByte
      bytes(cursor + 3) = (x >>> 32).toByte
      bytes(cursor + 4) = (x >>> 24).toByte
      bytes(cursor + 5) = (x >>> 16).toByte
      bytes(cursor + 6) = (x >>> 8).toByte
      bytes(cursor + 7) = x.toByte
      cursor += 8
    }
  }

  final class StringInterner {
    private val ids            = mutable.LinkedHashMap.empty[String, Int]
    def id(x: String): Int     = ids.getOrElseUpdate(x, ids.size)
    def entries: Array[String] = ids.keysIterator.toArray
  }

  final class Writer(output: ByteOutput = ArrayByteOutput(), private var interner: StringInterner | Null = null) {
    def size: Int                                  = output.size
    def setStringInterner(x: StringInterner): Unit = interner = x
    def toByteArray: Array[Byte]                   = output.toByteArray

    private def byte(x: Int): Unit              = output.writeByte(x)
    private def rawBytes(xs: Array[Byte]): Unit = output.writeBytes(xs)
    private def raw16(x: Int): Unit             = output.writeShortBE(x)
    private def raw32(x: Int): Unit             = output.writeIntBE(x)
    private def raw64(x: Long): Unit            = output.writeLongBE(x)

    def writeNil(): Unit = byte(0xc0)

    def writeBoolean(x: Boolean): Unit = byte(if (x) 0xc3 else 0xc2)

    def writeInt32(x: Int): Unit =
      if (x >= 0 && x <= 0x7f) byte(x)
      else if (x >= -32 && x < 0) byte(x & 0xff)
      else if (x >= Byte.MinValue && x <= Byte.MaxValue) {
        byte(0xd0)
        byte(x)
      } else if (x >= Short.MinValue && x <= Short.MaxValue) {
        byte(0xd1)
        raw16(x)
      } else {
        byte(0xd2)
        raw32(x)
      }

    def writeInt64(x: Long): Unit =
      if (x >= Int.MinValue && x <= Int.MaxValue) writeInt32(x.toInt)
      else {
        byte(0xd3)
        raw64(x)
      }

    def writeFloat32(x: Float): Unit = {
      byte(0xca)
      raw32(java.lang.Float.floatToRawIntBits(x))
    }

    def writeFloat64(x: Double): Unit = {
      byte(0xcb)
      raw64(java.lang.Double.doubleToRawLongBits(x))
    }

    def writeString(x: String): Unit = interner match {
      case null              => writeStringLiteral(x)
      case s: StringInterner => writeInt32(s.id(x))
    }

    def writeStringLiteral(x: String): Unit = {
      val data = x.getBytes(StandardCharsets.UTF_8)
      val n    = data.length
      if (n <= 31) byte(0xa0 | n)
      else if (n <= 0xff) {
        byte(0xd9)
        byte(n)
      } else if (n <= 0xffff) {
        byte(0xda)
        raw16(n)
      } else {
        byte(0xdb)
        raw32(n)
      }
      rawBytes(data)
    }

    def writeArrayHeader(n: Int): Unit =
      if (n < 0) throw MsgPackException(s"Negative array size: $n")
      else if (n <= 15) byte(0x90 | n)
      else if (n <= 0xffff) {
        byte(0xdc)
        raw16(n)
      } else {
        byte(0xdd)
        raw32(n)
      }
  }

  final class Reader(private val bytes: ByteInput, private var stringTable: Array[String] | Null = null) {
    private var cursor = 0

    def offset: Int                             = cursor
    def isAtEnd: Boolean                        = cursor == bytes.length
    def setStringTable(xs: Array[String]): Unit = stringTable = xs
    def nextIsArray: Boolean =
      cursor < bytes.length && {
        val m = bytes.unsignedByteAt(cursor)
        ((m & 0xf0) == 0x90) || m == 0xdc || m == 0xdd
      }

    private def fail(message: String): Nothing =
      throw MsgPackException(s"$message at byte $cursor")

    private def require(n: Int): Unit =
      if (cursor + n > bytes.length) fail(s"Unexpected end of input, need $n byte(s)")

    private def u8(): Int = {
      require(1)
      val x = bytes.unsignedByteAt(cursor)
      cursor += 1
      x
    }

    private def i8(): Int = u8().toByte.toInt

    private def u16(): Int = {
      require(2)
      val x = bytes.unsignedShortBE(cursor)
      cursor += 2
      x
    }

    private def i16(): Int = u16().toShort.toInt

    private def i32(): Int = {
      require(4)
      val x = bytes.int32BE(cursor)
      cursor += 4
      x
    }

    private def i64(): Long = {
      require(8)
      val x = bytes.int64BE(cursor)
      cursor += 8
      x
    }

    private def u32ToInt(): Int = {
      val x = i32()
      if (x < 0) fail("Length exceeds Int.MaxValue")
      x
    }

    private def markerName(marker: Int): String = f"0x$marker%02x"

    def readNil(): Unit = {
      val m = u8()
      if (m != 0xc0) fail(s"Expected nil, got ${markerName(m)}")
    }

    def tryReadNil(): Boolean =
      if (cursor < bytes.length && bytes.unsignedByteAt(cursor) == 0xc0) {
        cursor += 1
        true
      } else false

    def readBoolean(): Boolean = u8() match {
      case 0xc2 => false
      case 0xc3 => true
      case m    => fail(s"Expected boolean, got ${markerName(m)}")
    }

    private def readIntegralLong(): Long = {
      val m = u8()
      if (m <= 0x7f) m.toLong
      else if (m >= 0xe0) m.toByte.toLong
      else
        m match {
          case 0xcc => u8().toLong
          case 0xcd => u16().toLong
          case 0xce => i32().toLong & 0xffffffffL
          case 0xcf =>
            val x = i64()
            if (x < 0) fail("uint64 value exceeds Long.MaxValue")
            x
          case 0xd0 => i8().toLong
          case 0xd1 => i16().toLong
          case 0xd2 => i32().toLong
          case 0xd3 => i64()
          case _    => fail(s"Expected integer, got ${markerName(m)}")
        }
    }

    def readInt32(): Int = {
      val x = readIntegralLong()
      if (x < Int.MinValue || x > Int.MaxValue) fail(s"Integer out of Int range: $x")
      x.toInt
    }

    def readInt64(): Long = readIntegralLong()

    def readFloat32(): Float = u8() match {
      case 0xca => java.lang.Float.intBitsToFloat(i32())
      case 0xcb =>
        val d = java.lang.Double.longBitsToDouble(i64())
        val f = d.toFloat
        if (java.lang.Double.isNaN(d)) f
        else if (f.toDouble == d) f
        else fail(s"Float64 to Float32 conversion with loss of precision: $d")
      case m => fail(s"Expected Float32/Float64, got ${markerName(m)}")
    }

    def readFloat64(): Double = u8() match {
      case 0xcb => java.lang.Double.longBitsToDouble(i64())
      case m    => fail(s"Expected Float64, got ${markerName(m)}")
    }

    def readString(): String =
      if (stringTable == null) readStringLiteral()
      else {
        val xs = stringTable.asInstanceOf[Array[String]]
        val id = readInt32()
        if (id < 0 || id >= xs.length) fail(s"Bad string table id: $id")
        xs(id)
      }

    def readStringLiteral(): String = {
      val m = u8()
      val n =
        if ((m & 0xe0) == 0xa0) m & 0x1f
        else
          m match {
            case 0xd9 => u8()
            case 0xda => u16()
            case 0xdb => u32ToInt()
            case _    => fail(s"Expected string, got ${markerName(m)}")
          }
      require(n)
      val out = bytes.utf8String(cursor, n)
      cursor += n
      out
    }

    def readArrayHeader(): Int = {
      val m = u8()
      if ((m & 0xf0) == 0x90) m & 0x0f
      else
        m match {
          case 0xdc => u16()
          case 0xdd => u32ToInt()
          case _    => fail(s"Expected array, got ${markerName(m)}")
        }
    }
  }

  object Codec {

    def apply[A](f: (Writer, A) => Unit, g: Reader => A): Codec[A] = new Codec[A] {
      def encode(w: Writer, x: A): Unit = f(w, x)
      def decode(r: Reader): A          = g(r)
    }

    private inline def summonOrDerive[A]: Codec[A] = summonFrom {
      case c: Codec[A]     => c
      case m: Mirror.Of[A] => derived[A](using m)
    }

    given Codec[Boolean] = Codec(_.writeBoolean(_), _.readBoolean())
    given Codec[String]  = Codec(_.writeString(_), _.readString())
    given Codec[Byte]    = Codec((w, x) => w.writeInt32(x.toInt), _.readInt32().toByte)
    given Codec[Char]    = Codec((w, x) => w.writeInt32(x.toInt), _.readInt32().toChar)
    given Codec[Short]   = Codec((w, x) => w.writeInt32(x.toInt), _.readInt32().toShort)
    given Codec[Int]     = Codec(_.writeInt32(_), _.readInt32())
    given Codec[Long]    = Codec(_.writeInt64(_), _.readInt64())
    given Codec[Float]   = Codec(_.writeFloat32(_), _.readFloat32())
    given Codec[Double]  = Codec(_.writeFloat64(_), _.readFloat64())

    inline given [A]: Codec[List[A]] = {
      val C = summonOrDerive[A]
      Codec(
        (w, xs) => {
          w.writeArrayHeader(xs.size)
          var rest = xs
          while (rest.nonEmpty) {
            C.encode(w, rest.head)
            rest = rest.tail
          }
        },
        r => {
          val n = r.readArrayHeader()
          val b = List.newBuilder[A]
          var i = 0
          while (i < n) {
            b += C.decode(r)
            i += 1
          }
          b.result()
        }
      )
    }

    inline given [A]: Codec[Set[A]] = {
      val C = summonOrDerive[A]
      Codec(
        (w, xs) => {
          w.writeArrayHeader(xs.size)
          xs.foreach(C.encode(w, _))
        },
        r => {
          val n = r.readArrayHeader()
          val b = Set.newBuilder[A]
          var i = 0
          while (i < n) {
            b += C.decode(r)
            i += 1
          }
          b.result()
        }
      )
    }

    inline given [A: ClassTag]: Codec[ArraySeq[A]] = {
      val C = summonOrDerive[A]
      Codec(
        (w, xs) => {
          w.writeArrayHeader(xs.length)
          var i = 0
          while (i < xs.length) {
            C.encode(w, xs(i))
            i += 1
          }
        },
        r => {
          val n  = r.readArrayHeader()
          val xs = new Array[A](n)
          var i  = 0
          while (i < n) {
            xs(i) = C.decode(r)
            i += 1
          }
          ArraySeq.unsafeWrapArray(xs)
        }
      )
    }

    inline given [A, B]: Codec[Map[A, B]] = {
      val C = summonOrDerive[(A, B)]
      Codec(
        (w, xs) => {
          w.writeArrayHeader(xs.size)
          xs.foreach(C.encode(w, _))
        },
        r => {
          val n = r.readArrayHeader()
          val b = Map.newBuilder[A, B]
          var i = 0
          while (i < n) {
            b += C.decode(r)
            i += 1
          }
          b.result()
        }
      )
    }

    inline given [T0, T1]: Codec[(T0, T1)] = {
      val C0 = summonOrDerive[T0]
      val C1 = summonOrDerive[T1]
      Codec(
        (w, x) => {
          w.writeArrayHeader(2)
          C0.encode(w, x._1)
          C1.encode(w, x._2)
        },
        r => {
          val n = r.readArrayHeader()
          if (n != 2) throw MsgPackException(s"Expected tuple array of size 2, got $n")
          (C0.decode(r), C1.decode(r))
        }
      )
    }

    inline given [A]: Codec[Option[A]] = {
      val C = summonOrDerive[A]
      Codec(
        (w, x) => x.fold(w.writeNil())(C.encode(w, _)),
        r => if (r.tryReadNil()) None else Some(C.decode(r))
      )
    }

    private trait SumCase {
      def arity: Int
      def encodeFields(w: Writer, x: Any): Unit
      def decodeFields(r: Reader, fieldCount: Int): Any
    }

    private def mkSumCase[A](m: Mirror.ProductOf[A], codecs: Array[Codec[?]]): SumCase =
      new SumCase {
        def arity: Int = codecs.length
        def encodeFields(w: Writer, x: Any): Unit = {
          val p = x.asInstanceOf[Product]
          var i = 0
          while (i < codecs.length) {
            codecs(i).asInstanceOf[Codec[Any]].encode(w, p.productElement(i))
            i += 1
          }
        }
        def decodeFields(r: Reader, n: Int): Any = {
          if (n != codecs.length)
            throw MsgPackException(s"Expected sum case with ${codecs.length} field(s), got $n")
          val arr = new Array[Any](codecs.length)
          var i   = 0
          while (i < codecs.length) {
            arr(i) = codecs(i).decode(r)
            i += 1
          }
          m.fromProduct(Tuple.fromArray(arr))
        }
      }

    private inline def summonCodecs[T <: Tuple]: Array[Codec[?]] = inline erasedValue[T] match {
      case _: EmptyTuple => Array.empty
      case _: (t *: ts) =>
        val head = summonOrDerive[t]
        val tail = summonCodecs[ts]
        val out  = new Array[Codec[?]](tail.length + 1)
        out(0) = head
        Array.copy(tail, 0, out, 1, tail.length)
        out
    }

    private inline def sumCase[A]: SumCase = summonFrom { case p: Mirror.ProductOf[A] =>
      mkSumCase[A](p, summonCodecs[p.MirroredElemTypes])
    }

    private inline def summonCases[T <: Tuple]: Array[SumCase] = inline erasedValue[T] match {
      case _: EmptyTuple => Array.empty
      case _: (t *: ts) =>
        val head = sumCase[t]
        val tail = summonCases[ts]
        val out  = new Array[SumCase](tail.length + 1)
        out(0) = head
        Array.copy(tail, 0, out, 1, tail.length)
        out
    }

    inline def derived[T](using m: Mirror.Of[T]): Codec[T] = inline m match {
      case s: Mirror.SumOf[T] =>
        lazy val cases = summonCases[s.MirroredElemTypes]
        Codec[T](
          (w, x) => {
            val ord = s.ordinal(x)
            val c   = cases(ord)
            if (c.arity == 0) w.writeInt32(ord)
            else {
              w.writeArrayHeader(c.arity + 1)
              w.writeInt32(ord)
              c.encodeFields(w, x)
            }
          },
          r =>
            if (r.nextIsArray) {
              val n   = r.readArrayHeader()
              val ord = r.readInt32()
              if (ord < 0 || ord >= cases.length) throw MsgPackException(s"Bad sum ordinal: $ord")
              cases(ord).decodeFields(r, n - 1).asInstanceOf[T]
            } else {
              val ord = r.readInt32()
              if (ord < 0 || ord >= cases.length) throw MsgPackException(s"Bad sum ordinal: $ord")
              val c = cases(ord)
              if (c.arity != 0) throw MsgPackException(s"Expected array payload for non-nullary sum ordinal: $ord")
              c.decodeFields(r, 0).asInstanceOf[T]
            }
        )
      case p: Mirror.ProductOf[T] =>
        val arity = constValue[Tuple.Size[p.MirroredElemTypes]]
        Codec[T](
          (w, x) => {
            w.writeArrayHeader(arity)
            writeFields[p.MirroredElemTypes](w, x.asInstanceOf[Product], 0)
          },
          r => {
            val n = r.readArrayHeader()
            if (n != arity) throw MsgPackException(s"Expected product array of size $arity, got $n")
            p.fromProduct(readFields[p.MirroredElemTypes](r))
          }
        )
    }

    private inline def writeFields[T <: Tuple](w: Writer, p: Product, idx: Int): Unit =
      inline erasedValue[T] match {
        case _: EmptyTuple => ()
        case _: (t *: ts) =>
          summonOrDerive[t].encode(w, p.productElement(idx).asInstanceOf[t])
          writeFields[ts](w, p, idx + 1)
      }

    private inline def readFields[T <: Tuple](r: Reader): Tuple =
      inline erasedValue[T] match {
        case _: EmptyTuple => EmptyTuple
        case _: (t *: ts)  => summonOrDerive[t].decode(r) *: readFields[ts](r)
      }

  }

  case class Versioned[T](hash: String, t: T) derives MsgPack.Codec

  private val InternedMagic = 0x4d504349 // "MPCI"

  private def isInternedEnvelope(xs: ByteInput): Boolean =
    xs.length >= 6 &&
      xs.unsignedByteAt(0) == 0x93 &&
      xs.unsignedByteAt(1) == 0xd2 &&
      xs.unsignedByteAt(2) == 0x4d &&
      xs.unsignedByteAt(3) == 0x50 &&
      xs.unsignedByteAt(4) == 0x43 &&
      xs.unsignedByteAt(5) == 0x49

  def encodeRawTo[A: Encoder](x: A, out: ByteOutput): Unit =
    summon[Encoder[A]].encode(Writer(out), x)

  def encodeRaw[A: Encoder](x: A): Array[Byte] = {
    val out = ArrayByteOutput()
    encodeRawTo(x, out)
    out.toByteArray
  }

  def encodeInternedTo[A: Encoder](x: A, out: ByteOutput): Unit = {
    val C     = summon[Encoder[A]]
    val table = new StringInterner
    C.encode(Writer(NullByteOutput, table), x)

    val entries = table.entries
    val w       = Writer(out)
    w.writeArrayHeader(3)
    w.writeInt32(InternedMagic)
    w.writeArrayHeader(entries.length)
    var i = 0
    while (i < entries.length) {
      w.writeStringLiteral(entries(i))
      i += 1
    }
    w.setStringInterner(table)
    C.encode(w, x)
  }

  def encodeInterned[A: Encoder](x: A): Array[Byte] = {
    val out = ArrayByteOutput()
    encodeInternedTo(x, out)
    out.toByteArray
  }

  def encodeTo[A: Encoder](x: A, out: ByteOutput): Unit = encodeInternedTo(x, out)

  def encode[A: Encoder](x: A): Array[Byte] = encodeInterned(x)

  def decodeRawInput[A: Decoder](xs: ByteInput): Either[Exception, A] =
    try {
      val r = Reader(xs)
      val a = summon[Decoder[A]].decode(r)
      if (!r.isAtEnd) throw MsgPackException(s"Trailing bytes after MessagePack value at byte ${r.offset}")
      Right(a)
    } catch {
      case e: Exception => Left(e)
    }

  def decodeRaw[A: Decoder](xs: Array[Byte]): Either[Exception, A] =
    decodeRawInput(ArrayByteInput(xs))

  def decodeInternedInput[A: Decoder](xs: ByteInput): Either[Exception, A] =
    try {
      val r = Reader(xs)
      val n = r.readArrayHeader()
      if (n != 3) throw MsgPackException(s"Expected interned envelope array of size 3, got $n")
      val magic = r.readInt32()
      if (magic != InternedMagic) throw MsgPackException(s"Bad interned envelope magic: $magic")
      val tableSize = r.readArrayHeader()
      val table     = new Array[String](tableSize)
      var i         = 0
      while (i < tableSize) {
        table(i) = r.readStringLiteral()
        i += 1
      }
      r.setStringTable(table)
      val a = summon[Decoder[A]].decode(r)
      if (!r.isAtEnd) throw MsgPackException(s"Trailing bytes after MessagePack value at byte ${r.offset}")
      Right(a)
    } catch {
      case e: Exception => Left(e)
    }

  def decodeInterned[A: Decoder](xs: Array[Byte]): Either[Exception, A] =
    decodeInternedInput(ArrayByteInput(xs))

  def decodeInput[A: Decoder](xs: ByteInput): Either[Exception, A] =
    if (isInternedEnvelope(xs)) decodeInternedInput(xs) else decodeRawInput(xs)

  def decode[A: Decoder](xs: Array[Byte]): Either[Exception, A] =
    decodeInput(ArrayByteInput(xs))
}
