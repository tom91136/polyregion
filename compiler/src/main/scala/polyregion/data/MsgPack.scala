package polyregion.data

import scala.deriving.*
import scala.compiletime.{constValue, erasedValue, summonInline}
import scala.collection.mutable.LinkedHashMap

object MsgPack {

  enum Kind        { case Compact, Verbose         }
  trait Encoder[A] { def encode(a: A): upack.Msg   }
  trait Decoder[A] { def decode(msg: upack.Msg): A }
  trait Codec[A] extends Encoder[A], Decoder[A]
  object Codec {

    import upack._

    def apply[A](f: A => upack.Msg, g: upack.Msg => A) = new Codec[A] {
      def encode(x: A): upack.Msg = f(x)
      def decode(x: upack.Msg): A = g(x)
    }

    given Codec[Boolean] = Codec(upack.Bool(_), _.bool)
    given Codec[String]  = Codec(upack.Str(_), _.str)
    given Codec[Byte]    = Codec(upack.Int32(_), _.int32.toByte)
    given Codec[Char]    = Codec(upack.Int32(_), _.int32.toChar)
    given Codec[Short]   = Codec(upack.Int32(_), _.int32.toShort)
    given Codec[Int]     = Codec(upack.Int32(_), _.int32)
    given Codec[Long]    = Codec(upack.Int64(_), _.int64)
    given Codec[Float] = Codec(
      upack.Float32(_),
      {
        case upack.Float32(v)                            => v
        case upack.Float64(v) if v.toFloat.toDouble == v => v.toFloat
        case upack.Float64(v) => throw new Exception(s"Float64 to Float32 conversion with loss of precision: $v")
        case x                => throw new Exception(s"Expected Float32/Float64, got $x")
      }
    )
    given Codec[Double] = Codec(
      upack.Float64(_),
      {
        case upack.Float64(v) => v
        case x                => throw new Exception(s"Expected Float32/Float64, got $x")
      }
    )
    given [A](using C: Codec[A]): Codec[List[A]] =
      Codec(xs => upack.Arr(xs.map(C.encode(_))*), _.arr.map(m => C.decode(m)).toList)
    given [A](using C: Codec[A]): Codec[Option[A]] =
      Codec(_.fold(upack.Null)(C.encode(_)), x => if (x.isNull) None else Some(C.decode(x)))

    private inline def summonAll[T <: Tuple, TC[_]]: Vector[TC[Any]] = inline erasedValue[T] match {
      case _: EmptyTuple => Vector()
      case _: (t *: ts)  => summonInline[TC[t]].asInstanceOf[TC[Any]] +: summonAll[ts, TC]
    }

    inline given derived[T](using m: Mirror.Of[T]): Codec[T] = inline m match {
      case s: Mirror.SumOf[T] =>
        val sum      = constValue[s.MirroredLabel].toString
        lazy val tcs = summonAll[s.MirroredElemTypes, Codec]
        val kind     = Kind.Compact
        Codec[T](
          { x =>
            val ord = s.ordinal(x)
            val t   = tcs(ord).encode(x)
            kind match {
              case Kind.Compact => Arr(Int32(ord), t)
              case Kind.Verbose => Obj(Str("sum") -> Str(sum), Str("ord") -> Int32(ord), Str("val") -> t)
            }
          },
          msg =>
            kind match {
              case Kind.Compact =>
                msg.arr.toList match {
                  case Int32(ord) :: t :: Nil =>
                    tcs(ord).decode(t).asInstanceOf[T]
                }
              case Kind.Verbose =>
                msg.obj.toList match {
                  case (Str("sum"), Str(sum)) :: (Str("ord"), Int32(ord)) :: (Str("val"), t) :: Nil =>
                    tcs(ord).decode(t).asInstanceOf[T]
                }
            }
        )
      case p: Mirror.ProductOf[T] =>
        lazy val xs = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
        val kind    = Kind.Compact
        Codec[T](
          x =>
            kind match {
              case Kind.Compact => Arr(xs.zip(iterator(x)).map { case ((_, e), v) => e.encode(v) }*)
              case Kind.Verbose =>
                xs.zip(iterator(x)).map { case ((k, e), v) => (Str(k), e.encode(v)) } match {
                  case (x :: xs) => Obj(x, xs*)
                  case Nil       => Obj()
                }
            },
          msg =>
            p.fromProduct(Tuple.fromArray(kind match {
              case Kind.Compact => msg.arr.zip(xs).map { case (msg, (_, c)) => c.decode(msg) }.toArray
              case Kind.Verbose => xs.map((field, c) => c.decode(msg.obj(Str(field)))).toArray
            }))
        )
    }

    private def iterator[T](p: T) = p.asInstanceOf[Product].productIterator

    private inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(String, Codec[Any])] =
      inline (erasedValue[L], erasedValue[T]) match {
        case (_: EmptyTuple, _: EmptyTuple) => Nil
        case (_: (l *: ls), _: (t *: ts)) =>
          ((constValue[l].toString), summonInline[Codec[t]].asInstanceOf[Codec[Any]]) ::
            deriveProduct[ls, ts]
      }

  }

  def decodeMsg(xs: Array[Byte]): Either[Exception, upack.Msg] =
    try Right(upack.read(xs))
    catch { case e: Exception => Left(e) }
  def encodeMsg[A: Encoder](x: A): upack.Msg = summon[Encoder[A]].encode(x)

  def encode[A: Encoder](x: A): Array[Byte] = upack.write(encodeMsg[A](x))
  def decode[A: Decoder](xs: Array[Byte]): Either[Exception, A] =
    decodeMsg(xs).map(summon[Decoder[A]].decode(_))

}
