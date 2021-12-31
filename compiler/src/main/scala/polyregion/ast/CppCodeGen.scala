package polyregion.ast

import polyregion.data.Cpp._
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.lang.annotation.Target

import ujson.Arr
import scala.collection.mutable.ArrayBuffer

object CppCodeGen {

  import scala.deriving.*
  import scala.compiletime.{constValue, erasedValue, summonInline}

  inline def summonAll[T <: Tuple, TC[_]]: Vector[TC[Any]] = inline erasedValue[T] match {
    case _: EmptyTuple => Vector()
    case _: (t *: ts)  => summonInline[TC[t]].asInstanceOf[TC[Any]] +: summonAll[ts, TC]
  }

  trait MsgPackEncoder[A] extends (A => upack.Msg)
  object MsgPackEncoder {
    import upack._

    given MsgPackEncoder[Boolean]                                  = upack.Bool(_)
    given MsgPackEncoder[String]                                   = upack.Str(_)
    given MsgPackEncoder[Byte]                                     = upack.Int32(_)
    given MsgPackEncoder[Char]                                     = upack.Int32(_)
    given MsgPackEncoder[Short]                                    = upack.Int32(_)
    given MsgPackEncoder[Int]                                      = upack.Int32(_)
    given MsgPackEncoder[Long]                                     = upack.Int64(_)
    given MsgPackEncoder[Float]                                    = upack.Float32(_)
    given MsgPackEncoder[Double]                                   = upack.Float64(_)
    given [A](using E: MsgPackEncoder[A]): MsgPackEncoder[List[A]] = xs => upack.Arr(xs.map(E(_))*)

    private def iterator[T](p: T) = p.asInstanceOf[Product].productIterator

    object Verbose {
      inline given derived[T](using m: Mirror.Of[T]): MsgPackEncoder[T] = {
        inline m match {
          case s: Mirror.SumOf[T] =>
            // { sum:SumName, ord : Int, value : X }
            val sum = constValue[s.MirroredLabel].toString
            x =>
              val ord = s.ordinal(x)
              upack.Obj(
                upack.Str("sum") -> upack.Str(sum),
                upack.Str("ord") -> upack.Int32(ord),
                upack.Str("val") -> summonAll[s.MirroredElemTypes, MsgPackEncoder](ord)(x)
              )
          case p: Mirror.ProductOf[T] =>
            val xs = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
            x =>
              xs.zip(iterator(x)).map { case ((k, e), v) => (upack.Str(k), e(v)) } match {
                case (x :: xs) => upack.Obj(x, xs*)
                case Nil       => upack.Obj()
              }
        }
      }
    }

    object Compact {
      inline given derived[T](using m: Mirror.Of[T]): MsgPackEncoder[T] = {
        inline m match {
          case s: Mirror.SumOf[T] =>
            // { sum:SumName, ord : Int, value : X }
            val sum = constValue[s.MirroredLabel].toString
            x =>
              val ord = s.ordinal(x)
              upack.Arr(
                upack.Int32(ord),
                summonAll[s.MirroredElemTypes, MsgPackEncoder](ord)(x)
              )
          case p: Mirror.ProductOf[T] =>
            val xs = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
            x => upack.Arr(xs.zip(iterator(x)).map { case ((_, e), v) => e(v) }*)
        }
      }
    }

    inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(String, MsgPackEncoder[Any])] =
      inline (erasedValue[L], erasedValue[T]) match {
        case (_: EmptyTuple, _: EmptyTuple) => Nil
        case (_: (l *: ls), _: (t *: ts)) =>
          ((constValue[l].toString), summonInline[MsgPackEncoder[t]].asInstanceOf[MsgPackEncoder[Any]]) ::
            deriveProduct[ls, ts]
      }

    def msg[A: MsgPackEncoder](x: A): upack.Msg      = summon[MsgPackEncoder[A]](x)
    def encode[A: MsgPackEncoder](x: A): Array[Byte] = write(msg[A](x))

  }

  trait MsgPackDecoder[A] extends (upack.Msg => A)

  object MsgPackDecoder {
    import upack._

    // def iterator[T](p: T) = p.asInstanceOf[Product].productIterator

    // def eqSum[T](s: Mirror.SumOf[T], elems: => List[MsgPackDecoder[_]]): MsgPackDecoder[T] = { xs =>
    //   Array.emptyByteArray
    // }

    // inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(String, CppType)] =
    //   inline (erasedValue[L], erasedValue[T]) match
    //     case (_: EmptyTuple, _: EmptyTuple) => Nil
    //     case (_: (l *: ls), _: (t *: ts)) =>
    //       (s"${constValue[l]}", summonInline[ToCppType[t]]()) :: deriveProduct[ls, ts]

    // inline given derived[T](using m: Mirror.Of[T]): MsgPackDecoder[T] = {
    //   lazy val elemInstances = summonAll[m.MirroredElemTypes, MsgPackDecoder]
    //   inline m match
    //     case s: Mirror.SumOf[T] => eqSum(s, elemInstances)
    //     case p: Mirror.ProductOf[T] =>
    //       deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
    //       eqProduct(p, elemInstances)
    // }
  }
  case class AA(s: String)

  @main def main(): Unit = {

    println("\n=========\n")

    import PolyAstUnused._
    val s = Sym("a.b")

    import MsgPackEncoder.Verbose.derived

    given MsgPackEncoder[Sym]      = derived
    given MsgPackEncoder[TypeKind] = derived
    given MsgPackEncoder[Type]     = derived
    given MsgPackEncoder[Term]     = derived
    given MsgPackEncoder[Named]    = derived
    given MsgPackEncoder[Position] = derived
    given MsgPackEncoder[Intr]     = derived
    given MsgPackEncoder[Stmt]     = derived
    given MsgPackEncoder[Expr]     = derived
    given MsgPackEncoder[Tree]     = derived

    val ast = Stmt.Cond(
      Expr.Alias(Term.BoolConst(true)),
      Stmt.Return(Expr.Alias(Term.Select(Nil, Named("a", Type.Float)))) :: Nil,
      Stmt.Return(Expr.Alias(Term.FloatConst(1))) :: Nil
    )

    println(MsgPackEncoder.msg(ast))
    println(MsgPackEncoder.encode(ast).length)
    // given ReadWriter[TypeKind] = macroRW
    // given ReadWriter[Sym]      = macroRW
    // given ReadWriter[Type]     = macroRW

    // println(write(s, indent = 2, escapeUnicode = true))

    val alts = deriveStruct[PolyAstUnused.Sym]().emit //
      ::: deriveStruct[PolyAstUnused.TypeKind]().emit
      ::: deriveStruct[PolyAstUnused.Type]().emit
      ::: deriveStruct[PolyAstUnused.Named]().emit
      ::: deriveStruct[PolyAstUnused.Position]().emit
      ::: deriveStruct[PolyAstUnused.Term]().emit
      ::: deriveStruct[PolyAstUnused.Tree]().emit
      ::: deriveStruct[PolyAstUnused.Function]().emit
      ::: deriveStruct[PolyAstUnused.StructDef]().emit

    val header = StructSource.emitHeader("polyregion::polyast", alts)
    // println(header)
    println("\n=========\n")
    val impl = StructSource.emitImpl("polyregion::polyast", "polyast", alts)
    // println(impl)

    val target = Paths.get(".").resolve("native/src/generated/").normalize.toAbsolutePath

    Files.createDirectories(target)
    println(s"Dest=${target}")
    println("\n=========\n")

    Files.writeString(
      target.resolve("polyast.cpp"),
      impl,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )
    Files.writeString(
      target.resolve("polyast.h"),
      header,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    println(summon[ToCppType[PolyAstUnused.TypeKind.Fractional.type]]().qualified)
//    import Cpp.*
//    println(T1Mid.T1ALeaf(Nil, List("a", "b"), 23, T1Mid.T1BLeaf))

    // println(Cpp.deriveStruct[Alt]().map(_.emitSource).mkString("\n"))
    // println(Cpp.deriveStruct[FirstTop]().map(_.emitSource).mkString("\n"))
    // println(Cpp.deriveStruct[First]().map(_.emitSource).mkString("\n"))
//    println(Cpp.deriveStruct[Foo]().map(_.emitSource).mkString("\n"))
    ()
  }

}
