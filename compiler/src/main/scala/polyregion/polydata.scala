package polyregion

import scala.deriving.*
import scala.compiletime.{constValue, erasedValue, summonInline}

import scala.reflect.ClassTag

//class CppCodeGenTarget extends CodeGenTarget[String]

trait CodeGenTarget[A] { self =>
  def ingest[A]: this.type
  def result: String
}

trait CodeGen[A, B] {
  def gen(target: CodeGenTarget[B], a: A): target.type
}

trait Form[A] {
  def show(a: A): String
}
object Form {

  inline def deriveSum[N <: Tuple, T <: Tuple]: List[String] =
    inline (erasedValue[N], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (n *: ns), _: (t *: ts)) =>
        val x = derived[t](using summonInline[Mirror.Of[t]])




        val template = s"""
		  |struct ${constValue[n]} {
		  |
		  |}
		  |""".stripMargin


        s"struct ${constValue[n]} (${summonInline[ClassTag[t]]}) = \n\t$x}" :: deriveSum[ns, ts]

  inline def deriveProduct[L <: Tuple, T <: Tuple]: List[String] =
    inline (erasedValue[L], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (l *: ls), _: (t *: ts))   => s"[${constValue[l]}:${summonInline[ClassTag[t]]}]" :: deriveProduct[ls, ts]

  inline def derived[T](using m: Mirror.Of[T]): String =
    inline m match
      case s: Mirror.SumOf[T] =>
        lazy val elemInstances = deriveSum[s.MirroredElemLabels, s.MirroredElemTypes]


        s" Sum: ${constValue[s.MirroredLabel]}  \n${elemInstances.mkString("\n")}"
      case p: Mirror.ProductOf[T] =>
        val xs = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]

        s"$p ${xs}"

}

// tree.generate[CppCodeGen] : String
// CodeGen.create[MyAst] : String
// def create[A : CodeGen]

// tree.serialise() : Array[Byte] // msgpack

trait Eq[T] {
  def eqv(x: T, y: T): Boolean
}

inline def summonAll[T <: Tuple]: List[Eq[_]] =
  inline erasedValue[T] match
    case _: EmptyTuple => Nil
    case _: (t *: ts)  => summonInline[Eq[t]] :: summonAll[ts]

object Eq {

  given Eq[Int] with
    def eqv(x: Int, y: Int) = x == y

  def check(elem: Eq[_])(x: Any, y: Any): Boolean =
    elem.asInstanceOf[Eq[Any]].eqv(x, y)

  def iterator[T](p: T) = p.asInstanceOf[Product].productIterator

  def eqSum[T](s: Mirror.SumOf[T], elems: => List[Eq[_]]): Eq[T] =
    new Eq[T] {
      def eqv(x: T, y: T): Boolean = {

        val ordx = s.ordinal(x)
        (s.ordinal(y) == ordx) && check(elems(ordx))(x, y)
      }

    }

  def eqProduct[T](p: Mirror.ProductOf[T], elems: => List[Eq[_]]): Eq[T] =
    new Eq[T] {
      def eqv(x: T, y: T): Boolean =
        iterator(x).zip(iterator(y)).zip(elems.iterator).forall { case ((x, y), elem) =>
          check(elem)(x, y)
        }
    }

  inline given derived[T](using m: Mirror.Of[T]): Eq[T] = {
    lazy val elemInstances = summonAll[m.MirroredElemTypes]
    inline m match
      case s: Mirror.SumOf[T]     => eqSum(s, elemInstances)
      case p: Mirror.ProductOf[T] => eqProduct(p, elemInstances)
  }

}

object Foo {

  enum Opt[+T] derives Eq {
    case Sm(t: T, u: T)
    case Nn
  }

  enum Base(val u : Int) {
    case This(x: Int, s: String) extends Base(x)
    case That(a: String) extends Base(42)
  }

  @main def main2(): Unit = {

//    println(Opt.Sm(2, 7) == Opt.Sm(1 + 1, 7))
//    import Form.*
//    val x = summon[Form[Opt[Int]]]

    println(Form.derived[Base])
    ()
  }
}
