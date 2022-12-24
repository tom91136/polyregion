package polyregion.ast

import scala.util.NotGiven

trait Traversal[A, B] {
  extension (a: A) {
    def collect: List[B]
    def modify(f: B => B): A
    def modifyAccumulate[C](f: B => (B, C)): (A, List[C])
  }
}

object Traversal {
  import scala.deriving.*
  import scala.compiletime.{constValue, erasedValue, summonInline}
  import scala.compiletime.ops.int.*

  private inline def summonWithHole[T <: Tuple, A]: List[Option[Traversal[_, A]]] = inline erasedValue[T] match {
    case _: EmptyTuple => Nil
    case _: (A *: ts)  => None :: summonWithHole[ts, A]
    case _: (t *: ts)  => Some(summonInline[Traversal[t, A]]) :: summonWithHole[ts, A]
  }

  private inline def summonAll[T <: Tuple, B]: List[Traversal[_, B]] = inline erasedValue[T] match {
    case _: EmptyTuple => Nil
    case _: (t *: ts)  => summonInline[Traversal[t, B]] :: summonAll[ts, B]
  }

  inline given derived[A, B](using m: Mirror.Of[A]): Traversal[A, B] = inline m match {
    case p: Mirror.ProductOf[A] =>
      new Traversal[A, B] {
        private lazy val instances = summonWithHole[p.MirroredElemTypes, B]
        extension (a: A) {
          def collect: List[B] =
            instances.zip(a.asInstanceOf[Product].productIterator).flatMap {
              case (None, x)           => x.asInstanceOf[B] :: Nil
              case (Some(instance), x) => instance.asInstanceOf[Traversal[Any, B]].collect(x)
            }
          def modify(f: B => B): A = {
            val xs = instances
              .zip(a.asInstanceOf[Product].productIterator)
              .map {
                case (None, x)           => f(x.asInstanceOf[B])
                case (Some(instance), x) => instance.asInstanceOf[Traversal[Any, B]].modify(x)(f)
              }
              .toArray
            p.fromProduct(Tuple.fromArray(xs))
          }
          def modifyAccumulate[C](f: B => (B, C)): (A, List[C]) = {
            val (xs, cs) = instances
              .zip(a.asInstanceOf[Product].productIterator)
              .map {
                case (None, x) =>
                  val (b, c) = f(x.asInstanceOf[B])
                  (b, c :: Nil)
                case (Some(instance), x) => instance.asInstanceOf[Traversal[Any, B]].modifyAccumulate(x)(f)
              }
              .toArray
              .unzip
            (p.fromProduct(Tuple.fromArray(xs)), cs.toList.flatten)
          }
        }
      }
    case s: Mirror.SumOf[A] =>
      new Traversal[A, B] {
        private val instances = summonAll[m.MirroredElemTypes, B].toArray
        extension (a: A) {
          def collect: List[B]     = instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collect(a)
          def modify(f: B => B): A = instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modify(a)(f)
          def modifyAccumulate[C](f: B => (B, C)): (A, List[C]) =
            instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyAccumulate(a)(f)
        }
      }
  }

  inline def apply[A: Mirror.Of, B]: Traversal[A, B] = derived[A, B]

  extension [A](a: A) {

    inline def collect[B](using inline t: Traversal[A, B]): List[B]            = t.collect(a)
    inline def modify[B](using inline t: Traversal[A, B])(inline f: B => B): A = t.modify(a)(f)
    inline def modifyAccumulate[B, C](using inline t: Traversal[A, B])(inline f: B => (B, C)): (A, List[C]) =
      t.modifyAccumulate(a)(f)

  }

  private inline def nilTraversal[A, B] = new Traversal[A, B] {
    extension (a: A) {
      def collect: List[B]                                  = Nil
      def modify(f: B => B): A                              = a
      def modifyAccumulate[C](f: B => (B, C)): (A, List[C]) = (a, Nil)
    }
  }

    given [B](using NotGiven[Boolean =:= B]): Traversal[Boolean, B] = nilTraversal
    given [B](using NotGiven[String =:= B]): Traversal[String, B]   = nilTraversal
    given [B](using NotGiven[Byte =:= B]): Traversal[Byte, B]       = nilTraversal
    given [B](using NotGiven[Char =:= B]): Traversal[Char, B]       = nilTraversal
    given [B](using NotGiven[Short =:= B]): Traversal[Short, B]     = nilTraversal
    given [B](using NotGiven[Int =:= B]): Traversal[Int, B]         = nilTraversal
    given [B](using NotGiven[Long =:= B]): Traversal[Long, B]       = nilTraversal
    given [B](using NotGiven[Float =:= B]): Traversal[Float, B]     = nilTraversal
    given [B](using NotGiven[Double =:= B]): Traversal[Double, B]   = nilTraversal

  // base case
    given [B]: Traversal[B, B] = new Traversal[B, B] {
    extension (a: B) {
      def collect: List[B]     = a :: Nil
      def modify(f: B => B): B = f(a)
      def modifyAccumulate[C](f: B => (B, C)): (B, List[C]) = {
        val (b, c) = f(a)
        (b, c :: Nil)
      }
    }
  }

    given [A, B](using t: Traversal[A, B]): Traversal[List[A], B] = new Traversal[List[A], B] {
    extension (xs: List[A]) {
      def collect: List[B]           = xs.flatMap(t.collect(_))
      def modify(f: B => B): List[A] = xs.map(t.modify(_)(f))
      def modifyAccumulate[C](f: B => (B, C)): (List[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyAccumulate(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

    given [A, B](using t: Traversal[A, B]): Traversal[Vector[A], B] = new Traversal[Vector[A], B] {
    extension (xs: Vector[A]) {
      def collect: List[B]             = xs.flatMap(t.collect(_)).toList
      def modify(f: B => B): Vector[A] = xs.map(t.modify(_)(f))
      def modifyAccumulate[C](f: B => (B, C)): (Vector[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyAccumulate(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

    given [A, B](using t: Traversal[A, B]): Traversal[Option[A], B] = new Traversal[Option[A], B] {
    extension (xs: Option[A]) {
      def collect: List[B]             = xs.fold(Nil)(t.collect(_))
      def modify(f: B => B): Option[A] = xs.map(t.modify(_)(f))
      def modifyAccumulate[C](f: B => (B, C)): (Option[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyAccumulate(_)(f)).unzip
        (bs, css.getOrElse(Nil))
      }
    }
  }

}
