package polyregion.ast

import scala.util.NotGiven

trait Traversal[A, B] {
  extension (a: A) {
    def collectAll: List[B]
    def collectWhere[C](f: PartialFunction[B, C]): List[C]
    def modifyAll(f: B => B): A
    def modifyCollect[C](f: B => (B, C)): (A, List[C])
  }
}

object Traversal {
  import scala.compiletime.{constValue, erasedValue, summonInline}
  import scala.deriving.*

  private def prod[A, B](p: Mirror.ProductOf[A], instances: => List[Traversal[?, B]]) =
    new Traversal[A, B] {
      extension (a: A) {
        def collectAll: List[B] =
          instances.zip(a.asInstanceOf[Product].productIterator).flatMap { (instance, x) =>
            instance.asInstanceOf[Traversal[Any, B]].collectAll(x)
          }
        def collectWhere[C](f: PartialFunction[B, C]): List[C] =
          instances.zip(a.asInstanceOf[Product].productIterator).flatMap { (instance, x) =>
            instance.asInstanceOf[Traversal[Any, B]].collectWhere(x)(f)
          }
        def modifyAll(f: B => B): A = {
          val xs = instances
            .zip(a.asInstanceOf[Product].productIterator)
            .map { (instance, x) =>
              instance.asInstanceOf[Traversal[Any, B]].modifyAll(x)(f)
            }
            .toArray
          p.fromProduct(Tuple.fromArray(xs))
        }
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) = {
          val (xs, cs) = instances
            .zip(a.asInstanceOf[Product].productIterator)
            .map { (instance, x) =>
              instance.asInstanceOf[Traversal[Any, B]].modifyCollect(x)(f)
            }
            .toArray
            .unzip
          (p.fromProduct(Tuple.fromArray(xs)), cs.toList.flatten)
        }
      }
    }

  private inline def summonAll[T <: Tuple, B]: List[Traversal[_, B]] = inline erasedValue[T] match {
    case _: EmptyTuple => Nil
    case _: (t *: ts)  => summonInline[Traversal[t, B]] :: summonAll[ts, B]
  }

  private def sum[A, B](s: Mirror.SumOf[A], instances: => Array[Traversal[?, B]]) =
    new Traversal[A, B] {
      extension (a: A) {
        def collectAll: List[B] =
          instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectAll(a)
        def collectWhere[C](f: PartialFunction[B, C]): List[C] =
          instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectWhere(a)(f)
        def modifyAll(f: B => B): A = instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyAll(a)(f)
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) =
          instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyCollect(a)(f)
      }
    }

  inline given derived[A, B](using inline m: Mirror.Of[A], inline ev: NotGiven[A =:= B]): Traversal[A, B] =
    inline m match {
      case p: Mirror.ProductOf[A] => prod[A, B](p, summonAll[m.MirroredElemTypes, B])
      case s: Mirror.SumOf[A]     => sum[A, B](s, summonAll[m.MirroredElemTypes, B].toArray)
    }

  extension [A](a: A) {

    inline def collectAll[B](using inline t: Traversal[A, B]): List[B] = t.collectAll(a)
    inline def collectWhere[B](using inline t: Traversal[A, B]): [C] => PartialFunction[B, C] => List[C] = [C] =>
      (f: PartialFunction[B, C]) => t.collectWhere(a)(f)
    inline def modifyAll[B](using inline t: Traversal[A, B])(inline f: B => B): A = t.modifyAll(a)(f)
    inline def modifyCollect[B, C](using inline t: Traversal[A, B])(inline f: B => (B, C)): (A, List[C]) =
      t.modifyCollect(a)(f)
  }

  // non product base case
  inline given [A, B](using inline ev: NotGiven[A <:< Product]): Traversal[A, B] = new Traversal[A, B] {
    extension (a: A) {
      def collectAll: List[B]                                = Nil
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = Nil
      def modifyAll(f: B => B): A                            = a
      def modifyCollect[C](f: B => (B, C)): (A, List[C])     = (a, Nil)
    }
  }

  // eq case
  inline given [B]: Traversal[B, B] = new Traversal[B, B] {
    extension (a: B) {
      def collectAll: List[B]                                = a :: Nil
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = f.lift(a).toList
      def modifyAll(f: B => B): B                            = f(a)
      def modifyCollect[C](f: B => (B, C)): (B, List[C]) = {
        val (b, c) = f(a)
        (b, c :: Nil)
      }
    }
  }

  inline given [A, B](using inline t: Traversal[A, B]): Traversal[List[A], B] = new Traversal[List[A], B] {
    extension (xs: List[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f))
      def modifyAll(f: B => B): List[A]                      = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (List[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

  inline given [A, B](using inline t: Traversal[A, B]): Traversal[Vector[A], B] = new Traversal[Vector[A], B] {
    extension (xs: Vector[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_)).toList
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f)).toList
      def modifyAll(f: B => B): Vector[A]                    = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Vector[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

  inline given [A, B](using inline t: Traversal[A, B]): Traversal[Option[A], B] = new Traversal[Option[A], B] {
    extension (xs: Option[A]) {
      def collectAll: List[B]                                = xs.fold(Nil)(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.fold(Nil)(t.collectWhere(_)(f))
      def modifyAll(f: B => B): Option[A]                    = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Option[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.getOrElse(Nil))
      }
    }
  }

}