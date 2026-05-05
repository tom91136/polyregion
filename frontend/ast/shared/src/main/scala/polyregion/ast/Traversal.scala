package polyregion.ast

import scala.annotation.targetName
import scala.util.NotGiven

trait Traversal[A, B] {
  extension (a: A) {
    def collectAll: List[B]
    def collectWhere[C](f: PartialFunction[B, C]): List[C]
    def collectFirst_[C](f: PartialFunction[B, C]): Option[C]
    def modifyAll(f: B => B): A
    def modifyCollect[C](f: B => (B, C)): (A, List[C])
  }
}

object Traversal {
  import scala.compiletime.{constValue, erasedValue, summonInline}
  import scala.deriving.*

  private def prod[A, B](p: Mirror.ProductOf[A], instances: => List[Traversal[?, B]], same: => A => Option[B]) =
    new Traversal[A, B] {
      extension (a: A) {
        def collectAll: List[B] =
          same(a).toList ::: instances.zip(a.asInstanceOf[Product].productIterator).flatMap { (instance, x) =>
            instance.asInstanceOf[Traversal[Any, B]].collectAll(x)
          }
        def collectWhere[C](f: PartialFunction[B, C]): List[C] =
          same(a).collect(f).toList ::: instances.zip(a.asInstanceOf[Product].productIterator).flatMap {
            (instance, x) =>
              instance.asInstanceOf[Traversal[Any, B]].collectWhere(x)(f)
          }
        def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
          same(a)
            .collect(f)
            .orElse(
              instances.view
                .zip(a.asInstanceOf[Product].productIterator)
                .map((instance, x) => instance.asInstanceOf[Traversal[Any, B]].collectFirst_(x)(f))
                .collectFirst { case Some(x) => x }
            )
        def modifyAll(f: B => B): A = {
          val xs = instances
            .zip(a.asInstanceOf[Product].productIterator)
            .map { (instance, x) =>
              instance.asInstanceOf[Traversal[Any, B]].modifyAll(x)(f)
            }
            .toArray
          val a0 = p.fromProduct(Tuple.fromArray(xs))
          same(a0).fold(a0)(f(_).asInstanceOf[A])
        }
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) = {
          val (xs, cs) = instances
            .zip(a.asInstanceOf[Product].productIterator)
            .map { (instance, x) =>
              instance.asInstanceOf[Traversal[Any, B]].modifyCollect(x)(f)
            }
            .toArray
            .unzip
          val a0 = p.fromProduct(Tuple.fromArray(xs))
          same(a0).fold((a0, cs.toList.flatten)) { x =>
            val (b0, c0) = f(x)
            (b0.asInstanceOf[A], c0 :: cs.toList.flatten)
          }
        }
      }
    }

  private inline def summonAll[T <: Tuple, B]: List[Traversal[?, B]] = inline erasedValue[T] match {
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
        def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
          instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectFirst_(a)(f)
        def modifyAll(f: B => B): A = instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyAll(a)(f)
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) =
          instances(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyCollect(a)(f)
      }
    }

  inline given derived[A, B](using inline m: Mirror.Of[A]): Traversal[A, B] =
    inline m match {
      case p: Mirror.ProductOf[A] =>
        prod[A, B](
          p,
          summonAll[p.MirroredElemTypes, B],
          inline (erasedValue[A], erasedValue[B]) match {
            case (_: B, _) => (a: A) => Some(a.asInstanceOf[B])
            case _         => (a: A) => None
          }
        )
      case s: Mirror.SumOf[A] => sum[A, B](s, summonAll[s.MirroredElemTypes, B].toArray)
    }

  extension [A](a: A) {

    def collectAll[B](using t: Traversal[A, B]): List[B] = t.collectAll(a)
    def collectWhere[B](using t: Traversal[A, B]): [C] => PartialFunction[B, C] => List[C] = [C] =>
      (f: PartialFunction[B, C]) => t.collectWhere(a)(f)
    def collectFirst_[B](using t: Traversal[A, B]): [C] => PartialFunction[B, C] => Option[C] = [C] =>
      (f: PartialFunction[B, C]) => t.collectFirst_(a)(f)

    def collectWhereOption[B](using t: Traversal[A, B]): [C] => (B => Option[C]) => List[C] = [C] =>
      (f: B => Option[C]) => t.collectWhere(a)(f.unlift)

    def collectFirstOption[B](using t: Traversal[A, B]): [C] => (B => Option[C]) => Option[C] = [C] =>
      (f: B => Option[C]) => t.collectFirst_(a)(f.unlift)

    def modifyAll[B](using t: Traversal[A, B])(f: B => B): A = t.modifyAll(a)(f)
    def modifyCollect[B, C](using t: Traversal[A, B])(f: B => (B, C)): (A, List[C]) =
      t.modifyCollect(a)(f)
  }

  // non product base case

  given [A, B](using ev: NotGiven[A <:< Product]): Traversal[A, B] = new Traversal[A, B] {
    extension (a: A) {
      def collectAll: List[B]                                   = Nil
      def collectWhere[C](f: PartialFunction[B, C]): List[C]    = Nil
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] = None
      def modifyAll(f: B => B): A                               = a
      def modifyCollect[C](f: B => (B, C)): (A, List[C])        = (a, Nil)
    }
  }

  given [A, B](using t: Traversal[A, B]): Traversal[List[A], B] = new Traversal[List[A], B] {
    extension (xs: List[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f))
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
        xs.view.map(t.collectFirst_(_)(f)).collectFirst { case Some(x) => x }
      def modifyAll(f: B => B): List[A] = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (List[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

  given [A, B](using t: Traversal[A, B]): Traversal[Vector[A], B] = new Traversal[Vector[A], B] {
    extension (xs: Vector[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_)).toList
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f)).toList
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
        xs.view.map(t.collectFirst_(_)(f)).collectFirst { case Some(x) => x }
      def modifyAll(f: B => B): Vector[A] = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Vector[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.flatten.toList)
      }
    }
  }

  given [A, B](using t: Traversal[A, B]): Traversal[Option[A], B] = new Traversal[Option[A], B] {
    extension (xs: Option[A]) {
      def collectAll: List[B]                                   = xs.fold(Nil)(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C]    = xs.fold(Nil)(t.collectWhere(_)(f))
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] = xs.fold(None)(t.collectFirst_(_)(f))
      def modifyAll(f: B => B): Option[A]                       = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Option[A], List[C]) = {
        val (bs, css) = xs.map(t.modifyCollect(_)(f)).unzip
        (bs, css.getOrElse(Nil))
      }
    }
  }

}
