package polyregion.ast

import scala.annotation.targetName
import scala.deriving.Mirror
import scala.compiletime.{erasedValue, summonInline}
import scala.util.NotGiven
import scala.quoted.*

trait Traversal[A, B] {
  extension (a: A) {
    def collectAll: List[B]
    def collectWhere[C](f: PartialFunction[B, C]): List[C]
    def collectFirst_[C](f: PartialFunction[B, C]): Option[C]
    def modifyAll(f: B => B): A
    def modifyCollect[C](f: B => (B, C)): (A, List[C])
    def modifyAllInternal(f: B => B): A
    def modifyCollectInternal[C](f: B => (B, C)): (A, List[C])
  }
}

object Traversal {

  inline def constructProduct[A](inline tup: Tuple)(using inline m: Mirror.ProductOf[A]): A =
    ${ constructProductImpl[A]('tup) }

  private def constructProductImpl[A: Type](tup: Expr[Tuple])(using Quotes): Expr[A] = {
    import quotes.reflect.*
    val tpe = TypeRepr.of[A]
    val sym = tpe.typeSymbol
    val fs  = sym.caseFields

    val tas = tpe match {
      case AppliedType(_, xs) => xs
      case _                  => Nil
    }
    val tats = tas.map(t => TypeTree.of(using t.asType.asInstanceOf[Type[Any]]))

    val as: List[Term] = fs.zipWithIndex.map { case (f, idx) =>
      tpe.memberType(f).asType match {
        case '[ft] => '{ ${ tup }.productElement(${ Expr(idx) }).asInstanceOf[ft] }.asTerm
      }
    }

    val ctor: Term = Select(New(TypeTree.of[A]), sym.primaryConstructor)
    val mk: Term   = if (tats.nonEmpty) TypeApply(ctor, tats) else ctor
    Apply(mk, as).asExprOf[A]
  }

  private inline def collectAllFields[T <: Tuple, B](prod: Product, idx: Int): List[B] =
    inline erasedValue[T] match {
      case _: EmptyTuple => Nil
      case _: (t *: ts) =>
        val tr: Traversal[t, B] = summonInline[Traversal[t, B]]
        tr.collectAll(prod.productElement(idx).asInstanceOf[t]) :::
          collectAllFields[ts, B](prod, idx + 1)
    }
  private inline def collectWhereFields[T <: Tuple, B, C](prod: Product, idx: Int, f: PartialFunction[B, C]): List[C] =
    inline erasedValue[T] match {
      case _: EmptyTuple => Nil
      case _: (t *: ts) =>
        val tr: Traversal[t, B] = summonInline[Traversal[t, B]]
        tr.collectWhere(prod.productElement(idx).asInstanceOf[t])(f) :::
          collectWhereFields[ts, B, C](prod, idx + 1, f)
    }
  private inline def collectFirstFields[T <: Tuple, B, C](
      prod: Product,
      idx: Int,
      f: PartialFunction[B, C]
  ): Option[C] =
    inline erasedValue[T] match {
      case _: EmptyTuple => None
      case _: (t *: ts) =>
        val tr: Traversal[t, B] = summonInline[Traversal[t, B]]
        tr.collectFirst_(prod.productElement(idx).asInstanceOf[t])(f)
          .orElse(collectFirstFields[ts, B, C](prod, idx + 1, f))
    }
  private inline def modifyAllFields[T <: Tuple, B](prod: Product, idx: Int, f: B => B): Tuple =
    inline erasedValue[T] match {
      case _: EmptyTuple => EmptyTuple
      case _: (t *: ts) =>
        val tr: Traversal[t, B] = summonInline[Traversal[t, B]]
        val nx                  = tr.modifyAll(prod.productElement(idx).asInstanceOf[t])(f)
        nx *: modifyAllFields[ts, B](prod, idx + 1, f)
    }
  private inline def modifyCollectFields[T <: Tuple, B, C](
      prod: Product,
      idx: Int,
      f: B => (B, C)
  ): (Tuple, List[C]) =
    inline erasedValue[T] match {
      case _: EmptyTuple => (EmptyTuple, Nil)
      case _: (t *: ts) =>
        val tr: Traversal[t, B] = summonInline[Traversal[t, B]]
        val (nx, cx)            = tr.modifyCollect(prod.productElement(idx).asInstanceOf[t])(f)
        val (rest, cs)          = modifyCollectFields[ts, B, C](prod, idx + 1, f)
        (nx *: rest, cx ::: cs)
    }

  private inline def isB[A, B]: Boolean = inline erasedValue[A] match {
    case _: B => true
    case _    => false
  }

  private inline def summonAll[T <: Tuple, B]: List[Traversal[?, B]] = inline erasedValue[T] match {
    case _: EmptyTuple => Nil
    case _: (t *: ts)  => summonInline[Traversal[t, B]] :: summonAll[ts, B]
  }

  private def sum[A, B](
      s: Mirror.SumOf[A],
      tss: => Array[Traversal[?, B]],
      aSubB: Boolean
  ) =
    new Traversal[A, B] {
      extension (a: A) {
        def collectAll: List[B] =
          tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectAll(a)
        def collectWhere[C](f: PartialFunction[B, C]): List[C] =
          tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectWhere(a)(f)
        def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
          tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].collectFirst_(a)(f)
        def modifyAll(f: B => B): A = {
          val a1 = tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyAllInternal(a)(f)
          if (aSubB) f(a1.asInstanceOf[B]).asInstanceOf[A] else a1
        }
        def modifyAllInternal(f: B => B): A =
          tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyAllInternal(a)(f)
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) = {
          val (a1, cs) = tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyCollectInternal(a)(f)
          if (aSubB) {
            val (b0, c0) = f(a1.asInstanceOf[B])
            (b0.asInstanceOf[A], c0 :: cs)
          } else (a1, cs)
        }
        def modifyCollectInternal[C](f: B => (B, C)): (A, List[C]) =
          tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].modifyCollectInternal(a)(f)
      }
    }

  inline given derived[A, B](using inline m: Mirror.Of[A]): Traversal[A, B] =
    inline m match {
      case _: Mirror.Singleton =>
        new Traversal[A, B] {
          extension (a: A) {
            def collectAll: List[B] =
              inline if (isB[A, B]) a.asInstanceOf[B] :: Nil else Nil
            def collectWhere[C](f: PartialFunction[B, C]): List[C] =
              inline if (isB[A, B]) f.lift(a.asInstanceOf[B]).toList else Nil
            def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
              inline if (isB[A, B]) f.lift(a.asInstanceOf[B]) else None
            def modifyAll(f: B => B): A =
              inline if (isB[A, B]) {
                val fb = f(a.asInstanceOf[B])
                if (a.getClass.isInstance(fb)) fb.asInstanceOf[A] else a
              } else a
            def modifyAllInternal(f: B => B): A = a
            def modifyCollect[C](f: B => (B, C)): (A, List[C]) =
              inline if (isB[A, B]) {
                val (b0, c0) = f(a.asInstanceOf[B])
                if (a.getClass.isInstance(b0)) (b0.asInstanceOf[A], c0 :: Nil) else (a, c0 :: Nil)
              } else (a, Nil)
            def modifyCollectInternal[C](f: B => (B, C)): (A, List[C]) = (a, Nil)
          }
        }
      case p: Mirror.ProductOf[A] =>
        new Traversal[A, B] {
          extension (a: A) {
            def collectAll: List[B] = {
              val prod = a.asInstanceOf[Product]
              val bss  = collectAllFields[p.MirroredElemTypes, B](prod, 0)
              inline if (isB[A, B]) a.asInstanceOf[B] :: bss else bss
            }
            def collectWhere[C](f: PartialFunction[B, C]): List[C] = {
              val prod = a.asInstanceOf[Product]
              val css  = collectWhereFields[p.MirroredElemTypes, B, C](prod, 0, f)
              inline if (isB[A, B]) f.lift(a.asInstanceOf[B]).toList ::: css else css
            }
            def collectFirst_[C](f: PartialFunction[B, C]): Option[C] = {
              val prod = a.asInstanceOf[Product]
              inline if (isB[A, B])
                f.lift(a.asInstanceOf[B]).orElse(collectFirstFields[p.MirroredElemTypes, B, C](prod, 0, f))
              else
                collectFirstFields[p.MirroredElemTypes, B, C](prod, 0, f)
            }
            def modifyAll(f: B => B): A = {
              val a0 = modifyAllInternal(f)
              inline if (isB[A, B]) {
                val fb = f(a0.asInstanceOf[B])
                if (a0.getClass.isInstance(fb)) fb.asInstanceOf[A] else a0
              } else a0
            }
            def modifyAllInternal(f: B => B): A = {
              val prod = a.asInstanceOf[Product]
              val tup  = modifyAllFields[p.MirroredElemTypes, B](prod, 0, f)
              constructProduct[A](tup)(using p)
            }
            def modifyCollect[C](f: B => (B, C)): (A, List[C]) = {
              val (a0, cs) = modifyCollectInternal(f)
              inline if (isB[A, B]) {
                val (b0, c0) = f(a0.asInstanceOf[B])
                if (a0.getClass.isInstance(b0)) (b0.asInstanceOf[A], c0 :: cs) else (a0, c0 :: cs)
              } else (a0, cs)
            }
            def modifyCollectInternal[C](f: B => (B, C)): (A, List[C]) = {
              val prod      = a.asInstanceOf[Product]
              val (tup, cs) = modifyCollectFields[p.MirroredElemTypes, B, C](prod, 0, f)
              (constructProduct[A](tup)(using p), cs)
            }
          }
        }
      case s: Mirror.SumOf[A] => sum[A, B](s, summonAll[s.MirroredElemTypes, B].toArray, isB[A, B])
    }

  private val nullTraversal: Traversal[Any, Any] = new Traversal[Any, Any] {
    extension (a: Any) {
      def collectAll: List[Any]                                        = Nil
      def collectWhere[C](f: PartialFunction[Any, C]): List[C]         = Nil
      def collectFirst_[C](f: PartialFunction[Any, C]): Option[C]      = None
      def modifyAll(f: Any => Any): Any                                = a
      def modifyAllInternal(f: Any => Any): Any                        = a
      def modifyCollect[C](f: Any => (Any, C)): (Any, List[C])         = (a, Nil)
      def modifyCollectInternal[C](f: Any => (Any, C)): (Any, List[C]) = (a, Nil)
    }
  }

  given [A, B](using NotGiven[A <:< Product]): Traversal[A, B] =
    nullTraversal.asInstanceOf[Traversal[A, B]]

  given [A, B](using t: Traversal[A, B]): Traversal[List[A], B] = new Traversal[List[A], B] {
    extension (xs: List[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f))
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
        xs.view.map(t.collectFirst_(_)(f)).collectFirst { case Some(x) => x }
      def modifyAll(f: B => B): List[A]         = xs.map(t.modifyAll(_)(f))
      def modifyAllInternal(f: B => B): List[A] = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (List[A], List[C]) = {
        val bsBuf  = scala.collection.mutable.ListBuffer.empty[A]
        val cssBuf = scala.collection.mutable.ListBuffer.empty[C]
        var it     = xs
        while (it.nonEmpty) {
          val r = t.modifyCollect(it.head)(f)
          bsBuf += r._1
          cssBuf ++= r._2
          it = it.tail
        }
        (bsBuf.toList, cssBuf.toList)
      }
      def modifyCollectInternal[C](f: B => (B, C)): (List[A], List[C]) = modifyCollect(f)
    }
  }

  given [A, B](using t: Traversal[A, B]): Traversal[Vector[A], B] = new Traversal[Vector[A], B] {
    extension (xs: Vector[A]) {
      def collectAll: List[B]                                = xs.flatMap(t.collectAll(_)).toList
      def collectWhere[C](f: PartialFunction[B, C]): List[C] = xs.flatMap(t.collectWhere(_)(f)).toList
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
        xs.view.map(t.collectFirst_(_)(f)).collectFirst { case Some(x) => x }
      def modifyAll(f: B => B): Vector[A]         = xs.map(t.modifyAll(_)(f))
      def modifyAllInternal(f: B => B): Vector[A] = xs.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Vector[A], List[C]) = {
        val bsBuf  = scala.collection.mutable.ArrayBuffer.empty[A]
        val cssBuf = scala.collection.mutable.ListBuffer.empty[C]
        xs.foreach { x =>
          val r = t.modifyCollect(x)(f)
          bsBuf += r._1
          cssBuf ++= r._2
        }
        (bsBuf.toVector, cssBuf.toList)
      }
      def modifyCollectInternal[C](f: B => (B, C)): (Vector[A], List[C]) = modifyCollect(f)
    }
  }

  given [A, B](using t: Traversal[A, B]): Traversal[Option[A], B] = new Traversal[Option[A], B] {
    extension (o: Option[A]) {
      def collectAll: List[B]                                   = o.fold(Nil)(t.collectAll(_))
      def collectWhere[C](f: PartialFunction[B, C]): List[C]    = o.fold(Nil)(t.collectWhere(_)(f))
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] = o.fold(None)(t.collectFirst_(_)(f))
      def modifyAll(f: B => B): Option[A]                       = o.map(t.modifyAll(_)(f))
      def modifyAllInternal(f: B => B): Option[A]               = o.map(t.modifyAll(_)(f))
      def modifyCollect[C](f: B => (B, C)): (Option[A], List[C]) = o match {
        case None    => (None, Nil)
        case Some(x) => val (b, cs) = t.modifyCollect(x)(f); (Some(b), cs)
      }
      def modifyCollectInternal[C](f: B => (B, C)): (Option[A], List[C]) = modifyCollect(f)
    }
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
}
