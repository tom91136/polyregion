package polyregion.ast

import scala.deriving.Mirror
import scala.compiletime.{erasedValue, summonInline}
import scala.util.NotGiven
import scala.quoted.*

trait Traversal[A, B] {
  extension (a: A) {
    def visitAll(f: B => Unit): Unit
    def collectAll: List[B] = {
      val out = List.newBuilder[B]
      visitAll(out += _)
      out.result()
    }
    def collectWhere[C](f: PartialFunction[B, C]): List[C] = {
      val out     = List.newBuilder[C]
      val collect = f.runWith(out += _)
      visitAll { value =>
        collect(value)
        ()
      }
      out.result()
    }
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

  private inline def isB[A, B]: Boolean = inline erasedValue[A] match {
    case _: B => true
    case _    => false
  }

  private inline def summonAll[T <: Tuple, B]: List[Traversal[?, B]] = inline erasedValue[T] match {
    case _: EmptyTuple => Nil
    case _: (t *: ts)  => summonInline[Traversal[t, B]] :: summonAll[ts, B]
  }

  private def singleton[A, B](aSubB: Boolean): Traversal[A, B] =
    new Traversal[A, B] {
      extension (a: A) {
        def visitAll(f: B => Unit): Unit = if (aSubB) f(a.asInstanceOf[B])
        def collectFirst_[C](f: PartialFunction[B, C]): Option[C] =
          if (aSubB) f.lift(a.asInstanceOf[B]) else None
        def modifyAll(f: B => B): A =
          if (aSubB) {
            val fb = f(a.asInstanceOf[B])
            if (a.getClass.isInstance(fb)) fb.asInstanceOf[A] else a
          } else a
        def modifyAllInternal(f: B => B): A = a
        def modifyCollect[C](f: B => (B, C)): (A, List[C]) =
          if (aSubB) {
            val (b0, c0) = f(a.asInstanceOf[B])
            if (a.getClass.isInstance(b0)) (b0.asInstanceOf[A], c0 :: Nil) else (a, c0 :: Nil)
          } else (a, Nil)
        def modifyCollectInternal[C](f: B => (B, C)): (A, List[C]) = (a, Nil)
      }
    }

  private def product[A, B](
      m: Mirror.ProductOf[A],
      tssThunk: => Array[Traversal[?, B]],
      aSubB: Boolean
  ): Traversal[A, B] = new Traversal[A, B] {
    private lazy val tss = tssThunk
    extension (a: A) {
      def visitAll(f: B => Unit): Unit = {
        if (aSubB) f(a.asInstanceOf[B])
        val prod = a.asInstanceOf[Product]
        val arr  = tss
        var i    = 0
        while (i < arr.length) {
          arr(i).asInstanceOf[Traversal[Any, B]].visitAll(prod.productElement(i))(f)
          i += 1
        }
      }
      def collectFirst_[C](f: PartialFunction[B, C]): Option[C] = {
        val prod = a.asInstanceOf[Product]
        val arr  = tss
        val n    = arr.length
        def loop(i: Int): Option[C] =
          if (i >= n) None
          else
            arr(i)
              .asInstanceOf[Traversal[Any, B]]
              .collectFirst_(prod.productElement(i))(f)
              .orElse(loop(i + 1))
        if (aSubB) f.lift(a.asInstanceOf[B]).orElse(loop(0)) else loop(0)
      }
      def modifyAll(f: B => B): A = {
        val a0 = modifyAllInternal(f)
        if (aSubB) {
          val fb = f(a0.asInstanceOf[B])
          if (a0.getClass.isInstance(fb)) fb.asInstanceOf[A] else a0
        } else a0
      }
      def modifyAllInternal(f: B => B): A = {
        val prod = a.asInstanceOf[Product]
        val arr  = tss
        val n    = arr.length
        val out  = new Array[Any](n)
        var i    = 0
        while (i < n) {
          out(i) = arr(i).asInstanceOf[Traversal[Any, B]].modifyAll(prod.productElement(i))(f)
          i += 1
        }
        m.fromProduct(Tuple.fromArray(out))
      }
      def modifyCollect[C](f: B => (B, C)): (A, List[C]) = {
        val (a0, cs) = modifyCollectInternal(f)
        if (aSubB) {
          val (b0, c0) = f(a0.asInstanceOf[B])
          if (a0.getClass.isInstance(b0)) (b0.asInstanceOf[A], c0 :: cs) else (a0, c0 :: cs)
        } else (a0, cs)
      }
      def modifyCollectInternal[C](f: B => (B, C)): (A, List[C]) = {
        val prod = a.asInstanceOf[Product]
        val arr  = tss
        val n    = arr.length
        val out  = new Array[Any](n)
        val cs   = scala.collection.mutable.ListBuffer.empty[C]
        var i    = 0
        while (i < n) {
          val (nx, cx) = arr(i).asInstanceOf[Traversal[Any, B]].modifyCollect(prod.productElement(i))(f)
          out(i) = nx
          cs ++= cx
          i += 1
        }
        (m.fromProduct(Tuple.fromArray(out)), cs.toList)
      }
    }
  }

  private def sum[A, B](
      s: Mirror.SumOf[A],
      tssThunk: => Array[Traversal[?, B]],
      aSubB: Boolean
  ): Traversal[A, B] = new Traversal[A, B] {
    private lazy val tss = tssThunk
    extension (a: A) {
      def visitAll(f: B => Unit): Unit =
        tss(s.ordinal(a)).asInstanceOf[Traversal[A, B]].visitAll(a)(f)
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
        singleton[A, B](isB[A, B])
      case p: Mirror.ProductOf[A] =>
        product[A, B](p, summonAll[p.MirroredElemTypes, B].toArray, isB[A, B])
      case s: Mirror.SumOf[A] =>
        sum[A, B](s, summonAll[s.MirroredElemTypes, B].toArray, isB[A, B])
    }

  private val nullTraversal: Traversal[Any, Any] = new Traversal[Any, Any] {
    extension (a: Any) {
      def visitAll(f: Any => Unit): Unit                               = ()
      def collectFirst_[C](f: PartialFunction[Any, C]): Option[C]      = None
      def modifyAll(f: Any => Any): Any                                = a
      def modifyAllInternal(f: Any => Any): Any                        = a
      def modifyCollect[C](f: Any => (Any, C)): (Any, List[C])         = (a, Nil)
      def modifyCollectInternal[C](f: Any => (Any, C)): (Any, List[C]) = (a, Nil)
    }
  }

  def empty[A, B]: Traversal[A, B] = nullTraversal.asInstanceOf[Traversal[A, B]]

  given [A, B](using NotGiven[A <:< Product]): Traversal[A, B] =
    empty

  given [A, B](using t: Traversal[A, B]): Traversal[List[A], B] = new Traversal[List[A], B] {
    extension (xs: List[A]) {
      def visitAll(f: B => Unit): Unit = xs.foreach(t.visitAll(_)(f))
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
      def visitAll(f: B => Unit): Unit = xs.foreach(t.visitAll(_)(f))
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
      def visitAll(f: B => Unit): Unit                          = o.foreach(t.visitAll(_)(f))
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

    inline def visitAll[B](using t: Traversal[A, B])(f: B => Unit): Unit = t.visitAll(a)(f)
    def collectAll[B](using t: Traversal[A, B]): List[B]                 = t.collectAll(a)
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
