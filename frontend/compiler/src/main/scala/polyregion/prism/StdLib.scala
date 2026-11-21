package polyregion.prism

import polyregion.ast.{PolyAST as p, *}
import polyregion.prism.compiletime.*
import polyregion.scalalang.intrinsics
import polyregion.scalalang.intrinsics.TypedBuffer

import scala.Predef as _
import scala.annotation.targetName
import scala.quoted.{Expr, Quotes, Type}

object StdLib {

  class Tuple2[T1, T2](val v1: T1, val v2: T2) {
    def _1 = v1
    def _2 = v2
  }

  class Range(val start: Int, val end: Int, val step: Int, val isInclusive: Boolean, val isEmpty: Boolean) {
    def by(step: Int): Range = {
      if (step == 0) intrinsics.raise[java.lang.IllegalArgumentException]("step cannot be 0.")
      new Range(
        start,
        end,
        step,
        isInclusive,
        (start > end && step > 0) || (start < end && step < 0) || (start == end && !isInclusive)
      )
    }

    def foreach[U](f: Int => U): Unit = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var k = 0L
      while (k < n) {
        f((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
    }

    def length: Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      n.toInt
    }

    def size: Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      n.toInt
    }

    def nonEmpty: Boolean = !isEmpty

    def contains(x: Int): Boolean =
      if (x == end && !isInclusive) false
      else if (step > 0) {
        if (x < start || x > end) false else step == 1 || (x.toLong - start.toLong) % step.toLong == 0L
      } else {
        if (x < end || x > start) false else step == -1 || (start.toLong - x.toLong) % (-step.toLong) == 0L
      }

    def apply(idx: Int): Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      if (idx < 0 || idx >= n) intrinsics.raise[java.lang.IndexOutOfBoundsException]("index out of bounds")
      (start.toLong + idx.toLong * step.toLong).toInt
    }

    def head: Int = {
      if (isEmpty) intrinsics.raise[java.util.NoSuchElementException]("head on empty Range")
      start
    }

    def last: Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.util.NoSuchElementException]("last on empty Range")
      (start.toLong + (n - 1L) * step.toLong).toInt
    }

    def foldLeft[B](z: B)(op: (B, Int) => B): B = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var acc = z
      var k   = 0L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def exists(p: Int => Boolean): Boolean = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var found = false
      var k     = 0L
      while (k < n && !found) {
        found = p((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      found
    }

    def forall(p: Int => Boolean): Boolean = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var all = true
      var k   = 0L
      while (k < n && all) {
        all = p((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      all
    }

    def count(p: Int => Boolean): Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var found = 0
      var k     = 0L
      while (k < n) {
        if (p((start.toLong + k * step.toLong).toInt)) found += 1
        k += 1L
      }
      found
    }

    def reduce[B >: Int](op: (B, B) => B): B = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.lang.UnsupportedOperationException]("empty.reduceLeft")
      var acc: B = start
      var k      = 1L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def reduceLeft[B >: Int](op: (B, Int) => B): B = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.lang.UnsupportedOperationException]("empty.reduceLeft")
      var acc: B = start
      var k      = 1L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def map[B](f: Int => B): IndexedSeq[B] = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      val out = intrinsics.array[B](n.toInt)
      var k   = 0L
      while (k < n) {
        out.update(k.toInt, f((start.toLong + k * step.toLong).toInt))
        k += 1L
      }
      new IndexedSeq[B](n.toInt, out)
    }

    def filter(pred: Int => Boolean): IndexedSeq[Int] = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      val out = intrinsics.array[Int](n.toInt)
      var k   = 0L
      var m   = 0
      while (k < n) {
        val x = (start.toLong + k * step.toLong).toInt
        if (pred(x)) { out.update(m, x); m += 1 }
        k += 1L
      }
      new IndexedSeq[Int](m, out)
    }

    def sum[B >: Int](num: Numeric[B]): Int = {
      val hi: Long = if (isInclusive) end.toLong else if (step > 0) end.toLong - 1L else end.toLong + 1L
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      if (isEmpty) 0
      else if (n == 1L) start
      else ((n * (start.toLong + (start.toLong + (n - 1L) * step.toLong))) / 2L).toInt
    }
  }

  class Inclusive(val start: Int, val end: Int, val step: Int, val isEmpty: Boolean) {
    def by(step: Int): Range = {
      if (step == 0) intrinsics.raise[java.lang.IllegalArgumentException]("step cannot be 0.")
      new Range(start, end, step, isInclusive = true, (start > end && step > 0) || (start < end && step < 0))
    }

    def foreach[U](f: Int => U): Unit = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var k = 0L
      while (k < n) {
        f((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
    }

    def length: Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      n.toInt
    }

    def size: Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      n.toInt
    }

    def nonEmpty: Boolean = !isEmpty

    def contains(x: Int): Boolean =
      if (step > 0) {
        if (x < start || x > end) false else step == 1 || (x.toLong - start.toLong) % step.toLong == 0L
      } else {
        if (x < end || x > start) false else step == -1 || (start.toLong - x.toLong) % (-step.toLong) == 0L
      }

    def apply(idx: Int): Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      if (idx < 0 || idx >= n) intrinsics.raise[java.lang.IndexOutOfBoundsException]("index out of bounds")
      (start.toLong + idx.toLong * step.toLong).toInt
    }

    def head: Int = {
      if (isEmpty) intrinsics.raise[java.util.NoSuchElementException]("head on empty Range")
      start
    }

    def last: Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.util.NoSuchElementException]("last on empty Range")
      (start.toLong + (n - 1L) * step.toLong).toInt
    }

    def foldLeft[B](z: B)(op: (B, Int) => B): B = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var acc = z
      var k   = 0L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def exists(p: Int => Boolean): Boolean = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var found = false
      var k     = 0L
      while (k < n && !found) {
        found = p((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      found
    }

    def forall(p: Int => Boolean): Boolean = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var all = true
      var k   = 0L
      while (k < n && all) {
        all = p((start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      all
    }

    def count(p: Int => Boolean): Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      var found = 0
      var k     = 0L
      while (k < n) {
        if (p((start.toLong + k * step.toLong).toInt)) found += 1
        k += 1L
      }
      found
    }

    def reduce[B >: Int](op: (B, B) => B): B = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.lang.UnsupportedOperationException]("empty.reduceLeft")
      var acc: B = start
      var k      = 1L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def reduceLeft[B >: Int](op: (B, Int) => B): B = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (isEmpty) intrinsics.raise[java.lang.UnsupportedOperationException]("empty.reduceLeft")
      var acc: B = start
      var k      = 1L
      while (k < n) {
        acc = op(acc, (start.toLong + k * step.toLong).toInt)
        k += 1L
      }
      acc
    }

    def map[B](f: Int => B): IndexedSeq[B] = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      val out = intrinsics.array[B](n.toInt)
      var k   = 0L
      while (k < n) {
        out.update(k.toInt, f((start.toLong + k * step.toLong).toInt))
        k += 1L
      }
      new IndexedSeq[B](n.toInt, out)
    }

    def filter(pred: Int => Boolean): IndexedSeq[Int] = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      val out = intrinsics.array[Int](n.toInt)
      var k   = 0L
      var m   = 0
      while (k < n) {
        val x = (start.toLong + k * step.toLong).toInt
        if (pred(x)) { out.update(m, x); m += 1 }
        k += 1L
      }
      new IndexedSeq[Int](m, out)
    }

    def sum[B >: Int](num: Numeric[B]): Int = {
      val hi: Long = end.toLong
      val n: Long =
        if (isEmpty) 0L
        else if (step > 0) (hi - start.toLong) / step.toLong + 1L
        else (start.toLong - hi) / (-step.toLong) + 1L
      if (n > Int.MaxValue.toLong)
        intrinsics.raise[java.lang.IllegalArgumentException]("seqs cannot contain more than Int.MaxValue elements.")
      if (isEmpty) 0
      else if (n == 1L) start
      else ((n * (start.toLong + (start.toLong + (n - 1L) * step.toLong))) / 2L).toInt
    }
  }

  class IndexedSeq[A](val length_ : Int, val data: intrinsics.TypedBuffer[A]) {
    def length: Int       = length_
    def size: Int         = length_
    def isEmpty: Boolean  = length_ == 0
    def nonEmpty: Boolean = length_ != 0

    def apply(i: Int): A = {
      if (i < 0 || i >= length_) intrinsics.raise[java.lang.IndexOutOfBoundsException]("index out of bounds")
      data(i)
    }

    def foreach[U](f: A => U): Unit = {
      var i = 0
      while (i < length_) {
        f(data(i))
        i += 1
      }
    }

    def foldLeft[B](z: B)(op: (B, A) => B): B = {
      var acc = z
      var i   = 0
      while (i < length_) {
        acc = op(acc, data(i))
        i += 1
      }
      acc
    }

    def exists(p: A => Boolean): Boolean = {
      var found = false
      var i     = 0
      while (i < length_ && !found) {
        found = p(data(i))
        i += 1
      }
      found
    }

    def forall(p: A => Boolean): Boolean = {
      var all = true
      var i   = 0
      while (i < length_ && all) {
        all = p(data(i))
        i += 1
      }
      all
    }

    def count(p: A => Boolean): Int = {
      var found = 0
      var i     = 0
      while (i < length_) {
        if (p(data(i))) found += 1
        i += 1
      }
      found
    }

    def sum[B >: A](num: Numeric[B]): B = {
      var acc: B = intrinsics.zero[B]
      var i      = 0
      while (i < length_) {
        acc = intrinsics.add[B](acc, data(i))
        i += 1
      }
      acc
    }

    def map[B](f: A => B): IndexedSeq[B] = {
      val out = intrinsics.array[B](length_)
      var i   = 0
      while (i < length_) {
        out.update(i, f(data(i)))
        i += 1
      }
      new IndexedSeq[B](length_, out)
    }

    def filter(pred: A => Boolean): IndexedSeq[A] = {
      val out = intrinsics.array[A](length_)
      var i   = 0
      var m   = 0
      while (i < length_) {
        val x = data(i)
        if (pred(x)) { out.update(m, x); m += 1 }
        i += 1
      }
      new IndexedSeq[A](m, out)
    }
  }

  class Class[T] {}

  class Numeric[T]

  class ClassTag[T] { // extends scala.Equals {
    // override def canEqual(that: Any): Boolean = true
    // TODO
  }
  object ClassTag {
    def apply[T](runtimeClass1: java.lang.Class[?]): ClassTag[T] = new ClassTag[T]
  }

  class RichInt(val x: Int) {
    def min(that: Int)         = java.lang.Math.min(x, that)
    def max(that: Int)         = java.lang.Math.max(x, that)
    def until(end: Int): Range = new Range(x, end, 1, isInclusive = false, x >= end)
    def until(end: Int, step: Int): Range = {
      if (step == 0) intrinsics.raise[java.lang.IllegalArgumentException]("step cannot be 0.")
      new Range(x, end, step, isInclusive = false, (x > end && step > 0) || (x < end && step < 0) || x == end)
    }
    def to(end: Int): Inclusive = Inclusive(x, end, 1, x > end)
    def to(end: Int, step: Int): Inclusive = {
      if (step == 0) intrinsics.raise[java.lang.IllegalArgumentException]("step cannot be 0.")
      Inclusive(x, end, step, (x > end && step > 0) || (x < end && step < 0))
    }
  }

  class ArrayOps[A](val xs: scala.Array[A]) {
    def size: Int = xs.length
  }

  class Predef {
    def intWrapper(x: Int): RichInt                      = RichInt(x)
    def intArrayOps(xs: scala.Array[Int]): ArrayOps[Int] = new ArrayOps[Int](xs)
  }

//   object Array {
//     def ofDim[T](n1: Int)(implicit ev: ClassTag[T]): Array[T] = new Array[T](n1, intrinsics.array[T](n1))
//   }
//   class Array[T](val length: Int, val data: intrinsics.Arr[T]) extends SizedArr[T] {
// //    def length: Int                = data.length
//     def apply(i: Int): T           = data.apply(i)
//     def update(i: Int, x: T): Unit = data.update(i, x)
//   }

  object MutableSeq {
    def onDim[T](N: Int): MutableSeq[T] = new MutableSeq[T](N, intrinsics.array[T](N))
  }
  class MutableSeq[A](val length_ : Int, val data: intrinsics.TypedBuffer[A]) {
    def length: Int                = length_
    def apply(i: Int): A           = data.apply(i)
    def update(i: Int, x: A): Unit = data.update(i, x)
  }

  // trait Function0[+R] { def apply(): R }

  // trait Function1[-T1, +R] {
  //   def apply(v1: T1): R
  //   def compose[A](g: A => T1): A => R = { x => apply(g(x)) }
  //   def andThen[A](g: R => A): T1 => A = { x => g(apply(x)) }
  // }

  final def Mirrors: List[Prism] = derivePackedMirrors(
    (
      // Unsupported

      witness[scala.collection.ArrayOps[?], ArrayOps[?]]( //
        { case (q @ given Quotes, _) => '{ throw new java.lang.AssertionError("No") } },
        { case (q @ given Quotes, _) => '{ throw new java.lang.AssertionError("No") } },
        { case (q @ given Quotes, ys, _) => '{ throw new java.lang.AssertionError("No") } }
      ),
//       // Mutable collections
//       witness[scala.Array.type, Array.type](_ => Array, (_, _) => scala.Array),
//       witness[scala.Array, Array](
//         [X] =>
//           (xs: scala.Array[X]) =>
//             Array[X](
//               xs.length,
//               new intrinsics.Arr[X] {
// //              override def length: Int                    = xs.length
//                 override def apply(i: scala.Int): X             = xs(i)
//                 override def update(i: scala.Int, x: X): scala.Unit = xs(i) = x
//               }
//           ),
//         [X] =>
//           (ys: scala.Array[X], xs: Array[X]) => {
//             var i = 0; while (i < xs.length) ys(i) = xs(i); ys
//         }
//       ),
      // witness[scala.collection.mutable.Seq.type, MutableSeq.type](_ => MutableSeq, (_, _) => scala.collection.mutable.Seq),
      // witness[scala.collection.mutable.Seq[?], MutableSeq[?]](
      //   { case (q @ given Quotes, xs) =>
      //     xs match {
      //       case '{ $xs: scala.collection.mutable.Seq[a] } =>
      //         '{
      //           new MutableSeq[a](
      //             $xs.length,
      //             new intrinsics.TypedBuffer[a] {
      //               override def apply(i: scala.Int): a             = $xs(i)
      //               override def update(i: scala.Int, x: a): scala.Unit = $xs(i) = x
      //             }
      //           )
      //         }
      //     }
      //   },
      //   { case (q @ given Quotes, xs) =>
      //     '{
      //       val ys = scala.collection.mutable.Seq[Any]($xs.length_)
      //       var i  = 0; while (i < $xs.length) ys(i) = $xs(i); ys
      //     }
      //   },
      //   { case (q @ given Quotes, ys, xs) =>
      //     (ys, xs) match {
      //       case ('{ $ys: scala.collection.mutable.Seq[Any] }, '{ $xs: MutableSeq[Any] }) =>
      //         '{ var i = 0; while (i < $xs.length) $ys(i) = $xs(i) }
      //     }
      //   }
      // ),
      // polyregion.scalalang.Buffer (sealed trait with primitive-specific subclasses) shares the
      // MutableSeq mirror shape with ListBuffer. The per-witness mirror table override (in
      // derivePackedTypeMirrorsImpl) lets both witnesses target MutableSeq without conflict.
      witness[polyregion.scalalang.Buffer[?], MutableSeq[?]](
        { case (q @ given Quotes, xs) =>
          xs match {
            case '{ $xs: polyregion.scalalang.Buffer[a] } =>
              '{
                new MutableSeq[a](
                  $xs.length,
                  new intrinsics.TypedBuffer[a] {
                    override def apply(i: scala.Int): a                 = $xs(i)
                    override def update(i: scala.Int, x: a): scala.Unit = $xs(i) = x
                  }
                )
              }
          }
        },
        { case (q @ given Quotes, xs) =>
          xs match {
            case '{ $xs: MutableSeq[t] } =>
              '{ polyregion.scalalang.Buffer.tabulate[t]($xs.length_)($xs(_)) }
          }
        },
        { case (q @ given Quotes, ys, xs) =>
          ((ys, xs): @unchecked) match {
            case ('{ $ys: polyregion.scalalang.Buffer[Any] }, '{ $xs: MutableSeq[Any] }) =>
              '{
                var i = 0
                while (i < $xs.length) {
                  $ys(i) = $xs(i)
                  i += 1
                }
              }
          }
        }
      ),
      witness[scala.collection.mutable.ListBuffer[?], MutableSeq[?]](
        { case (q @ given Quotes, xs) =>
          xs match {
            case '{ $xs: scala.collection.mutable.ListBuffer[a] } =>
              '{
                new MutableSeq[a](
                  $xs.length,
                  new intrinsics.TypedBuffer[a] {
                    override def apply(i: scala.Int): a                 = $xs(i)
                    override def update(i: scala.Int, x: a): scala.Unit = $xs(i) = x
                  }
                )
              }
          }
        },
        { case (q @ given Quotes, xs) =>
          xs match {
            case '{ $xs: MutableSeq[t] } =>
              '{
                scala.collection.mutable.ListBuffer.tabulate[t]($xs.length_)($xs(_))
              }
          }

        },
        { case (q @ given Quotes, ys, xs) =>
          ((ys, xs): @unchecked) match {
            case ('{ $ys: scala.collection.mutable.ListBuffer[Any] }, '{ $xs: MutableSeq[Any] }) =>
              '{
                var i = 0
                while (i < $xs.length) {
                  $ys(i) = $xs(i)
                  i += 1
                }
              }
          }
        }
      ),
      // FIXME ArrayBuffer mirror: same shape as ListBuffer would work, but adding a third
      // source-class binding to MutableSeq breaks downstream uses of the global
      // mirrorToSourceTable (only one source per mirror is kept, dropping the existing Buffer
      // / ListBuffer entries in some rewrite paths). Re-enable once the rewriter consults
      // per-call-site bindings everywhere.
      // witness[scala.Tuple2[?, ?], Tuple2[?, ?]]( //
      //   { case (_ @ given Quotes, x) =>
      //     x match { case '{ $x: scala.Tuple2[t0, t1] } => '{ new Tuple2[t0, t1]($x._1, $x._2) } }
      //   },
      //   { case (_ @ given Quotes, x) =>
      //     x match { case '{ $x: Tuple2[t0, t1] } => '{ scala.Tuple2[t0, t1]($x._1, $x._2) } }
      //   },
      //   { case (_ @ given Quotes, ys, _) => '{ () } }
      // ),

      // Immutable types, restore simply uses the original instance
      witness[scala.reflect.ClassTag[?], ClassTag[?]]( //
        { case (_ @ given Quotes, x) => '{ new ClassTag[Any]() } },
        { case (_ @ given Quotes, x) =>
          x match { // TODO not sure if this will actually work
            case '{ $cls: ClassTag[t] } => '{ scala.compiletime.summonInline[scala.reflect.ClassTag[t]] }
          }
        },
        { case (_ @ given Quotes, ys, _) => '{ () } }
      ),
      witness[scala.collection.immutable.IndexedSeq[?], IndexedSeq[?]]( //
        { case (q @ given Quotes, _) => '{ throw new java.lang.AssertionError("No") } },
        { case (q @ given Quotes, _) => '{ throw new java.lang.AssertionError("No") } },
        { case (q @ given Quotes, ys, _) => '{ throw new java.lang.AssertionError("No") } }
      ),
      witness[scala.math.Numeric[?], Numeric[?]]( //
        { case (_ @ given Quotes, _) => '{ new Numeric[Any]() } },
        { case (_ @ given Quotes, _) => '{ throw new java.lang.AssertionError("No") } },
        { case (_ @ given Quotes, ys, _) => '{ () } }
      ),
      witness[java.lang.Class[?], Class[?]]( //
        { case (_ @ given Quotes, _) => '{ new Class[Any]() } },
        { case (_ @ given Quotes, x) =>
          x match { // TODO not sure if this will actually work
            case '{ $cls: Class[t] } => '{ scala.compiletime.summonInline[java.lang.Class[t]] }
          }
        },
        { case (_ @ given Quotes, ys, _) => '{ () } }
      ),
      witness[scala.collection.immutable.Range, Range](
        { case (_ @ given Quotes, r) => '{ new Range($r.start, $r.end, $r.step, $r.isInclusive, $r.isEmpty) } },
        { case (_ @ given Quotes, r) =>
          '{
            if ($r.isInclusive) scala.collection.immutable.Range.inclusive($r.start, $r.end, $r.step)
            else scala.collection.immutable.Range($r.start, $r.end, $r.step)
          }
        },
        { case (_ @ given Quotes, _, _) => '{ () } }
      ),
      witness[scala.collection.immutable.Range.Inclusive, Inclusive](
        { case (_ @ given Quotes, r) => '{ Inclusive($r.start, $r.end, $r.step, $r.isEmpty) } },
        { case (_ @ given Quotes, r) => '{ scala.collection.immutable.Range.inclusive($r.start, $r.end, $r.step) } },
        { case (_ @ given Quotes, _, _) => '{ () } }
      ),
      witness[scala.reflect.ClassTag.type, ClassTag.type](
        { case (_ @ given Quotes, _) => '{ ClassTag } },
        { case (_ @ given Quotes, _) => '{ scala.reflect.ClassTag } },
        { case (_ @ given Quotes, _, _) => '{ scala.reflect.ClassTag } }
      ),
      witness[scala.runtime.RichInt, RichInt](
        { case (_ @ given Quotes, x) => '{ RichInt($x.self) } },
        { case (_ @ given Quotes, x) => '{ new scala.runtime.RichInt($x.x) } },
        { case (_ @ given Quotes, x, _) => '{ () } }
      ),
      witness[scala.Predef.type, Predef](
        { case (_ @ given Quotes, _) => '{ Predef() } },
        { case (_ @ given Quotes, _) => '{ scala.Predef } },
        { case (_ @ given Quotes, x, _) => '{ () } }
      )
    )
  )

  final def Tables: PrismTables = PrismTables(Mirrors, Functions, StructDefs)

  final def Functions: Map[p.Signature, (p.Function, Set[p.StructDef])] =
    Mirrors
      .map(_._1)
      .flatMap(m => m.functions.map(f => f -> Set(m.structDef.copy(name = m.source))))
      .map { case (f, clsDeps) => f.signature -> (f, clsDeps) }
      .toMap

  final def StructDefs: Map[p.Sym, p.StructDef] =
    Mirrors
      .map(_._1)
      .map { x =>
        x.source -> x.structDef.copy(name = x.source)
      }
      .toMap

  @main def main(): Unit = {
    Functions.values.foreach { case (fn, deps) =>
      println(s"${fn.repr.linesIterator.map("\t" + _).mkString("\n")}")
    }
    ()
  }

}
