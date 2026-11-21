package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.ParLowering

import scala.annotation.compileTimeOnly

object par {

  class Par[T] private ()

  object stages {

    private inline val Stored = "a .par chain must end in a terminal operation within the same expression"

    @compileTimeOnly(Stored) def array[T](xs: Array[T], context: rt.Context): Par[T] = ???
    @compileTimeOnly(Stored) def range(xs: Range, context: rt.Context): Par[Int]     = ???
    @compileTimeOnly(Stored) def on[T](xs: Par[T], context: rt.Context): Par[T]      = ???
    @compileTimeOnly(Stored) def map[T, U](xs: Par[T], f: T => U): Par[U]            = ???
    @compileTimeOnly(Stored) def filter[T](xs: Par[T], f: T => Boolean): Par[T]      = ???
  }

  extension [T](inline xs: Array[T]) inline def par: Par[T] = stages.array(xs, rt.Context.current())

  extension (inline xs: Range) inline def par: Par[Int] = stages.range(xs, rt.Context.current())

  extension [T](inline xs: Par[T]) {
    inline def on(inline context: rt.Context): Par[T]           = stages.on(xs, context)
    inline def map[U](inline f: T => U): Par[U]                 = stages.map(xs, f)
    inline def filter(inline f: T => Boolean): Par[T]           = stages.filter(xs, f)
    inline def filterNot(inline f: T => Boolean): Par[T]        = stages.filter(xs, (x: T) => !f(x))
    inline def toArray: Array[T]                                = ${ ParLowering.toArray[T]('xs) }
    inline def foreach(inline f: T => Unit): Unit               = ${ ParLowering.foreach('xs, 'f) }
    inline def sum: T                                           = ${ ParLowering.sum[T]('xs) }
    inline def product: T                                       = ${ ParLowering.product[T]('xs) }
    inline def min: T                                           = ${ ParLowering.min[T]('xs) }
    inline def max: T                                           = ${ ParLowering.max[T]('xs) }
    inline def size: Int                                        = ${ ParLowering.size('xs) }
    inline def count(inline f: T => Boolean): Int               = ${ ParLowering.count('xs, 'f) }
    inline def exists(inline f: T => Boolean): Boolean          = ${ ParLowering.exists('xs, 'f) }
    inline def forall(inline f: T => Boolean): Boolean          = ${ ParLowering.forall('xs, 'f) }
    inline def fold(inline z: T)(inline op: (T, T) => T): T     = ${ ParLowering.fold[T]('xs, 'z, 'op) }
    inline def reduce(inline op: (T, T) => T): T                = ${ ParLowering.reduce[T]('xs, 'op) }
    inline def sorted: Array[T]                                 = ${ ParLowering.sorted[T]('xs) }
    inline def sortWith(inline lt: (T, T) => Boolean): Array[T] = ${ ParLowering.sortWith[T]('xs, 'lt) }
  }
}
