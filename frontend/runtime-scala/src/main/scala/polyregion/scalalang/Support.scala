package polyregion.scalalang

import scala.reflect.ClassTag
import scala.math.Integral
import scala.collection.immutable.ArraySeq
import scala.collection.mutable

object Support {

  inline def splitStatic(inline range: Range)(inline n: Int): Seq[Range] =
    if (range.isEmpty) Seq()
    else if (n == 1) Seq(range)
    else if (range.size < n) range.map(n => n to n)
    else {
      val xn = range.size
      val k  = xn / n
      val m  = xn % n

      val start = range.start
      val bound = if (range.isInclusive) range.end - 1 else range.end
      val step  = range.step

      (0 until n).map { i =>

        val a = i * k + i.min(m)
        val b = (i + 1) * k + (i + 1).min(m)

        ((start + a) * step until (start + b) * step) by step
      }
    }

  inline def linearise(inline start: Int, inline step: Int)(inline index: Int) = start + (index * step)


}
