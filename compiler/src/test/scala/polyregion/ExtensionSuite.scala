package polyregion

import polyregion.compiletime.*

import scala.compiletime.*
import scala.reflect.ClassTag

implicit class RI(i: Int) {
  //  private val x : Int = 3
  //  val y : Int = 2
  //  println("1")
  def max3(j: Int) = i + j
}

class ExtensionSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.ExtensionSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

//  extension (i : Int){
//    infix def max2(j : Int) = j
//  }

//  import RI._

  case class V3(a: Float, b: Float, c: Float) {
    infix def add(that: V3) = V3(a + that.a, b + that.b, c + that.c)
  }

  inline given NativeStruct[V3] = nativeStructOf

  {
    val a = 1
    val b = 1
    testExpr {
      val x  = a.toFloat
      val xs = Array.ofDim[Float](3)
      xs(0) = x
      xs(1) = x + 1f
      xs(2) = x + 2f
      val y  = math.abs(x)
      val z  = V3(xs(0), 2f, 3f)
      val aa = z add z
      // val m = (1d,2f)
      a.max3(b) + z.a.toInt
      // V3(1,2,2)
    }
  }

  {}

}
