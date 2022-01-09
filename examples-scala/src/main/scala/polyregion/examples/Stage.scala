package polyregion.examples

import com.kenai.jffi.HeapInvocationBuffer

import java.nio.ByteBuffer
import scala.collection.mutable.ArrayBuffer
import polyregion.PolyregionRuntime

object Stage {

  object Foo {

    class U {
      def x = b

    }
    def a = 3
    def b = {
      var x = 2
      x *= 12345
      x
    }
    def z = {
      val xx = "FOOBARBAZ"
    }
    val c = "a"
  }

  case class Bar(to: Int) {
    def go = {
      var a = 0
      for (i <- (0 to to))
        a += i
    }
  }

  object Bar {

    def work(n: Int) = {
      val xs = 0 to n
      val ys = 0 to n

    }

  }

  case class Vec2[T](x: T, y: T) {
    inline def +(that: Vec2[T]): Vec2[T]     = Vec2(x, y)
    def noInlinePlus(that: Vec2[T]): Vec2[T] = Vec2(x, y)
  }

  val CONST = 42

  import polyregion.compileTime._

  // showTpe[Bar]

  class In {
    val n = { (n: Int) =>
      n + 1
      println(s"A = $n")
    }
  }

  class Out {
    println("B")
    println("B")

    val u = In()
    val x = Option(u)
    x.foreach { y =>
      // showExpr(y.n)
    }
  }

  // val ys = Vector[Int]()
  // val zs = ArrayBuffer[Int]()

  // showExpr { (n: Int) =>
  //   n + 1
  //   val u                 = xs(422)
  //   val bad: Array[Float] = xs.map(_ * 2f)

  //   ys
  //   zs
  //   xs(n + 2) = CONST.toFloat + 42f + bad(0)
  // }

  val vv = Vec2(2, 2)
//  foreach(0 to 10) { foo =>
//    val lambda2 = foo
//  }

  // val xs : List[Pointers] =
  // engine.invoke(LLVMIR, xs)
  //

//  Runtime.LibFfi.invoke(
//    1,
//    polyregion.Runtime.NullPtr -> Runtime.LibFfi.Type.Void,
//    xs.pointer                 -> Runtime.LibFfi.Type.Float,
//    ys.pointer                 -> Runtime.LibFfi.Type.Float
//  )

  // val V1 = refOut(n)
  // val V2 = V1 * scalar
  // val V3 = ys(n)
  // val V4 = V3 + V2
  // val V5 = xs(n)
  // val VO = V4 + V5
  // xs.update(n, VO)

  val scalar = 42.69f

  def main(args: Array[String]) = {
    PolyregionRuntime.load();
    PolyregionRuntime.invoke(
      Array.emptyByteArray,
      "abc",
      1,
      java.nio.ByteBuffer.allocate(0),
      Array.emptyByteArray,
      Array.emptyObjectArray
    );
  }

  def main2(args: Array[String]): Unit = {
    val xs = polyregion.Buffer.ofDim[Float](10)
    val ys = polyregion.Buffer.ofDim[Float](10)

    def printAndreset(name: String) = {
      println(s"[$name]")
      println(s" xs = ${xs.toList}")
      println(s" ys = ${ys.toList}")
      println("---")
      for (x <- xs.indices) {

        xs(x) = x.toFloat
        ys(x) = x.toFloat
      }
    }

    // foreach(1 to 10) { n =>
    //   var foo = n
    //   foo = 0

    // }

    printAndreset("none")

    foreachJVM(0 until 10) { n =>
      xs(n) += 2f
      val scalarLambda = 321.1f
      val scalarF      = scalarLambda + 123f
      var refOut       = xs
      xs(n) += ys(n) + refOut(n) * scalar + scalarLambda + scalarF
    }
    printAndreset("JVM")

    offload {
      var a = 1f
      xs(0) = 42f
      xs(1) = a + xs(1)
    }

    // foreach(0 until 10) { n =>
    //   xs(n) += 2f
    //   val scalarLambda = 321.1f
    //   val scalarF      = scalarLambda + 123f
    //   var refOut       = xs
    //   xs(n) += ys(n) + refOut(n) * scalar + scalarLambda + scalarF
    // }
    import com.kenai.jffi.Library

    val handle =
      com.kenai.jffi.Library.openLibrary("/home/tom/polyregion/native/obj.so", Library.LAZY);

    println("h=" + handle)
    val b = polyregion.Runtime.FFIInvocationBuilder()
    b.arg(1)
    b.arg(scalar)
    b.arg(ys.buffer)
    b.arg(xs.buffer)
    b.arg(10)
    b.arg(0)
    // b.invoke(140462809608192L)
    printAndreset("Native")

//
//    foreach(0 until 10) { n =>
////      xs(n) = ys(n)
//
//      var i = 0
//      while(i < bound){
//
//        i += step
//      }
//
//    }
//

//    foreach(0 until 10) { n =>
//      xs(n) = 42f
//      ys(n) = 42f
//    }

//
//    System.setProperty("a", "42")
//
//    val hidden = System.getProperty("a").toFloat
//    val x =
//      foreach(0 until 10) { n =>
//        xs(n) = hidden * 2f
//      }
//
//    printAndreset()
  }

}
