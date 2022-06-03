package polyregion.examples

import scala.collection.IterableOnceOps
import scala.concurrent.{Await, Future, Promise}
import scala.util.Try

//object Foo {
//  var x                         = 10
//  def bar                       = x + 1
//  def baz                       = bar + 1
//  def nBar(n: Int)              = n * bar + baz
//  inline def nBarInline(n: Int) = n * bar + baz + nBar(baz)
//  object Bar { object Baz }
//}

object Simple {
  //  object Bar { val y = Foo.bar }
//  val a         = 10
//  val b         = 20
//  private def c = (a + b) * 2

  def main(args: Array[String]): Unit = {
    println("Start")
    val x = 1
    val z = 1

//    object Local {
//      val g      = Foo.bar + Foo.baz + a + b + c + x
//      val levelB = LevelA.LevelB
//      object LevelA {
//        val a1 = Foo.bar
//        object LevelB { val a2 = g + a1 + x }
//      }
//    }
    val y = 50
    val n = 10

    val xs = polyregion.scala.Buffer.range[Int](0, n)

    val cow = Array(42, 100)

    // offload("-O3") { 1+1 }

    // Task API +JIT (multiple async dispatch supported; CPU=pooled, GPU=multiple command queue)
    // import polyregion.scala.backends.{Host: Device, JVM: Device, GPU: Runtime}
    //
    // singular : device.task[A](a: => A)
    // parallel :
    //    device.foreach(x: Range)(f: Int => Unit)
    //    device.foreachND(x: Range, l : Range)(f: Int => Unit)
    //    device.foreach(Intel.Haswell, AMD.Znver2)(x: Range, y: Range)(f: (Int, Int) => Unit)
    //    device.foreach(x: Range, y: Range, z: Range)(f: (Int, Int, Int) => Unit)
    //    device.reduce[A](x: Range)(c: (A, A) => A)(f: Int => A)
    //    device.reduce[A](x: Range, y: Range)(c: (A, A) => A)(f: (Int, Int) => A)
    //    device.reduce[A](x: Range, y: Range, z: Range)(c: (A, A) => A)(f: (Int, Int, Int) => A)

    //    device.reduce[A](combine : (A, A) => A)(x: Int)(f : Int => A)
    //    device.reduce[A](combine : (A, A) => A)(x: Int, y: Int)(f : (Int, Int) => A)
    //    device.reduce[A](combine : (A, A) => A)(x: Int, y: Int, z: Int)(f : (Int, Int, Int) => A)

    //  collection extensions: extension (xs : Seq[T]) {  def offload(d: Device) ...  }

    //  Task API AOT

    //  singular :

//    val l = Local

    println(xs)
    println(cow.toList)

    val rel = polyregion.jvm.runtime.Platform.Relocatable()
    val dev = rel.devices()(0)

    val q = dev.createQueue()

    import polyregion.scala.blocking.*
    import polyregion.scala.*

    type M = Config[Target.NVPTX64.SM61.type, Opt.O0]

    val result = CUDA.devices(0).aot.task[Config[Target.NVPTX64.SM61.type, Opt.O0], Int](43 )
    // val result = OpenCL.devices(0).aot.task[Config[Target.OpenCL_C.type, Opt.O0], Int](42)
    // val result = Host.aot.task[Config[Target.Host.type , Opt.O0], Int](   42)
    println("   R  =" + result)

    // val Const  = scala.compiletime.constValue[Target.X86.Znver.Arch]
    // val Const1 = scala.compiletime.constValue[c.Target#Arch]
//     val result = polyregion.scala.compiletime.offload[M, Unit](q, () => println("Done!")) {
// //       //      val objRef = Foo
// //       //      val c      = Local.LevelA.LevelB
// //       //      val c2     = Foo.Bar.Baz
// //       //      val m      = c.a2
// //       //      val n      = l
// //       //      val a40    = l.LevelA
// //       //      val a41    = l.LevelA.a1
// //       //      val m1     = l.levelB.a2
// //       //
// //       //      val i       = n.g
// //       //      val objRef2 = objRef
// //       //      val a       = Local
// //       //      val a1      = a.LevelA
// //       //      val a2      = a.LevelA.a1
// //       //      val m2      = Foo.bar + objRef.nBar(2) + objRef2.nBarInline(42)
// //       //      val a3      = Local.LevelA.a1
// //       //
// //       //      // aliases
// //       //      val alias1      = Simple
// //       //      val alias2      = alias1
// //       //      val alias3      = alias2
// //       //      val aliasX      = alias3
// //       //      val aliasY      = alias3
// //       //      val aliasResult = aliasX.a + aliasX.b + aliasY.a + aliasY.b
// //       //
// //       //      val y =
// //       //        objRef.bar + Foo.baz + x + b + m + Local.g + x + Local.LevelA.a1 + Local.LevelA.LevelB.a2 + aliasResult + m1
// //       //      val z = 42 + x + y + Foo.nBar(y) + Foo.nBarInline(y) + Bar.y + Simple.a + Simple.b + i + objRef2.x + a1.LevelB.a2
// //       //      val out = a3 + m2 + a2 + m + a41 + z
// //       //      Foo.bar + out + Foo.nBarInline(m)

//       var max = 0
//       var i   = 0
//       while (i < n) {
//         max = math.max(max, xs(i))
//         //        max = if(i < y) max*2 else max
// //         xs(i) = 42 // i-1
//         i += 1
//       }
//       xs(0) = max
//       // xs(1) = max * 2
//       // val c0 = cow(0)
//       // cow(0) = max
//       // cow(1) = c0 + cow(1)
// //       // cow(0) = cow(100)

// // //       val x = n
//       ()
// // //       max
//     }
    println(xs)
    println(cow.toList)

    // inline def foo[AA >: String]() = polyregion.scala.compiletime.showTpe[AA]

    // foo()
    // foo[Int]()
    // foo["X"]()
    // foo[Any]()
    // foo[String]()

    // println(s"result = $result")
    println("Done!")
  }

}
