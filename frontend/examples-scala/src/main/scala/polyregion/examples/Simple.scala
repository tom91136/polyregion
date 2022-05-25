package polyregion.examples

import scala.collection.IterableOnceOps
import scala.concurrent.{Await, Future, Promise}
import scala.util.Try

object Foo {
  var x                         = 10
  def bar                       = x + 1
  def baz                       = bar + 1
  def nBar(n: Int)              = n * bar + baz
  inline def nBarInline(n: Int) = n * bar + baz + nBar(baz)
  object Bar { object Baz }
}

object Simple {
  object Bar { val y = Foo.bar }
  val a         = 10
  val b         = 20
  private def c = (a + b) * 2

  def main(args: Array[String]): Unit = {
    println("Start")
    val x = 1
    val z = 1

    object Local {
      val g      = Foo.bar + Foo.baz + a + b + c + x
      val levelB = LevelA.LevelB
      object LevelA {
        val a1 = Foo.bar
        object LevelB { val a2 = g + a1 + x }
      }
    }
    val y = 50
    val n = 100

    val xs = polyregion.scala.Buffer.range[Int](0, n)

    println(xs)

    enum Optimisation {
      case O0, O1, O2, O3, Ofast
    }

    sealed trait Arch
    enum CUDAArch extends Arch {
      case SM51, SM80
    }



    sealed trait CompileOptions
    object CompileOptions {
      case class JIT(opt: Optimisation, archHint: Option[Arch] = None) extends CompileOptions
      case class AOT[A <: Arch](arch: A, opt: Optimisation)            extends CompileOptions
    }

    trait Runtime[F[_], O] {
      def name: String
      def properties: Map[String, String]
      def devices: Vector[Device[F, O]]
      def jit: Runtime[F, CompileOptions.JIT]
    }

    trait SuspendedRuntime[F[_], O](f: [A] => ((Either[Throwable, A] => Unit) => Unit) => F[A]) extends Runtime[F, O] {

      val nn = f[Int](cb => cb(Right(1)))

      override def devices: Vector[Device[F, O]] = ???
    }


    abstract class F2[O]
        extends SuspendedRuntime[Future, O](
          [A] => { (cb: (Either[Throwable, A] => Unit) => Unit) =>
            val p = Promise[A]
            cb((e: Either[Throwable, A]) => p.tryComplete(e.toTry): Unit)
            p.future
          }
        ) {}

    trait Device[F[_], O] {

      def jit: Device[F, CompileOptions.JIT] = ???

      def name: String
      def properties(): Map[String, String]
      def task[A](using o: O)(f: => A): F[A]

      def foreach(x: Range)(using o: O)(f: Int => Unit): F[Unit]
      def foreach(x: Range, y: Range)(using o: O)(f: (Int, Int) => Unit): F[Unit]
      def foreach(x: Range, y: Range, z: Range)(using o: O)(f: (Int, Int, Int) => Unit): F[Unit]

      def reduce[A](x: Range)(using o: O)(g: (A, A) => A)(f: Int => A): F[A]
      def reduce[A](x: Range, y: Range)(using o: O)(g: (A, A) => A)(f: (Int, Int) => A): F[A]
      def reduce[A](x: Range, y: Range, z: Range)(using o: O)(g: (A, A) => A)(f: (Int, Int, Int) => A): F[A]

      inline def foreach(x: Int)(using o: O)(f: Int => Unit): F[Unit] =
        foreach(0 until x)(f)
      inline def foreach(x: Int, y: Int)(using o: O)(f: (Int, Int) => Unit): F[Unit] =
        foreach(0 until x, 0 until y)(f)
      inline def foreach(x: Int, y: Int, z: Int)(using o: O)(f: (Int, Int, Int) => Unit): F[Unit] =
        foreach(0 until x, 0 until y, 0 until z)(f)

      inline def reduce[A](x: Int)(using o: O)(g: (A, A) => A)(f: Int => A): F[A] =
        reduce[A](0 until x)(g)(f)
      inline def reduce[A](x: Int, y: Int)(using o: O)(g: (A, A) => A)(f: (Int, Int) => A): F[A] =
        reduce[A](0 until x, 0 until y)(g)(f)
      inline def reduce[A](x: Int, y: Int, z: Int)(using o: O)(g: (A, A) => A)(f: (Int, Int, Int) => A): F[A] =
        reduce[A](0 until x, 0 until y, 0 until z)(g)(f)

    }



    trait CUDADevice[F[_]] extends Device[F, CompileOptions.AOT[CUDAArch]]

    class CUDARuntime[F[_]] extends Runtime[F, CompileOptions.AOT[CUDAArch]] {
      override inline def name: String                    = "CUDA"
      override def properties: Map[String, String] = Map.empty
      override def devices: Vector[CUDADevice[F]]  = Vector.empty
      def jit: Runtime[F, CompileOptions.JIT]      = ???
    }


    //

    val HostRuntime: Device[Future, CompileOptions.JIT] = ???
    val Native: Device[Future, CompileOptions]                         = ???

    // val HIP: Runtime    = ???
    val CUDA: CUDARuntime[Future] = ???
    // val OpenCL: Runtime = ???

    implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global

    CUDA.devices(0).task(using CompileOptions.AOT(CUDAArch.SM80, Optimisation.Ofast))(1 + 1)
    CUDA.jit.devices(0).task(using CompileOptions.JIT(Optimisation.Ofast))(1 + 1)

//    CUDA.devices(0).queue

    def runM[A](d:  Device[Future, CompileOptions.JIT])  =
      d.task(1 + 1)


    // Native.task(1+1)

    HostRuntime.task(1 + 1).map(x => x * 2)
    runM(HostRuntime)

    Promise[Int]().tryComplete(Try(1))

    given CompileOptions.JIT = CompileOptions.JIT(Optimisation.O3)

    Native.foreach(1 to 3)(x => ())
    Native.foreach(1 to 3)(using CompileOptions.JIT(Optimisation.O3))(x => ())

    Native.foreach(1 to 3, 10 to 20, 3 to 3)((x, y, z) => ())
    Native.foreach(1 to 3, 10 to 20, 3 to 3)(using CompileOptions.JIT(Optimisation.O3))((x, y, z) => ())

    trait Offload[A] extends scala.collection.Seq[A] {}

    extension [A](xs: Seq[A]) {
      def offload: Offload[A] = ???
    }

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

    val l = Local
//     val result = polyregion.scala.compiletime.offload {
// //      val objRef = Foo
// //      val c      = Local.LevelA.LevelB
// //      val c2     = Foo.Bar.Baz
// //      val m      = c.a2
// //      val n      = l
// //      val a40    = l.LevelA
// //      val a41    = l.LevelA.a1
// //      val m1     = l.levelB.a2
// //
// //      val i       = n.g
// //      val objRef2 = objRef
// //      val a       = Local
// //      val a1      = a.LevelA
// //      val a2      = a.LevelA.a1
// //      val m2      = Foo.bar + objRef.nBar(2) + objRef2.nBarInline(42)
// //      val a3      = Local.LevelA.a1
// //
// //      // aliases
// //      val alias1      = Simple
// //      val alias2      = alias1
// //      val alias3      = alias2
// //      val aliasX      = alias3
// //      val aliasY      = alias3
// //      val aliasResult = aliasX.a + aliasX.b + aliasY.a + aliasY.b
// //
// //      val y =
// //        objRef.bar + Foo.baz + x + b + m + Local.g + x + Local.LevelA.a1 + Local.LevelA.LevelB.a2 + aliasResult + m1
// //      val z = 42 + x + y + Foo.nBar(y) + Foo.nBarInline(y) + Bar.y + Simple.a + Simple.b + i + objRef2.x + a1.LevelB.a2
// //      val out = a3 + m2 + a2 + m + a41 + z
// //      Foo.bar + out + Foo.nBarInline(m)

//       var max = 0
//       var i   = 0
//       while (i < n) {
//         max = math.max(max, xs(i))
// //        max = if(i < y) max*2 else max
// //        xs(i) = i // i-1
//         i += 1
//       }
//       max
//     }
    println(xs)

    // println(s"result = $result")
    println("Done!")
  }

}
