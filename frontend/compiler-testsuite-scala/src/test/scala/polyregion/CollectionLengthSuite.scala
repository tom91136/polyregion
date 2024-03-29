package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.collection.mutable.ArrayBuffer
import scala.compiletime.*

class CollectionLengthSuite extends BaseSuite {

  private inline def testExpr[A](inline r: A) = if (Toggles.LengthSuite) {
    test(s"${codeOf(r)}=${r}")(assertOffloadValue(offload1(r)))
  }

  case class U(a: Int)
  case class V(a: Int, u: U)

  {
    val xs = Buffer[Float](41, 42, 43)
    // val n = 42.toShort

    val v = V(1, U(2))
    // testExpr{
    //   val x = v
    //   x.a+x.a + x.u.a
    //  }
//    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
  }

  {
    val xs = Array[Int](1, 2, 3)
//    testExpr {
//      xs.size
//    }
    //    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
  }
//
//  {
//    val xs = ArrayBuffer[Int](1, 2, 3)
//    testExpr(xs.size)
//    testExpr(xs.length)
//    testExpr(xs.length + xs.size + 10)
//    testExpr {
//      var i = 0;
//      while (i < xs.length) { xs(0) = i; i += 1 }
//    }
//  }

  trait Base(val a: Int) {
    def foo(n: Int): Int // = a + n
  }

  class ClassA(a: Int) extends Base(a) {
    override def foo(n: Int): Int = 42 + n // + super.foo(n)
  }

  class ClassB(val b: Int) extends Base(42) {
    override def foo(n: Int): Int = b
    def bar                       = foo(2)
    def baz                       = b
  }

//  trait XFunction1  {
//    def apply(v1: Int): Int
//    // def compose[A](g: A => T1): A => R = { x => apply(g(x)) }
//    // def andThen[A](g: R => A): T1 => A = { x => g(apply(x)) }
//  }

  class F0 extends Function0[Int] {
    override def apply() = 42
  }

//  class F1 extends XFunction1  {
//    override def apply(a: Int) = a + 42
//  }
//
//  class F2 extends XFunction1  {
//    override def apply(a: Int) = a + 1
//  }

  trait XFunction1[-T1, +R] {
    def apply(v1: T1): R
//     def compose[A](g: XFunction1[A ,  T1]): A => R = { x => apply(g(x)) }
//     def andThen [A](g: XFunction1[R, A]): XFunction1[T1 ,  A] = { x => g(apply(x)) }
  }
  class F1 extends XFunction1[Int, Int] {
    override def apply(a: Int) = a + 42
  }

  class F2 extends XFunction1[Int, Int] {
    override def apply(a: Int) = a + 1
  }

  trait Foo{
    def foo : Int
  }

//  val f1 = F1()
// f1.andThen(f1).apply(42)
// take(f1)

  def take(f: Int => Int) =
    f.apply(42)
// struct Anon0 <: F<Int, Int> {  }
// (this: Anon0) apply(a : Int) {  return a + 42; }
  {
    val i = 0
    testExpr {

//      val ext = 2
//      val f0  = (i: Int) => i + 42 + ext
//      val f1  = (i: Int) => i
//      f0.apply(1)
//      f1.apply(1)

//      def foo(i:Int) = i
//      foo(ext)


//       val f0 = F0()
//       f0()

//      val f1: XFunction1[Int, Int] = if (i == 0) F1() else F2()
//

//      F1().andThen(F2())
//      f1.andThen(f1)
//      // take(f1)
//      // take(f1)
//       f1.andThen(f1).apply(42)
//
//      val m = ClassB(123)

//      val ext = 128
//      type A = Int

      i+1
//      class Foo2{
//        def innerFn : Int = 42
//      }
//
//      val f2 = new Foo2{
//        override def innerFn = 43
//      }
//      f2.innerFn
//      1
//      val f = new Foo{
//        override def foo = 42
//      }
//      f.foo


//        val m = new XFunction1[Int, Int]{
//          override def apply(v1: Int): Int = v1+v1
//        }
//        m.apply(1)
//
//      val o = if (i != 0) ClassA(2) else ClassB(9)
//
//      // val m = 42
//
//      // val f0            = (argA: Int) => argA + 42 + m // scala.Function1[scala.Int, scala.Int]
//
//      // def f1(argB: Int) = argB + 42
//
//      // val f2 = f1(_)
//
//      // f0(1) + f1(1)
//
////      o.foo(42)
//      // o.a
//
//      // m.baz
//      // o.foo(2)
//      // m.bar
//       o.foo(2) + m.foo(12345) + m.foo(42) + m.bar + f1.apply(2)
      // 1

    }
  }

}
