package polyregion

import scala.quoted.{Expr, Quotes}
import polyregion.scalalang.{Compiler, Quoted}
import polyregion.ast.*

class CompilerSpec extends munit.FunSuite {

  test("check") {

//    println("A")
//    val m = 0
//    val u = CompilerTests.compilerAssert {
//      val u = m
//    }

    def fn0 = 12
    class Foo {
      val m = 2
      def a() = m + m
      def b(x : Int) = m + x
    }


    CompilerTests.compilerAssert({
      val a   = 1
      val b   = 2
      val uuu = a + 2L
      //      a + b
      //      val u = (1,2)

      val c   = fn0
      val foo = new Foo
//      val m = (1,2)
      val u   = foo.a()

//      val m = u+1

      val m = foo.b(1)
//      val m2 = foo.b(_)


    })

//    assertEquals(1, 1)

  }

}