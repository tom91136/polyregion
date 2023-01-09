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

    CompilerTests.compilerAssert()

//    assertEquals(1, 1)

  }

}
