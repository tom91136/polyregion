package polyregion

import polyregion.ast.*
import polyregion.scalalang.{Compiler, Quoted}

import scala.quoted.{Expr, Quotes}

class CompilerSpec extends munit.FunSuite {

//  test("empty") {
//    CompilerTests.compilerAssert {}
//  }
//
//  test("unit") {
//    CompilerTests.compilerAssert(())
//  }
//
//  test("constant") {
//    CompilerTests.compilerAssert(1)
//  }
//
//  test("simple") {
//    CompilerTests.compilerAssert {
//      val a = 1
//      val b = 2
//      a + b
//    }
//  }
//
//  test("local function single") {
//    CompilerTests.compilerAssert {
//      val c                  = 12
//      def f1(a: Int, b: Int) = a + b + c
//      f1(1, 2)
//    }
//  }
//
//  test("local function single") {
//    CompilerTests.compilerAssert {
//      val c                  = 12
//      def f1(a: Int, b: Int) = a + b + c
//      f1(1, 2) + f1(1, 2) // FIXME inline fails for multiple invocations
//    }
//  }
//
//  test("local function multiple") {
//    CompilerTests.compilerAssert {
//      val c                  = 12
//      def f1(a: Int, b: Int) = a + b + c
//      def f2()               = f1(2, 1)
//      f1(1, 2) + f2() + f2() // FIXME inline fails for multiple invocations
//    }
//  }
//
//  test("nested local function") {
//    CompilerTests.compilerAssert {
//      val c = 12
//      def f2(a: Int, b: Int) = {
//        def f1(a: Int, b: Int) = a + b + c
//        a+b+f1(a,b)
//      }
//      f2(1, 2)
//    }
//  }
//
//  test("external function") {
//    def fn0 = 12
//    class Foo {
//      val m         = 2
//      def a()       = m + m
//      def b(x: Int) = m + x
//    }
//
//    CompilerTests.compilerAssert {
//      val f = Foo()
//      f.m + f.a() + f.b(f.m) + fn0
//    }
//  }

  test("fns") {
    CompilerTests.compilerAssert {

      val twice: Int => Int = (a: Int) => a + a

      twice(1)

    }
  }
//
}
