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

  trait X {
    def m: Int
  }
  trait Y {
    def m: Int = 1
  }

  class Base(val a: Int) {
//    override def m = 2
    def foo(n: Int): Int = a + n
  }

  class ClassA(a: Int) extends Base(a) {
    override def foo(n: Int): Int = 42 + n + super.foo(n)
  }

  class ClassB(val b: Int) extends Base(42) {
    override def foo(n: Int): Int = b
  }

  // struct Base   { cls: Int, a: Int          } :> ClassA;ClassB;ClassX
  // struct ClassA { cls: Int, a: Int          } <: Base
  // struct ClassB { cls: Int, a: Int, b : Int } <: Base
  // struct ClassX { cls: Int, a, Int}           <: ClassA
  //
  // foo^(cls: Int, obj : Base, n : Int){ // this == Base|ClassA|ClassB|ClassX, root = Base
  //        if(cls == #Base)   return obj.to[Base]   .foo(n);
  //   else if(cls == #ClassA) return obj.to[ClassA] .foo(n);
  //   else if(cls == #ClassB) return obj.to[ClassB] .foo(n);
  //   else assert
  // }
  //
  // (this: Base).foo(n : Int){ // this == Base
  //   return this.a + n;
  // }
  // (this: ClassA).foo(n : Int){ // this == ClassA
  //   return 42 + n + (this : Base).foo(n) ; // super = Base; if(  fn.recv.tpe <:< recv ) static dispatch else dynamic
  // }
  // (this: ClassB).foo(n : Int){ // this == ClassB
  //   return this.b;
  // }

  // var x : Base = ???
  //
  // x.foo(123)
  // ---
  // foo^(x.cls, x, 123)  // x = Base|ClassA|ClassB, dynamic dispatch ^

  // var x : ClassA = ???
  // x.foo()  // x =  ClassA

  //  val o : Base = if(true) ClassA(2) else ClassB(1)

  class Unrelated

  test("fns") {
    CompilerTests.compilerAssert {

      val un = Unrelated()
      val m = ClassB(9)
//      m.foo(3)
      val o  = if(true) ClassA(2) else ClassB(9)
      o.foo(2)

//      val m = o.foo(2)

//      val twice: Int => Int = (a: Int) => a + a
//
//      twice(1)

    }
  }
//
}
