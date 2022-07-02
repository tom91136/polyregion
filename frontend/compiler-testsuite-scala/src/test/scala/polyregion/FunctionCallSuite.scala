package polyregion

import polyregion.scala.compiletime.*

// import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class FunctionCallSuite extends BaseSuite {

  inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.FunctionCallSuite) {
    test(name)(assertOffload(r))
  }

  {
    object A {
      val x                     = 42d
      def double(x: Double)     = x * 2.0
      def double2(y: Double)    = x * x * 2.0
      def first(a: Int, b: Int) = a

      inline def doubleInline(x: Double)                     = x * 2.0
      inline def doubleInlineAll(inline x: Double)           = x * 2.0
      inline def timesInlineMix(inline x: Double, y: Double) = x * y
    }

    testExpr("module-inline-all-const")(A.doubleInlineAll(1f))
    testExpr("module-inline-all")(A.doubleInlineAll(A.x))
    testExpr("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(A.x))))
    testExpr("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(1f))))

    testExpr("module-inline-const")(A.doubleInline(1f))
    testExpr("module-inline")(A.doubleInline(A.x))
    testExpr("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(A.x))))
    testExpr("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(1f))))

    testExpr("module-inlinemix")(A.timesInlineMix(A.x, 2f))
    testExpr("module-inlinemix-nest")(A.timesInlineMix(A.timesInlineMix(A.timesInlineMix(A.x, 2f), 3f), 4f))

    testExpr("module-const-1")(A.double(1.0))
    testExpr("module-const-2")(A.double2(1.0))
    testExpr("module")(A.double(A.x))

    testExpr("module-const-mix")(A.double(1f) * A.double(A.x))
    testExpr("module-nest")(A.double(A.double(A.double(A.x))))

    testExpr("module-named-first")(A.first(a = 1, b = 2))
    testExpr("module-named-first-partial")(A.first(a = 1, 2))
    testExpr("module-named-second")(A.first(b = 1, a = 2))
    testExpr("module-named-second-partial")(A.first(1, b = 2))
  }

  {
    // Local-scope defs, part of this test class.
    // This mainly tests references w.r.t class initialisers (called local dummy).
    val x                     = 42d
    def double(x: Double)     = x * 2.0
    def double2(y: Double)    = x * y * 2.0
    def first(a: Int, b: Int) = a

    inline def doubleInline(x: Double)                     = x * 2.0
    inline def doubleInlineAll(inline x: Double)           = x * 2.0
    inline def timesInlineMix(inline x: Double, y: Double) = x * y

    testExpr("class-init-inline-all-const")(doubleInlineAll(1f))
    testExpr("class-init-inline-all")(doubleInlineAll(x))
    testExpr("class-init-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
    testExpr("class-init-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))

    testExpr("class-init-inline-const")(doubleInline(1f))
    testExpr("class-init-inline")(doubleInline(x))
    testExpr("class-init-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
    testExpr("class-init-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))

    testExpr("class-init-inlinemix")(timesInlineMix(x, 2f))
    testExpr("class-init-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))

    testExpr("class-init-const-1")(double(1.0))
    testExpr("class-init-const-2")(double2(1.0))
    testExpr("class-init")(double(x))

    testExpr("class-init-const-mix")(double(1f) * double(x))
    testExpr("class-init-nest")(double(double(double(x))))

    testExpr("class-init-named-first")(first(a = 1, b = 2))
    testExpr("class-init-named-first-partial")(first(a = 1, 2))
    testExpr("class-init-named-second")(first(b = 1, a = 2))
    testExpr("class-init-named-second-partial")(first(1, b = 2))
  }

  // class-scope defs, part of this test class
  val x                     = 42d
  def double(x: Double)     = x * 2.0
  def double2(y: Double)    = x * y * 2.0
  def first(a: Int, b: Int) = a

  inline def doubleInline(x: Double)                     = x * 2.0
  inline def doubleInlineAll(inline x: Double)           = x * 2.0
  inline def timesInlineMix(inline x: Double, y: Double) = x * y

  testExpr("class-inline-all-const")(doubleInlineAll(1f))
  testExpr("class-inline-all")(doubleInlineAll(x))
  testExpr("class-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
  testExpr("class-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))

  testExpr("class-inline-const")(doubleInline(1f))
  testExpr("class-inline")(doubleInline(x))
  testExpr("class-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
  testExpr("class-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))

  testExpr("class-inlinemix")(timesInlineMix(x, 2f))
  testExpr("class-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))

  testExpr("class-const-1")(double(1.0))
  testExpr("class-const-2")(double2(1.0))
  testExpr("class")(double(x))

  testExpr("class-const-mix")(double(1f) * double(x))
  testExpr("class-nest")(double(double(double(x))))

  testExpr("class-named-first")(first(a = 1, b = 2))
  testExpr("class-named-first-partial")(first(a = 1, 2))
  testExpr("class-named-second")(first(b = 1, a = 2))
  testExpr("class-named-second-partial")(first(1, b = 2))

  // overloading
  def fn(a: Int)    = a + a
  def fn(a: Double) = a * a
  {
    val x = 2
    val y = 5.0
    testExpr("overload")(fn(x).toDouble + fn(y))
    testExpr("overload-const")(fn(x).toDouble + fn(y))
  }

  // multiple args
  def a(a: Int)(b: Int) = a + b
  {
    val x = 1
    val y = 2
    testExpr("multiple-args")(a(x)(y))
    testExpr("multiple-args-const")(a(1)(2))
  }

}
