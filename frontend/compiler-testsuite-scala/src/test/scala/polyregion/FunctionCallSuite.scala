package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class FunctionCallSuite extends BaseSuite {

  inline def testCapture[A <:AnyVal : ClassTag](inline name: String)(inline r: => A) = if (Toggles.FunctionCallSuite) {
    test(name)(assertOffload(r))
  }

//  {
//    object A {
//      def double(x: Double)                                  = x * 2.0
//      inline def doubleInline(x: Double)                     = x * 2.0
//      inline def doubleInlineAll(inline x: Double)           = x * 2.0
//      inline def timesInlineMix(inline x: Double, y: Double) = x * y
//      def first(a: Int, b: Int)                              = a
//      val x                                                  = 42d
//    }
//
//    testCapture("module-inline-all-const")(A.doubleInlineAll(1f))
//    testCapture("module-inline-all")(A.doubleInlineAll(A.x))
//    testCapture("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(A.x))))
//    testCapture("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(1f))))
//
//    testCapture("module-inline-const")(A.doubleInline(1f))
//    testCapture("module-inline")(A.doubleInline(A.x))
//    testCapture("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(A.x))))
//    testCapture("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(1f))))
//
//    testCapture("module-inlinemix")(A.timesInlineMix(A.x, 2f))
//    testCapture("module-inlinemix-nest")(A.timesInlineMix(A.timesInlineMix(A.timesInlineMix(A.x, 2f), 3f), 4f))
//
//    testCapture("module-const")(A.double(1f))
//    testCapture("module-const-mix")(A.double(1f) * A.double(A.x))
//    testCapture("module-nest")(A.double(A.double(A.double(A.x))))
//
//    testCapture("module-named-first")(A.first(a = 1, b = 2))
//    testCapture("module-named-first-partial")(A.first(a = 1, 2))
//    testCapture("module-named-second")(A.first(b = 1, a = 2))
//    testCapture("module-named-second-partial")(A.first(1, b = 2))
//  }

  {
    //  local-scope defs, part of this test class
    def double(x: Double)                                  = x * 2.0
    inline def doubleInline(x: Double)                     = x * 2.0
    inline def doubleInlineAll(inline x: Double)           = x * 2.0
    inline def timesInlineMix(inline x: Double, y: Double) = x * y
    def first(a: Int, b: Int)                              = a
    val x                                                  = 42d

//    testCapture("module-inline-all-const")(doubleInlineAll(1f))
//    testCapture("module-inline-all")(doubleInlineAll(x))
//    testCapture("module-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
//    testCapture("module-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))
//
//    testCapture("module-inline-const")(doubleInline(1f))
//    testCapture("module-inline")(doubleInline(x))
//    testCapture("module-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
//    testCapture("module-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))
//
//    testCapture("module-inlinemix")(timesInlineMix(x, 2f))
//    testCapture("module-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))
//
//    testCapture("module-const")(double(1f))
//    testCapture("module-const-mix")( double(1f) *  double( x))
//    testCapture("module-nest")( double( double( double( x))))
//
//    testCapture("module-named-first")( first(a = 1, b = 2))
//    testCapture("module-named-first-partial")( first(a = 1, 2))
//    testCapture("module-named-second")( first(b = 1, a = 2))
//    testCapture("module-named-second-partial")( first(1, b = 2))
  }

  //  class-scope defs, part of this test class
  def double(x: Double)                                  = x * 2.0
  inline def doubleInline(x: Double)                     = x * 2.0
  inline def doubleInlineAll(inline x: Double)           = x * 2.0
  inline def timesInlineMix(inline x: Double, y: Double) = x * y
  def first(a: Int, b: Int)                              = a
  val x                                                  = 42d

//      testCapture("module-inline-all-const")(doubleInlineAll(1f))
//      testCapture("module-inline-all")(doubleInlineAll(x))
//      testCapture("module-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
//      testCapture("module-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))
//
//      testCapture("module-inline-const")(doubleInline(1f))
//      testCapture("module-inline")(doubleInline(x))
//      testCapture("module-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
//      testCapture("module-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))
//
//      testCapture("module-inlinemix")(timesInlineMix(x, 2f))
//      testCapture("module-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))

  testCapture("module-const")(double(x))
//      testCapture("module-const-mix")( double(1f) *  double( x))
//      testCapture("module-nest")( double( double( double( x))))
//
//      testCapture("module-named-first")( first(a = 1, b = 2))
//      testCapture("module-named-first-partial")( first(a = 1, 2))
//      testCapture("module-named-second")( first(b = 1, a = 2))
//      testCapture("module-named-second-partial")( first(1, b = 2))

}
