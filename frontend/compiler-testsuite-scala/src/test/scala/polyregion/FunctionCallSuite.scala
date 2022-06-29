package polyregion

import polyregion.scala.compiletime.*

// import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class FunctionCallSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.FunctionCallSuite) {
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

    testCapture("module-inline-all-const")(A.doubleInlineAll(1f))
    testCapture("module-inline-all")(A.doubleInlineAll(A.x))
    testCapture("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(A.x))))
    testCapture("module-inline-all-nest")(A.doubleInlineAll(A.doubleInlineAll(A.doubleInlineAll(1f))))

    testCapture("module-inline-const")(A.doubleInline(1f))
    testCapture("module-inline")(A.doubleInline(A.x))
    testCapture("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(A.x))))
    testCapture("module-inline-nest")(A.doubleInline(A.doubleInline(A.doubleInline(1f))))

    testCapture("module-inlinemix")(A.timesInlineMix(A.x, 2f))
    testCapture("module-inlinemix-nest")(A.timesInlineMix(A.timesInlineMix(A.timesInlineMix(A.x, 2f), 3f), 4f))

    testCapture("module-const-1")(A.double(1.0))
    testCapture("module-const-2")(A.double2(1.0))
    testCapture("module")(A.double(A.x))

    testCapture("module-const-mix")(A.double(1f) * A.double(A.x))
    testCapture("module-nest")(A.double(A.double(A.double(A.x))))

    testCapture("module-named-first")(A.first(a = 1, b = 2))
    testCapture("module-named-first-partial")(A.first(a = 1, 2))
    testCapture("module-named-second")(A.first(b = 1, a = 2))
    testCapture("module-named-second-partial")(A.first(1, b = 2))
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

    testCapture("class-init-inline-all-const")(doubleInlineAll(1f))
    testCapture("class-init-inline-all")(doubleInlineAll(x))
    testCapture("class-init-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
    testCapture("class-init-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))

    testCapture("class-init-inline-const")(doubleInline(1f))
    testCapture("class-init-inline")(doubleInline(x))
    testCapture("class-init-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
    testCapture("class-init-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))

    testCapture("class-init-inlinemix")(timesInlineMix(x, 2f))
    testCapture("class-init-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))

    testCapture("class-init-const-1")(double(1.0))
    testCapture("class-init-const-2")(double2(1.0))
    testCapture("class-init")(double(x))

    testCapture("class-init-const-mix")(double(1f) * double(x))
    testCapture("class-init-nest")(double(double(double(x))))

    testCapture("class-init-named-first")(first(a = 1, b = 2))
    testCapture("class-init-named-first-partial")(first(a = 1, 2))
    testCapture("class-init-named-second")(first(b = 1, a = 2))
    testCapture("class-init-named-second-partial")(first(1, b = 2))
  }

  // class-scope defs, part of this test class
  val x                     = 42d
  def double(x: Double)     = x * 2.0
  def double2(y: Double)    = x * y * 2.0
  def first(a: Int, b: Int) = a

  inline def doubleInline(x: Double)                     = x * 2.0
  inline def doubleInlineAll(inline x: Double)           = x * 2.0
  inline def timesInlineMix(inline x: Double, y: Double) = x * y

  testCapture("class-inline-all-const")(doubleInlineAll(1f))
  testCapture("class-inline-all")(doubleInlineAll(x))
  testCapture("class-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(x))))
  testCapture("class-inline-all-nest")(doubleInlineAll(doubleInlineAll(doubleInlineAll(1f))))

  testCapture("class-inline-const")(doubleInline(1f))
  testCapture("class-inline")(doubleInline(x))
  testCapture("class-inline-nest")(doubleInline(doubleInline(doubleInline(x))))
  testCapture("class-inline-nest")(doubleInline(doubleInline(doubleInline(1f))))

  testCapture("class-inlinemix")(timesInlineMix(x, 2f))
  testCapture("class-inlinemix-nest")(timesInlineMix(timesInlineMix(timesInlineMix(x, 2f), 3f), 4f))

  testCapture("class-const-1")(double(1.0))
  testCapture("class-const-2")(double2(1.0))
  testCapture("class")(double(x))

  testCapture("class-const-mix")(double(1f) * double(x))
  testCapture("class-nest")(double(double(double(x))))

  testCapture("class-named-first")(first(a = 1, b = 2))
  testCapture("class-named-first-partial")(first(a = 1, 2))
  testCapture("class-named-second")(first(b = 1, a = 2))
  testCapture("class-named-second-partial")(first(1, b = 2))

}
