package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*

class FunctionCallSuite extends BaseSuite {

  inline def testCapture[A](inline name: String)(inline r: => A) = if (Toggles.FunctionCallSuite) {
    test(name)(assertOffload(r))
  }

  {
    object A {
      def double(f: Double) = f * 2
    }
    testCapture("module")(A.double(1f))
  }

  {
    def double(x: Double)         = x * 2
    inline def double2(x: Double) = x * 2
    val x                         = 42d
    testCapture("inlined-const")(double2(1f))
    testCapture("inlined")(double2(x))
    testCapture("inlined-nest")(double2(double2(double2(x))))
    testCapture("const")(double(1f))
    testCapture("const-mix")(double(1f) * double(x))
    testCapture("nest")(double(double(double(x))))
  }

}
