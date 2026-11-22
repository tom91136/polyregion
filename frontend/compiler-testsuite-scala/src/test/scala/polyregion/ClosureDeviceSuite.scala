package polyregion

import ClosureFixtures.*

class ClosureDeviceSuite extends RangeSuiteBase {

  same("hof-lambda-captures-callable-parameter") {
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    def quad(g: Int => Int, x: Int): Int  = twice(y => g(y) + 1, x)
    quad(_ * 2, 3)
  }
  same("hof-lambda-captures-callable-parameter-and-local") {
    val k                                 = 10
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    def quad(g: Int => Int, x: Int): Int  = twice(y => g(y) + k, x)
    quad(_ * 2, 3) + quad(_ - 1, 4)
  }
  same("this-lambda-field")(new Scaler(3).byField(5))
  same("this-lambda-method")(new Scaler(3).byMethod(5))
  same("this-lambda-field-and-method")(new Scaler(3).byBoth(5))
  same("this-lambda-nested")(new Scaler(3).nested(5))
  same("this-lambda-explicit-this")(new Scaler(3).viaThis(5))
}
