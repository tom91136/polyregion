package polyregion

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
}
