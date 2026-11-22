package polyregion

object ClosureFixtures {

  class Scaler(val k: Int) {
    def scale(x: Int): Int                = x * k
    def twice(f: Int => Int, x: Int): Int = f(f(x))
    def byField(x: Int): Int              = twice(y => y * k, x)
    def byMethod(x: Int): Int             = twice(y => scale(y) + 1, x)
    def byBoth(x: Int): Int               = twice(y => scale(y) + k, x)
    def nested(x: Int): Int               = twice(y => twice(z => z + k, y), x)
    def viaThis(x: Int): Int              = twice(y => this.scale(y), x)
    def plain(x: Int): Int                = (0 until 3).map(i => i * k).foldLeft(x)(_ + _)
  }
}
