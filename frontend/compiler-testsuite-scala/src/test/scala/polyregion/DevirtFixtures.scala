package polyregion

object DevirtFixtures {

  trait Op {
    def apply(x: Int): Int
  }

  class Outsider(val k: Int) extends Op {
    def apply(x: Int): Int = x * k + 1000
  }
}
