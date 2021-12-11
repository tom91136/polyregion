package polyregion

object Foo {

  class U {
    def x = b

  }
  def a = 3
  def b = {
    var x = 2
    x *= 12345
    x
  }
  def z = {
    val xx = "FOOBARBAZ"
  }
  val c = "a"
}

case class Bar(to: Int) {
  def go = {
    var a = 0
    for (i <- (0 to to))
      a += i
  }
}

object Bar {

  def work(n: Int) = {
    val xs = 0 to n
    val ys = 0 to n

  }

}

case class Vec2[T](x: T, y: T) {
  inline def +(that: Vec2[T]): Vec2[T]     = Vec2(x, y)
  def noInlinePlus(that: Vec2[T]): Vec2[T] = Vec2(x, y)
}

val CONST = 42

object Stage {

  import polyregion.compileTime._

  // showTpe[Bar]
  
  class In{
    val n = {(n : Int) =>
      n+1
      println(s"A = $n")
    }
  }
   

  class Out {
    println("B")
    println("B")

    val u = In()
    val x = Option(u)
    x.foreach{y => 
      // showExpr(y.n)
    }
  }
 


  val xs = Array[Float](42.3)
  showExpr({(n: Int) => 
    n+1
    val u = xs(422)
    val bad : Array[Float] = xs.map(_*2f)

    xs(n+2) = CONST.toFloat + 42f + bad(0)
  })
 



}
