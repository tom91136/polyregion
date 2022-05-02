package polyregion.examples
import polyregion.scala.{Buffer, NativeStruct}

object A {
  given NativeStruct[Simple.type]    = polyregion.scala.compiletime.nativeStructOf
  given NativeStruct[Simple]    = polyregion.scala.compiletime.nativeStructOf
  given NativeStruct[FooProper.type] = polyregion.scala.compiletime.nativeStructOf
}

object FooProper {
  var out1 = 1
  def bar = {
    out1 += 1
    1 + Simple.out + out1
  }
}

class Simple(val a: Int)
object Simple {
  import A.{given, *}

  val out = 1

  def say = 2 + 1 + out

  def main(args: Array[String]): Unit = {

    summon[NativeStruct[FooProper.type]]
//    summon[NativeStruct[FooProper.type ]]


    polyregion.scala.compiletime.showExpr{
      summon[NativeStruct[polyregion.examples.FooProper.type]]
    }
    println("Enter")
    val aa = Simple(2)

    val a = polyregion.scala.compiletime.offload {
      import A.{given, *}

//      val x = FooProper.bar
      // val y = say
      val y = FooProper.bar
      y
//      val y = 1
//      val m = 1 + 1 + out + y
//      m
    }

    println(s"result = $a")
    println("Done!")
  }

}
