package polyregion.examples
import polyregion.scala.{Buffer, NativeStruct}

object A {
  given NativeStruct[Simple.type]    = polyregion.scala.compiletime.nativeStructOf
  given NativeStruct[FooProper.type] = polyregion.scala.compiletime.nativeStructOf
}

object FooProper {
  var out1 = 1
  def bar = {
    out1 += 1
    1 + Simple.out + out1
  }
}

object Simple {
  import A.{given, *}

  val out = 1

  def say = 2 + 1 + out

  def main(args: Array[String]): Unit = {

    summon[NativeStruct[Simple.type]]
//    summon[NativeStruct[FooProper.type ]]

    println("Enter")
    val a = polyregion.scala.compiletime.offload {
//      val x = FooProper.bar
      val y = say
      y
//      val y = 1
//      val m = 1 + 1 + out + y
//      m
    }

    println(s"result = $a")
    println("Done!")
  }

}
