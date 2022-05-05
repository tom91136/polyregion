package polyregion.examples
import polyregion.scala.{Buffer, NativeStruct}
import scala.collection.mutable.ArrayBuffer

object A {
//  given NativeStruct[Simple.type]    = polyregion.scala.compiletime.nativeStructOf
//  given NativeStruct[Simple]    = polyregion.scala.compiletime.nativeStructOf
//  given NativeStruct[FooProper.type] = polyregion.scala.compiletime.nativeStructOf
}

object FooProper {
  var out1 = 1
  def bar = {
    // out1 += 1
    // 1 + Simple.out + out1
    1
  }
}

class Simple(val a: Int)
object Simple {
  import A.{given, *}

  scala.collection.immutable.Seq

  val out = 10

  def say = 2 + 1 + out

  def main(args: Array[String]): Unit = {

//    summon[NativeStruct[FooProper.type]]
//    summon[NativeStruct[FooProper.type ]]

    // polyregion.scala.compiletime.showExpr{
    //   val m = FooProper
    //   val n = m.out1
    // }
    println("Enter")
    val aa = Simple(2)

      val x = 1

    val a = polyregion.scala.compiletime.offload {
      val y = Simple.out
      val z = 42 + x + y
      z
    }


    println(s"result = $a")
    println("Done!")
  }

}
