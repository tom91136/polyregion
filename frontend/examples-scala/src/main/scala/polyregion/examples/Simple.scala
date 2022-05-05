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
    out1 += 1
    1 + Simple.out + out1
  }
}

class Simple(val a: Int)
object Simple {
  import A.{given, *}

  scala.collection.immutable.Seq

  val out = 1

  def say = 2 + 1 + out

  def main(args: Array[String]): Unit = {


//    summon[NativeStruct[FooProper.type]]
//    summon[NativeStruct[FooProper.type ]]

    // polyregion.scala.compiletime.showExpr{
    //   1+1
    // }
    println("Enter")
    val aa = Simple(2)

    val x = 42
    val y = 10

    val xs = Buffer[Int](12, 2, 3)
    println(s"in=${xs.toList}")


    val a = polyregion.scala.compiletime.offload {
//      val y = FooProper.bar
      val z = 42 + x + y

      xs(0) = 43
      xs(1) = 43
//      xs(2) = 43

      xs(0)
      //      val m = 1 + 1 + out + y
//      m
    }
    val b = x + y

    println(s"in=${xs.toList}")

    println(s"result = $a $b")
    println("Done!")
  }

}
