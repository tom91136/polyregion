package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.collection.mutable.ArrayBuffer
import _root_.scala.collection.{BuildFrom, Factory, mutable}
import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class CastSuite extends BaseSuite {

  inline def testExpr(inline name: String)(inline r: Any) = if (Toggles.CastSuite) {
    test(name)(assertOffload(r))
  }

  inline def testCastDouble(inline x: Double): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastFloat(inline x: Float): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastLong(inline x: Long): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastInt(inline x: Int): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastShort(inline x: Short): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastChar(inline x: Char): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }
  inline def testCastByte(inline x: Byte): Unit = {
    testExpr(s"${x.getClass}-to-double-$x=${x.toDouble}") { val y = x; val z = y.toDouble; x.toDouble }
    testExpr(s"${x.getClass}-to-float-$x=${x.toFloat}") { val y = x; val z = y.toFloat; x.toFloat }
    testExpr(s"${x.getClass}-to-long-$x=${x.toLong}") { val y = x; val z = y.toLong; x.toLong }
    testExpr(s"${x.getClass}-to-int-$x=${x.toInt}") { val y = x; val z = y.toInt; x.toInt }
    testExpr(s"${x.getClass}-to-short-$x=${x.toShort}") { val y = x; val z = y.toShort; x.toShort }
    testExpr(s"${x.getClass}-to-char-$x=${x.toChar}") { val y = x; val z = y.toChar; x.toChar }
    testExpr(s"${x.getClass}-to-byte-$x=${x.toByte}") { val y = x; val z = y.toByte; x.toByte }
  }

  testExpr(s"literal-to-prim") {

    // Scala's L/F/D literals
    // TODO support scala.language.experimental.genericNumberLiterals when it takes off

    val double2Double = 42d.toDouble
    val double2Float  = 42d.toFloat
    val double2Long   = 42d.toLong
    val double2Int    = 42d.toInt
    val double2Short  = 42d.toShort
    val double2Char   = 42d.toChar
    val double2Byte   = 42d.toByte

    val float2Double = 42f.toDouble
    val float2Float  = 42f.toFloat
    val float2Long   = 42f.toLong
    val float2Int    = 42f.toInt
    val float2Short  = 42f.toShort
    val float2Char   = 42f.toChar
    val float2Byte   = 42f.toByte

    val int2Double = 42.toDouble
    val int2Float  = 42.toFloat
    val int2Long   = 42.toLong
    val int2Int    = 42.toInt
    val int2Short  = 42.toShort
    val int2Char   = 42.toChar
    val int2Byte   = 42.toByte

    val long2Double = 42L.toDouble
    val long2Float  = 42L.toFloat
    val long2Long   = 42L.toLong
    val long2Int    = 42L.toInt
    val long2Short  = 42L.toShort
    val long2Char   = 42L.toChar
    val long2Byte   = 42L.toByte
  }

  Ints.foreach(testCastInt(_))
  Bytes.foreach(testCastByte(_))
  Chars.foreach(testCastChar(_))
  Shorts.foreach(testCastShort(_))
  Ints.foreach(testCastInt(_))
  Longs.foreach(testCastLong(_))
  Floats.foreach(testCastFloat(_))
  Doubles.foreach(testCastDouble(_))

}
