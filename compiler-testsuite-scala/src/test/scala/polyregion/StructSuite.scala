package polyregion

import polyregion.scala.*
import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag

class StructSuite extends BaseSuite {

  // case class T3[A <: AnyVal](val a: A, b: A, c: A)
  // given [A <:AnyVal ]: NativeStruct[T3[A]]    = nativeStructOf

  case class Char3(a: Char, b: Char, c: Char)
  case class Byte3(a: Byte, b: Byte, c: Byte)
  case class Short3(a: Short, b: Short, c: Short)
  case class Int3(a: Int, b: Int, c: Int)
  case class Long3(a: Long, b: Long, c: Long)
  case class Float3(a: Float, b: Float, c: Float)
  case class Double3(a: Double, b: Double, c: Double)
  case class Boolean3(a: Boolean, b: Boolean, c: Boolean)
  case class Mixture(a: Char, b: Byte, c: Short, d: Int, e: Long, f: Float, g: Double, h: Boolean)

  case class Char3x3(a: Char3, b: Char3, c: Char3)
  case class Byte3x3(a: Byte3, b: Byte3, c: Byte3)
  case class Short3x3(a: Short3, b: Short3, c: Short3)
  case class Int3x3(a: Int3, b: Int3, c: Int3)
  case class Long3x3(a: Long3, b: Long3, c: Long3)
  case class Float3x3(a: Float3, b: Float3, c: Float3)
  case class Double3x3(a: Double3, b: Double3, c: Double3)
  case class Boolean3x3(a: Boolean3, b: Boolean3, c: Boolean3)
  case class MixtureMixture(
      m: Mixture,
      a: Char3,
      b: Byte3,
      c: Short3,
      d: Int3,
      e: Long3,
      f: Float3,
      g: Double3,
      h: Boolean3
  )

  inline def dummyNativeStruct[A] = new NativeStruct[A] {
    def name: String                                                       = ""
    def sizeInBytes: Int                                                   = 1
    def encode(buffer: _root_.java.nio.ByteBuffer, index: Int, a: A): Unit = ()
    def decode(buffer: _root_.java.nio.ByteBuffer, index: Int): A          = ???
  }

  inline given NativeStruct[Char3]    = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Byte3]    = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Short3]   = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Int3]     = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Long3]    = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Float3]   = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Double3]  = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct
  inline given NativeStruct[Boolean3] = if (Toggles.StructSuite) nativeStructOf else dummyNativeStruct

  inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.StructSuite) {
    test(name)(assertOffload(r))
  }

  // val xs = Buffer.range(0, 10)

  // assertOffload {
  //   val outside = 4
  //   xs.foreach2(i => i + 1 + outside)
  //   // val f =  { (i : Int) => i+1}
  //   // val x = 2
  //   // f(x)
  //   ()

  // }

  //       [_:Range]
  //         | [filter]                                                          ''.map((a, b) => a+b)
  // (0 to 10).withFilter(i => i > 5).map(x => (x, x+1)).flatMap(x => Array(x,x,x)).foreach((a, b) => a+b)
  // var _i = 0
  // while(_i < 10){
  //
  //   if( !( _i > 5 ) ) continue;  # (i => i > 5)(_i)         =:= if(!f(_i))
  //
  //   val _v' = (_i, _i+1)         # (x => (x, x+1))(_i)      =:= val _v' = f(_i)
  //
  //   val _jx = Array(_v',_v',_v') # (x => Array(x,x,x))(_v') =:= val _jx = f(_v')
  //   var _j = 0
  //   while(_j < 10) {
  //     val v2 = jx(_j)
  //     v2.*1 + v2.*2              # ((a,b) => a+b)(_v')      =:= f(_v')
  //   }
  //
  //

  //   _i+=1
  // }

  // assertOffload{
  //   var x = 1
  //   val arr = List[Int](1,2,3)

  //   // withFilter(f) === if(f(x)) continue;
  //   // flatMap(f)    === val x = _; while(true){ f(x) }
  //   // foreach(f)    === f(x)
  //   // map(f)        === val x' = f(x)

  //   arr.withFilter(c => c > 5).flatMap{c => arr.map{d => c+d}}
  //   for{
  //     a <- arr
  //     if a > 5
  //       n = a
  //       o = n
  //     b <- arr
  //   } yield  a+b
  //   1

  // }

  {
    val xs = Buffer.tabulate(10)(x =>
      Float3(
        x * math.Pi.toFloat * 1, //
        x * math.Pi.toFloat * 2, //
        x * math.Pi.toFloat * 3  //
      )
    )
    testExpr("buffer-param")(xs(1).a + xs(3).b + xs(5).c)
  }

  {
    val x = Float3(42.0, 1.0, 2.0)
    testExpr("passthrough") {
      val y = x
      val z = y
      z
    }
  }

  {
    val x = Float3(42.0, 1.0, 2.0)
    testExpr("arg-deref-member-mix") {
      val y = x // ref elision
      y.a + y.b + y.c
    }
  }

  testExpr("deref-member-mix") {
    val x = Float3(42.0, 1.0, 2.0)
    x.a + x.b + x.c
  }

  {
    val x = Float3(42.0, 1.0, 2.0)
    testExpr("arg-deref-member") {
      val y = x // ref elision
      y.c
    }
  }

  testExpr("deref-member") {
    val x = Float3(42.0, 1.0, 2.0)
    x.c
  }

  testExpr("deref-member-direct") {
    Float3(42.0, 1.0, 2.0).c
  }

  testExpr("return")(Float3(42.0, 1.0, 2.0))

  {
    val a = 0.1f
    val b = 0.2f
    val c = 0.3f
    testExpr("ctor-arg-return")(Float3(a, b, c))
  }

  testExpr("ctor-return") {
    val a = 0.1f
    val b = 0.2f
    val c = 0.3f
    Float3(a, b, c)
  }

  // testExpr("nested-buffer-param") {
  //   val xs = Buffer.tabulate(10)(x =>
  //     Vec33(
  //       Vec3(
  //         x * math.Pi.toFloat * 1, //
  //         x * math.Pi.toFloat * 2, //
  //         x * math.Pi.toFloat * 3  //
  //       ),
  //       Vec3(
  //         x * math.Pi.toFloat * 4, //
  //         x * math.Pi.toFloat * 5, //
  //         x * math.Pi.toFloat * 6  //
  //       ),
  //       Vec3(
  //         x * math.Pi.toFloat * 7, //
  //         x * math.Pi.toFloat * 8, //
  //         x * math.Pi.toFloat * 9  //
  //       )
  //     )
  //   )
  //   assertOffload {
  //     xs(0).a.a + xs(0).b.b + xs(0).c.c
  //   }
  // }

  // testExpr("nested-return") {
  //   assertOffload(Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0)))
  // }

  // testExpr("nested-use-and-return") {
  //   assertOffload {
  //     val x = Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0))
  //     x.a.a + x.b.b + x.c.c
  //   }
  // }

  // testExpr("nested-return-ref") {
  //   val a = 0.1f
  //   val b = 0.2f
  //   val c = 0.3f
  //   assertOffload(Vec33(Vec3(a, a, a), Vec3(b, b, b), Vec3(c, c, c)))
  // }

  // testExpr("nested-param") {
  //   val v = Vec33(Vec3(0.0, 1.0, 2.0), Vec3(3.0, 4.0, 5.0), Vec3(6.0, 7.0, 8.0))
  //   assertOffload {
  //     val x = v // ref elision, otherwise we'll be passing x.a, x.b, x.c as params
  //     x.a.a + x.b.b + x.c.c
  //   }
  // }

}
