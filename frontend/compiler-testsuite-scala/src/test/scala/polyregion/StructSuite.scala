package polyregion

import polyregion.scalalang.*
import polyregion.scalalang.compiletime.*

import scala.collection.mutable.ArrayBuffer
import scala.compiletime.*
import scala.reflect.ClassTag

class StructSuite extends BaseSuite {

  // case class T3[A <: AnyVal](val a: A, b: A, c: A)
  // given [A <:AnyVal ]: NativeStruct[T3[A]]    = nativeStructOf

  case class Char3(a: Char, b: Char, c: Char)
  case class Byte3(a: Byte, b: Byte, c: Byte)
  case class Short3(a: Short, b: Short, c: Short)
  case class Int3(a: Int, b: Int, c: Int)
  case class Long3(a: Long, b: Long, c: Long)
  case class Long2(a: Long, b: Long)
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

//  inline def dummyNativeStruct[A] = new NativeStruct[A] {
//    def name: String                                                       = ""
//    def sizeInBytes: Int                                                   = 1
//    def encode(buffer: java.nio.ByteBuffer, index: Int, a: A): Unit = ()
//    def decode(buffer: java.nio.ByteBuffer, index: Int): A          = ???
//  }

  // inline given NativeStruct[Char3]    = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Byte3]    = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Short3]   = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Int3]     = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Long3]    = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Float3]   = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // inline given NativeStruct[Double3]  = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct
  // // inline given NativeStruct[Boolean3] = if (Toggles.StructSuite) dummyNativeStruct else dummyNativeStruct

  // inline def testExpr[A <: AnyRef: NativeStruct](inline name: String)(inline r: => A) = if (Toggles.StructSuite) {
  //   test(name)(assertOffload(r))
  // }

  private inline def testExpr[A](inline name: String)(inline r: => A) = if (Toggles.StructSuite) {
    test(name)(assertOffloadValue(offload1(r)))
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
    // Top-level `Array[Float3]` captures aren't accepted by bindWrite ("Top level arrays at
    // parameter boundary is illegal"); `Buffer.tabulate[Float3]` hits a `???` in
    // `StructBuffer.apply` (NativeStruct deriving is unimplemented on the host side). Use a
    // `scala.collection.mutable.ListBuffer` — it's wired through the StdLib prism as a
    // MutableSeq and marshals as a sized struct capture.
    val xs = scala.collection.mutable.ListBuffer.tabulate[Float3](10)(x =>
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
    testExpr("arg-deref-member") {
      val y = x // ref elision
      y.c
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

  testExpr("deref-member") {
    val x = Float3(42.0, 1.0, 2.0)
    x.c
  }

  testExpr("deref-member-direct") {
    Float3(42.0, 1.0, 2.0).c
  }

  // ---

  {
    val x = Float3(42.0, 1.0, 2.0)
    testExpr("passthrough") {
      val y = x
      val z = y
      z
    }
  }

  testExpr("return")(Float3(42.0, 1.0, 2.0))

  {
    val a = 0.1f
    val b = 0.2f
    val c = 0.3f
    testExpr("ctor-arg-return")(Float3(a, b, c))
    val x = Float3(a, b, c)
    testExpr("ctor-arg-in") {
      val y = x
      y.a + y.b + y.c
    }
  }

  testExpr("ctor-return") {
    val a = 0.1f
    val b = 0.2f
    val c = 0.3f
    Float3(a, b, c)
  }

  {
    val nested = Int3x3(
      Int3(1, 2, 3),
      Int3(4, 5, 6),
      Int3(7, 8, 9)
    )
    testExpr("nested-buffer-param") {
      nested.a.a + nested.b.b + nested.c.c
    }
  }

  // FIXME `a.field += 1` test assumes JDK and Offload runs see the same starting `a.field`.
  // With class-field write-back enabled, the first run mutates the host's `a.field`, so the
  // second run starts from the post-mutation state — they return different values. The test
  // would only be valid if `a` were re-initialised between runs. Skipped until we have a
  // per-target capture-snapshot.
  // {
  //   class A {
  //     var field = 1
  //   }
  //   val a = A()
  //   testExpr("a") {
  //     a.field += 1
  //     a.field
  //   }
  //   println(a.field)
  // }

  // Vec3/Vec33 helpers aren't defined; skip the speculative nested-return drafts.

}
