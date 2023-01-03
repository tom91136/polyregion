package polyregion

import polyregion.scala.compiletime.*

import _root_.scala.compiletime.*
import _root_.scala.reflect.ClassTag
import _root_.scala.collection.mutable.ListBuffer

class ValueReturnSuite extends BaseSuite {

  inline def testValueReturn[A](inline r: A) = if (Toggles.ValueReturnSuite) {
    test(s"${r.getClass}-const=$r")(assertOffloadValue(offload1(r)))
    val x: A = r
    test(s"${r.getClass}-ref1=$r")(assertOffloadValue(offload1(x)))
    val y: A = x
    test(s"${r.getClass}-ref2=$r")(assertOffloadValue(offload1(y)))
    val z: A = y
    test(s"${r.getClass}-ref3=$r")(assertOffloadValue(offload1(z)))
  }

//  assertOffloadEffect {
//    class MyCls {
//      var fieldA = 42
//      var fieldB = 0.1
//    }
//    val a = new MyCls
//    offload0 {
//      a.fieldA += 1
//      a.fieldB *= a.fieldA
//    }
//    assertEquals(a.fieldA, 43)
//    assertEquals(a.fieldB, 4.3)
//  }
//
//  assertOffloadEffect {
//    class MyClsA {
//      var fieldA = 10
//    }
//    val a = new MyClsA
//    class MyClsB {
//      var fieldA = a
//      var fieldB = 42
//    }
//    val b = new MyClsB
//    val c = new MyClsA
//    offload0 {
//      a.fieldA *= 10
//      b.fieldB += b.fieldA.fieldA
//      c.fieldA = 20
//    }
//    assertEquals(a.fieldA, 100)
//    assertEquals(b.fieldA.fieldA, 100)
//    assertEquals(b.fieldB, 142)
//    assertEquals(c.fieldA, 20)
//  }

  assertOffloadEffect {

    class MyCls {
      var fieldA = 10
    }

    var a = new MyCls

    class MyBox {
      var value = a
    }

    val box = new MyBox

    val m = ListBuffer(1)
    // val m = new polyregion.ValueReturnSuite#MyCls
    offload0 {
      val b = new MyCls
      b.fieldA = 42 + m(0)
      box.value = b
    }
    assertEquals(box.value.fieldA, 43)
  }

  // assert values survives round trip

//  testValueReturn[Char](0)
//  testValueReturn[Char](255)
//  testValueReturn[Char](1)
//  testValueReturn[Char](0)
//  testValueReturn[Char](42)

//  testValueReturn[Byte](-128)
//  testValueReturn[Byte](127)
//  testValueReturn[Byte](1)
//  testValueReturn[Byte](-1)
//  testValueReturn[Byte](0)
//  testValueReturn[Byte](42)
//
//  testValueReturn[Short](-32768)
//  testValueReturn[Short](32767)
//  testValueReturn[Short](1)
//  testValueReturn[Short](-1)
//  testValueReturn[Short](0)
//  testValueReturn[Short](42)
//
//  testValueReturn[Int](0x80000000)
//  testValueReturn[Int](0x7fffffff)
//  testValueReturn[Int](1)
//  testValueReturn[Int](-1)
//  testValueReturn[Int](0)
//  testValueReturn[Int](42)
//
//  testValueReturn[Long](0x8000000000000000L)
//  testValueReturn[Long](0x7fffffffffffffffL)
//  testValueReturn[Long](1)
//  testValueReturn[Long](-1)
//  testValueReturn[Long](0)
//  testValueReturn[Long](42)
//
//  testValueReturn[Float](1.4e-45f)
//  testValueReturn[Float](1.17549435e-38f)
//  testValueReturn[Float](3.4028235e+38) // XXX 3.4028235e+38f appears to not fit!
//  testValueReturn[Float](0.0f / 0.0f)
//  testValueReturn[Float](1)
//  testValueReturn[Float](-1)
//  testValueReturn[Float](0)
//  testValueReturn[Float](42)
//  testValueReturn[Float](3.14159265358979323846)
//
//  testValueReturn[Double](4.9e-324d)
//  testValueReturn[Double](2.2250738585072014e-308d)
//  testValueReturn[Double](1.7976931348623157e+308d)
//  testValueReturn[Double](0.0f / 0.0d)
//  testValueReturn[Double](1)
//  testValueReturn[Double](-1)
//  testValueReturn[Double](0)
//  testValueReturn[Double](42)
//  testValueReturn[Double](3.14159265358979323846d)
//
//  testValueReturn(true)
//  testValueReturn(false)
//
//  testValueReturn[Unit](())

}
