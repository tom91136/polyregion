package polyregion

import polyregion.scala.Support

class SupportSuite extends munit.FunSuite {

  extension (x: Seq[Range]) {
    inline def list: List[List[Int]] = x.map(_.toList).toList
  }

  test("linearise") {

    def assertLinearise(r: Range) =
      assertEquals(r.toList, List.tabulate(r.size)(Support.linearise(r.start, r.step)(_)))

    val values = List(1, 2, 3, 10, -1, -2, -3, -10, 0)
    for {
      start <- values
      stop  <- values
      step  <- values.filter(_ != 0)
    } {
      assertLinearise(start to stop by step)
      assertLinearise(start until stop by step)
    }

  }

  test("split even excl") {
    assertEquals(
      Support.splitStatic(0 until 3)(3).list,
      List(List(0), List(1), List(2))
    )
    assertEquals(
      Support.splitStatic(0 until 1)(1).list,
      List(List(0))
    )
    assertEquals(
      Support.splitStatic(0 until 0)(1).list,
      List()
    )
    assertEquals(
      Support.splitStatic(0 until 2)(1).list,
      List(List(0, 1))
    )
    assertEquals(
      Support.splitStatic(0 until 0)(2).list,
      List()
    )
    assertEquals(
      Support.splitStatic(-5 until 5)(5).list,
      List(List(-5, -4), List(-3, -2), List(-1, 0), List(1, 2), List(3, 4))
    )
    assertEquals(
      Support.splitStatic(-5 until 5)(2).list,
      List(List(-5, -4, -3, -2, -1), List(0, 1, 2, 3, 4))
    )
  }

  test("split even incl") {
    assertEquals(
      Support.splitStatic(0 to 3)(4).list,
      List(List(0), List(1), List(2), List(3))
    )
    assertEquals(
      Support.splitStatic(0 to 1)(1).list,
      List(List(0, 1))
    )
    assertEquals(
      Support.splitStatic(0 to 0)(1).list,
      List(List(0))
    )
    assertEquals(
      Support.splitStatic(0 to 2)(1).list,
      List(List(0, 1, 2))
    )
    assertEquals(
      Support.splitStatic(0 to 0)(2).list,
      List(List(0))
    )
    assertEquals(
      Support.splitStatic(-5 to 4)(5).list,
      List(List(-5, -4), List(-3, -2), List(-1, 0), List(1, 2), List(3, 4))
    )
    assertEquals(
      Support.splitStatic(-5 to 4)(2).list,
      List(List(-5, -4, -3, -2, -1), List(0, 1, 2, 3, 4))
    )
  }

  test("split with odd") {
    assertEquals(
      Support.splitStatic(0 until 11)(3).list,
      List(List(0, 1, 2, 3), List(4, 5, 6, 7), List(8, 9, 10))
    )
    assertEquals(
      Support.splitStatic(0 until 4)(3).list,
      List(List(0, 1), List(2), List(3))
    )
    assertEquals(
      Support.splitStatic(0 until 4)(5).list,
      List(List(0), List(1), List(2), List(3))
    )
  }

  test("split even with spacing") {
    assertEquals(
      Support.splitStatic(0 until 11 by 2)(3).list,
      List(List(0, 2), List(4, 6), List(8, 10))
    )
    assertEquals(
      Support.splitStatic(0 until 2 by 2)(3).list,
      List(List(0))
    )
  }

  test("split odd with spacing") {
    assertEquals(
      Support.splitStatic(0 until 13 by 2)(3).list,
      List(List(0, 2, 4), List(6, 8), List(10, 12))
    )
  }

}
