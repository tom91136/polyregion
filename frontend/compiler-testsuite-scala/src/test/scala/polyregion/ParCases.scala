package polyregion

import polyregion.jvm.runtime as rt
import polyregion.par.*

import scala.compiletime.testing.typeCheckErrors

abstract class ParCases extends munit.FunSuite {

  protected case class Backend(label: String, open: () => Option[rt.Context], owned: Boolean)

  protected def backends: List[Backend]

  protected def hostBackend: Backend = Backend(
    "host",
    () =>
      rt.Context
        .platforms()
        .flatMap(_.devices)
        .find(d => d.platformKind == 1 && d.moduleFormat == 2)
        .map(rt.Context.create),
    owned = true
  )

  protected def cudaBackend: Backend =
    Backend("cuda", () => Option(rt.Context.current()).filterNot(_.sharedAddressSpace()), owned = false)

  protected def check(name: String)(f: rt.Context => Unit): Unit =
    backends.foreach { backend =>
      test(s"$name on ${backend.label}") {
        val ctx = backend.open()
        assume(ctx.isDefined, s"no ${backend.label} context")
        try f(ctx.get)
        finally if (backend.owned) ctx.get.close()
      }
    }

  private def launches(ctx: rt.Context)(f: => Unit): Long = {
    val before = ctx.stats().launches
    f
    ctx.stats().launches - before
  }

  private def ints(n: Int): Array[Int] = Array.tabulate(n)(i => (i * 7 + 3) % 101 - 40)

  check("map then sum") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).map(_ * 2).sum, xs.map(_ * 2).sum)
  }

  check("map and map compose") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).map(_ + 1).map(_ * 3).sum, xs.map(x => (x + 1) * 3).sum)
  }

  check("filter then sum") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 0).sum, xs.filter(_ > 0).sum)
  }

  check("map then filter then sum") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).map(_ * 3).filter(_ % 2 == 0).sum, xs.map(_ * 3).filter(_ % 2 == 0).sum)
  }

  check("product") { ctx =>
    val xs = Array.tabulate(12)(i => if (i % 3 == 0) 2 else 1)
    assertEquals(xs.par.on(ctx).product, xs.product)
  }

  check("count with a predicate") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).count(_ % 5 == 0), xs.count(_ % 5 == 0))
  }

  check("count after a filter") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 10).count(_ % 2 == 0), xs.filter(_ > 10).count(_ % 2 == 0))
  }

  check("size without a filter") { ctx =>
    val xs = ints(321)
    assertEquals(xs.par.on(ctx).map(_ + 1).size, 321)
  }

  check("size after a filter") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 0).size, xs.count(_ > 0))
  }

  check("exists") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).exists(_ == 60), xs.exists(_ == 60))
    assertEquals(xs.par.on(ctx).exists(_ == 1000), xs.exists(_ == 1000))
  }

  check("forall") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).forall(_ < 100), xs.forall(_ < 100))
    assertEquals(xs.par.on(ctx).forall(_ > 0), xs.forall(_ > 0))
  }

  check("forall ignores filtered elements") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 0).forall(_ > 0), true)
  }

  check("fold with a neutral element") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).fold(0)(_ + _), xs.sum)
    assertEquals(xs.par.on(ctx).map(_ + 100).fold(1)(_ * _ % 7), xs.map(_ + 100).fold(1)(_ * _ % 7))
  }

  check("reduce needs no neutral element") { ctx =>
    val xs = ints(1000).map(_ + 100)
    assertEquals(xs.par.on(ctx).reduce(_ + _), xs.reduce(_ + _))
    assertEquals(xs.par.on(ctx).map(_ - 5).reduce(_ + _), xs.map(_ - 5).reduce(_ + _))
  }

  check("reduce of a filtered chain materialises first") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 0).reduce(_ + _), xs.filter(_ > 0).reduce(_ + _))
  }

  check("min and max") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).min, xs.min)
    assertEquals(xs.par.on(ctx).max, xs.max)
    assertEquals(xs.par.on(ctx).map(x => -x).filter(_ < 20).max, xs.view.map(x => -x).filter(_ < 20).max)
  }

  check("toArray after map") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).map(_ * 2 + 1).toArray.toList, xs.map(_ * 2 + 1).toList)
  }

  check("toArray without stages copies") { ctx =>
    val xs  = ints(100)
    val out = xs.par.on(ctx).toArray
    assertEquals(out.toList, xs.toList)
    out(0) = 12345
    assertNotEquals(xs(0), 12345)
  }

  check("toArray after a filter") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ > 0).toArray.toList, xs.filter(_ > 0).toList)
  }

  check("toArray after filter and map") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).filter(_ % 2 == 0).map(_ * 3).toArray.toList, xs.filter(_ % 2 == 0).map(_ * 3).toList)
  }

  check("toArray after map and filter") { ctx =>
    val xs = ints(1000)
    assertEquals(
      xs.par.on(ctx).map(_ * 3).filter(_ % 2 == 0).toArray.toList,
      xs.view.map(_ * 3).filter(_ % 2 == 0).toList
    )
  }

  check("map changes the element type") { ctx =>
    val xs = ints(500)
    assertEquals(xs.par.on(ctx).map(_.toDouble * 0.5).toArray.toList, xs.map(_.toDouble * 0.5).toList)
    assertEquals(xs.par.on(ctx).map(_.toLong * 1000000L).sum, xs.map(_.toLong * 1000000L).sum)
  }

  check("sorted") { ctx =>
    val xs     = ints(1000)
    val before = xs.toList
    assertEquals(xs.par.on(ctx).sorted.toList, xs.sorted.toList)
    assertEquals(xs.toList, before)
  }

  check("sorted after map and filter") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).map(_ * 5).filter(_ > 0).sorted.toList, xs.map(_ * 5).filter(_ > 0).sorted.toList)
  }

  check("sortWith orders by a comparator") { ctx =>
    val xs = ints(1000)
    assertEquals(xs.par.on(ctx).sortWith((a, b) => a > b).toList, xs.sorted.reverse.toList)
  }

  check("lambdas capture scalars and arrays") { ctx =>
    val xs     = ints(1000)
    val scale  = 3
    val offset = Array(11)
    assertEquals(
      xs.par.on(ctx).map(x => x * scale + offset(0)).filter(_ > scale).sum,
      xs.map(x => x * scale + offset(0)).filter(_ > scale).sum
    )
  }

  check("long and double sources") { ctx =>
    val ls = Array.tabulate(1000)(i => i.toLong * 4000000L)
    val ds = Array.tabulate(1000)(i => i.toDouble * 0.5)
    assertEquals(ls.par.on(ctx).sum, ls.sum)
    assertEquals(ds.par.on(ctx).map(_ * 2.0).sum, ds.map(_ * 2.0).sum)
    assertEquals(ds.par.on(ctx).max, ds.max)
  }

  check("empty sources") { ctx =>
    val xs = Array.ofDim[Int](0)
    assertEquals(xs.par.on(ctx).map(_ + 1).sum, 0)
    assertEquals(xs.par.on(ctx).map(_ + 1).toArray.toList, Nil)
    assertEquals(xs.par.on(ctx).exists(_ > 0), false)
    assertEquals(xs.par.on(ctx).forall(_ > 0), true)
    assertEquals(xs.par.on(ctx).sorted.toList, Nil)
    intercept[UnsupportedOperationException](xs.par.on(ctx).reduce(_ + _))
  }

  check("a filter that keeps nothing") { ctx =>
    val xs = ints(100)
    assertEquals(xs.par.on(ctx).filter(_ > 1000).toArray.toList, Nil)
    assertEquals(xs.par.on(ctx).filter(_ > 1000).sum, 0)
  }

  check("range sum") { ctx =>
    assertEquals((0 until 100000).par.on(ctx).map(_ % 7).sum, (0 until 100000).map(_ % 7).sum)
    assertEquals((1 to 100 by 3).par.on(ctx).sum, (1 to 100 by 3).sum)
    assertEquals((10 until 0 by -3).par.on(ctx).map(_ * 2).sum, (10 until 0 by -3).map(_ * 2).sum)
  }

  check("range count and exists") { ctx =>
    assertEquals((0 until 10000).par.on(ctx).count(_ % 7 == 0), (0 until 10000).count(_ % 7 == 0))
    assertEquals((0 until 10000).par.on(ctx).exists(_ == 9999), true)
    assertEquals((0 until 10000).par.on(ctx).forall(_ < 9999), false)
  }

  check("range toArray") { ctx =>
    assertEquals((0 until 1000).par.on(ctx).map(_ * 3).toArray.toList, (0 until 1000).map(_ * 3).toList)
    assertEquals((0 until 1000).par.on(ctx).filter(_ % 3 == 0).toArray.toList, (0 until 1000).filter(_ % 3 == 0).toList)
    assertEquals((0 until 1000).par.on(ctx).toArray.toList, (0 until 1000).toList)
  }

  check("range reduce and max") { ctx =>
    assertEquals((1 to 50).par.on(ctx).reduce(_ + _), (1 to 50).reduce(_ + _))
    assertEquals((0 until 1000).par.on(ctx).map(x => (x * 37) % 101).max, (0 until 1000).map(x => (x * 37) % 101).max)
  }

  check("range foreach writes a captured array") { ctx =>
    val n   = 1000
    val xs  = ints(n)
    val out = Array.ofDim[Int](n)
    (0 until n).par.on(ctx).foreach(i => out(i) = xs(i) * 2)
    assertEquals(out.toList, xs.map(_ * 2).toList)
  }

  check("array foreach reads and writes captured arrays") { ctx =>
    val n   = 1000
    val xs  = ints(n)
    val out = Array.ofDim[Int](n)
    val ref = Array.ofDim[Int](n)
    xs.par.on(ctx).filter(_ > 0).foreach(x => out(x) = 1)
    xs.filter(_ > 0).foreach(x => ref(x) = 1)
    assertEquals(out.toList, ref.toList)
  }

  check("a chain over a method parameter and a computed source") { ctx =>
    def total(xs: Array[Int], k: Int): Int = xs.par.on(ctx).map(_ * k).filter(_ != 0).sum
    val built                              = java.util.concurrent.atomic.AtomicInteger()
    def source(): Array[Int]               = { built.incrementAndGet(); ints(200) }
    assertEquals(total(ints(200), 3), ints(200).map(_ * 3).filter(_ != 0).sum)
    assertEquals(source().par.on(ctx).map(_ + 1).sum, ints(200).map(_ + 1).sum)
    assertEquals(built.get, 1)
  }

  check("a boolean inside a chain needs no array") { ctx =>
    val xs = ints(300)
    assertEquals(xs.par.on(ctx).map(_ > 0).count(b => b), xs.count(_ > 0))
    assertEquals(xs.par.on(ctx).map(_ * 2).map(_ > 20).exists(b => b), xs.exists(_ * 2 > 20))
  }

  check("the default context runs the chain") { _ =>
    val xs = ints(1000)
    assertEquals(xs.par.map(_ * 2).filter(_ > 0).sum, xs.map(_ * 2).filter(_ > 0).sum)
    assertEquals((0 until 1000).par.count(_ % 3 == 0), (0 until 1000).count(_ % 3 == 0))
  }

  private def baseline(ctx: rt.Context)(f: => Unit): Long = {
    val n = launches(ctx)(f)
    assert(n > 0, "a single call launched nothing")
    n
  }

  check("map, map and sum cost one reduction") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(ctx)(xs.par.on(ctx).map(_ + 1).map(_ * 2).sum), baseline(ctx)(xs.par.on(ctx).sum))
  }

  check("filter, map and sum cost one reduction") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(ctx)(xs.par.on(ctx).filter(_ > 0).map(_ * 2).sum), baseline(ctx)(xs.par.on(ctx).sum))
  }

  check("map, filter and a predicate terminal cost one predicate launch") { ctx =>
    val xs = ints(1000)
    assertEquals(
      launches(ctx)(xs.par.on(ctx).map(_ * 3).filter(_ > 0).count(_ % 2 == 0)),
      baseline(ctx)(xs.par.on(ctx).count(_ > 0))
    )
    assertEquals(
      launches(ctx)(xs.par.on(ctx).map(_ * 3).filter(_ > 0).exists(_ > 100)),
      baseline(ctx)(xs.par.on(ctx).exists(_ > 100))
    )
    assertEquals(
      launches(ctx)(xs.par.on(ctx).map(_ * 3).filter(_ > 0).forall(_ > 0)),
      baseline(ctx)(xs.par.on(ctx).forall(_ > 0))
    )
  }

  check("map and map to an array cost one transform") { ctx =>
    val xs = ints(1000)
    assertEquals(
      launches(ctx)(xs.par.on(ctx).map(_ + 1).map(_ * 2).toArray),
      baseline(ctx)(xs.par.on(ctx).map(_ + 1).toArray)
    )
  }

  check("a filter to an array costs a copy_if and, with a map, a transform") { ctx =>
    val xs        = ints(1000)
    val copy      = baseline(ctx)(xs.par.on(ctx).filter(_ > 0).toArray)
    val transform = baseline(ctx)(xs.par.on(ctx).map(_ * 2).toArray)
    assertEquals(launches(ctx)(xs.par.on(ctx).filter(_ > 0).map(_ * 2).toArray), copy + transform)
  }

  check("sorted costs a sort and, after a map, a transform") { ctx =>
    val xs        = ints(1000)
    val sort      = baseline(ctx)(xs.par.on(ctx).sorted)
    val transform = baseline(ctx)(xs.par.on(ctx).map(_ * 2).toArray)
    assertEquals(launches(ctx)(xs.par.on(ctx).map(_ + 1).sorted), transform + sort)
  }

  check("range chains cost one launch per fused stage group") { ctx =>
    val reduction = baseline(ctx)((0 until 1000).par.on(ctx).sum)
    val tabulate  = baseline(ctx)((0 until 1000).par.on(ctx).toArray)
    val copy      = baseline(ctx)(ints(1000).par.on(ctx).filter(_ > 0).toArray)
    assertEquals(launches(ctx)((0 until 1000).par.on(ctx).map(_ * 2).sum), reduction)
    assertEquals(launches(ctx)((0 until 1000).par.on(ctx).map(_ * 2).toArray), tabulate)
    assertEquals(launches(ctx)((0 until 1000).par.on(ctx).filter(_ % 3 == 0).toArray), tabulate + copy)
  }

  check("reduce without a neutral element costs one reduction and, after a filter, a copy_if") { ctx =>
    val xs        = ints(1000)
    val reduction = baseline(ctx)(xs.par.on(ctx).reduce(_ + _))
    val copy      = baseline(ctx)(xs.par.on(ctx).filter(_ > 0).toArray)
    assertEquals(launches(ctx)(xs.par.on(ctx).map(_ + 100).reduce(_ + _)), reduction)
    assertEquals(launches(ctx)(xs.par.on(ctx).filter(_ > 0).reduce(_ + _)), copy + reduction)
  }

  check("a million elements") { ctx =>
    val xs = Array.tabulate(1000000)(i => (i * 31) % 1009)
    assertEquals(xs.par.on(ctx).map(_ + 1).sum, xs.map(_ + 1).sum)
    assertEquals(xs.par.on(ctx).filter(_ % 7 == 0).toArray.toList, xs.filter(_ % 7 == 0).toList)
    assertEquals(xs.par.on(ctx).sorted.toList, xs.sorted.toList)
    assertEquals(xs.par.on(ctx).max, xs.max)
  }

  check("float sources") { ctx =>
    val fs = Array.tabulate(1000)(i => i.toFloat * 0.25f)
    assertEquals(fs.par.on(ctx).map(_ * 2f).sum, fs.map(_ * 2f).sum)
    assertEquals(fs.par.on(ctx).filter(_ > 100f).toArray.toList, fs.filter(_ > 100f).toList)
    assertEquals(fs.par.on(ctx).min, fs.min)
  }

  check("an empty chain launches nothing") { ctx =>
    val xs = Array.ofDim[Int](0)
    assertEquals(launches(ctx)(xs.par.on(ctx).map(_ + 1).sum), 0L)
    assertEquals(launches(ctx)(xs.par.on(ctx).map(_ + 1).toArray), 0L)
    assertEquals(launches(ctx)(xs.par.on(ctx).sorted), 0L)
  }

  private val total = Ordering.Double.TotalOrdering

  private val totalFloat = Ordering.Float.TotalOrdering

  private def bits(xs: Array[Double]): List[Long] = xs.map(java.lang.Double.doubleToLongBits(_)).toList

  private def bits(x: Double): Long = java.lang.Double.doubleToLongBits(x)

  private def bits(xs: Array[Float]): List[Int] = xs.map(java.lang.Float.floatToIntBits(_)).toList

  private def bits(x: Float): Int = java.lang.Float.floatToIntBits(x)

  private val awkward = Array(
    Double.NaN,
    2.0,
    1.0,
    0.0,
    -0.0,
    Double.NegativeInfinity,
    3.5,
    Double.NaN,
    -1.0,
    0.0,
    -0.0,
    Double.PositiveInfinity,
    5.0,
    4.0,
    7.0,
    6.0,
    9.0,
    8.0,
    -3.0
  )

  private val awkwardFloats = awkward.map(_.toFloat)

  check("sortWith is stable") { ctx =>
    val xs = Array.tabulate(5000)(i => (i * 7919) % 5000)
    assertEquals(
      xs.par.on(ctx).sortWith((a, b) => a % 7 < b % 7).toList,
      xs.sortWith((a, b) => a % 7 < b % 7).toList
    )
  }

  check("sorted orders NaN last and negative zero first") { ctx =>
    assertEquals(bits(awkward.par.on(ctx).sorted), bits(awkward.sorted(using total)))
    assertEquals(bits(awkwardFloats.par.on(ctx).sorted), bits(awkwardFloats.sorted(using totalFloat)))
    assertEquals(bits(Array(Double.NaN, 2.0, 1.0).par.on(ctx).sorted), bits(Array(1.0, 2.0, Double.NaN)))
    assertEquals(bits(Array(0.0, -0.0).par.on(ctx).sorted), bits(Array(-0.0, 0.0)))
  }

  check("min and max follow the total order on NaN and signed zero") { ctx =>
    assertEquals(bits(awkward.par.on(ctx).min), bits(awkward.min(using total)))
    assertEquals(bits(awkward.par.on(ctx).max), bits(awkward.max(using total)))
    assertEquals(bits(awkwardFloats.par.on(ctx).min), bits(awkwardFloats.min(using totalFloat)))
    assertEquals(bits(awkwardFloats.par.on(ctx).max), bits(awkwardFloats.max(using totalFloat)))
    assertEquals(bits(Array(-0.0, 0.0).par.on(ctx).min), bits(-0.0))
    assertEquals(bits(Array(0.0, -0.0).par.on(ctx).min), bits(-0.0))
    assertEquals(bits(Array(-0.0, 0.0).par.on(ctx).max), bits(0.0))
    assertEquals(bits(Array(0.0, -0.0).par.on(ctx).max), bits(0.0))
    assertEquals(bits(Array(1.0, Double.NaN, 2.0).par.on(ctx).max), bits(Double.NaN))
    assertEquals(bits(Array(1.0, Double.NaN, 2.0).par.on(ctx).min), bits(1.0))
  }

  check("empty ranges") { ctx =>
    assertEquals((5 until 5).par.on(ctx).map(_ + 1).sum, 0)
    assertEquals((5 until 5).par.on(ctx).map(_ + 1).toArray.toList, Nil)
    assertEquals((5 until 5).par.on(ctx).filter(_ > 0).toArray.toList, Nil)
    assertEquals((5 until 5).par.on(ctx).count(_ > 0), 0)
    assertEquals((5 until 5).par.on(ctx).exists(_ > 0), false)
    assertEquals((5 until 5).par.on(ctx).forall(_ > 0), true)
    intercept[UnsupportedOperationException]((5 until 5).par.on(ctx).max)
  }

  check("integer bounds") { ctx =>
    val ints =
      Array(Int.MaxValue, 1, Int.MinValue, 0, -1, Int.MaxValue, Int.MinValue, 7, 3, 2, 9, 8, 5, 4, 6, 11, 10, 13)
    val longs = Array(
      Long.MaxValue,
      1L,
      Long.MinValue,
      0L,
      -1L,
      Long.MaxValue,
      Long.MinValue,
      7L,
      3L,
      2L,
      9L,
      8L,
      5L,
      4L,
      6L,
      11L,
      10L,
      13L
    )
    assertEquals(ints.par.on(ctx).sorted.toList, ints.sorted.toList)
    assertEquals(ints.par.on(ctx).min, ints.min)
    assertEquals(ints.par.on(ctx).max, ints.max)
    assertEquals(ints.par.on(ctx).sum, ints.sum)
    assertEquals(longs.par.on(ctx).sorted.toList, longs.sorted.toList)
    assertEquals(longs.par.on(ctx).min, longs.min)
    assertEquals(longs.par.on(ctx).max, longs.max)
    assertEquals(longs.par.on(ctx).sum, longs.sum)
    assertEquals(longs.par.on(ctx).map(_ + 1).sum, longs.map(_ + 1).sum)
  }

  check("empty reductions throw the sequential messages") { ctx =>
    val xs = Array.ofDim[Int](0)
    assertEquals(
      intercept[UnsupportedOperationException](xs.par.on(ctx).min).getMessage,
      intercept[UnsupportedOperationException](xs.min).getMessage
    )
    assertEquals(
      intercept[UnsupportedOperationException](xs.par.on(ctx).max).getMessage,
      intercept[UnsupportedOperationException](xs.max).getMessage
    )
    assertEquals(
      intercept[UnsupportedOperationException](xs.par.on(ctx).reduce(_ + _)).getMessage,
      intercept[UnsupportedOperationException](xs.reduce(_ + _)).getMessage
    )
  }

  test("a second context selection is rejected") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array(1, 2, 3)
      val a = polyregion.jvm.runtime.Context.current()
      xs.par.on(a).map(_ + 1).on(a).sum
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("a .par chain selects its context once"), messages)
  }

  test("a stored chain is rejected") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array(1, 2, 3)
      val chain = xs.par.map(_ + 1)
      chain.sum
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("a .par chain must be a single expression ending in a terminal operation"), messages)
  }

  test("an unsupported element type is rejected") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array("a", "b")
      xs.par.map(_.length).sum
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains(".par arrays hold Int, Long, Float or Double elements"), messages)
  }

  test("a short array is rejected naming the supported elements") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array[Short](1, 2, 3)
      xs.par.map(_ + 1).sum
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains(".par arrays hold Int, Long, Float or Double elements"), messages)
  }

  test("an array result of a narrow element type is rejected") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array(1, 2, 3)
      xs.par.map(_.toShort).toArray
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains(".par arrays hold Int, Long, Float or Double elements"), messages)
  }

  test("sum of a non-numeric element is rejected") {
    val errors = typeCheckErrors("""
      import polyregion.par.*
      val xs = Array(1, 2, 3)
      xs.par.map(_ > 1).sum
    """)
    val messages = errors.map(_.message).mkString("\n")
    assert(messages.contains("sum is only defined for"), messages)
  }
}
