package polyregion

import polyregion.jvm.runtime as rt
import polyregion.par.*
import polyregion.scalalang.LaunchRuntime

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

  private def launches(f: => Unit): Long = {
    val before = LaunchRuntime.launchCount
    f
    LaunchRuntime.launchCount - before
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

  check("map, map and sum are one launch") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).map(_ + 1).map(_ * 2).sum), 1L)
  }

  check("filter, map and sum are one launch") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).filter(_ > 0).map(_ * 2).sum), 1L)
  }

  check("map, filter and count are one launch") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).map(_ * 3).filter(_ > 0).count(_ % 2 == 0)), 1L)
    assertEquals(launches(xs.par.on(ctx).map(_ * 3).filter(_ > 0).exists(_ > 100)), 1L)
    assertEquals(launches(xs.par.on(ctx).map(_ * 3).filter(_ > 0).forall(_ > 0)), 1L)
  }

  check("map and map to an array is one launch") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).map(_ + 1).map(_ * 2).toArray), 1L)
  }

  check("a filter to an array is one launch without a map and two with one") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).filter(_ > 0).toArray), 1L)
    assertEquals(launches(xs.par.on(ctx).filter(_ > 0).map(_ * 2).toArray), 2L)
  }

  check("sorted copies without a launch other than the sort") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).sorted), 1L)
    assertEquals(launches(xs.par.on(ctx).map(_ + 1).sorted), 2L)
  }

  check("a range reduction and a range map to an array are one launch") { ctx =>
    assertEquals(launches((0 until 1000).par.on(ctx).map(_ * 2).sum), 1L)
    assertEquals(launches((0 until 1000).par.on(ctx).map(_ * 2).toArray), 1L)
    assertEquals(launches((0 until 1000).par.on(ctx).filter(_ % 3 == 0).toArray), 2L)
  }

  check("reduce without a neutral element is one launch") { ctx =>
    val xs = ints(1000)
    assertEquals(launches(xs.par.on(ctx).map(_ + 100).reduce(_ + _)), 1L)
    assertEquals(launches(xs.par.on(ctx).filter(_ > 0).reduce(_ + _)), 2L)
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
    assertEquals(launches(xs.par.on(ctx).map(_ + 1).sum), 0L)
    assertEquals(launches(xs.par.on(ctx).map(_ + 1).toArray), 0L)
    assertEquals(launches(xs.par.on(ctx).sorted), 0L)
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
