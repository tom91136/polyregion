package polyregion

import polyregion.jvm.compiler as ct
import polyregion.jvm.runtime as rt
import polyregion.jvm.twin.{FieldAccess, Layout, Marshal, ReflectiveTwin, TwinException, Twins}
import polyregion.jvm.twin.Layout.{Field, Kind}

import java.nio.{ByteBuffer, ByteOrder}
import scala.jdk.CollectionConverters.*

object TwinSuite {
  class Prims(
      var b: Byte,
      var s: Short,
      var c: Char,
      var i: Int,
      var j: Long,
      var f: Float,
      var d: Double,
      var z: Boolean
  )
  class Inner(var a: Float, var b: Double)
  class Outer(var id: Int, var inner: Inner, var tail: Long)
  class Holder(var items: Array[Inner], var ints: Array[Int], var more: Array[Int])
  class Grid(var rows: Array[Array[Int]])
  class Tagged(var x: Int)
  class Sealed(val x: Int)

  def prim(name: String, offset: Int, size: Int, descriptor: String) =
    Field(name, descriptor, Kind.PRIM, offset, size, "")

  val primsLayout = Layout(
    "Prims",
    classOf[Prims].getName,
    48,
    8,
    -1,
    -1,
    Array(
      prim("b", 0, 1, "B"),
      prim("s", 2, 2, "S"),
      prim("c", 4, 2, "C"),
      prim("i", 8, 4, "I"),
      prim("j", 16, 8, "J"),
      prim("f", 24, 4, "F"),
      prim("d", 32, 8, "D"),
      prim("z", 40, 1, "Z")
    )
  )
  val innerLayout = Layout(
    "Inner",
    classOf[Inner].getName,
    16,
    8,
    -1,
    -1,
    Array(prim("a", 0, 4, "F"), prim("b", 8, 8, "D"))
  )
  val outerLayout = Layout(
    "Outer",
    classOf[Outer].getName,
    32,
    8,
    -1,
    -1,
    Array(
      prim("id", 0, 4, "I"),
      Field("inner", "", Kind.STRUCT, 8, 16, "Inner"),
      prim("tail", 24, 8, "J")
    )
  )
  val holderLayout = Layout(
    "Holder",
    classOf[Holder].getName,
    24,
    8,
    -1,
    -1,
    Array(
      Field("items", "", Kind.ARRAY, 0, 8, "Inner"),
      Field("ints", "", Kind.ARRAY, 8, 8, ""),
      Field("more", "", Kind.ARRAY, 16, 8, "")
    )
  )
  val gridLayout = Layout("Grid", classOf[Grid].getName, 8, 8, -1, -1, Array(Field("rows", "", Kind.ARRAY, 0, 8, "")))
  val taggedLayout =
    Layout("Tagged", classOf[Tagged].getName, 8, 4, 4, 7, Array(prim("x", 0, 4, "I")))

  val all = List(primsLayout, innerLayout, outerLayout, holderLayout, gridLayout, taggedLayout)
}

class TwinSuite extends munit.FunSuite {
  import TwinSuite.*

  private case class Row(label: String, device: Option[rt.DeviceInfo], target: ct.Target, arch: String)

  private val rows = {
    val devices = rt.Context.platforms().flatMap(_.devices)
    val host = Row("host", devices.find(d => d.platformKind == 1 && d.moduleFormat == 2), ct.Target.LLVM_HOST, "native")
    val nvidia = devices.find(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0)
    List(
      host,
      Row(
        "nvidia",
        nvidia,
        ct.Target.LLVM_NVPTX64,
        nvidia.map(d => s"sm_${d.cudaArchitectureMajor}${d.cudaArchitectureMinor}").getOrElse("sm_80")
      )
    )
  }

  private def twins(layouts: List[Layout] = all) = Twins(
    java.util.HashMap(layouts.map(l => l.className -> l).toMap.asJava)
  )

  private class Fixture(val ctx: rt.Context) {
    val queue = ctx.queue()
    def device(pointer: Long, bytes: Int): ByteBuffer = {
      val b = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)
      ctx.copyFromDevice(pointer, b, bytes)
      b
    }
    def poke(pointer: Long, at: Int, write: ByteBuffer => Unit, bytes: Int): Unit = {
      val b = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)
      write(b)
      ctx.copyToDevice(b, pointer + at, bytes)
    }
  }

  private def withRow(row: Row)(f: Fixture => Unit): Unit = row.device match {
    case None => assume(false, s"no ${row.label} device in the catalogue")
    case Some(device) =>
      val ctx = rt.Context.create(device)
      try f(Fixture(ctx))
      finally ctx.close()
  }

  test("layouts round trip through their text encoding") {
    val decoded = Layout.decodeAll(Layout.encodeAll(all.asJava))
    assertEquals(decoded.size, all.size)
    all.foreach { l =>
      val d = decoded.get(l.name)
      assertEquals(d.encode, l.encode)
      assertEquals(d.fields.map(_.toString).toList, l.fields.map(_.toString).toList)
    }
    val awkward = Layout("a:b,c~d|e@f;g", "x.Y", 4, 4, -1, -1, Array(Field("n@m:o", "[Lx/Y;", Kind.ARRAY, 0, 8, "x:y")))
    assertEquals(Layout.decode(awkward.encode).encode, awkward.encode)
    assertEquals(Layout.decode(awkward.encode).fields.head.name, "n@m:o")
  }

  test("an unknown layout encoding is rejected") {
    intercept[TwinException](Layout.decodeAll("TL9~x"))
  }

  rows.foreach { row =>
    test(s"[${row.label}] primitives round trip") {
      withRow(row) { fx =>
        val p = Prims(1, 2, 'c', 4, 5L, 6f, 7.0, true)
        val m = Marshal(fx.queue, twins())
        try {
          val ptr = m.instance(p)
          val img = fx.device(ptr, 48)
          assertEquals((img.get(0), img.getShort(2), img.getChar(4), img.getInt(8)), (1.toByte, 2.toShort, 'c', 4))
          assertEquals((img.getLong(16), img.getFloat(24), img.getDouble(32), img.get(40)), (5L, 6f, 7.0, 1.toByte))
          fx.poke(ptr, 8, _.putInt(0, 40).putLong(8, 50L), 16)
          fx.poke(ptr, 40, _.put(0, 0.toByte), 1)
          m.settle()
          assertEquals((p.i, p.j, p.z, p.b, p.d), (40, 50L, false, 1.toByte, 7.0))
        } finally m.close()
      }
    }

    test(s"[${row.label}] nested structs sit inline and update in place") {
      withRow(row) { fx =>
        val o     = Outer(9, Inner(1.5f, 2.5), 77L)
        val inner = o.inner
        val m     = Marshal(fx.queue, twins())
        try {
          val ptr = m.instance(o)
          val img = fx.device(ptr, 32)
          assertEquals((img.getInt(0), img.getFloat(8), img.getDouble(16), img.getLong(24)), (9, 1.5f, 2.5, 77L))
          fx.poke(ptr, 8, _.putFloat(0, 3.5f).putDouble(8, 4.5), 16)
          m.settle()
          assert(o.inner eq inner)
          assertEquals((o.inner.a, o.inner.b, o.id, o.tail), (3.5f, 4.5, 9, 77L))
        } finally m.close()
      }
    }

    test(s"[${row.label}] arrays of structs, primitives and rows hold device pointers") {
      withRow(row) { fx =>
        val items = Array(Inner(1f, 2.0), Inner(3f, 4.0), Inner(5f, 6.0))
        val h     = Holder(items, Array(10, 20, 30), Array(7))
        val g     = Grid(Array(Array(1, 2), Array(3, 4, 5)))
        val m     = Marshal(fx.queue, twins())
        try {
          val hp   = m.instance(h)
          val gp   = m.instance(g)
          val img  = fx.device(hp, 24)
          val ip   = img.getLong(0)
          val ints = fx.device(img.getLong(8), 12)
          assertEquals((ints.getInt(0), ints.getInt(4), ints.getInt(8)), (10, 20, 30))
          val elems = fx.device(ip, 48)
          assertEquals((elems.getFloat(16), elems.getDouble(24)), (3f, 4.0))
          val table = fx.device(fx.device(gp, 8).getLong(0), 16)
          val row1  = fx.device(table.getLong(8), 12)
          assertEquals((row1.getInt(0), row1.getInt(8)), (3, 5))

          fx.poke(ip, 32, _.putFloat(0, 9f), 4)
          fx.poke(img.getLong(8), 4, _.putInt(0, 99), 4)
          fx.poke(table.getLong(8), 0, _.putInt(0, 33), 4)
          m.settle()
          assertEquals(items(2).a, 9f)
          assertEquals(h.ints.toList, List(10, 99, 30))
          assertEquals(g.rows.map(_.toList).toList, List(List(1, 2), List(33, 4, 5)))
        } finally m.close()
      }
    }

    test(s"[${row.label}] an array reachable twice is staged once") {
      withRow(row) { fx =>
        val shared = Array(1, 2, 3, 4)
        val a      = Holder(Array(), shared, shared)
        val b      = Holder(Array(), shared, Array(9))
        val m      = Marshal(fx.queue, twins())
        try {
          val ap = m.instance(a)
          val bp = m.instance(b)
          val ai = fx.device(ap, 24)
          val bi = fx.device(bp, 24)
          assertEquals(ai.getLong(8), ai.getLong(16))
          assertEquals(ai.getLong(8), bi.getLong(8))
          assertNotEquals(bi.getLong(16), bi.getLong(8))
          assertEquals(m.instance(a), ap)
          // a, b, their empty item arrays, shared, and b.more
          assertEquals(m.stagedBuffers, 6)
          fx.poke(ai.getLong(8), 12, _.putInt(0, 44), 4)
          m.settle()
          assertEquals(shared.toList, List(1, 2, 3, 44))
        } finally m.close()
      }
    }

    test(s"[${row.label}] the dispatch tag is written at its offset") {
      withRow(row) { fx =>
        val m = Marshal(fx.queue, twins())
        try {
          val img = fx.device(m.instance(Tagged(5)), 8)
          assertEquals((img.getInt(0), img.getInt(4)), (5, 7))
        } finally m.close()
      }
    }

    test(s"[${row.label}] close releases every staged buffer") {
      withRow(row) { fx =>
        val before = fx.queue.stats().releases
        val m      = Marshal(fx.queue, twins())
        m.instance(Holder(Array(Inner(1f, 2.0)), Array(1), Array(2)))
        assertEquals(m.stagedBuffers, 4)
        m.close()
        assertEquals(m.stagedBuffers, 0)
        assertEquals(fx.queue.stats().releases - before, if (fx.ctx.sharedAddressSpace) 0L else 4L)
      }
    }

    test(s"[${row.label}] a null nested object and a mixed array are rejected") {
      withRow(row) { fx =>
        val m = Marshal(fx.queue, twins())
        try {
          val e = intercept[TwinException](m.instance(Outer(1, null, 2L)))
          assert(e.getMessage.contains("is null"), e.getMessage)
          val h =
            intercept[TwinException](m.array(Array[AnyRef](Inner(1f, 1.0), Prims(1, 1, 'a', 1, 1L, 1f, 1.0, true))))
          assert(h.getMessage.contains("one concrete type"), h.getMessage)
        } finally m.close()
      }
    }
  }

  rows.foreach { row =>
    test(s"[${row.label}] readBack copies one staged array and settle copies the rest") {
      withRow(row) { fx =>
        val a = Array(1, 2)
        val b = Array(3, 4)
        val m = Marshal(fx.queue, twins())
        try {
          val ap = m.array(a)
          val bp = m.array(b)
          fx.poke(ap, 0, _.putInt(0, 10), 4)
          fx.poke(bp, 0, _.putInt(0, 30), 4)
          m.sync()
          m.readBack(a)
          assertEquals((a.toList, b.toList), (List(10, 2), List(3, 4)))
          m.settle()
          assertEquals(b.toList, List(30, 4))
        } finally m.close()
      }
    }
  }

  test("a class without a layout is rejected with the known names") {
    val e = intercept[TwinException](twins(List(innerLayout)).of(classOf[Prims]))
    assert(e.getMessage.contains("No device layout for " + classOf[Prims].getName), e.getMessage)
  }

  private def mismatch(layout: Layout, cls: Class[?], expect: String): Unit = {
    val e = intercept[TwinException](ReflectiveTwin(layout, cls, Twins(java.util.HashMap())))
    assert(e.getMessage.contains(expect), e.getMessage)
  }

  test("a layout that has drifted from the class fails on load") {
    def one(f: Field) = Layout("L", classOf[Inner].getName, 16, 8, -1, -1, Array(f))
    mismatch(one(prim("a", 0, 4, "I")), classOf[Inner], "has type F but the layout expects I")
    mismatch(one(prim("gone", 0, 4, "F")), classOf[Inner], "has no field `gone`")
    mismatch(one(prim("a", 0, 8, "F")), classOf[Inner], "is 8 bytes in the layout")
    mismatch(one(Field("a", "", Kind.ARRAY, 0, 8, "")), classOf[Inner], "is an array in the layout")
    mismatch(one(Field("a", "", Kind.STRUCT, 0, 8, "")), classOf[Inner], "is an object in the layout")
  }

  test("a private JDK field is refused without --add-opens and names the flag") {
    val e =
      intercept[TwinException](FieldAccess.of(classOf[java.util.ArrayList[?]], "elementData", "[Ljava/lang/Object;"))
    assert(e.getMessage.contains("--add-opens java.base/java.util=ALL-UNNAMED"), e.getMessage)
  }

  test("private JDK fields read and write back through VarHandles with --add-opens") {
    val (code, out) = ChildJvm.run(
      "polyregion.TwinVarHandleMain",
      Map.empty,
      List("--add-opens", "java.base/java.util=ALL-UNNAMED")
    )
    assertEquals(code, 0, out)
    assert(out.contains("twin varhandle ok"), out)
  }
}

object TwinVarHandleMain {
  def main(args: Array[String]): Unit = {
    val xs = java.util.ArrayList[Integer]()
    (1 to 3).foreach(i => xs.add(i))
    val data  = FieldAccess.of(classOf[java.util.ArrayList[?]], "elementData", "[Ljava/lang/Object;")
    val size  = FieldAccess.of(classOf[java.util.ArrayList[?]], "size", "I")
    val elems = data.get(xs).asInstanceOf[Array[AnyRef]]
    assertEq(elems.take(3).toList.map(x => x: Any), List[Any](1, 2, 3))
    assertEq(size.get(xs), 3)
    val replaced = Array[AnyRef](Integer.valueOf(7), Integer.valueOf(8), Integer.valueOf(9), null)
    data.set(xs, replaced)
    size.set(xs, 3)
    assertEq(xs.toArray.toList.map(x => x: Any), List[Any](7, 8, 9))
    val words = FieldAccess.of(classOf[java.util.BitSet], "words", "[J")
    val bits  = java.util.BitSet(); bits.set(3)
    assertEq(words.get(bits).asInstanceOf[Array[Long]](0), 8L)
    println("twin varhandle ok")
  }
  private def assertEq(a: Any, b: Any): Unit = if (a != b) throw AssertionError(s"$a != $b")
}
