package polyregion.javalang

import polyregion.ast.PolyAST as p

class ClassLayoutsSuite extends munit.FunSuite {

  private def struct(members: (String, p.Type)*) =
    p.StructDef(p.Sym("a.Box"), Nil, members.toList.map((n, t) => p.Named(n, t)), Nil)

  private val source   = struct("v" -> p.Type.IntS32, "w" -> p.Type.Float64)
  private val other    = p.StructDef(p.Sym("a.Other"), Nil, Nil, Nil)
  private val subset   = struct("v" -> p.Type.IntS32)
  private val mismatch = struct("v" -> p.Type.IntS64)
  private val extra    = struct("v" -> p.Type.IntS32, "z" -> p.Type.IntS32)

  test("a class file layout that is part of the source layout yields to it") {
    assertEquals(ClassLayouts.merge(List(source), List(subset)), Right(List(source)))
  }

  test("a class the source does not know keeps its class file layout") {
    assertEquals(ClassLayouts.merge(List(source), List(other)), Right(List(source, other)))
  }

  test("a member the source layout lacks is refused") {
    val message = ClassLayouts.merge(List(source), List(extra)).left.map(_.getMessage)
    assert(message.left.exists(_.contains("`z: ")), message.toString)
  }

  test("a member whose type differs is refused") {
    assert(ClassLayouts.merge(List(source), List(mismatch)).isLeft)
  }
}
