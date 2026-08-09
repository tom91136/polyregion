package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class DeadStructEliminationSuite extends munit.FunSuite {

  // Spec: structs that are never referenced anywhere in the program (entry, helper functions,
  // or other struct definitions) should be removed.

  test("orphan struct should be removed when no function references it") {
    val orphan = p.StructDef(sym("orphan"), Nil, Nil, Nil)
    val out    = DeadStructElimination(program(entry(), defs = List(orphan)), NoopLog)
    assert(!out.defs.exists(_.name == orphan.name), s"orphan struct ${orphan.name} should be removed")
  }

  test("referenced struct should be kept") {
    val s    = p.StructDef(sym("Point"), Nil, List(named("x", p.Type.IntS32)), Nil)
    val sTpe = p.Type.Struct(s.name, Nil)
    val a    = arg("p", sTpe)
    val out  = DeadStructElimination(program(entry(args = List(a)), defs = List(s)), NoopLog)
    assert(out.defs.exists(_.name == s.name), "referenced struct must be kept")
  }
}
