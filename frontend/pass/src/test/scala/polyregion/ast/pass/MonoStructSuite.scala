package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class MonoStructSuite extends munit.FunSuite {

  // Spec: each generic struct that appears used with concrete type arguments should be cloned
  // into a monomorphic StructDef whose members have the type variables substituted; the program
  // is rewritten to reference the monomorphic name; the boundary value is a `Map[Sym, Sym]`
  // from monomorphic name back to the original generic name (for downstream pickling).

  test("non-generic program: no rename mappings produced") {
    val sd          = p.StructDef(sym("Pt"), Nil, List(named("x", p.Type.IntS32)), Nil)
    val (lookup, _) = MonoStruct(program(entry(), defs = List(sd)), NoopLog)
    assert(
      lookup.isEmpty || lookup.forall { case (mono, orig) => mono == orig },
      s"non-generic program should produce no rename mappings, got: $lookup"
    )
  }

  test("monomorphic instantiation produces a renamed StructDef and reverse-lookup entry") {
    // Generic struct Box[T] { v: T }, used as Box[Int] in entry.
    val genericName   = sym("Box")
    val genericDef    = p.StructDef(genericName, List("T"), List(named("v", p.Type.Var("T"))), Nil)
    val mono          = p.Type.Struct(genericName, List(p.Type.IntS32))
    val a             = arg("b", mono)
    val (lookup, out) = MonoStruct(program(entry(args = List(a)), defs = List(genericDef)), NoopLog)
    val newNames      = out.defs.map(_.name)
    val renamed       = newNames.find(_ != genericName)
    assert(renamed.isDefined, s"expected a monomorphic StructDef for Box[Int], got defs: $newNames")
    assertEquals(lookup.get(renamed.get), Some(genericName))
  }
}
