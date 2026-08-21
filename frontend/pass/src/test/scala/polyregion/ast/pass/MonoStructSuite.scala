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

  test("monomorphic instantiation preserves member identities in declarations and selects") {
    val genericName = sym("Box")
    val genericDef  = p.StructDef(genericName, List("T"), List(named("Box::value", p.Type.Var("T"))), Nil)
    val mono        = p.Type.Struct(genericName, List(p.Type.IntS32))
    val box         = arg("box", mono)
    val read        = p.Term.Select(box.named, List(p.PathStep.Field("Box::value")), p.Type.IntS32)

    val (_, out) = MonoStruct(
      program(entry(args = List(box), body = List(p.Stmt.Return(p.Expr.Alias(read)))), defs = List(genericDef)),
      NoopLog
    )

    val renamed = out.defs.find(_.name != genericName).getOrElse(fail("missing monomorphic struct"))
    assertEquals(renamed.members.map(_.symbol), List("Box::value"))
    val selected = out.entry.body.collect {
      case p.Stmt.Return(p.Expr.Alias(p.Term.Select(_, List(p.PathStep.Field(name)), _))) => name
    }
    assertEquals(selected, List("Box::value"))
  }

  test("monomorphises generic structs reached only through another struct") {
    val innerName = sym("Inner")
    val outerName = sym("Outer")
    val inner = p.StructDef(
      innerName,
      List("T"),
      List(named("Inner::value", p.Type.Var("T"))),
      Nil
    )
    val outer = p.StructDef(
      outerName,
      List("T"),
      List(named("Outer::inner", p.Type.Struct(innerName, List(p.Type.Var("T"))))),
      Nil
    )
    val outerInt = p.Type.Struct(outerName, List(p.Type.IntS32))

    val (_, out) = MonoStruct(program(entry(args = List(arg("outer", outerInt))), defs = List(inner, outer)), NoopLog)

    assertEquals(out.defs.flatMap(_.collectAll[p.Type].collect { case x: p.Type.Var => x }), Nil)
    assert(out.defs.exists(_.name.repr == p.Type.Struct(innerName, List(p.Type.IntS32)).monomorphicName))
  }

  test("retains non-generic inherited struct identities") {
    val baseName  = sym("Base")
    val childName = sym("Child")
    val leafName  = sym("Leaf")
    val base      = p.StructDef(baseName, Nil, List(named("x", p.Type.IntS32)), Nil)
    val child = p.StructDef(
      childName,
      Nil,
      List(named("#base_Base", p.Type.Struct(baseName, Nil))),
      List(p.Type.Struct(baseName, Nil))
    )
    val leaf = p.StructDef(
      leafName,
      Nil,
      List(named("#base_Child", p.Type.Struct(childName, Nil))),
      List(p.Type.Struct(childName, Nil))
    )

    val (_, out) = MonoStruct(
      program(entry(args = List(arg("leaf", p.Type.Struct(leafName, Nil)))), defs = List(base, child, leaf)),
      NoopLog
    )

    assertEquals(out.defs.map(_.name).toSet, Set(baseName, childName, leafName))
    assertEquals(out.defs.find(_.name == childName).map(_.members), Some(child.members))
  }

  test("substitutes nested generic arguments in parent types") {
    val boxName    = sym("Box")
    val parentName = sym("Parent")
    val childName  = sym("Child")
    val box        = p.StructDef(boxName, List("T"), List(named("value", p.Type.Var("T"))), Nil)
    val parent = p.StructDef(
      parentName,
      List("T"),
      List(named("value", p.Type.Var("T"))),
      Nil
    )
    val child = p.StructDef(
      childName,
      List("T"),
      Nil,
      List(p.Type.Struct(parentName, List(p.Type.Struct(boxName, List(p.Type.Var("T"))))))
    )
    val childInt = p.Type.Struct(childName, List(p.Type.IntS32))

    val (_, out) = MonoStruct(
      program(entry(args = List(arg("child", childInt))), defs = List(box, parent, child)),
      NoopLog
    )

    assertEquals(out.defs.flatMap(_.collectAll[p.Type].collect { case x: p.Type.Var => x }), Nil)
  }

  test("rejects expanding polymorphic struct recursion with a bounded diagnostic") {
    val nodeName = sym("Node")
    val boxName  = sym("Box")
    val nested = p.Type.Struct(
      nodeName,
      List(p.Type.Struct(boxName, List(p.Type.Var("T"))))
    )
    val node = p.StructDef(
      nodeName,
      List("T"),
      List(named("next", p.Type.Ptr(nested, p.Type.Space.Private))),
      Nil
    )
    val nodeInt = p.Type.Struct(nodeName, List(p.Type.IntS32))

    val error = intercept[IllegalStateException] {
      MonoStruct(program(entry(args = List(arg("node", nodeInt))), defs = List(node)), NoopLog)
    }

    assert(error.getMessage.contains("polymorphic recursion"))
  }

  test("allows many independent instantiations of one generic struct") {
    val boxName = sym("ManyBox")
    val box     = p.StructDef(boxName, List("T"), List(named("value", p.Type.Var("T"))), Nil)
    val uses = List.tabulate(65) { index =>
      arg(s"box$index", p.Type.Struct(boxName, List(p.Type.Struct(sym(s"Value$index"), Nil))))
    }

    val (_, out) = MonoStruct(program(entry(args = uses), defs = List(box)), NoopLog)

    assertEquals(out.defs.count(_.name != boxName), 65)
  }

  test("allows type-changing recursion that stabilises at a concrete instantiation") {
    val nodeName = sym("StableNode")
    val boxedInt = p.Type.Struct(sym("StableBox"), List(p.Type.IntS32))
    val node = p.StructDef(
      nodeName,
      List("T"),
      List(named("next", p.Type.Ptr(p.Type.Struct(nodeName, List(boxedInt)), p.Type.Space.Private))),
      Nil
    )

    val (_, out) = MonoStruct(
      program(entry(args = List(arg("node", p.Type.Struct(nodeName, List(p.Type.IntS32))))), defs = List(node)),
      NoopLog
    )

    assertEquals(out.defs.size, 2)
  }

  test("generic uses in retained concrete structs seed the monomorphic closure") {
    val boxName    = sym("RetainedBox")
    val holderName = sym("RetainedHolder")
    val box        = p.StructDef(boxName, List("T"), List(named("value", p.Type.Var("T"))), Nil)
    val holder = p.StructDef(
      holderName,
      Nil,
      List(named("box", p.Type.Struct(boxName, List(p.Type.IntS32)))),
      Nil
    )

    val (_, out) = MonoStruct(program(entry(), defs = List(box, holder)), NoopLog)

    assert(out.defs.exists(_.name == holderName))
    assertEquals(out.defs.size, 2)
    assertEquals(out.defs.flatMap(_.collectAll[p.Type].collect { case p.Type.Struct(`boxName`, _) => () }), Nil)
  }

  test("does not substitute type variables shadowed by callable binders") {
    val holderName = sym("CallableHolder")
    val callable   = p.Type.Exec(List("T"), List(p.Type.Var("T")), p.Type.Var("T"))
    val holder = p.StructDef(
      holderName,
      List("T"),
      List(named("callback", callable), named("value", p.Type.Var("T"))),
      Nil
    )

    val (_, out) = MonoStruct(
      program(entry(args = List(arg("holder", p.Type.Struct(holderName, List(p.Type.IntS32))))), defs = List(holder)),
      NoopLog
    )

    val specialised = out.defs.find(_.name != holderName).getOrElse(fail("missing specialised holder"))
    assertEquals(specialised.members.map(_.tpe), List(callable, p.Type.IntS32))
  }
}
