package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class SourceNameNormaliseSuite extends munit.FunSuite {

  test("definitions and their references receive the same collision-free source names") {
    val symbol = sym("ns", "Box")
    val tpe    = p.Type.Struct(symbol, Nil)
    val root   = named("root", tpe)
    val definition = p.StructDef(
      symbol,
      Nil,
      List(named("kernel"), named("a-b"), named("a_b")),
      Nil
    )
    val kernel = entry(
      args = List(p.Arg(root)),
      body = List(
        p.Stmt
          .Var(named("value"), Some(p.Expr.Alias(p.Term.Select(root, List(p.PathStep.Field("a-b")), p.Type.IntS32)))),
        p.Stmt.Var(named("offset", p.Type.IntU64), Some(p.Expr.OffsetOf(tpe, "a_b"))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out    = SourceNameNormalise()(program(kernel, defs = List(definition)), NoopLog)
    val result = out.defs.head
    assertEquals(result.name.fqcn, "ns_Box")
    assertEquals(result.members.map(_.symbol), List("_kernel", "a_b", "a_b_1"))
    assertEquals(
      out.entry.required.collectAll[p.Term].flatMap {
        case p.Term.Select(_, steps, _) => steps.collect { case p.PathStep.Field(name) => name }
        case _                          => Nil
      },
      List("a_b")
    )
    assertEquals(
      out.entry.required.collectAll[p.Expr].collect { case p.Expr.OffsetOf(_, field) => field },
      List("a_b_1")
    )
  }

  test("host abi metadata follows the renamed entry") {
    val kernel = fn("ns.kernel", convention = p.CallConvention.OffloadEntry)
    val other  = p.MetaEntry(p.Conventions.HostAbiMeta + "elsewhere", "2")
    val input = program(kernel).copy(metadata =
      List(
        p.MetaEntry(p.Conventions.HostAbiMeta + "ns.kernel", "2"),
        p.MetaEntry(p.Conventions.HostAbiNoteMeta + "ns.kernel", "spin"),
        other
      )
    )
    val out = SourceNameNormalise()(input, NoopLog)
    assertEquals(out.entry.required.name.fqcn, "ns_kernel")
    assertEquals(
      out.metadata,
      List(
        p.MetaEntry(p.Conventions.HostAbiMeta + "ns_kernel", "2"),
        p.MetaEntry(p.Conventions.HostAbiNoteMeta + "ns_kernel", "spin"),
        other
      )
    )
  }

  test("Metal keywords are target-selective") {
    val symbol     = sym("thread")
    val definition = p.StructDef(symbol, Nil, Nil, Nil)
    assertEquals(SourceNameNormalise()(program(None, List(definition)), NoopLog).defs.head.name.fqcn, "thread")
    assertEquals(
      SourceNameNormalise(metalKeywords = true)(program(None, List(definition)), NoopLog).defs.head.name.fqcn,
      "_thread"
    )
  }
}
