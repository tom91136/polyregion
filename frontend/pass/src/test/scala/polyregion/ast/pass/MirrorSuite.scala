package polyregion.ast.pass

import polyregion.ast.PolyAST as p
import polyregion.ast.PolyAST.Conventions.RuntimeAbi

class MirrorSuite extends munit.FunSuite {
  import PassTest.*

  test("emits host-affinity mirror/unmirror for a scalar-only capture") {
    val in  = program(entry(args = List(arg("capture", p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global)))))
    val out = Mirror()(in, PassTest.NoopLog)

    val added = out.functions.filterNot(in.functions.contains)
    assertEquals(added.map(_.name.fqn.last).toSet, Set(Mirror.PreludeName, Mirror.PostludeName))

    val mirror   = out.functions.find(_.name.fqn.last == Mirror.PreludeName).get
    val unmirror = out.functions.find(_.name.fqn.last == Mirror.PostludeName).get

    for (f <- List(mirror, unmirror)) {
      assertEquals(f.affinity, p.Function.Affinity.Host)
      assertEquals(f.visibility, p.Function.Visibility.Exported)
      assertEquals(f.args.map(_.named.tpe), List(p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global), p.Type.IntU64))
    }
    assertEquals(mirror.rtn, p.Type.IntU64)
    assertEquals(unmirror.rtn, p.Type.Unit0)

    assertEquals(out.entry, in.entry)
    assertEquals(out.offloadFunctions.map(_.name), in.functions.map(_.name))

    def foreignNames(f: p.Function): Set[String] =
      f.body.flatMap {
        case p.Stmt.Var(_, Some(p.Expr.ForeignCall(n, _, _)), _) => List(n)
        case _                                                   => Nil
      }.toSet
    assertEquals(foreignNames(mirror), Set(RuntimeAbi.SmaAlloc))
    assertEquals(foreignNames(unmirror), Set(RuntimeAbi.SmaReadAlloc))
  }

  test("mirrors pointers nested inside by-value structs") {
    val g       = p.Type.Space.Global
    val i32p    = p.Type.Ptr(p.Type.IntS32, g)
    val vecImpl = p.StructDef(sym("vec_impl"), Nil, List(named("begin", i32p), named("end", i32p)), Nil)
    val vec     = p.StructDef(sym("vec"), Nil, List(named("impl", p.Type.Struct(sym("vec_impl"), Nil))), Nil)
    val capture = p.StructDef(sym("Capture"), Nil, List(named("value", p.Type.Struct(sym("vec"), Nil))), Nil)

    val in = program(
      entry(args = List(arg(p.Conventions.ThisReceiver, p.Type.Ptr(p.Type.Struct(sym("Capture"), Nil), g)))),
      defs = List(capture, vec, vecImpl)
    )
    val out = Mirror()(in, PassTest.NoopLog)

    def foreignCalls(f: p.Function): List[String] = {
      def go(ss: List[p.Stmt]): List[String] = ss.flatMap {
        case p.Stmt.Var(_, Some(p.Expr.ForeignCall(n, _, _)), _) => List(n)
        case p.Stmt.Cond(_, t, e)                                => go(t) ++ go(e)
        case _                                                   => Nil
      }
      go(f.body)
    }
    val mirror   = out.functions.find(_.name.fqn.last == Mirror.PreludeName).get
    val unmirror = out.functions.find(_.name.fqn.last == Mirror.PostludeName).get
    val m        = foreignCalls(mirror)
    val u        = foreignCalls(unmirror)

    assertEquals(m.count(_ == RuntimeAbi.SmaAlloc), 1)
    assertEquals(m.count(_.startsWith(RuntimeAbi.SmaEnsure)), 2)
    assertEquals(m.count(_ == RuntimeAbi.SmaPatch), 2)
    assertEquals(u.count(_ == RuntimeAbi.SmaReadAlloc), 2)
    assertEquals(u.count(_ == RuntimeAbi.SmaVisitClear), 1)
  }

  test("uses ensure_deep for multi-indirection (array-of-pointers) capture fields") {
    val g     = p.Type.Space.Global
    val i32   = p.Type.IntS32
    val pp    = p.Type.Ptr(p.Type.Ptr(i32, g), g)
    val ppp   = p.Type.Ptr(p.Type.Ptr(p.Type.Ptr(i32, g), g), g)
    val cap   = p.StructDef(sym("Capture"), Nil, List(named("pp", pp), named("ppp", ppp)), Nil)
    val thisN = named(p.Conventions.ThisReceiver, p.Type.Ptr(p.Type.Struct(sym("Capture"), Nil), g))
    val in    = program(entry(args = List(p.Arg(thisN))), defs = List(cap))
    val out   = Mirror()(in, PassTest.NoopLog)

    def calls(f: p.Function): List[(String, Long)] = {
      def go(ss: List[p.Stmt]): List[(String, Long)] = ss.flatMap {
        case p.Stmt.Var(_, Some(p.Expr.ForeignCall(n, args, _)), _) =>
          val depth = args.collectFirst { case p.Term.IntU64Const(d) => d }.getOrElse(-1L)
          List(n -> depth)
        case p.Stmt.Cond(_, t, e) => go(t) ++ go(e)
        case _                    => Nil
      }
      go(f.body)
    }
    val m    = calls(out.functions.find(_.name.fqn.last == Mirror.PreludeName).get)
    val deep = m.filter(_._1 == RuntimeAbi.SmaEnsureDeep).map(_._2).sorted
    assertEquals(deep, List(1L, 2L))
  }
}
