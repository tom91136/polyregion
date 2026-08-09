package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class PartialEvalAliasSuite extends munit.FunSuite {

  private val sTpe     = p.Type.Struct(sym("S"), Nil)
  private val innerTpe = p.Type.Struct(sym("Inner"), Nil)

  private def ptrTo(t: p.Type): p.Type = p.Type.Ptr(t, p.Type.Space.Private)

  private def refTo(that: p.Term.Select): p.Expr =
    p.Expr.RefTo(that, None, that.tpe, p.Type.Space.Private, p.Region.Rooted(that.root))

  private def pe(body: List[p.Stmt], rtn: p.Type = p.Type.Unit0, args: List[p.Arg] = Nil): List[p.Stmt] =
    PartialEval()(program(entry(args = args, body = body).copy(rtn = rtn)), NoopLog).entry.body

  private def fx(root: p.Named): p.Term.Select = fieldOf(root)("x", p.Type.IntS32)

  private def returnedTerm(body: List[p.Stmt]): Option[p.Term] = body.collectFirst {
    case p.Stmt.Return(p.Expr.Alias(t)) => t
  }

  private def decls(body: List[p.Stmt]): List[p.Stmt] = body.collectWhere[p.Stmt] { case v: p.Stmt.Var => v }

  test("read of a field through a pointer-to-local becomes a read of the pointee") {
    val s = named("s", sTpe)
    val q = named("q", ptrTo(sTpe))
    val out = pe(
      List(
        p.Stmt.Var(s, None, isMutable = true),
        p.Stmt.Var(q, Some(refTo(selectT(s)))),
        p.Stmt.Return(p.Expr.Alias(fx(q)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(returnedTerm(out), Some(fx(s)))
  }

  test("write of a field through a pointer-to-local becomes a write to the pointee") {
    val s = named("s", sTpe)
    val q = named("q", ptrTo(sTpe))
    val out = pe(
      List(
        p.Stmt.Var(s, None, isMutable = true),
        p.Stmt.Var(q, Some(refTo(selectT(s)))),
        p.Stmt.Mut(fx(q), p.Expr.Alias(p.Term.IntS32Const(5))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    assertEquals(
      out.collectFirst { case p.Stmt.Mut(lhs, _) => lhs },
      Some(fx(s))
    )
  }

  test("pointer to a nested field keeps the base steps ahead of the forwarded step") {
    val s = named("s", sTpe)
    val q = named("q", ptrTo(innerTpe))
    val out = pe(
      List(
        p.Stmt.Var(s, None, isMutable = true),
        p.Stmt.Var(q, Some(refTo(fieldOf(s)("inner", innerTpe)))),
        p.Stmt.Return(p.Expr.Alias(fieldOf(q)("y", p.Type.IntS32)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(
      returnedTerm(out),
      Some(p.Term.Select(s, List(p.PathStep.Field("inner"), p.PathStep.Field("y")), p.Type.IntS32))
    )
  }

  test("bare use of a pointer-to-local is left for DCE") {
    val s = named("s", sTpe)
    val q = named("q", ptrTo(sTpe))
    val body = List(
      p.Stmt.Var(s, None, isMutable = true),
      p.Stmt.Var(q, Some(refTo(selectT(s)))),
      p.Stmt.Return(p.Expr.Alias(selectT(q)))
    )
    assertEquals(pe(body, rtn = ptrTo(sTpe)), body)
  }

  test("pointer into a re-aimed base is not forwarded") {
    val x = named("x", ptrTo(sTpe))
    val y = named("y", ptrTo(sTpe))
    val q = named("q", ptrTo(innerTpe))
    val body = List(
      p.Stmt.Var(q, Some(refTo(fieldOf(x)("inner", innerTpe)))),
      p.Stmt.Mut(selectT(x), p.Expr.Alias(selectT(y))),
      p.Stmt.Return(p.Expr.Alias(fieldOf(q)("z", p.Type.IntS32)))
    )
    assertEquals(pe(body, rtn = p.Type.IntS32, args = List(p.Arg(x), p.Arg(y))), body)
  }

  test("alias of a struct forwards into a field read and the decl is dropped") {
    val a = named("a", sTpe)
    val b = named("b", sTpe)
    val out = pe(
      List(
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
        p.Stmt.Return(p.Expr.Alias(fx(b)))
      ),
      rtn = p.Type.IntS32,
      args = List(p.Arg(a))
    )
    assertEquals(returnedTerm(out), Some(fx(a)))
    assertEquals(decls(out), Nil)
  }

  test("chained aliases resolve to the root in one run") {
    val a = named("a", sTpe)
    val b = named("b", sTpe)
    val c = named("c", sTpe)
    val out = pe(
      List(
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
        p.Stmt.Var(c, Some(p.Expr.Alias(selectT(b)))),
        p.Stmt.Return(p.Expr.Alias(fx(c)))
      ),
      rtn = p.Type.IntS32,
      args = List(p.Arg(a))
    )
    assertEquals(returnedTerm(out), Some(fx(a)))
    assertEquals(decls(out), Nil)
  }

  test("alias of a reassigned root is not forwarded") {
    val a = named("a", sTpe)
    val z = named("z", sTpe)
    val b = named("b", sTpe)
    val body = List(
      p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
      p.Stmt.Mut(selectT(a), p.Expr.Alias(selectT(z))),
      p.Stmt.Return(p.Expr.Alias(fx(b)))
    )
    assertEquals(pe(body, rtn = p.Type.IntS32, args = List(p.Arg(a), p.Arg(z))), body)
  }

  test("alias of a root written through a pointer is not forwarded") {
    val a = named("a", sTpe)
    val b = named("b", sTpe)
    val q = named("q", ptrTo(sTpe))
    val out = pe(
      List(
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
        p.Stmt.Var(q, Some(refTo(selectT(a)))),
        p.Stmt.Mut(fx(q), p.Expr.Alias(p.Term.IntS32Const(5))),
        p.Stmt.Return(p.Expr.Alias(fx(b)))
      ),
      rtn = p.Type.IntS32,
      args = List(p.Arg(a))
    )
    assertEquals(returnedTerm(out), Some(fx(b)))
    assertEquals(out.collectFirst { case p.Stmt.Mut(lhs, _) => lhs }, Some(fx(a)))
  }

  test("mutable alias binding is not forwarded") {
    val a = named("a", sTpe)
    val b = named("b", sTpe)
    val body = List(
      p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a))), isMutable = true),
      p.Stmt.Return(p.Expr.Alias(fx(b)))
    )
    assertEquals(pe(body, rtn = p.Type.IntS32, args = List(p.Arg(a))), body)
  }

  test("alias declared inside a branch is forwarded and dropped there") {
    val a = named("a", sTpe)
    val b = named("b", sTpe)
    val c = named("c", p.Type.Bool1)
    val out = pe(
      List(
        p.Stmt.Cond(
          selectT(c),
          List(
            p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
            p.Stmt.Return(p.Expr.Alias(fx(b)))
          ),
          Nil
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(0)))
      ),
      rtn = p.Type.IntS32,
      args = List(p.Arg(a), p.Arg(c))
    )
    assertEquals(
      out.collectFirst { case p.Stmt.Cond(_, t, _) => t },
      Some(List(p.Stmt.Return(p.Expr.Alias(fx(a)))))
    )
  }

  private def structDef(name: String, members: (String, p.Type)*): p.StructDef =
    p.StructDef(sym(name), Nil, members.toList.map((n, t) => named(n, t)), Nil)

  private def peDefs(
      body: List[p.Stmt],
      defs: List[p.StructDef],
      rtn: p.Type,
      args: List[p.Arg]
  ): List[p.Stmt] =
    PartialEval()(program(entry(args = args, body = body).copy(rtn = rtn), Nil, defs), NoopLog).entry.body

  private def reinterpret(src: p.Named, dst: p.Named, field: String, srcDef: p.StructDef, dstDef: p.StructDef) = {
    val fieldTpe = dstDef.members.find(_.symbol == field).get.tpe
    peDefs(
      List(
        p.Stmt.Var(dst, Some(p.Expr.Cast(selectT(src), dst.tpe))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Select(dst, List(p.PathStep.Field(field)), fieldTpe)))
      ),
      List(srcDef, dstDef),
      fieldTpe,
      List(p.Arg(src))
    )
  }

  test("a reinterpret forwards a field read to the source field at the same ordinal") {
    val srcDef = structDef("Src", "a" -> p.Type.IntS32, "b" -> p.Type.IntS32)
    val dstDef = structDef("Dst", "x" -> p.Type.IntS32, "y" -> p.Type.IntS32)
    val s      = named("s", p.Type.Struct(sym("Src"), Nil))
    val d      = named("d", p.Type.Struct(sym("Dst"), Nil))
    assertEquals(
      returnedTerm(reinterpret(s, d, "y", srcDef, dstDef)),
      Some(p.Term.Select(s, List(p.PathStep.Field("b")), p.Type.IntS32))
    )
  }

  test("a reinterpret does not forward when a preceding member differs in type") {
    val srcDef = structDef("Src", "a" -> p.Type.IntS16, "b" -> p.Type.IntS16, "c" -> p.Type.IntS32)
    val dstDef = structDef("Dst", "x" -> p.Type.IntS32, "y" -> p.Type.IntS32, "z" -> p.Type.IntS32)
    val s      = named("s", p.Type.Struct(sym("Src"), Nil))
    val d      = named("d", p.Type.Struct(sym("Dst"), Nil))
    assertEquals(
      returnedTerm(reinterpret(s, d, "z", srcDef, dstDef)),
      Some(p.Term.Select(d, List(p.PathStep.Field("z")), p.Type.IntS32))
    )
  }

  test("a reinterpret does not forward through a union") {
    val srcDef =
      p.StructDef(sym("Src"), Nil, List(named("a", p.Type.IntS32), named("b", p.Type.IntS32)), Nil, isUnion = true)
    val dstDef = structDef("Dst", "x" -> p.Type.IntS32, "y" -> p.Type.IntS32)
    val s      = named("s", p.Type.Struct(sym("Src"), Nil))
    val d      = named("d", p.Type.Struct(sym("Dst"), Nil))
    assertEquals(
      returnedTerm(reinterpret(s, d, "y", srcDef, dstDef)),
      Some(p.Term.Select(d, List(p.PathStep.Field("y")), p.Type.IntS32))
    )
  }

  test("a reinterpret does not forward when the bases differ") {
    val srcDef = p.StructDef(
      sym("Src"),
      Nil,
      List(named("a", p.Type.IntS32), named("b", p.Type.IntS32)),
      List(p.Type.Struct(sym("Base"), Nil))
    )
    val dstDef = structDef("Dst", "x" -> p.Type.IntS32, "y" -> p.Type.IntS32)
    val s      = named("s", p.Type.Struct(sym("Src"), Nil))
    val d      = named("d", p.Type.Struct(sym("Dst"), Nil))
    assertEquals(
      returnedTerm(reinterpret(s, d, "y", srcDef, dstDef)),
      Some(p.Term.Select(d, List(p.PathStep.Field("y")), p.Type.IntS32))
    )
  }
}
