package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import PassTest.*

class ArithmeticGuardSuite extends munit.FunSuite {

  private val g    = p.Type.Space.Global
  private val errT = p.Type.Ptr(p.Type.IntS8, g)

  private def ret = p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))

  private def intr(op: (p.Term, p.Term, p.Type) => p.Intr)(x: p.Named, y: p.Named): p.Expr =
    p.Expr.IntrOp(op(selectT(x), selectT(y), x.tpe))

  private def program(tpe: p.Type, expr: (p.Named, p.Named) => p.Expr, args: List[p.Arg]): p.Program = {
    val out    = named("out", p.Type.Ptr(tpe, g))
    val x      = named("x", tpe)
    val y      = named("y", tpe)
    val result = named("r", tpe)
    PassTest.program(
      entry(
        args = p.Arg(out) :: args,
        body = List(
          p.Stmt.Var(result, Some(expr(x, y))),
          p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), selectT(result)),
          ret
        )
      )
    )
  }

  private def lowered(tpe: p.Type, op: (p.Term, p.Term, p.Type) => p.Intr): p.Program = {
    val x = named("x", tpe)
    val y = named("y", tpe)
    StructuredExit(ArithmeticGuard(program(tpe, intr(op), List(p.Arg(x), p.Arg(y))), NoopLog), NoopLog)
  }

  private def run(tpe: p.Type, op: (p.Term, p.Term, p.Type) => p.Intr, x: Long, y: Long): (Long, Int, String) = {
    val out = lowered(tpe, op)
    val vm  = Interpreter.Vm(out)
    val dst = vm.alloc(8L)
    val err = vm.alloc(4L + p.Conventions.assertMessageLimit)
    vm.store(dst, V.I(0L), tpe)
    val args = out.entry.required.args.map { arg =>
      arg.named.symbol match {
        case p.Conventions.ErrorArg => errT          -> V.I(err)
        case "out"                  => arg.named.tpe -> V.I(dst)
        case "x"                    => tpe           -> V.I(x)
        case "y"                    => tpe           -> V.I(y)
        case other                  => fail(s"unexpected argument $other")
      }
    }
    vm.call(p.Conventions.EntryName, args)
    val code = vm.load(err, p.Type.IntS32) match { case V.I(v) => v.toInt; case _ => 0 }
    val message = String(
      Iterator
        .from(4)
        .map(o => vm.load(err + o, p.Type.IntU8) match { case V.I(v) => v.toInt; case _ => 0 })
        .takeWhile(_ != 0)
        .map(_.toByte)
        .toArray,
      java.nio.charset.StandardCharsets.UTF_8
    )
    val value = vm.load(dst, tpe) match { case V.I(v) => v; case _ => fail("non-integral result") }
    (value, code, message)
  }

  private val Div = (x: p.Term, y: p.Term, t: p.Type) => p.Intr.Div(x, y, t)
  private val Rem = (x: p.Term, y: p.Term, t: p.Type) => p.Intr.Rem(x, y, t)

  private val signed = List(
    p.Type.IntS8  -> Byte.MinValue.toLong,
    p.Type.IntS16 -> Short.MinValue.toLong,
    p.Type.IntS32 -> Int.MinValue.toLong,
    p.Type.IntS64 -> Long.MinValue
  )

  signed.foreach { (tpe, min) =>
    test(s"${tpe.repr} division by -1 wraps and the remainder is zero") {
      assertEquals(run(tpe, Div, min, -1), (min, 0, ""))
      assertEquals(run(tpe, Rem, min, -1), (0L, 0, ""))
      assertEquals(run(tpe, Div, 7, -1), (-7L, 0, ""))
      assertEquals(run(tpe, Rem, 7, -1), (0L, 0, ""))
    }

    test(s"${tpe.repr} division and remainder by zero raise an arithmetic assert") {
      val arithmetic = p.Enums.AssertCode.Arithmetic.value
      assertEquals(run(tpe, Div, 7, 0), (0L, arithmetic, "/ by zero"))
      assertEquals(run(tpe, Rem, 7, 0), (0L, arithmetic, "/ by zero"))
      assertEquals(run(tpe, Div, min, 0), (0L, arithmetic, "/ by zero"))
    }

    test(s"${tpe.repr} other divisors are left to the plain operation") {
      assertEquals(run(tpe, Div, -7, 2), (-3L, 0, ""))
      assertEquals(run(tpe, Rem, -7, 2), (-1L, 0, ""))
      assertEquals(run(tpe, Div, min, 2), (min / 2, 0, ""))
    }
  }

  test("unsigned and floating point division are untouched") {
    val u = named("u", p.Type.IntU32)
    val f = named("f", p.Type.Float64)
    val body = List(
      p.Stmt.Var(named("a", p.Type.IntU32), Some(intr(Div)(u, u))),
      p.Stmt.Var(named("b", p.Type.Float64), Some(intr(Div)(f, f))),
      ret
    )
    val in = PassTest.program(entry(args = List(p.Arg(u), p.Arg(f)), body = body))
    assertEquals(ArithmeticGuard(in, NoopLog).entry, in.entry)
  }

  test("guarding twice leaves the first result unchanged") {
    val x   = named("x")
    val y   = named("y")
    val in  = PassTest.program(entry(args = List(p.Arg(x), p.Arg(y)), body = List(p.Stmt.Return(intr(Div)(x, y)))))
    val one = ArithmeticGuard(in, NoopLog)
    assertEquals(ArithmeticGuard(one, NoopLog), one)
  }

  test("a division in a return, an update, and nested control flow is guarded") {
    val x = named("x")
    val y = named("y")
    val r = named("r")
    val body = List(
      p.Stmt.Var(r, Some(p.Expr.Alias(p.Term.IntS32Const(0))), isMutable = true),
      p.Stmt.Cond(
        p.Term.Bool1Const(true),
        List(p.Stmt.Mut(selectT(r), intr(Rem)(x, y))),
        Nil
      ),
      p.Stmt.Return(intr(Div)(x, y))
    )
    val in    = PassTest.program(entry(args = List(p.Arg(x), p.Arg(y)), body = body))
    val out   = ArithmeticGuard(in, NoopLog).entry.required
    val plain = out.collectWhere[p.Expr] { case e @ p.Expr.IntrOp(p.Intr.Div(_, _, _) | p.Intr.Rem(_, _, _)) => e }
    assertEquals(plain.size, 2)
    val asserts = out.collectWhere[p.Expr] { case e @ p.Expr.SpecOp(p.Spec.Assert(_, _)) => e }
    assertEquals(asserts.size, 2)
  }
}
