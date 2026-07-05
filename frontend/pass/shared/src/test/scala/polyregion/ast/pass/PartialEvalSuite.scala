package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class PartialEvalSuite extends munit.FunSuite {

  private def pe(body: List[p.Stmt], rtn: p.Type = p.Type.Unit0, args: List[p.Arg] = Nil): List[p.Stmt] =
    PartialEval()(program(entry(args = args, body = body).copy(rtn = rtn)), NoopLog).entry.body

  private def aliasOf(s: p.Stmt): Option[p.Term] = s match {
    case p.Stmt.Var(_, Some(p.Expr.Alias(t)), _) => Some(t)
    case _                                       => None
  }

  private def returnedTerm(body: List[p.Stmt]): Option[p.Term] = body.collectFirst {
    case p.Stmt.Return(p.Expr.Alias(t)) => t
  }

  // --- constant folding ---

  test("fold i32(100) - i32(1) -> i32(99) at the binding") {
    val v = named("v", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.Sub(p.Term.IntS32Const(100), p.Term.IntS32Const(1), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(returnedTerm(out), Some(p.Term.IntS32Const(99)))
  }

  test("propagate const through a subsequent IntrOp to the return") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10)))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(5), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(returnedTerm(out), Some(p.Term.IntS32Const(15)))
  }

  test("integer divide-by-zero is preserved (residualised for a runtime trap)") {
    val v        = named("v", p.Type.IntS32)
    val expected = p.Expr.IntrOp(p.Intr.Div(p.Term.IntS32Const(10), p.Term.IntS32Const(0), p.Type.IntS32))
    val out      = pe(List(p.Stmt.Var(v, Some(expected)), p.Stmt.Return(p.Expr.Alias(selectT(v)))), rtn = p.Type.IntS32)
    val actual   = out.collectFirst { case s: p.Stmt.Var if s.name == v => s.expr }
    assertEquals(actual, Some(Some(expected)))
  }

  test("fold float arithmetic") {
    val v = named("v", p.Type.Float64)
    val out = pe(
      List(
        p.Stmt
          .Var(v, Some(p.Expr.IntrOp(p.Intr.Mul(p.Term.Float64Const(2.5), p.Term.Float64Const(4.0), p.Type.Float64)))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.Float64
    )
    assertEquals(returnedTerm(out), Some(p.Term.Float64Const(10.0)))
  }

  test("fold logical comparison and a cast") {
    val v = named("v", p.Type.IntS64)
    val out = pe(
      List(
        p.Stmt.Var(v, Some(p.Expr.Cast(p.Term.IntS32Const(7), p.Type.IntS64))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      rtn = p.Type.IntS64
    )
    assertEquals(returnedTerm(out), Some(p.Term.IntS64Const(7L)))
  }

  // --- copy / alias propagation ---

  test("copy-propagate a bare alias and drop the dead decl") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("a", p.Type.IntS32))
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(selectT(a)))))
  }

  test("copy-propagate a field path, concatenating steps") {
    val a     = named("a", p.Type.IntS32)
    val b     = named("b", p.Type.IntS32)
    val aDotP = fieldOf(a)("p", p.Type.IntS32)
    val bDotQ = p.Term.Select(b, List(p.PathStep.Field("q")), p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(b, Some(p.Expr.Alias(aDotP))),
        p.Stmt.Return(p.Expr.Alias(bDotQ))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("a", p.Type.IntS32))
    )
    // b.q -> a.p.q
    assertEquals(
      returnedTerm(out),
      Some(p.Term.Select(a, List(p.PathStep.Field("p"), p.PathStep.Field("q")), p.Type.IntS32))
    )
  }

  // --- dead pure-binding elimination ---

  test("drop an unreferenced unit alias binding") {
    val u = named("u", p.Type.Unit0)
    val g = named("g", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(u, Some(p.Expr.Alias(p.Term.Unit0Const))),
        p.Stmt.Var(
          g,
          Some(
            p.Expr.Index(
              selectT("buf", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)),
              p.Term.IntS32Const(0),
              p.Type.IntS32
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(selectT(g)))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("buf", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)))
    )
    assert(!out.exists { case p.Stmt.Var(n, _, _) => n == u; case _ => false }, out.map(_.repr).mkString("\n"))
  }

  test("keep a dead binding whose initialiser has effects (Invoke)") {
    val u   = named("u", p.Type.Unit0)
    val ivk = p.Expr.Invoke(sym("f"), Nil, None, Nil, p.Type.Unit0)
    val out = pe(
      List(p.Stmt.Var(u, Some(ivk)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
    )
    assert(out.exists { case p.Stmt.Var(n, Some(`ivk`), _) => n == u; case _ => false }, out.map(_.repr).mkString("\n"))
  }

  // --- static control-flow pruning ---

  test("Cond(true) keeps the true branch, Cond(false) keeps the false branch") {
    val v = named("v", p.Type.IntS32)
    def body(c: Boolean) = List(
      p.Stmt.Cond(
        p.Term.Bool1Const(c),
        List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
        List(p.Stmt.Var(v, Some(p.Expr.Alias(p.Term.IntS32Const(2)))))
      ),
      p.Stmt.Return(p.Expr.Alias(selectT(v)))
    )
    assertEquals(returnedTerm(pe(body(true), rtn = p.Type.IntS32)), Some(p.Term.IntS32Const(1)))
    assertEquals(returnedTerm(pe(body(false), rtn = p.Type.IntS32)), Some(p.Term.IntS32Const(2)))
  }

  test("while(false) and empty ForRange are dropped") {
    val out = pe(
      List(
        p.Stmt.While(p.Term.Bool1Const(false), List(p.Stmt.Break)),
        p.Stmt.ForRange(
          named("i", p.Type.IntS64),
          p.Term.IntS64Const(10),
          p.Term.IntS64Const(10),
          p.Term.IntS64Const(1),
          List(p.Stmt.Break)
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    assert(
      out.forall { case _: p.Stmt.While | _: p.Stmt.ForRange => false; case _ => true },
      out.map(_.repr).mkString("\n")
    )
  }

  test("mutable val is not propagated") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10))), isMutable = true),
        p.Stmt.Mut(selectT(a), p.Expr.Alias(p.Term.IntS32Const(20))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(5), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      rtn = p.Type.IntS32
    )
    val bRhs = out.collectFirst { case s: p.Stmt.Var if s.name == b => s }.flatMap(aliasOf)
    assert(bRhs.isEmpty, s"mutable a must not fold into b; got ${out.map(_.repr).mkString("\n")}")
  }

  // --- fusion: PE reaches in one walk what the pass sequence needed ordering for ---

  test("fold a constant through an alias through a statically-taken branch in one pass") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val t = named("t", p.Type.Bool1)
    val r = named("r", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10)))),
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))),                                         // alias b = a
        p.Stmt.Var(t, Some(p.Expr.IntrOp(p.Intr.LogicGt(selectT(b), p.Term.IntS32Const(5))))), // t = b > 5
        p.Stmt.Cond(
          selectT(t),
          List(p.Stmt.Var(r, Some(p.Expr.Alias(p.Term.IntS32Const(1))))),
          List(p.Stmt.Var(r, Some(p.Expr.Alias(p.Term.IntS32Const(2)))))
        ),
        p.Stmt.Return(p.Expr.Alias(selectT(r)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(returnedTerm(out), Some(p.Term.IntS32Const(1)))
    assert(out.forall { case _: p.Stmt.Cond => false; case _ => true }, out.map(_.repr).mkString("\n"))
  }

  // --- address lattice: fold *(&x) and (&x)[0] back to the lvalue ---

  private val ptrI32 = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
  private def addrOf(sel: p.Term.Select): p.Expr =
    p.Expr.RefTo(sel, None, p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque)

  test("deref of a reference: *(&x) folds to x") {
    val x  = named("x", p.Type.IntS32)
    val pp = named("p", ptrI32)
    val out = pe(
      List(
        p.Stmt.Var(pp, Some(addrOf(selectT(x)))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("x", p.Type.IntS32))
    )
    assertEquals(returnedTerm(out), Some(selectT(x)))
  }

  test("index-zero of a reference: (&x)[0] folds to x") {
    val x  = named("x", p.Type.IntS32)
    val pp = named("p", ptrI32)
    val out = pe(
      List(
        p.Stmt.Var(pp, Some(addrOf(selectT(x)))),
        p.Stmt.Return(p.Expr.Index(selectT(pp), p.Term.IntS32Const(0), p.Type.IntS32))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("x", p.Type.IntS32))
    )
    assertEquals(returnedTerm(out), Some(selectT(x)))
  }

  test("deref of &(a.f) folds to a.f, and the dead address-of decl is dropped") {
    val a     = named("a", p.Type.IntS32)
    val pp    = named("p", ptrI32)
    val aDotF = fieldOf(a)("f", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(pp, Some(p.Expr.RefTo(aDotF, None, p.Type.IntS32, p.Type.Space.Global, p.Region.Opaque))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("a", p.Type.IntS32))
    )
    assertEquals(returnedTerm(out), Some(p.Term.Select(a, List(p.PathStep.Field("f")), p.Type.IntS32)))
    assert(!out.exists { case p.Stmt.Var(n, _, _) => n == pp; case _ => false }, out.map(_.repr).mkString("\n"))
  }

  test("reassigned pointer is not address-folded") {
    val x     = named("x", p.Type.IntS32)
    val y     = named("y", p.Type.IntS32)
    val pp    = named("p", ptrI32)
    val deref = p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(pp, Some(addrOf(selectT(x))), isMutable = true),
        p.Stmt.Mut(selectT(pp), addrOf(selectT(y))),
        p.Stmt.Return(p.Expr.Alias(deref))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("x", p.Type.IntS32), arg("y", p.Type.IntS32))
    )
    // p is reassigned -> excluded -> the deref stays a deref of p, not folded to x or y
    assertEquals(returnedTerm(out), Some(deref))
  }

  test("address of a later-mutated root is not address-folded (p keeps its own slot)") {
    // a is mutated after its address is taken; folding *p to a anyway would erase the last RefTo to a
    val a     = named("a", p.Type.IntS32)
    val pp    = named("p", ptrI32)
    val deref = p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(pp, Some(addrOf(selectT(a)))),
        p.Stmt.Mut(selectT(a), p.Expr.Alias(p.Term.IntS32Const(42))),
        p.Stmt.Return(p.Expr.Alias(deref))
      ),
      rtn = p.Type.IntS32,
      args = List(arg("a", p.Type.IntS32))
    )
    // a is mutated after its address is taken -> excluded -> the deref stays a deref of p
    assertEquals(returnedTerm(out), Some(deref))
  }

  // --- canonicaliseAddresses mode: whole-value (&lvalue) field-alias canonicalisation ---

  private def peThenCanonicalise(body: List[p.Stmt], rtn: p.Type, args: List[p.Arg]): (List[p.Stmt], List[p.Stmt]) = {
    val peProg   = PartialEval()(program(entry(args = args, body = body).copy(rtn = rtn)), NoopLog)
    val anchored = PartialEval(canonicaliseAddresses = true)(peProg, NoopLog)
    (peProg.entry.body, anchored.entry.body)
  }

  test("canonicalise mode is a no-op on PE fold output: whole-value deref (&x)[0]") {
    val x = named("x", p.Type.IntS32)
    val v = named("v", ptrI32)
    val (peB, anB) = peThenCanonicalise(
      List(
        p.Stmt.Var(v, Some(addrOf(selectT(x)))),
        p.Stmt.Return(p.Expr.Index(selectT(v), p.Term.IntS32Const(0), p.Type.IntS32))
      ),
      p.Type.IntS32,
      List(arg("x", p.Type.IntS32))
    )
    assertEquals(anB, peB)
  }

  test("canonicalise mode is a no-op on PE fold output: whole-value store (&x)[0] = e") {
    val x = named("x", p.Type.IntS32)
    val v = named("v", ptrI32)
    val (peB, anB) = peThenCanonicalise(
      List(
        p.Stmt.Var(v, Some(addrOf(selectT(x)))),
        p.Stmt.Update(selectT(v), p.Term.IntS32Const(0), p.Term.IntS32Const(5)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      p.Type.Unit0,
      List(arg("x", p.Type.IntS32))
    )
    assertEquals(anB, peB)
    // and PE actually lowered the store to a direct Mut of x
    assert(
      peB.exists { case p.Stmt.Mut(p.Term.Select(`x`, Nil, _), _) => true; case _ => false },
      peB.map(_.repr).mkString("\n")
    )
  }

  test("canonicalise mode is a no-op on PE fold output: deref-then-field (*v).f") {
    val s = named("s", p.Type.Struct(sym("S"), Nil))
    val v = named("v", p.Type.Ptr(p.Type.Struct(sym("S"), Nil), p.Type.Space.Global))
    val (peB, anB) = peThenCanonicalise(
      List(
        p.Stmt.Var(
          v,
          Some(p.Expr.RefTo(selectT(s), None, p.Type.Struct(sym("S"), Nil), p.Type.Space.Global, p.Region.Opaque))
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Select(v, List(p.PathStep.Deref, p.PathStep.Field("f")), p.Type.IntS32)))
      ),
      p.Type.IntS32,
      List(p.Arg(s))
    )
    assertEquals(anB, peB)
    assertEquals(returnedTerm(peB), Some(p.Term.Select(s, List(p.PathStep.Field("f")), p.Type.IntS32)))
  }

  test("the pipeline step PartialEval(canonicaliseAddresses=true) parses and builds (C++/DSO round-trip)") {
    val spec  = PassPipelineParser.parseStep("PartialEval(canonicaliseAddresses=true)")
    val built = spec.flatMap(PassRegistry.build)
    assert(built.isRight, built.toString)
  }

  test("canonicaliseAddresses mode root-anchors a whole-value deref (&x)[0] -> x") {
    val x = named("x", p.Type.IntS32)
    val v = named("v", ptrI32)
    val prog = program(
      entry(
        args = List(arg("x", p.Type.IntS32)),
        body = List(
          p.Stmt.Var(v, Some(addrOf(selectT(x)))),
          p.Stmt.Return(p.Expr.Index(selectT(v), p.Term.IntS32Const(0), p.Type.IntS32))
        )
      ).copy(rtn = p.Type.IntS32)
    )
    val out = PartialEval(canonicaliseAddresses = true)(prog, NoopLog)
    assertEquals(out.entry.body, List(p.Stmt.Return(p.Expr.Alias(selectT(x)))))
  }

  // --- Fold matches the Interpreter on unsigned Div/Rem/BSR (previously divergent) ---

  private def foldedU64(op: p.Expr): Long = {
    val v = named("v", p.Type.IntU64)
    returnedTerm(pe(List(p.Stmt.Var(v, Some(op)), p.Stmt.Return(p.Expr.Alias(selectT(v)))), p.Type.IntU64)) match {
      case Some(p.Term.IntU64Const(x)) => x
      case other                       => fail(s"not folded to a u64 const: $other")
    }
  }
  private def interpU64(op: p.Expr): Long = {
    val v = named("v", p.Type.IntU64)
    val prog = program(
      entry(body = List(p.Stmt.Var(v, Some(op)), p.Stmt.Return(p.Expr.Alias(selectT(v))))).copy(rtn = p.Type.IntU64)
    )
    new Interpreter.Vm(prog).call(p.Conventions.EntryName, Nil) match {
      case Interpreter.V.I(x) => x
      case other              => fail(s"interpreter returned non-int: $other")
    }
  }

  test("unsigned IntU64 Div folds with unsigned semantics, matching the Interpreter") {
    val hi = 0x8000000000000000L // as unsigned: 2^63
    val op = p.Expr.IntrOp(p.Intr.Div(p.Term.IntU64Const(hi), p.Term.IntU64Const(2L), p.Type.IntU64))
    assertEquals(foldedU64(op), 0x4000000000000000L) // signed / would give 0xC000000000000000
    assertEquals(foldedU64(op), interpU64(op))
  }

  test("unsigned IntU64 Rem folds with unsigned semantics, matching the Interpreter") {
    val hi = 0x8000000000000000L
    val op = p.Expr.IntrOp(p.Intr.Rem(p.Term.IntU64Const(hi), p.Term.IntU64Const(3L), p.Type.IntU64))
    assertEquals(foldedU64(op), java.lang.Long.remainderUnsigned(hi, 3L))
    assertEquals(foldedU64(op), interpU64(op))
  }

  test("unsigned IntU64 BSR folds as a logical shift, matching the Interpreter") {
    val hi = 0x8000000000000000L
    val op = p.Expr.IntrOp(p.Intr.BSR(p.Term.IntU64Const(hi), p.Term.IntU64Const(1L), p.Type.IntU64))
    assertEquals(foldedU64(op), 0x4000000000000000L) // arithmetic >> would give 0xC000000000000000
    assertEquals(foldedU64(op), interpU64(op))
  }

  // --- algebraic + boolean identities (safe strength reductions on a dynamic operand) ---

  private def simplifyOf(op: p.Expr, rtn: p.Type, args: List[p.Arg]): Option[p.Term] = {
    val v = named("v", rtn)
    returnedTerm(pe(List(p.Stmt.Var(v, Some(op)), p.Stmt.Return(p.Expr.Alias(selectT(v)))), rtn, args))
  }
  private def i32(n: String)  = arg(n, p.Type.IntS32)
  private def selI(n: String) = selectT(named(n, p.Type.IntS32))

  test("x + 0 -> x") {
    val op = p.Expr.IntrOp(p.Intr.Add(selI("x"), p.Term.IntS32Const(0), p.Type.IntS32))
    assertEquals(simplifyOf(op, p.Type.IntS32, List(i32("x"))), Some(selI("x")))
  }
  test("x * 0 -> 0 (integer)") {
    val op = p.Expr.IntrOp(p.Intr.Mul(selI("x"), p.Term.IntS32Const(0), p.Type.IntS32))
    assertEquals(simplifyOf(op, p.Type.IntS32, List(i32("x"))), Some(p.Term.IntS32Const(0)))
  }
  test("x * 1 -> x") {
    val op = p.Expr.IntrOp(p.Intr.Mul(selI("x"), p.Term.IntS32Const(1), p.Type.IntS32))
    assertEquals(simplifyOf(op, p.Type.IntS32, List(i32("x"))), Some(selI("x")))
  }
  test("x ^ x -> 0 and x & 0 -> 0") {
    val xor = p.Expr.IntrOp(p.Intr.BXor(selI("x"), selI("x"), p.Type.IntS32))
    assertEquals(simplifyOf(xor, p.Type.IntS32, List(i32("x"))), Some(p.Term.IntS32Const(0)))
    val and = p.Expr.IntrOp(p.Intr.BAnd(selI("x"), p.Term.IntS32Const(0), p.Type.IntS32))
    assertEquals(simplifyOf(and, p.Type.IntS32, List(i32("x"))), Some(p.Term.IntS32Const(0)))
  }
  test("boolean identities: true && b -> b, false && b -> false") {
    val b   = named("b", p.Type.Bool1)
    val and = p.Expr.IntrOp(p.Intr.LogicAnd(p.Term.Bool1Const(true), selectT(b)))
    assertEquals(simplifyOf(and, p.Type.Bool1, List(p.Arg(b))), Some(selectT(b)))
    val andF = p.Expr.IntrOp(p.Intr.LogicAnd(p.Term.Bool1Const(false), selectT(b)))
    assertEquals(simplifyOf(andF, p.Type.Bool1, List(p.Arg(b))), Some(p.Term.Bool1Const(false)))
  }
  test("x == x -> true (integer)") {
    val op = p.Expr.IntrOp(p.Intr.LogicEq(selI("x"), selI("x")))
    assertEquals(simplifyOf(op, p.Type.Bool1, List(i32("x"))), Some(p.Term.Bool1Const(true)))
  }
  test("float x + 0.0 is NOT simplified (IEEE signed zero)") {
    val f = named("f", p.Type.Float64)
    val v = named("v", p.Type.Float64)
    val out = pe(
      List(
        p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.Add(selectT(f), p.Term.Float64Const(0.0), p.Type.Float64)))),
        p.Stmt.Return(p.Expr.Alias(selectT(v)))
      ),
      p.Type.Float64,
      List(p.Arg(f))
    )
    assert(
      out.exists { case p.Stmt.Var(`v`, Some(_: p.Expr.IntrOp), _) => true; case _ => false },
      out.map(_.repr).mkString("\n")
    )
  }

  // --- unreachable code after a terminator ---

  test("code after a Return is dropped, even a side-effecting call") {
    val ivk = p.Expr.Invoke(sym("f"), Nil, None, Nil, p.Type.Unit0)
    val out = pe(
      List(
        p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(1))),
        p.Stmt.Var(named("dead", p.Type.Unit0), Some(ivk))
      ),
      p.Type.IntS32
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(1)))))
  }

  // --- identical-branch collapse ---

  test("identical branches collapse: if(c){return x}else{return x} -> return x") {
    val c = named("c", p.Type.Bool1)
    val out = pe(
      List(
        p.Stmt
          .Cond(selectT(c), List(p.Stmt.Return(p.Expr.Alias(selI("x")))), List(p.Stmt.Return(p.Expr.Alias(selI("x")))))
      ),
      p.Type.IntS32,
      List(p.Arg(c), i32("x"))
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(selI("x")))))
  }

  test("empty branches collapse: if(c){}else{} -> drop") {
    val c = named("c", p.Type.Bool1)
    val out = pe(
      List(p.Stmt.Cond(selectT(c), Nil, Nil), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      p.Type.Unit0,
      List(p.Arg(c))
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
  }

  // --- constant reassociation ---

  test("reassociation: (x + 1) + 2 -> x + 3, dead intermediate dropped") {
    val x = named("x", p.Type.IntS32); val a = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.IntrOp(p.Intr.Add(selectT(x), p.Term.IntS32Const(1), p.Type.IntS32)))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      p.Type.IntS32,
      List(i32("x"))
    )
    val bRhs = out.collectFirst { case p.Stmt.Var(`b`, Some(e), _) => e }
    assertEquals(bRhs, Some(p.Expr.IntrOp(p.Intr.Add(selectT(x), p.Term.IntS32Const(3), p.Type.IntS32))))
    assert(!out.exists { case p.Stmt.Var(`a`, _, _) => true; case _ => false }, out.map(_.repr).mkString("\n"))
  }

  test("reassociation: (i * 4) * 2 -> i * 8") {
    val i = named("i", p.Type.IntS32); val a = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.IntrOp(p.Intr.Mul(selectT(i), p.Term.IntS32Const(4), p.Type.IntS32)))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Mul(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      p.Type.IntS32,
      List(i32("i"))
    )
    assertEquals(
      out.collectFirst { case p.Stmt.Var(`b`, Some(e), _) => e },
      Some(p.Expr.IntrOp(p.Intr.Mul(selectT(i), p.Term.IntS32Const(8), p.Type.IntS32)))
    )
  }

  test("reassociation stops at a store: a=x+1; buf[0]=9; b=a+2 stays a+2") {
    val x   = named("x", p.Type.IntS32); val a = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32)
    val buf = named("buf", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.IntrOp(p.Intr.Add(selectT(x), p.Term.IntS32Const(1), p.Type.IntS32)))),
        p.Stmt.Update(selectT(buf), p.Term.IntS32Const(0), p.Term.IntS32Const(9)),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      p.Type.IntS32,
      List(i32("x"), p.Arg(buf))
    )
    assertEquals(
      out.collectFirst { case p.Stmt.Var(`b`, Some(e), _) => e },
      Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))
    )
  }

  // --- common subexpression elimination ---

  private def mulCount(body: List[p.Stmt]): Int =
    body.flatMap(_.collectWhere[p.Expr] { case e @ p.Expr.IntrOp(_: p.Intr.Mul) => e }).size

  test("CSE dedups an identical pure computation") {
    val x   = named("x", p.Type.IntS32); val y = named("y", p.Type.IntS32)
    val a   = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32); val s = named("s", p.Type.IntS32)
    val mul = p.Expr.IntrOp(p.Intr.Mul(selectT(x), selectT(y), p.Type.IntS32))
    val out = pe(
      List(
        p.Stmt.Var(a, Some(mul)),
        p.Stmt.Var(b, Some(mul)),
        p.Stmt.Var(s, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), selectT(b), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(s)))
      ),
      p.Type.IntS32,
      List(i32("x"), i32("y"))
    )
    assertEquals(mulCount(out), 1, out.map(_.repr).mkString("\n"))
  }

  test("CSE does not reuse across a store to an operand") {
    val x   = named("x", p.Type.IntS32); val y = named("y", p.Type.IntS32)
    val a   = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32); val s = named("s", p.Type.IntS32)
    val mul = p.Expr.IntrOp(p.Intr.Mul(selectT(x), selectT(y), p.Type.IntS32))
    val out = pe(
      List(
        p.Stmt.Var(a, Some(mul)),
        p.Stmt.Mut(selectT(x), p.Expr.Alias(p.Term.IntS32Const(5))), // x reassigned -> a's x*y is stale
        p.Stmt.Var(b, Some(mul)),
        p.Stmt.Var(s, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), selectT(b), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(s)))
      ),
      p.Type.IntS32,
      List(i32("x"), i32("y"))
    )
    assertEquals(mulCount(out), 2, out.map(_.repr).mkString("\n"))
  }

  // --- memory-read operands are excluded from CSE and reassoc (phase-independent soundness) ---

  test("CSE does not dedup an expression that reads memory") {
    val pp    = named("pp", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val y     = named("y", p.Type.IntS32)
    val a     = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32); val s = named("s", p.Type.IntS32)
    val deref = p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)
    val mul   = p.Expr.IntrOp(p.Intr.Mul(deref, selectT(y), p.Type.IntS32))
    val out = pe(
      List(
        p.Stmt.Var(a, Some(mul)),
        p.Stmt.Var(b, Some(mul)),
        p.Stmt.Var(s, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), selectT(b), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(s)))
      ),
      p.Type.IntS32,
      List(p.Arg(pp), i32("y"))
    )
    assertEquals(mulCount(out), 2, out.map(_.repr).mkString("\n"))
  }

  test("reassociation does not chain through a memory-read base") {
    val pp    = named("pp", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global))
    val a     = named("a", p.Type.IntS32); val b = named("b", p.Type.IntS32)
    val deref = p.Term.Select(pp, List(p.PathStep.Deref), p.Type.IntS32)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.IntrOp(p.Intr.Add(deref, p.Term.IntS32Const(1), p.Type.IntS32)))),
        p.Stmt.Var(b, Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))),
        p.Stmt.Return(p.Expr.Alias(selectT(b)))
      ),
      p.Type.IntS32,
      List(p.Arg(pp))
    )
    assertEquals(
      out.collectFirst { case p.Stmt.Var(`b`, Some(e), _) => e },
      Some(p.Expr.IntrOp(p.Intr.Add(selectT(a), p.Term.IntS32Const(2), p.Type.IntS32)))
    )
  }

  test("idempotent: running twice yields the same program") {
    val v = named("v", p.Type.IntS32)
    val body = List(
      p.Stmt.Var(v, Some(p.Expr.IntrOp(p.Intr.Add(p.Term.IntS32Const(1), p.Term.IntS32Const(2), p.Type.IntS32)))),
      p.Stmt.Return(p.Expr.Alias(selectT(v)))
    )
    val once  = PartialEval()(program(entry(body = body).copy(rtn = p.Type.IntS32)), NoopLog)
    val twice = PartialEval()(once, NoopLog)
    assertEquals(once.entry.body, twice.entry.body)
  }

  // --- one walk reaches the combined fixpoint the retired pass sequence needed ordering for ---

  test("fully folds a fused const/alias/dead-code/dead-loop program to a single constant") {
    val a = named("a", p.Type.IntS32)
    val b = named("b", p.Type.IntS32)
    val c = named("c", p.Type.IntS32)
    val u = named("u", p.Type.Unit0)
    val out = pe(
      List(
        p.Stmt.Var(a, Some(p.Expr.Alias(p.Term.IntS32Const(10)))),
        p.Stmt.Var(b, Some(p.Expr.Alias(selectT(a)))), // copy prop
        p.Stmt.Var(c, Some(p.Expr.IntrOp(p.Intr.Add(selectT(b), p.Term.IntS32Const(5), p.Type.IntS32)))),
        p.Stmt.Var(u, Some(p.Expr.Alias(p.Term.Unit0Const))),       // dead unit
        p.Stmt.While(p.Term.Bool1Const(false), List(p.Stmt.Break)), // dead loop
        p.Stmt.Return(p.Expr.Alias(selectT(c)))
      ),
      rtn = p.Type.IntS32
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(15)))))
  }

  // --- #empty_struct_storage dead-store peephole ---

  test("drops a store through a #empty_struct_storage padding field") {
    val s = named("s", p.Type.Ptr(p.Type.Struct(sym("S"), Nil), p.Type.Space.Global))
    val padding: p.Term.Select =
      p.Term.Select(s, List(p.PathStep.Field(p.Conventions.EmptyStructStorageField)), p.Type.IntS8)
    val out = pe(
      List(
        p.Stmt.Mut(padding, p.Expr.Alias(p.Term.IntS8Const(0))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      args = List(p.Arg(s))
    )
    assertEquals(out, List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
  }
}
