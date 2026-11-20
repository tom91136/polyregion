package polyregion.ast.pass

import polyregion.ast.{Interpreter, PolyAST as p, *, given}
import polyregion.ast.Interpreter.V
import polyregion.ast.Traversal.*
import PassTest.*

class EscapeAnalysisSuite extends munit.FunSuite {

  private val i32 = p.Type.IntS32
  private val g   = p.Type.Space.Global

  private val pairSym = sym("Pair")
  private val pairT   = p.Type.Struct(pairSym, Nil)
  private val pairDef = p.StructDef(pairSym, Nil, List(named("a", i32), named("b", i32)), Nil)

  private val outerSym = sym("Outer")
  private val outerT   = p.Type.Struct(outerSym, Nil)
  private val outerDef = p.StructDef(outerSym, Nil, List(named("p", pairT), named("c", i32)), Nil)

  private val ptrI32  = p.Type.Ptr(i32, g)
  private val ptrPtr  = p.Type.Ptr(ptrI32, g)
  private val capture = arg("cap", ptrPtr)

  private def int(v: Int): p.Term                            = p.Term.IntS32Const(v)
  private def alias(t: p.Term)                               = p.Expr.Alias(t)
  private def add(a: p.Term, b: p.Term)                      = p.Expr.IntrOp(p.Intr.Add(a, b, i32))
  private def lt(a: p.Term, b: p.Term)                       = p.Expr.IntrOp(p.Intr.LogicLt(a, b))
  private def v(n: p.Named, e: p.Expr, mut: Boolean = false) = p.Stmt.Var(n, Some(e), mut)
  private def mut(n: p.Named, e: p.Expr)                     = p.Stmt.Mut(selectT(n), e)
  private def field(n: p.Named, f: String, t: p.Type = i32)  = fieldOf(n)(f, t)
  private def set(n: p.Named, f: String, e: p.Expr)          = p.Stmt.Mut(field(n, f), e)
  private def ret(e: p.Expr)                                 = p.Stmt.Return(e)
  private def allocOf(comp: p.Type, size: Int)               = p.Expr.Alloc(comp, int(size), g, p.Region.Opaque)

  private def withResult(e: p.Function, rtn: p.Type): p.Function = e.copy(decl = e.decl.copy(rtn = rtn))

  private def run(
      body: List[p.Stmt],
      args: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil,
      rtn: p.Type = p.Type.Unit0,
      functions: List[p.Function] = Nil,
      refuse: Boolean = false
  ): p.Program = {
    val in = program(
      withResult(entry(args = args, body = body, termCaptures = termCaptures), rtn),
      functions,
      List(pairDef, outerDef)
    )
    EscapeAnalysis(refuse)(in, NoopLog)
  }

  private def refused(
      body: List[p.Stmt],
      args: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil,
      rtn: p.Type = p.Type.Unit0,
      functions: List[p.Function] = Nil
  ): String =
    intercept[IllegalStateException](run(body, args, termCaptures, rtn, functions, refuse = true)).getMessage

  private def structLocals(f: p.Function): List[p.Named] = f.collectWhere[p.Stmt] {
    case p.Stmt.Var(n, _, _) if n.tpe.isInstanceOf[p.Type.Struct] => n
  }

  private def allocs(f: p.Function): List[p.Expr.Alloc] = f.collectWhere[p.Expr] { case a: p.Expr.Alloc => a }

  private def eval(in: p.Program, args: List[(p.Type, V)] = Nil): Long =
    Interpreter.Vm(in).call(p.Conventions.EntryName, args).asI

  private def sumPair: List[p.Stmt] = {
    val t = named("t", pairT)
    val r = named("r", i32)
    List(
      p.Stmt.Var(t, None),
      set(t, "a", alias(int(40))),
      set(t, "b", alias(int(2))),
      v(r, add(field(t, "a"), field(t, "b"))),
      ret(alias(selectT(r)))
    )
  }

  test("struct temporary is replaced by scalar locals") {
    val in  = program(withResult(entry(body = sumPair), i32), Nil, List(pairDef))
    val out = EscapeAnalysis()(in, NoopLog)
    assertEquals(structLocals(out.entry.required), Nil)
    assertEquals(eval(out), eval(in))
    assertEquals(eval(out), 42L)
  }

  test("struct temporary initialised once folds to a constant after PartialEval") {
    val out = PartialEval()(
      EscapeAnalysis()(program(withResult(entry(body = sumPair), i32), Nil, List(pairDef)), NoopLog),
      NoopLog
    )
    assertEquals(out.entry.body, List(ret(alias(int(42)))))
  }

  test("struct returned whole is kept") {
    val t   = named("t", pairT)
    val out = run(List(p.Stmt.Var(t, None), set(t, "a", alias(int(1))), ret(alias(selectT(t)))), rtn = pairT)
    assertEquals(structLocals(out.entry.required).map(_.symbol), List("t"))
  }

  test("struct whose address is taken is kept") {
    val t   = named("t", pairT)
    val ref = named("ref", p.Type.Ptr(pairT, p.Type.Space.Private))
    val out = run(
      List(
        p.Stmt.Var(t, None),
        set(t, "a", alias(int(1))),
        v(ref, p.Expr.RefTo(selectT(t), None, pairT, p.Type.Space.Private, p.Region.Rooted(t))),
        ret(alias(field(t, "a")))
      ),
      rtn = i32
    )
    assertEquals(structLocals(out.entry.required).map(_.symbol), List("t"))
  }

  test("struct written in both branches of a conditional is replaced") {
    val t = named("t", pairT)
    val c = arg("c", p.Type.Bool1)
    val r = named("r", i32)
    val body = List(
      p.Stmt.Var(t, None),
      p.Stmt.Cond(selectT(c.named), List(set(t, "a", alias(int(7)))), List(set(t, "a", alias(int(9))))),
      v(r, alias(field(t, "a"))),
      ret(alias(selectT(r)))
    )
    val in  = program(withResult(entry(args = List(c), body = body), i32), Nil, List(pairDef))
    val out = EscapeAnalysis()(in, NoopLog)
    assertEquals(structLocals(out.entry.required), Nil)
    List(true, false).foreach { flag =>
      assertEquals(eval(out, List(p.Type.Bool1 -> V.I(if (flag) 1 else 0))), if (flag) 7L else 9L)
    }
  }

  test("struct carrying loop state is replaced") {
    val t    = named("t", pairT)
    val n    = arg("n", i32)
    val r    = named("r", i32)
    val more = named("more", p.Type.Bool1)
    val body = List(
      p.Stmt.Var(t, None),
      set(t, "a", alias(int(0))),
      set(t, "b", alias(int(0))),
      v(more, lt(field(t, "a"), selectT(n.named)), mut = true),
      p.Stmt.While(
        selectT(more),
        List(
          set(t, "b", add(field(t, "b"), field(t, "a"))),
          set(t, "a", add(field(t, "a"), int(1))),
          mut(more, lt(field(t, "a"), selectT(n.named)))
        )
      ),
      v(r, alias(field(t, "b"))),
      ret(alias(selectT(r)))
    )
    val in  = program(withResult(entry(args = List(n), body = body), i32), Nil, List(pairDef))
    val out = EscapeAnalysis()(in, NoopLog)
    assertEquals(structLocals(out.entry.required), Nil)
    assertEquals(eval(out, List(i32 -> V.I(5))), eval(in, List(i32 -> V.I(5))))
    assertEquals(eval(out, List(i32 -> V.I(5))), 10L)
  }

  test("nested struct temporaries are flattened") {
    val t                    = named("t", outerT)
    val inner: p.Term.Select = p.Term.Select(t, List(p.PathStep.Field("p"), p.PathStep.Field("a")), i32)
    val r                    = named("r", i32)
    val body = List(
      p.Stmt.Var(t, None),
      p.Stmt.Mut(inner, alias(int(5))),
      set(t, "c", alias(int(6))),
      v(r, add(inner, field(t, "c"))),
      ret(alias(selectT(r)))
    )
    val in  = program(withResult(entry(body = body), i32), Nil, List(pairDef, outerDef))
    val out = EscapeAnalysis()(in, NoopLog)
    assertEquals(structLocals(out.entry.required), Nil)
    assertEquals(eval(out), 11L)
  }

  test("struct temporary built by an inlined constructor and read through a receiver pointer is replaced") {
    val priv = p.Type.Space.Private
    val x    = arg("x", i32)
    val self = named("t", pairT)
    val make = fn(
      "make",
      List(x),
      pairT,
      List(
        p.Stmt.Var(self, None),
        set(self, "a", alias(selectT(x.named))),
        set(self, "b", alias(int(1))),
        ret(alias(selectT(self)))
      )
    )
    val recv  = arg("self", p.Type.Ptr(pairT, priv))
    val readA = p.Term.Select(recv.named, List(p.PathStep.Deref, p.PathStep.Field("a")), i32)
    val readB = p.Term.Select(recv.named, List(p.PathStep.Deref, p.PathStep.Field("b")), i32)
    val total = fn("total", List(recv), i32, List(ret(add(readA, readB))))
    val obj   = named("obj", pairT)
    val ref   = named("ref", p.Type.Ptr(pairT, priv))
    val real = List(
      v(obj, p.Expr.Invoke(p.Type.FnRef(sym("make")), Nil, None, List(int(5)), pairT)),
      v(ref, p.Expr.RefTo(selectT(obj), None, pairT, priv, p.Region.Rooted(obj))),
      v(named("r", i32), p.Expr.Invoke(p.Type.FnRef(sym("total")), Nil, None, List(selectT(ref)), i32)),
      ret(alias(selectT("r")))
    )
    val in  = program(withResult(entry(body = real), i32), List(make, total), List(pairDef))
    val opt = FullOpt()(in, NoopLog)
    val out = EscapeAnalysis()(opt, NoopLog)
    assertEquals(structLocals(out.entry.required), Nil)
    assertEquals(eval(out), 6L)
  }

  test("struct copy is kept when the source is also used whole") {
    val t    = named("t", pairT)
    val copy = named("copy", pairT)
    val out = run(
      List(
        p.Stmt.Var(t, None),
        set(t, "a", alias(int(1))),
        v(copy, alias(selectT(t))),
        v(named("x", i32), alias(field(copy, "a"))),
        ret(alias(selectT(t)))
      ),
      rtn = pairT
    )
    assertEquals(structLocals(out.entry.required).map(_.symbol), List("t", "copy"))
  }

  test("constant-index allocation is replaced by cells") {
    val a = named("a", ptrI32)
    val r = named("r", i32)
    val body = List(
      v(a, allocOf(i32, 2)),
      p.Stmt.Update(selectT(a), int(0), int(40)),
      p.Stmt.Update(selectT(a), int(1), int(2)),
      v(r, p.Expr.Index(selectT(a), int(0), i32)),
      v(named("s", i32), p.Expr.Index(selectT(a), int(1), i32)),
      ret(add(selectT(r), selectT("s")))
    )
    val in  = program(withResult(entry(body = body), i32), Nil, Nil)
    val out = EscapeAnalysis()(in, NoopLog)
    assertEquals(allocs(out.entry.required), Nil)
    assertEquals(eval(out), 42L)
  }

  test("allocation with a dynamic index is kept and not refused") {
    val a = named("a", ptrI32)
    val n = arg("n", i32)
    val body = List(
      v(a, p.Expr.Alloc(i32, selectT(n.named), g, p.Region.Opaque)),
      p.Stmt.Update(selectT(a), int(0), int(1)),
      v(named("x", i32), p.Expr.Index(selectT(a), selectT(n.named), i32)),
      ret(alias(selectT("x")))
    )
    val out = run(body, args = List(n), rtn = i32, refuse = true)
    assertEquals(allocs(out.entry.required).size, 1)
  }

  test("allocation stored into a capture is refused") {
    val a = named("a", ptrI32)
    val body = List(
      v(a, allocOf(i32, 1)),
      p.Stmt.Update(selectT(capture.named), int(0), selectT(a)),
      ret(alias(p.Term.Unit0Const))
    )
    val msg = refused(body, termCaptures = List(capture))
    assert(msg.contains("allocation `a`"), msg)
    assert(msg.contains("stored into capture `cap`"), msg)
  }

  test("allocation returned from the entry is refused") {
    val a   = named("a", ptrI32)
    val msg = refused(List(v(a, allocOf(i32, 4)), ret(alias(selectT(a)))), rtn = ptrI32)
    assert(msg.contains("allocation `a`"), msg)
    assert(msg.contains("returned from the offload entry"), msg)
  }

  test("allocation reaching a capture through an alias is refused") {
    val a = named("a", ptrI32)
    val b = named("b", ptrI32)
    val body =
      List(v(a, allocOf(i32, 1)), v(b, alias(selectT(a))), p.Stmt.Update(selectT(capture.named), int(0), selectT(b)))
    val msg = refused(body :+ ret(alias(p.Term.Unit0Const)), termCaptures = List(capture))
    assert(msg.contains("allocation `a`"), msg)
  }

  test("allocation stored in a branch and a loop is refused") {
    val a = named("a", ptrI32)
    val c = arg("c", p.Type.Bool1)
    val body = List(
      v(a, allocOf(i32, 1)),
      p.Stmt.Cond(selectT(c.named), List(p.Stmt.Update(selectT(capture.named), int(0), selectT(a))), Nil),
      ret(alias(p.Term.Unit0Const))
    )
    val msg = refused(body, args = List(c), termCaptures = List(capture))
    assert(msg.contains("stored into capture `cap`"), msg)
  }

  test("allocation passed to a callee that stores it into a capture is refused") {
    val q    = arg("q", ptrI32)
    val hold = arg("hold", ptrPtr)
    val callee = fn(
      "keep",
      List(q, hold),
      p.Type.Unit0,
      List(p.Stmt.Update(selectT(hold.named), int(0), selectT(q.named)), ret(alias(p.Term.Unit0Const)))
    )
    val a = named("a", ptrI32)
    val call =
      p.Expr.Invoke(p.Type.FnRef(sym("keep")), Nil, None, List(selectT(a), selectT(capture.named)), p.Type.Unit0)
    val msg = refused(
      List(v(a, allocOf(i32, 1)), v(named("u", p.Type.Unit0), call), ret(alias(p.Term.Unit0Const))),
      termCaptures = List(capture),
      functions = List(callee)
    )
    assert(msg.contains("allocation `a`"), msg)
    assert(msg.contains("passed to `keep`"), msg)
  }

  test("allocation passed to a callee that only reads it is not refused") {
    val q      = arg("q", ptrI32)
    val callee = fn("peek", List(q), i32, List(ret(p.Expr.Index(selectT(q.named), int(0), i32))))
    val a      = named("a", ptrI32)
    val call   = p.Expr.Invoke(p.Type.FnRef(sym("peek")), Nil, None, List(selectT(a)), i32)
    val body = List(
      v(a, allocOf(i32, 1)),
      p.Stmt.Update(selectT(a), int(0), int(3)),
      v(named("r", i32), call),
      ret(alias(selectT("r")))
    )
    val out = run(body, rtn = i32, functions = List(callee), refuse = true)
    assertEquals(allocs(out.entry.required).size, 1)
  }

  test("escape through a callee is transitive") {
    val q    = arg("q", ptrI32)
    val hold = arg("hold", ptrPtr)
    val inner = fn(
      "keep",
      List(q, hold),
      p.Type.Unit0,
      List(p.Stmt.Update(selectT(hold.named), int(0), selectT(q.named)), ret(alias(p.Term.Unit0Const)))
    )
    val q2    = arg("q2", ptrI32)
    val hold2 = arg("hold2", ptrPtr)
    val call =
      p.Expr.Invoke(p.Type.FnRef(sym("keep")), Nil, None, List(selectT(q2.named), selectT(hold2.named)), p.Type.Unit0)
    val outer = fn(
      "forward",
      List(q2, hold2),
      p.Type.Unit0,
      List(v(named("u", p.Type.Unit0), call), ret(alias(p.Term.Unit0Const)))
    )
    val a = named("a", ptrI32)
    val top =
      p.Expr.Invoke(p.Type.FnRef(sym("forward")), Nil, None, List(selectT(a), selectT(capture.named)), p.Type.Unit0)
    val msg = refused(
      List(v(a, allocOf(i32, 1)), v(named("u", p.Type.Unit0), top), ret(alias(p.Term.Unit0Const))),
      termCaptures = List(capture),
      functions = List(inner, outer)
    )
    assert(msg.contains("passed to `forward`"), msg)
  }

  test("pointer stored into an escaping allocation escapes with it") {
    val a = named("a", ptrI32)
    val b = named("b", ptrPtr)
    val body = List(
      v(a, allocOf(i32, 1)),
      v(b, allocOf(ptrI32, 1)),
      p.Stmt.Update(selectT(b), int(0), selectT(a)),
      ret(alias(selectT(b)))
    )
    val msg = refused(body, rtn = ptrPtr)
    assert(msg.contains("allocation `a`"), msg)
    assert(msg.contains("stored into escaping allocation `b`"), msg)
    assert(msg.contains("allocation `b`"), msg)
  }

  test("pointer stored into a non-escaping allocation does not escape") {
    val a = named("a", ptrI32)
    val b = named("b", ptrPtr)
    val body = List(
      v(a, allocOf(i32, 1)),
      v(b, allocOf(ptrI32, 1)),
      p.Stmt.Update(selectT(b), int(0), selectT(a)),
      ret(alias(p.Term.Unit0Const))
    )
    run(body, refuse = true)
  }

  test("diagnostic names the source position of the allocation") {
    val pos = p.SourcePosition("Kernel.scala", 12, Some(3))
    val a   = named("a", ptrI32).copy(origin = p.Origin(pos = Some(pos), source = Some("buf")))
    val msg = refused(List(v(a, allocOf(i32, 4)), ret(alias(selectT(a)))), rtn = ptrI32)
    assert(msg.contains("allocation `buf`"), msg)
    assert(msg.contains("Kernel.scala:12:3"), msg)
  }

  test("diagnostic uses the enclosing statement annotation when the name has no position") {
    val a   = named("a", ptrI32)
    val pos = p.SourcePosition("Kernel.scala", 30, None)
    val msg =
      refused(List(p.Stmt.Annotated(v(a, allocOf(i32, 4)), Some(pos), None), ret(alias(selectT(a)))), rtn = ptrI32)
    assert(msg.contains("Kernel.scala:30"), msg)
  }

  test("escaping allocation is left alone when refusal is off") {
    val a   = named("a", ptrI32)
    val out = run(List(v(a, allocOf(i32, 1)), ret(alias(selectT(a)))), rtn = ptrI32)
    assertEquals(allocs(out.entry.required).size, 1)
  }

  test("local-space allocation is not a heap site") {
    val a = named("a", p.Type.Ptr(i32, p.Type.Space.Local))
    val body =
      List(v(a, p.Expr.Alloc(i32, int(4), p.Type.Space.Local, p.Region.Rooted(a))), ret(alias(p.Term.Unit0Const)))
    val out = run(body, refuse = true)
    assertEquals(allocs(out.entry.required).size, 1)
  }

  test("unwritten constant cell reads as zero") {
    val a = named("a", ptrI32)
    val r = named("r", i32)
    val x = named("x", i32)
    val y = named("y", i32)
    val real = List(
      v(a, allocOf(i32, 2)),
      p.Stmt.Update(selectT(a), int(0), int(1)),
      v(x, p.Expr.Index(selectT(a), int(0), i32)),
      v(y, p.Expr.Index(selectT(a), int(1), i32)),
      v(r, add(selectT(x), selectT(y))),
      ret(alias(selectT(r)))
    )
    val out = PartialEval()(run(real, rtn = i32), NoopLog)
    assertEquals(allocs(out.entry.required), Nil)
    assertEquals(out.entry.body, List(ret(alias(int(1)))))
  }

  test("unwritten struct-element cell reads as a zeroed struct") {
    val a = named("a", p.Type.Ptr(pairT, g))
    val t = named("t", pairT)
    val real = List(
      v(a, p.Expr.Alloc(pairT, int(1), g, p.Region.Opaque)),
      v(t, p.Expr.Index(selectT(a), int(0), pairT)),
      ret(alias(field(t, "b")))
    )
    val out = PartialEval()(run(real, rtn = i32), NoopLog)
    assertEquals(allocs(out.entry.required), Nil)
    assertEquals(out.entry.body, List(ret(alias(int(0)))))
  }

  test("allocation of array elements is not replaced") {
    val arr = p.Type.Arr(i32, 2, g)
    val a   = named("a", p.Type.Ptr(arr, g))
    val out = run(
      List(
        v(a, p.Expr.Alloc(arr, int(1), g, p.Region.Opaque)),
        v(named("e", arr), p.Expr.Index(selectT(a), int(0), arr)),
        ret(alias(p.Term.Unit0Const))
      )
    )
    assertEquals(allocs(out.entry.required).size, 1)
  }

  test("reassigned array variable is not replaced") {
    val a = named("a", ptrI32)
    val out = run(
      List(
        v(a, allocOf(i32, 2), mut = true),
        p.Stmt.Update(selectT(a), int(0), int(1)),
        mut(a, allocOf(i32, 4)),
        v(named("r", i32), p.Expr.Index(selectT(a), int(0), i32)),
        ret(alias(selectT("r")))
      ),
      rtn = i32
    )
    assertEquals(allocs(out.entry.required).size, 2)
  }

  test("same-name overloads keep separate escape summaries") {
    val hold = arg("hold", ptrPtr)
    val q32  = arg("q", ptrI32)
    val q64  = arg("q", p.Type.Ptr(p.Type.IntS64, g))
    val keep32 = fn(
      "keep",
      List(q32, hold),
      p.Type.Unit0,
      List(p.Stmt.Update(selectT(hold.named), int(0), selectT(q32.named)), ret(alias(p.Term.Unit0Const)))
    )
    val keep64 = fn("keep", List(q64, hold), p.Type.Unit0, List(ret(alias(p.Term.Unit0Const))))
    val wide   = named("wide", p.Type.Ptr(p.Type.IntS64, g))
    val narrow = named("narrow", ptrI32)
    def call(a: p.Named) =
      p.Expr.Invoke(p.Type.FnRef(sym("keep")), Nil, None, List(selectT(a), selectT(capture.named)), p.Type.Unit0)
    val ok = List(
      v(wide, p.Expr.Alloc(p.Type.IntS64, int(4), g, p.Region.Opaque)),
      v(named("u", p.Type.Unit0), call(wide)),
      ret(alias(p.Term.Unit0Const))
    )
    run(ok, termCaptures = List(capture), functions = List(keep32, keep64), refuse = true)
    val bad = List(v(narrow, allocOf(i32, 4)), v(named("u", p.Type.Unit0), call(narrow)), ret(alias(p.Term.Unit0Const)))
    val msg = refused(bad, termCaptures = List(capture), functions = List(keep32, keep64))
    assert(msg.contains("passed to `keep`"), msg)
  }

  test("allocation stored into a field of a struct parameter escapes through the callee") {
    val holderSym = sym("Holder")
    val holderT   = p.Type.Struct(holderSym, Nil)
    val holderDef = p.StructDef(holderSym, Nil, List(named("p", ptrI32)), Nil)
    val q         = arg("q", ptrI32)
    val h         = arg("h", p.Type.Ptr(holderT, g))
    val store = p.Stmt.Mut(
      p.Term.Select(h.named, List(p.PathStep.Deref, p.PathStep.Field("p")), ptrI32),
      alias(selectT(q.named))
    )
    val callee    = fn("stash", List(q, h), p.Type.Unit0, List(store, ret(alias(p.Term.Unit0Const))))
    val holderCap = arg("holder", p.Type.Ptr(holderT, g))
    val a         = named("a", ptrI32)
    val call =
      p.Expr.Invoke(p.Type.FnRef(sym("stash")), Nil, None, List(selectT(a), selectT(holderCap.named)), p.Type.Unit0)
    val in = program(
      withResult(
        entry(
          body = List(v(a, allocOf(i32, 1)), v(named("u", p.Type.Unit0), call), ret(alias(p.Term.Unit0Const))),
          termCaptures = List(holderCap)
        ),
        p.Type.Unit0
      ),
      List(callee),
      List(holderDef)
    )
    val msg = intercept[IllegalStateException](EscapeAnalysis(refuse = true)(in, NoopLog)).getMessage
    assert(msg.contains("passed to `stash`"), msg)
  }

  test("distinct anonymous allocations are reported separately") {
    val c = arg("c", p.Type.Bool1)
    val msg = refused(
      List(
        p.Stmt.Cond(
          selectT(c.named),
          List(ret(allocOf(i32, 4))),
          List(ret(p.Expr.Alloc(p.Type.IntS64, int(8), g, p.Region.Opaque)))
        )
      ),
      args = List(c),
      rtn = ptrI32
    )
    assertEquals(msg.linesIterator.count(_.contains("escapes:")), 2, msg)
  }

  test("pass is idempotent") {
    val once  = EscapeAnalysis()(program(withResult(entry(body = sumPair), i32), Nil, List(pairDef)), NoopLog)
    val twice = EscapeAnalysis()(once, NoopLog)
    assertEquals(twice, once)
  }

  test("registry builds the pass with its refuse argument") {
    val built = PassRegistry.build(p.Pass.Spec("EscapeAnalysis", List(p.Pass.Arg("refuse", "true"))))
    assertEquals(built.map(_.asInstanceOf[EscapeAnalysis].refuse), Right(true))
  }
}
