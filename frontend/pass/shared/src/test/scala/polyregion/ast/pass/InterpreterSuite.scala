package polyregion.ast.pass

import polyregion.ast.PolyAST as p
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V

class InterpreterSuite extends munit.FunSuite {
  import PassTest.*

  private val i32 = p.Type.IntS32
  private val f32 = p.Type.Float32
  private val g   = p.Type.Space.Global

  private def onDevice[A](vm: Interpreter.Vm)(f: => A): A = {
    vm.deviceMode = true
    try f
    finally vm.deviceMode = false
  }

  private def ret(e: p.Expr)                                    = p.Stmt.Return(e)
  private def vlet(n: p.Named, e: p.Expr, mut: Boolean = false) = p.Stmt.Var(n, Some(e), mut)
  private def add(a: p.Term, b: p.Term, t: p.Type = i32)        = p.Expr.IntrOp(p.Intr.Add(a, b, t))
  private def mul(a: p.Term, b: p.Term, t: p.Type = i32)        = p.Expr.IntrOp(p.Intr.Mul(a, b, t))
  private def alias(t: p.Term)                                  = p.Expr.Alias(t)

  test("scalar arithmetic and direct call") {
    val sq = fn("sq", List(arg("x", i32)), i32, List(ret(mul(selectT("x", i32), selectT("x", i32)))))
    val vm = Interpreter.Vm(program(entry(), List(sq)))
    assertEquals(vm.call("sq", List(i32 -> V.I(7))), V.I(49))
  }

  test("ForRange accumulator loop") {
    val sum = fn(
      "sum",
      List(arg("n", i32)),
      i32,
      List(
        vlet(named("acc", i32), alias(p.Term.IntS32Const(0)), mut = true),
        p.Stmt.ForRange(
          named("i", i32),
          p.Term.IntS32Const(0),
          selectT("n", i32),
          p.Term.IntS32Const(1),
          List(p.Stmt.Mut(selectT("acc", i32), add(selectT("acc", i32), selectT("i", i32))))
        ),
        ret(alias(selectT("acc", i32)))
      )
    )
    val vm = Interpreter.Vm(program(entry(), List(sum)))
    assertEquals(vm.call("sum", List(i32 -> V.I(5))), V.I(10))
  }

  test("struct + pointer memory: b->p[0] += 1") {
    val box                 = p.StructDef(sym("Box"), Nil, List(named("p", p.Type.Ptr(i32, g))), Nil)
    val boxT                = p.Type.Struct(sym("Box"), Nil)
    val b                   = named("b", p.Type.Ptr(boxT, g))
    val pSel: p.Term.Select = p.Term.Select(b, List(p.PathStep.Field("p")), p.Type.Ptr(i32, g))
    val inc = fn(
      "inc",
      List(p.Arg(b)),
      p.Type.Unit0,
      List(
        vlet(named("cur", i32), p.Expr.Index(pSel, p.Term.IntS64Const(0), i32)),
        vlet(named("nx", i32), add(selectT("cur", i32), p.Term.IntS32Const(1))),
        p.Stmt.Update(pSel, p.Term.IntS64Const(0), selectT("nx", i32)),
        ret(alias(p.Term.Unit0Const))
      )
    )
    val vm  = Interpreter.Vm(program(entry(), List(inc), List(box)))
    val arr = vm.allocOf(i32)
    vm.store(arr, V.I(41), i32)
    val boxAddr = vm.allocOf(boxT)
    vm.store(boxAddr, V.I(arr), p.Type.Ptr(i32, g))
    vm.call("inc", List(p.Type.Ptr(boxT, g) -> V.I(boxAddr)))
    assertEquals(vm.load(arr, i32), V.I(42))
  }

  test("float multiply") {
    val sq = fn("fsq", List(arg("x", f32)), f32, List(ret(mul(selectT("x", f32), selectT("x", f32), f32))))
    assertEquals(Interpreter.Vm(program(entry(), List(sq))).call("fsq", List(f32 -> V.D(3.0))), V.D(9.0))
  }

  test("ConstantFold preserves semantics") {
    val k = fn(
      "k",
      Nil,
      i32,
      List(
        vlet(named("a", i32), add(p.Term.IntS32Const(2), p.Term.IntS32Const(3))),
        vlet(named("b", i32), mul(selectT("a", i32), p.Term.IntS32Const(4))),
        ret(alias(selectT("b", i32)))
      )
    )
    val in  = program(entry(), List(k))
    val out = ConstantFold(in, NoopLog)
    assertEquals(Interpreter.Vm(in).call("k", Nil), Interpreter.Vm(out).call("k", Nil))
    assertEquals(Interpreter.Vm(out).call("k", Nil), V.I(20))
  }

  test("device-mode guard faults on an unmarshalled host pointer") {
    val f32p    = p.Type.Ptr(f32, g)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("xs", f32p), named("n", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    def fld(name: String, t: p.Type): p.Term.Select = p.Term.Select(thisN, List(p.PathStep.Field(name)), t)
    val body = List(
      vlet(named("acc", f32), alias(p.Term.Float32Const(0f)), mut = true),
      p.Stmt.ForRange(
        named("i", i32),
        p.Term.IntS32Const(0),
        fld("n", i32),
        p.Term.IntS32Const(1),
        List(
          vlet(named("x", f32), p.Expr.Index(fld("xs", f32p), selectT("i", i32), f32)),
          p.Stmt.Mut(selectT("acc", f32), add(selectT("acc", f32), selectT("x", f32), f32))
        )
      ),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(capture))
    val vm = Interpreter.Vm(in)
    val xs = vm.allocOf(f32, 1)
    vm.store(xs, V.D(1.0), f32)
    val cap = vm.allocOf(capT)
    vm.store(cap + vm.offsetOf(capName, "xs"), V.I(xs), f32p)
    vm.store(cap + vm.offsetOf(capName, "n"), V.I(1), i32)
    // a real device kernel run against the unmirrored host capture must trip the host-access guard
    intercept[RuntimeException] {
      onDevice(vm)(vm.call(p.Conventions.EntryName, List(p.Type.Ptr(capT, g) -> V.I(cap))))
    }
  }

}
