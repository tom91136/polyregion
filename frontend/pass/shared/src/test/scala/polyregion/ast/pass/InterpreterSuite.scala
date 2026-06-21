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

  private def differential[A](
      in: p.Program,
      capT: p.Type,
      pass: p.Program => p.Program
  )(setup: Interpreter.Vm => Long)(observe: (Interpreter.Vm, Long) => A): (A, A) = {
    val capPtr = p.Type.Ptr(capT, g)
    val vmD    = Interpreter.Vm(in)
    val capD   = setup(vmD)
    vmD.call(p.Conventions.EntryName, List(capPtr -> V.I(capD)))
    val direct = observe(vmD, capD)

    val out = pass(in)
    val vmM = Interpreter.Vm(out)
    val sma = Interpreter.Sma(vmM)
    vmM.foreign = sma.handler
    val capM = setup(vmM)
    val size = vmM.sizeOf(capT)
    val i8p  = p.Type.Ptr(p.Type.IntS8, g)
    val dev  = vmM.call(preludeName(""), List(i8p -> V.I(capM), p.Type.IntU64 -> V.I(size)))
    onDevice(vmM)(vmM.call(p.Conventions.EntryName, List(capPtr -> dev)))
    vmM.call(postludeName(""), List(i8p -> V.I(capM), p.Type.IntU64 -> V.I(size)))
    (direct, observe(vmM, capM))
  }

  // capture {result, xs, n}; kernel sums xs[0..n) into *result. xs = [1,2,3,4] -> 10.0
  private def sumOverXs: (p.Program, p.Type, Interpreter.Vm => Long, (Interpreter.Vm, Long) => V) = {
    val f32p    = p.Type.Ptr(f32, g)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("result", f32p), named("xs", f32p), named("n", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld     = fieldOf(thisN)
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
      p.Stmt.Update(fld("result", f32p), p.Term.IntS64Const(0), selectT("acc", f32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(capture))
    def setup(vm: Interpreter.Vm): Long = {
      val xs = vm.allocOf(f32, 4)
      (0 until 4).foreach(k => vm.store(xs + k * 4, V.D((k + 1).toDouble), f32))
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + vm.offsetOf(capName, "xs"), V.I(xs), f32p)
      vm.store(cap + vm.offsetOf(capName, "n"), V.I(4), i32)
      cap
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)
    (in, capT, setup, result)
  }

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

  test("ConstantFold preserves semantics (differential)") {
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

  test("Mirror round-trips a pointer capture through a simulated device (differential)") {
    val (in, capT, setup, result) = sumOverXs
    val (direct, mirrored)        = differential(in, capT, Mirror()(_, NoopLog))(setup)(result)
    assertEquals(direct, V.D(10.0))
    assertEquals(mirrored, direct)
  }

  test("device-mode guard faults on an unmarshalled host pointer (the model has teeth)") {
    val f32p    = p.Type.Ptr(f32, g)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("xs", f32p), named("n", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld     = fieldOf(thisN)
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

  test("Mirror deep-copies a 2-level pointer matrix and reads mutations back (differential)") {
    val f32p    = p.Type.Ptr(f32, g)
    val f32pp   = p.Type.Ptr(f32p, g)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("rows", f32pp), named("nr", i32), named("nc", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld     = fieldOf(thisN)
    val nr      = 2
    val nc      = 3
    val body = List(
      p.Stmt.ForRange(
        named("i", i32),
        p.Term.IntS32Const(0),
        fld("nr", i32),
        p.Term.IntS32Const(1),
        List(
          vlet(named("row", f32p), p.Expr.Index(fld("rows", f32pp), selectT("i", i32), f32p)),
          p.Stmt.ForRange(
            named("j", i32),
            p.Term.IntS32Const(0),
            fld("nc", i32),
            p.Term.IntS32Const(1),
            List(
              vlet(named("v", f32), p.Expr.Index(selectT("row", f32p), selectT("j", i32), f32)),
              vlet(named("v2", f32), mul(selectT("v", f32), p.Term.Float32Const(2f), f32)),
              p.Stmt.Update(selectT("row", f32p), selectT("j", i32), selectT("v2", f32))
            )
          )
        )
      ),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(capture))

    def setup(vm: Interpreter.Vm): Long = {
      val rows = vm.allocOf(f32p, nr.toLong)
      (0 until nr).foreach { i =>
        val row = vm.allocOf(f32, nc.toLong)
        (0 until nc).foreach(j => vm.store(row + j * 4, V.D((i * nc + j + 1).toDouble), f32))
        vm.store(rows + i * 8, V.I(row), f32p)
      }
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "rows"), V.I(rows), f32pp)
      vm.store(cap + vm.offsetOf(capName, "nr"), V.I(nr), i32)
      vm.store(cap + vm.offsetOf(capName, "nc"), V.I(nc), i32)
      cap
    }
    def dump(vm: Interpreter.Vm, cap: Long): List[Double] = {
      val rows = vm.loadPtr(cap + vm.offsetOf(capName, "rows"))
      (0 until nr).toList.flatMap { i =>
        val row = vm.loadPtr(rows + i * 8)
        (0 until nc).toList.map(j => vm.load(row + j * 4, f32) match { case V.D(d) => d; case _ => Double.NaN })
      }
    }

    val (direct, mirrored) = differential(in, capT, Mirror()(_, NoopLog))(setup)(dump)
    assertEquals(direct, List(2.0, 4.0, 6.0, 8.0, 10.0, 12.0))
    assertEquals(mirrored, direct)
  }

  test("Mirror deep-copies a 3-level pointer cube (differential)") {
    val f32p    = p.Type.Ptr(f32, g)
    val f32pp   = p.Type.Ptr(f32p, g)
    val f32ppp  = p.Type.Ptr(f32pp, g)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture =
      p.StructDef(capName, Nil, List(named("result", f32p), named("cube", f32ppp), named("d", i32)), Nil)
    val thisN = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld   = fieldOf(thisN)
    val d     = 2
    def loop(ind: String, body: List[p.Stmt]) =
      p.Stmt.ForRange(named(ind, i32), p.Term.IntS32Const(0), fld("d", i32), p.Term.IntS32Const(1), body)
    val body = List(
      vlet(named("acc", f32), alias(p.Term.Float32Const(0f)), mut = true),
      loop(
        "i",
        List(
          vlet(named("plane", f32pp), p.Expr.Index(fld("cube", f32ppp), selectT("i", i32), f32pp)),
          loop(
            "j",
            List(
              vlet(named("row", f32p), p.Expr.Index(selectT("plane", f32pp), selectT("j", i32), f32p)),
              loop(
                "k",
                List(
                  vlet(named("x", f32), p.Expr.Index(selectT("row", f32p), selectT("k", i32), f32)),
                  p.Stmt.Mut(selectT("acc", f32), add(selectT("acc", f32), selectT("x", f32), f32))
                )
              )
            )
          )
        )
      ),
      p.Stmt.Update(fld("result", f32p), p.Term.IntS64Const(0), selectT("acc", f32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(capture))

    def setup(vm: Interpreter.Vm): Long = {
      var v    = 1
      val cube = vm.allocOf(f32pp, d.toLong)
      (0 until d).foreach { i =>
        val plane = vm.allocOf(f32p, d.toLong)
        (0 until d).foreach { j =>
          val row = vm.allocOf(f32, d.toLong)
          (0 until d).foreach { k =>
            vm.store(row + k * 4, V.D(v.toDouble), f32); v += 1
          }
          vm.store(plane + j * 8, V.I(row), f32p)
        }
        vm.store(cube + i * 8, V.I(plane), f32pp)
      }
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + vm.offsetOf(capName, "cube"), V.I(cube), f32ppp)
      vm.store(cap + vm.offsetOf(capName, "d"), V.I(d), i32)
      cap
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)

    val (direct, mirrored) = differential(in, capT, Mirror()(_, NoopLog))(setup)(result)
    assertEquals(direct, V.D(36.0))
    assertEquals(mirrored, direct)
  }

  test("Mirror patches a pointer nested inside by-value structs (std::vector shape, differential)") {
    val f32p    = p.Type.Ptr(f32, g)
    val implSym = sym("vec_impl")
    val implT   = p.Type.Struct(implSym, Nil)
    val vecSym  = sym("vec")
    val vecT    = p.Type.Struct(vecSym, Nil)
    val implDef = p.StructDef(implSym, Nil, List(named("data", f32p), named("n", i32)), Nil)
    val vecDef  = p.StructDef(vecSym, Nil, List(named("impl", implT)), Nil)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capDef  = p.StructDef(capName, Nil, List(named("result", f32p), named("v", vecT)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val dataSel =
      p.Term.Select(thisN, List(p.PathStep.Field("v"), p.PathStep.Field("impl"), p.PathStep.Field("data")), f32p)
    val nSel = p.Term.Select(thisN, List(p.PathStep.Field("v"), p.PathStep.Field("impl"), p.PathStep.Field("n")), i32)
    val resSel: p.Term.Select = p.Term.Select(thisN, List(p.PathStep.Field("result")), f32p)
    val body = List(
      vlet(named("acc", f32), alias(p.Term.Float32Const(0f)), mut = true),
      p.Stmt.ForRange(
        named("i", i32),
        p.Term.IntS32Const(0),
        nSel,
        p.Term.IntS32Const(1),
        List(
          vlet(named("x", f32), p.Expr.Index(dataSel, selectT("i", i32), f32)),
          p.Stmt.Mut(selectT("acc", f32), add(selectT("acc", f32), selectT("x", f32), f32))
        )
      ),
      p.Stmt.Update(resSel, p.Term.IntS64Const(0), selectT("acc", f32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(capDef, vecDef, implDef))

    def setup(vm: Interpreter.Vm): Long = {
      val n    = 4
      val data = vm.allocOf(f32, n.toLong)
      (0 until n).foreach(k => vm.store(data + k * 4, V.D((k + 1).toDouble), f32))
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap     = vm.allocOf(capT)
      val implOff = vm.offsetOf(capName, "v") + vm.offsetOf(vecSym, "impl")
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + implOff + vm.offsetOf(implSym, "data"), V.I(data), f32p)
      vm.store(cap + implOff + vm.offsetOf(implSym, "n"), V.I(n), i32)
      cap
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)

    val (direct, mirrored) = differential(in, capT, Mirror()(_, NoopLog))(setup)(result)
    assertEquals(direct, V.D(10.0))
    assertEquals(mirrored, direct)
  }

  test("Mirror deep-copies an array of structs-with-pointers element-by-element (differential)") {
    val f32p    = p.Type.Ptr(f32, g)
    val nodeSym = sym("Node")
    val nodeT   = p.Type.Struct(nodeSym, Nil)
    val nodeP   = p.Type.Ptr(nodeT, g)
    val node    = p.StructDef(nodeSym, Nil, List(named("v", f32), named("w", f32p)), Nil)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("result", f32p), named("items", nodeP), named("n", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld     = fieldOf(thisN)
    def item(i: String, field: String, t: p.Type) =
      p.Term.Select(
        thisN,
        List(p.PathStep.Field("items"), p.PathStep.IndexDyn(selectT(i, i32)), p.PathStep.Field(field)),
        t
      )
    val n = 4
    val body = List(
      vlet(named("acc", f32), alias(p.Term.Float32Const(0f)), mut = true),
      p.Stmt.ForRange(
        named("i", i32),
        p.Term.IntS32Const(0),
        fld("n", i32),
        p.Term.IntS32Const(1),
        List(
          vlet(named("vi", f32), alias(item("i", "v", f32))),
          vlet(named("wi", f32), p.Expr.Index(item("i", "w", f32p), p.Term.IntS64Const(0), f32)),
          vlet(named("s", f32), add(selectT("vi", f32), selectT("wi", f32), f32)),
          p.Stmt.Mut(selectT("acc", f32), add(selectT("acc", f32), selectT("s", f32), f32))
        )
      ),
      p.Stmt.Update(fld("result", f32p), p.Term.IntS64Const(0), selectT("acc", f32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(node, capture))

    def setup(vm: Interpreter.Vm): Long = {
      val nodeSz = vm.sizeOf(nodeT)
      val items  = vm.allocOf(nodeT, n.toLong)
      (0 until n).foreach { i =>
        val w = vm.allocOf(f32, 1)
        vm.store(w, V.D((i + 1) * 10.0), f32)
        vm.store(items + i * nodeSz + vm.offsetOf(nodeSym, "v"), V.D((i + 1).toDouble), f32)
        vm.store(items + i * nodeSz + vm.offsetOf(nodeSym, "w"), V.I(w), f32p)
      }
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + vm.offsetOf(capName, "items"), V.I(items), nodeP)
      vm.store(cap + vm.offsetOf(capName, "n"), V.I(n), i32)
      cap
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)

    val (direct, mirrored) = differential(in, capT, Mirror()(_, NoopLog))(setup)(result)
    // sum_i (i+1) + (i+1)*10 = 11 * (1+2+3+4) = 110; element [1..] needing their w patched is the regression
    assertEquals(direct, V.D(110.0))
    assertEquals(mirrored, direct)
  }

  test("Mirror deep-copies an array of structs with NESTED pointers (std::string shape, differential)") {
    // mimic libstdc++ std::string: pointer nested in a by-value sub-struct (_Alloc_hider), + a length
    // member + a 16-byte SSO union, total 32 bytes
    val f32p     = p.Type.Ptr(f32, g)
    val i64      = p.Type.IntS64
    val innerSym = sym("Hider")
    val innerT   = p.Type.Struct(innerSym, Nil)
    val inner    = p.StructDef(innerSym, Nil, List(named("w", f32p)), Nil)
    val nodeSym  = sym("Str")
    val nodeT    = p.Type.Struct(nodeSym, Nil)
    val nodeP    = p.Type.Ptr(nodeT, g)
    val node = p.StructDef(
      nodeSym,
      Nil,
      List(named("d", innerT), named("len", i64), named("sso0", i64), named("sso1", i64)),
      Nil
    )
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("result", f32p), named("items", nodeP), named("n", i32)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))
    val fld     = fieldOf(thisN)
    val n       = 4
    val body = List(
      vlet(named("acc", f32), alias(p.Term.Float32Const(0f)), mut = true),
      p.Stmt.ForRange(
        named("i", i32),
        p.Term.IntS32Const(0),
        fld("n", i32),
        p.Term.IntS32Const(1),
        List(
          vlet(
            named("wi", f32),
            p.Expr.Index(
              p.Term.Select(
                thisN,
                List(
                  p.PathStep.Field("items"),
                  p.PathStep.IndexDyn(selectT("i", i32)),
                  p.PathStep.Field("d"),
                  p.PathStep.Field("w")
                ),
                f32p
              ),
              p.Term.IntS64Const(0),
              f32
            )
          ),
          p.Stmt.Mut(selectT("acc", f32), add(selectT("acc", f32), selectT("wi", f32), f32))
        )
      ),
      p.Stmt.Update(fld("result", f32p), p.Term.IntS64Const(0), selectT("acc", f32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(inner, node, capture))

    def setup(vm: Interpreter.Vm): Long = {
      val nodeSz = vm.sizeOf(nodeT)
      val items  = vm.allocOf(nodeT, n.toLong)
      (0 until n).foreach { i =>
        val w = vm.allocOf(f32, 1)
        vm.store(w, V.D((i + 1) * 10.0), f32)
        vm.store(items + i * nodeSz + vm.offsetOf(nodeSym, "d") + vm.offsetOf(innerSym, "w"), V.I(w), f32p)
      }
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + vm.offsetOf(capName, "items"), V.I(items), nodeP)
      vm.store(cap + vm.offsetOf(capName, "n"), V.I(n), i32)
      cap
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)

    val (direct, mirrored) = differential(in, capT, Mirror()(_, NoopLog))(setup)(result)
    assertEquals(direct, V.D(100.0)) // 10+20+30+40
    assertEquals(mirrored, direct)
  }

  test("Mirror deep-copies a linked list via mirror_graph and reads node mutations back (differential)") {
    val f32p    = p.Type.Ptr(f32, g)
    val nodeSym = sym("Node")
    val nodeT   = p.Type.Struct(nodeSym, Nil)
    val nodeP   = p.Type.Ptr(nodeT, g)
    val node    = p.StructDef(nodeSym, Nil, List(named("value", f32), named("next", nodeP)), Nil)
    val capName = sym("Capture")
    val capT    = p.Type.Struct(capName, Nil)
    val capture = p.StructDef(capName, Nil, List(named("result", f32p), named("head", nodeP)), Nil)
    val thisN   = named(p.Conventions.ThisReceiver, p.Type.Ptr(capT, g))

    val i64 = p.Type.IntS64
    val acc = named("acc", f32)
    val pv  = named("p", nodeP)
    val v1  = named("_v1", i64)
    val v2  = named("_v2", p.Type.Bool1)
    val v3  = named("_v3", p.Type.Bool1)
    val nv  = named("_nv", f32)
    val v5  = named("_v5", i64)
    val v6  = named("_v6", p.Type.Bool1)
    // walks via real (patched) device pointers - Mirror does not rewrite the kernel to indices
    val body = List(
      p.Stmt.Var(acc, Some(p.Expr.Alias(p.Term.Float32Const(0f))), isMutable = true),
      p.Stmt.Var(pv, Some(p.Expr.Alias(p.Term.Select(thisN, List(p.PathStep.Field("head")), nodeP))), isMutable = true),
      p.Stmt.Var(v1, Some(p.Expr.Cast(p.Term.Select(pv, Nil, nodeP), i64)), isMutable = false),
      p.Stmt.Var(
        v2,
        Some(p.Expr.IntrOp(p.Intr.LogicNeq(p.Term.Select(v1, Nil, i64), p.Term.IntS64Const(0)))),
        isMutable = false
      ),
      p.Stmt.Var(v3, Some(p.Expr.Alias(p.Term.Select(v2, Nil, p.Type.Bool1))), isMutable = true),
      p.Stmt.While(
        p.Term.Select(v3, Nil, p.Type.Bool1),
        List(
          p.Stmt.Mut(
            p.Term.Select(acc, Nil, f32),
            add(p.Term.Select(acc, Nil, f32), p.Term.Select(pv, List(p.PathStep.Field("value")), f32), f32)
          ),
          // double the node value in place to exercise read_graph write-back
          p.Stmt.Var(
            nv,
            Some(mul(p.Term.Select(pv, List(p.PathStep.Field("value")), f32), p.Term.Float32Const(2f), f32)),
            isMutable = false
          ),
          p.Stmt
            .Mut(p.Term.Select(pv, List(p.PathStep.Field("value")), f32), p.Expr.Alias(p.Term.Select(nv, Nil, f32))),
          p.Stmt
            .Mut(p.Term.Select(pv, Nil, nodeP), p.Expr.Alias(p.Term.Select(pv, List(p.PathStep.Field("next")), nodeP))),
          p.Stmt.Var(v5, Some(p.Expr.Cast(p.Term.Select(pv, Nil, nodeP), i64)), isMutable = false),
          p.Stmt.Var(
            v6,
            Some(p.Expr.IntrOp(p.Intr.LogicNeq(p.Term.Select(v5, Nil, i64), p.Term.IntS64Const(0)))),
            isMutable = false
          ),
          p.Stmt.Mut(p.Term.Select(v3, Nil, p.Type.Bool1), p.Expr.Alias(p.Term.Select(v6, Nil, p.Type.Bool1)))
        )
      ),
      p.Stmt.Update(
        p.Term.Select(thisN, List(p.PathStep.Field("result")), f32p),
        p.Term.IntS64Const(0),
        p.Term.Select(acc, Nil, f32)
      ),
      ret(alias(p.Term.Unit0Const))
    )
    val in = program(entry(args = List(p.Arg(thisN)), body = body), defs = List(node, capture))

    def setup(vm: Interpreter.Vm): Long = {
      val nodes = (1 to 3).map { vv =>
        val n = vm.allocOf(nodeT)
        vm.store(n + vm.offsetOf(nodeSym, "value"), V.D(vv.toDouble), f32)
        n
      }
      nodes.zip(nodes.tail).foreach { case (a, b) => vm.store(a + vm.offsetOf(nodeSym, "next"), V.I(b), nodeP) }
      vm.store(nodes.last + vm.offsetOf(nodeSym, "next"), V.I(0), nodeP)
      val res = vm.allocOf(f32, 1)
      vm.store(res, V.D(0.0), f32)
      val cap = vm.allocOf(capT)
      vm.store(cap + vm.offsetOf(capName, "result"), V.I(res), f32p)
      vm.store(cap + vm.offsetOf(capName, "head"), V.I(nodes.head), nodeP)
      cap
    }
    def values(vm: Interpreter.Vm, cap: Long): List[Double] = {
      var p   = vm.loadPtr(cap + vm.offsetOf(capName, "head"))
      val out = scala.collection.mutable.ListBuffer.empty[Double]
      while (p != 0) {
        out += (vm.load(p + vm.offsetOf(nodeSym, "value"), f32) match { case V.D(d) => d; case _ => Double.NaN })
        p = vm.loadPtr(p + vm.offsetOf(nodeSym, "next"))
      }
      out.toList
    }
    def result(vm: Interpreter.Vm, cap: Long): V = vm.load(vm.loadPtr(cap + vm.offsetOf(capName, "result")), f32)

    val (direct, mirrored) =
      differential(in, capT, Mirror()(_, NoopLog))(setup)((vm, c) => (result(vm, c), values(vm, c)))
    assertEquals(direct._1, V.D(6.0))
    assertEquals(direct._2, List(2.0, 4.0, 6.0))
    assertEquals(mirrored, direct)
  }

  test("FnInline preserves early-return semantics when inlining (differential)") {
    val bool = p.Type.Bool1
    val i32p = p.Type.Ptr(i32, g)
    // mymax(a, b): if (a < b) return b; return a  -- the early-return shape std::max lowers to
    val mymax = fn(
      "mymax",
      List(arg("a", i32), arg("b", i32)),
      i32,
      List(
        vlet(named("c", bool), p.Expr.IntrOp(p.Intr.LogicLt(selectT("a", i32), selectT("b", i32)))),
        p.Stmt.Cond(selectT("c", bool), List(ret(alias(selectT("b", i32)))), Nil),
        ret(alias(selectT("a", i32)))
      )
    )
    def invoke(x: Int, y: Int) =
      p.Expr.Invoke(p.Sym(List("mymax")), Nil, None, List(p.Term.IntS32Const(x), p.Term.IntS32Const(y)), i32)
    val res = named("res", i32p)
    val body = List(
      vlet(named("hi", i32), invoke(3, 7)),
      vlet(named("lo", i32), invoke(9, 2)),
      vlet(named("s", i32), add(selectT("hi", i32), selectT("lo", i32))),
      p.Stmt.Update(selectT("res", i32p), p.Term.IntS64Const(0), selectT("s", i32)),
      ret(alias(p.Term.Unit0Const))
    )
    val in  = program(entry(args = List(p.Arg(res)), body = body), List(mymax))
    val out = FnInline(in, NoopLog)
    assert(out.functions.isEmpty, "FnInline should leave no callable functions")

    def run(prog: p.Program): Long = {
      val vm   = Interpreter.Vm(prog)
      val cell = vm.allocOf(i32, 1)
      vm.call(p.Conventions.EntryName, List(i32p -> V.I(cell)))
      vm.load(cell, i32) match { case V.I(x) => x; case _ => Long.MinValue }
    }
    // mymax(3,7)=7, mymax(9,2)=9 -> 16; a clobbered phi (early return lost) would give 3+9=12
    assertEquals(run(in), 16L)
    assertEquals(run(out), 16L)
  }

}
